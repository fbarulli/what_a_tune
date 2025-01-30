# optimized_model.py
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AutoModel
import traceback
import os
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.optim import AdamW
import copy
from gpu_tracker import GPUTracker  # Add this import
import numpy as np
from model_components import Mixout, SpectralNorm, PoolingHead, FGM, RDropLoss, EMA, FocalLoss
from typing import Dict, List, Optional
from torch.nn.utils import clip_grad_norm_





logger = logging.getLogger(__name__)


class OptimizedModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.model_cache_dir = os.path.join(self.project_root, "results", "model_cache")
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self.validation_outputs = []
        
        logger.info(f"Initializing model with config: {config}")

        # Initialize base model with specific config
        self.model = AutoModel.from_pretrained(
            config['model_name'],
            cache_dir=self.model_cache_dir,
            hidden_dropout_prob=config['hidden_dropout_prob'],
            attention_probs_dropout_prob=config['regularization']['attention_dropout'],
        )
        
        # Initialize weights of unfrozen layers with proper dimension checking
        unfrozen_layers = config['model_architecture']['unfrozen_layers']
        for param in self.model.parameters():
            param.requires_grad = False
        
        for layer in self.model.encoder.layer[-unfrozen_layers:]:
            for name, param in layer.named_parameters():
                param.requires_grad = True
                if hasattr(param, 'data'):
                    # Only apply Xavier initialization to weight matrices
                    if len(param.shape) >= 2:
                        if 'weight' in name:
                            torch.nn.init.xavier_normal_(param.data)
                    elif len(param.shape) == 1:
                        # For biases and 1D parameters
                        torch.nn.init.zeros_(param.data)

        self.pooling_head = PoolingHead(
            self.model.config.hidden_size,
            config['num_labels'],
            dropout_prob=config['hidden_dropout_prob'],
            config=config
        )

        # Initialize pooling head weights with proper dimension checking
        for name, param in self.pooling_head.named_parameters():
            if hasattr(param, 'data'):
                if len(param.shape) >= 2 and 'weight' in name:
                    torch.nn.init.xavier_normal_(param.data)
                elif len(param.shape) == 1:
                    torch.nn.init.zeros_(param.data)

        if config['search_space']['gradient_checkpointing']['values'][0]:
            self.model.gradient_checkpointing_enable()

        self.save_hyperparameters(config)
        
        # Initialize training components
        self.rdrop_loss = RDropLoss(alpha=config['rdrop_alpha'])
        self.automatic_optimization = False
        
        # Initialize regularization with careful scaling
        self.focal_loss = FocalLoss(
            alpha=config['regularization']['focal_alpha'],
            gamma=config['regularization']['focal_gamma']
        )
        
        # Enable or disable regularizations based on config
        self.use_focal = config['regularization']['use_focal']
        self.mixup_alpha = config['regularization']['mixup_alpha']
        
        # Initialize EMA last to ensure proper weight initialization
        if config['regularization']['use_ema']:
            self.ema = EMA(self.model, decay=config['regularization']['ema_decay'])
            self.ema.register()
        
        # Track gradient norms for monitoring
        self.grad_norm_queue = []
        
        # Initialize loss scaling for numerical stability
        self.loss_scaler = torch.cuda.amp.GradScaler(enabled=config['training']['fp16_training'])


    def _prepare_inputs(self, inputs):
        """Prepare inputs with proper type casting and device placement."""
        prepared_inputs = {}
        for k, v in inputs.items():
            if not hasattr(v, 'to'):
                prepared_inputs[k] = v
                continue
                
            if k == 'input_ids':
                prepared_inputs[k] = v.to(self.device, dtype=torch.long, non_blocking=True)
            elif k == 'attention_mask':
                prepared_inputs[k] = v.to(self.device, dtype=torch.long, non_blocking=True)
            else:
                prepared_inputs[k] = v.to(self.device, non_blocking=True)
        return prepared_inputs


    def mixup_data(self, x, y, alpha):
        """Performs mixup on the input and label."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def forward(self, **inputs):
        model_inputs = self._prepare_inputs({
            k: v for k, v in inputs.items() if k not in ['labels']
        })

        outputs = self.model(**model_inputs, output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state
        
        # Apply layer norm before pooling for better stability
        last_hidden_states = F.layer_norm(
            last_hidden_states, 
            [self.model.config.hidden_size]
        )
        
        logits = self.pooling_head(last_hidden_states, model_inputs['attention_mask'])
        return logits

    def _compute_loss(self, logits1, logits2, labels, batch_idx):
        """Compute loss with proper scaling and regularization."""
        # Basic cross entropy loss
        ce_loss = F.cross_entropy(logits1, labels)
        
        # RDrop loss with temperature scaling for better gradients
        rdrop_loss = self.rdrop_loss(
            logits1 / 1.5,  # Temperature scaling
            logits2 / 1.5,  # Temperature scaling
            labels
        )
        
        # Combine losses with careful scaling
        if self.use_focal:
            focal_loss = self.focal_loss(logits1, labels)
            loss = ce_loss + 0.5 * rdrop_loss + 0.1 * focal_loss
        else:
            loss = ce_loss + 0.5 * rdrop_loss
            
        return loss / self.config['training']['gradient_accumulation_steps']

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute a single training step with gradient stabilization.
        
        Args:
            batch: Dictionary containing the batch data
            batch_idx: Index of the current batch
            
        Returns:
            torch.Tensor: The computed loss value
        """
        gpu_tracker = GPUTracker()
        gpu_tracker.track_step("start_of_step", model=self)
        
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        batch = self._prepare_inputs(batch)
        labels = batch.pop('labels')

        # Reset gradients at the start
        optimizer.zero_grad()

        # Forward pass with gradient scaling and stability improvements
        with torch.cuda.amp.autocast(enabled=self.config['training']['fp16_training']):
            if self.mixup_alpha > 0:
                mixed_x, y_a, y_b, lam = self.mixup_data(batch['input_ids'], labels, self.mixup_alpha)
                mixed_attn = batch['attention_mask']
                logits1 = self(**{'input_ids': mixed_x, 'attention_mask': mixed_attn})
            else:
                logits1 = self(**batch)
                y_a, y_b = labels, None

            # R-drop forward pass with stability improvements
            with torch.no_grad():
                logits2 = self(**batch)

            # Label smoothing for better stability
            smoothing = 0.1
            num_classes = logits1.size(-1)
            labels_one_hot = F.one_hot(labels, num_classes).float()
            labels_smooth = (1.0 - smoothing) * labels_one_hot + smoothing / num_classes
            
            # Compute stabilized loss
            loss = F.cross_entropy(logits1, labels_smooth)
            
            if self.rdrop_loss is not None:
                rdrop_loss = self.rdrop_loss(
                    F.log_softmax(logits1 / 2.0, dim=-1),
                    F.log_softmax(logits2 / 2.0, dim=-1),
                    labels
                ) * 0.5
                loss = loss + rdrop_loss

        gpu_tracker.track_step("after_loss", local_vars={'loss': loss})

        # Scale loss and compute gradients
        scaled_loss = self.loss_scaler.scale(loss)
        self.manual_backward(scaled_loss)

        gpu_tracker.track_step("after_backward", model=self)

        # Gradient norm tracking
        all_params = [p for p in self.parameters() if p.requires_grad]
        
        # Update on accumulation step
        if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
            # Track gradient norm before potential clipping
            with torch.no_grad():
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in all_params if p.grad is not None])
                )
                
                # Track gradient statistics
                if len(self.grad_norm_queue) >= 100:
                    self.grad_norm_queue.pop(0)
                self.grad_norm_queue.append(total_norm.item())
            
            # Handle gradient clipping inside scaler context
            clip_value = float(self.config['network_training']['gradient_control']['clip_value'])
            
            # Step with gradient scaler
            self.loss_scaler.step(optimizer)
            self.loss_scaler.update()
            
            # Step scheduler after optimizer
            if scheduler is not None:
                scheduler.step()
            
            # Reset gradients after stepping
            optimizer.zero_grad()

        # Compute metrics for logging
        if self.grad_norm_queue:
            avg_grad_norm = sum(self.grad_norm_queue) / len(self.grad_norm_queue)
            self.log('grad_norm', avg_grad_norm, prog_bar=True)
            self.log('grad_loss_ratio', avg_grad_norm / loss.item(), prog_bar=True)
        
        # Log base metrics
        self.log('train_loss', loss.item(), prog_bar=True, sync_dist=True)
        self.log('learning_rate', optimizer.param_groups[0]['lr'], prog_bar=True)

        return loss



    def on_train_epoch_end(self):
        if self.use_ema:
            self.ema.apply_shadow()
        
        # Existing epoch end code...
        
        if self.use_ema:
            self.ema.restore()

    def validation_step(self, batch, batch_idx):
        if hasattr(self, 'ema') and self.config['regularization']['use_ema']:
            self.ema.apply_shadow()
            
        try:
            batch = self._prepare_inputs(batch)
            
            with torch.cuda.amp.autocast(enabled=self.config['training']['fp16_training']):
                labels = batch.pop('labels')
                logits = self(**batch)
                loss = F.cross_entropy(logits, labels)
                preds = torch.argmax(logits, dim=-1)
                acc = torch.tensor(accuracy_score(labels.cpu(), preds.cpu()))

            self.log('val_loss', loss, prog_bar=True, sync_dist=True)
            self.log('val_acc', acc, prog_bar=True, sync_dist=True)
            
            self.validation_outputs.append({
                'val_loss': loss.detach(),
                'val_acc': acc,
                'logits_mean': logits.mean().item(),
                'logits_std': logits.std().item()
            })
            
        except Exception as e:
            logger.error(f"Error during validation step: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        finally:
            if hasattr(self, 'ema') and self.config['regularization']['use_ema']:
                self.ema.restore()
        
        return {'val_loss': loss, 'val_acc': acc}


    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def on_validation_epoch_end(self):
        try:
            if not self.validation_outputs:
                logger.warning("No validation outputs collected")
                avg_loss = torch.tensor(float('inf'))
                avg_acc = torch.tensor(0.0)
            else:
                avg_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
                avg_acc = torch.stack([x['val_acc'] for x in self.validation_outputs]).mean()

            self.log('avg_val_loss', avg_loss, prog_bar=True, sync_dist=True)
            self.log('avg_val_acc', avg_acc, prog_bar=True, sync_dist=True)
            self.validation_outputs = []
        except Exception as e:
            logger.error(f"Error during validation epoch end: {str(e)}")
            logger.error(traceback.format_exc())
            self.log('avg_val_loss', torch.tensor(float('inf')), prog_bar=True, sync_dist=True)
            self.log('avg_val_acc', torch.tensor(0.0), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """Configure optimizers with proper learning rate scheduling."""
        # Separate parameters for different learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                        if not any(nd in n for nd in no_decay)],
                'weight_decay': float(self.config['weight_decay']),
                'lr': float(self.config['training']['initial_learning_rate'])
            },
            {
                'params': [p for n, p in self.named_parameters() 
                        if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': float(self.config['training']['initial_learning_rate'])
            }
        ]

        # Initialize optimizer with normalized learning rate
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=float(self.config['training']['initial_learning_rate']),
            eps=float(self.config['network_training']['optimizer']['eps']),
            betas=(0.9, 0.999)  # Standard Adam betas
        )

        # Calculate training steps
        total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
        warmup_steps = total_steps // 10  # 10% warmup

        # Create scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(self.config['search_space']['max_lr']['max']),
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,  # Warm up for first 10%
            div_factor=10.0,  # Initial lr = max_lr / 10
            final_div_factor=1e4,  # Min lr = initial_lr / 1e4
            anneal_strategy='linear'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
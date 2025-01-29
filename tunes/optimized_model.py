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

logger = logging.getLogger(__name__)

class Mixout(nn.Module):
    def __init__(self, p=0.5, hidden_size=None, scale_to_dropout=True):
        super().__init__()
        self.p = p
        self.hidden_size = hidden_size
        self.scale_to_dropout = scale_to_dropout
        self.mask = None

    def forward(self, x, scale=1.0):
        if not self.training or self.p == 0:
            return x

        if self.mask is None or self.mask.shape != x.shape:
            self.mask = torch.ones_like(x).bernoulli_(1 - self.p)
            if self.scale_to_dropout:
                self.mask = self.mask / (1-self.p)

        return x * self.mask * scale

class SpectralNorm(nn.Module):
    def __init__(self, module, num_iters=1):
        super().__init__()
        self.module = module
        self.num_iters = num_iters
        self._u = None
        self._v = None

    def _l2normalize(self, x):
        return x / (torch.norm(x) + 1e-12)

    def _spectral_norm(self):
        w = self.module.weight.view(self.module.weight.size(0), -1)

        if self._v is None:
            self._v = self._l2normalize(torch.randn(w.size(1)).to(w.device))
        if self._u is None:
            self._u = self._l2normalize(torch.randn(w.size(0)).to(w.device))

        for _ in range(self.num_iters):
            self._v = self._l2normalize(torch.mv(w.t(), self._u))
            self._u = self._l2normalize(torch.mv(w, self._v))

        sigma = torch.dot(torch.mv(w, self._v), self._u)
        return sigma.detach()

    def forward(self, *args, **kwargs):
        sigma = self._spectral_norm()
        self._weight = self.module.weight / sigma
        original_weight = self.module.weight
        self.module.weight = nn.Parameter(self._weight)
        output = self.module(*args, **kwargs)
        self.module.weight = original_weight
        return output

class PoolingHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_prob, config):
        super().__init__()
        reduction_factor = config['model_architecture']['pooling_head']['dense_reduction_factor']
        enable_spectral_norm = config['model_architecture']['pooling_head']['enable_spectral_norm']
        
        # Define layers with configurable reduction
        dense1_size = hidden_size * 3
        dense2_size = hidden_size // reduction_factor
        
        # Create layers with optional spectral norm
        dense1 = nn.Linear(dense1_size, hidden_size)
        self.dense1 = SpectralNorm(dense1) if enable_spectral_norm else dense1
        
        dense2 = nn.Linear(hidden_size, dense2_size)
        self.dense2 = SpectralNorm(dense2) if enable_spectral_norm else dense2
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(dense2_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(dense2_size, num_classes)

    def forward(self, hidden_states, attention_mask):
        cls_token = hidden_states[:, 0]
        
        # Mean pooling
        mean_pool = (
            torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
            / attention_mask.sum(dim=1, keepdim=True)
        )

        # Max pooling
        max_pool, _ = torch.max(hidden_states * attention_mask.unsqueeze(-1), dim=1)

        # Concatenate and pass through MLP
        pooled = torch.cat([cls_token, mean_pool, max_pool], dim=1)
        x = self.layer_norm1(F.gelu(self.dense1(pooled)))
        x = self.layer_norm2(F.gelu(self.dense2(x)))
        x = self.dropout(x)
        return self.classifier(x)

class FGM():
    def __init__(self, model, config):
        self.model = model
        self.backup = {}
        self.default_epsilon = config['network_training']['fgm']['default_epsilon']
        self.emb_name = config['network_training']['fgm']['emb_name']

    def attack(self, epsilon=None):
        epsilon = epsilon or self.default_epsilon
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class RDropLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits1, logits2, labels):
        ce_loss = F.cross_entropy(logits1, labels)
        kl_loss = self._kl_loss(logits1, logits2)
        return ce_loss + self.alpha * kl_loss

    def _kl_loss(self, p, q):
        p_log_prob = F.log_softmax(p, dim=-1)
        q_prob = F.softmax(q, dim=-1)
        return F.kl_div(p_log_prob, q_prob, reduction='batchmean')

class OptimizedModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.model_cache_dir = os.path.join(self.project_root, "results", "model_cache")
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self.validation_outputs = []

        logger.info(f"Initializing model with config: {config}")

        self.model = AutoModel.from_pretrained(
            config['model_name'],
            cache_dir=self.model_cache_dir
        )

        # Unfreeze specified number of top layers
        unfrozen_layers = config['model_architecture']['unfrozen_layers']
        for param in self.model.parameters():
            param.requires_grad = False
        for layer in self.model.encoder.layer[-unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.pooling_head = PoolingHead(
            self.model.config.hidden_size,
            config['num_labels'],
            dropout_prob=config['hidden_dropout_prob'],
            config=config
        )

        if config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()

        self.save_hyperparameters(config)
        self.fgm = FGM(self.model, config)
        self.rdrop_loss = RDropLoss(alpha=config['rdrop_alpha'])
        self.automatic_optimization = False

    def forward(self, **inputs):
        model_inputs = {k: v for k, v in inputs.items()
                       if k not in ['labels'] and hasattr(v, 'to')}

        outputs = self.model(**model_inputs, output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state
        logits = self.pooling_head(last_hidden_states, inputs['attention_mask'])
        return logits

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        labels = batch.pop('labels')

        fp16_enabled = self.config['training'].get('fp16_training', True)

        with torch.amp.autocast('cuda', enabled=fp16_enabled):
            # Forward pass
            logits1 = self(**batch)
            batch['labels'] = labels

            # R-drop forward pass
            batch2 = copy.deepcopy(batch)
            logits2 = self(**{k: v for k, v in batch2.items() if k != 'labels'})

            # R-Drop loss
            loss = self.rdrop_loss(logits1, logits2, labels)
            self.log('train_loss', loss, prog_bar=True, sync_dist=True)
            self.manual_backward(loss / self.config['training']['gradient_accumulation_steps'], 
                               retain_graph=True)

        if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
            # Adversarial training
            if self.config['training'].get('adv_training', True):
                self.fgm.attack(epsilon=self.config['search_space'].get('adv_epsilon', {}).get('max', 0.5))
                with torch.amp.autocast('cuda', enabled=fp16_enabled):
                    adv_logits = self(**{k: v for k, v in batch.items() if k != 'labels'})
                    adv_loss = self.rdrop_loss(adv_logits, logits2, labels)
                self.manual_backward(adv_loss / self.config['training']['gradient_accumulation_steps'])
                self.fgm.restore()

            # Gradient clipping
            if self.config['training'].get('max_grad_norm', None):
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.config['training']['max_grad_norm']
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return loss

    def validation_step(self, batch, batch_idx):
        try:
            with torch.amp.autocast('cuda', enabled=self.config['training'].get('fp16_training', True)):
                labels = batch.pop('labels')
                inputs = {k: v for k, v in batch.items() if k not in ['labels']}
                logits = self(**inputs)
                loss = F.cross_entropy(logits, labels)
                preds = torch.argmax(logits, dim=-1)
                acc = torch.tensor(accuracy_score(labels.cpu(), preds.cpu()))

            self.log('val_loss', loss, prog_bar=True, sync_dist=True)
            self.log('val_acc', acc, prog_bar=True, sync_dist=True)
            self.validation_outputs.append({
                'val_loss': loss.detach(),
                'val_acc': acc
            })
            return {'val_loss': loss, 'val_acc': acc}
        except Exception as e:
            logger.error(f"Error during validation step: {str(e)}")
            logger.error(traceback.format_exc())
            return None

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
        try:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.named_parameters()
                              if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config['weight_decay']
                },
                {
                    'params': [p for n, p in self.named_parameters()
                              if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]

            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config['training']['initial_learning_rate'],
                eps=self.config['search_space']['adam_epsilon']['max'],
                weight_decay=self.config['weight_decay']
            )

            data_module = self.trainer.datamodule
            steps_per_epoch = len(data_module.train_dataloader()) // self.config['training']['gradient_accumulation_steps']
            total_steps = steps_per_epoch * self.trainer.max_epochs

            scheduler = OneCycleLR(
                optimizer,
                max_lr=float(self.config['search_space']['max_lr']['max']),
                total_steps=total_steps,
                pct_start=float(self.config['network_training']['optimizer']['pct_start']),
                div_factor=float(self.config['network_training']['optimizer']['div_factor']),
                final_div_factor=float(self.config['network_training']['optimizer']['final_div_factor']),
                anneal_strategy=self.config['network_training']['optimizer']['anneal_strategy']
            )

            reduce_lr_on_plateau = ReduceLROnPlateau(
                optimizer,
                mode=self.config['network_training']['reduce_lr_on_plateau']['mode'],
                factor=float(self.config['network_training']['reduce_lr_on_plateau']['factor']),
                patience=int(self.config['network_training']['reduce_lr_on_plateau']['patience']),
                verbose=bool(self.config['network_training']['reduce_lr_on_plateau']['verbose'])
            )

            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'step'
            }

            lr_scheduler['reduce_on_plateau'] = reduce_lr_on_plateau
            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler
            }
        except Exception as e:
            logger.error(f"Error configuring optimizers: {str(e)}")
            logger.error(traceback.format_exc())
            return None
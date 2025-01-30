# model_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    
class EMA:
    """Exponential Moving Average of model parameters with proper device handling."""
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        """Register shadow parameters on the same device as model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(param.device)

    def update(self):
        """Update shadow parameters while maintaining device placement."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # Ensure shadow parameter is on the same device as model parameter
                self.shadow[name] = self.shadow[name].to(param.device)
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow parameters while preserving device placement."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                # Ensure shadow parameter is on the same device as model parameter
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        """Restore original parameters while maintaining device placement."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
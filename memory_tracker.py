# gpu_tracker.py
import torch
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

class GPUTracker:
    def __init__(self):
        self.history = []
        
    def track_step(self, tag: str, model: Optional[torch.nn.Module] = None, local_vars: Optional[Dict] = None):
        """Track GPU usage and values at a given step."""
        # GPU Memory
        gpu_stats = {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
        }
        
        logger.info(f"\n=== GPU Stats [{tag}] ===")
        logger.info(f"GPU Memory Allocated: {gpu_stats['allocated']:.1f}MB")
        logger.info(f"GPU Memory Reserved: {gpu_stats['reserved']:.1f}MB")
        logger.info(f"GPU Max Memory: {gpu_stats['max_allocated']:.1f}MB")
        
        # Track model gradients if provided
        if model is not None:
            logger.info("\nGradient Stats:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_stats = {
                        'mean': param.grad.mean().item(),
                        'std': param.grad.std().item(),
                        'max': param.grad.max().item(),
                        'min': param.grad.min().item(),
                        'norm': param.grad.norm().item()
                    }
                    logger.info(f"\n{name}:")
                    logger.info(f"  Mean: {grad_stats['mean']:.6f}")
                    logger.info(f"  Std: {grad_stats['std']:.6f}")
                    logger.info(f"  Max: {grad_stats['max']:.6f}")
                    logger.info(f"  Min: {grad_stats['min']:.6f}")
                    logger.info(f"  Norm: {grad_stats['norm']:.6f}")
        
        # Track specific variables if provided
        if local_vars is not None:
            logger.info("\nVariable Stats:")
            for name, var in local_vars.items():
                if torch.is_tensor(var):
                    logger.info(f"\n{name}:")
                    logger.info(f"  Shape: {var.shape}")
                    logger.info(f"  Mean: {var.mean().item():.6f}")
                    logger.info(f"  Std: {var.std().item():.6f}")
                    logger.info(f"  Max: {var.max().item():.6f}")
                    logger.info(f"  Min: {var.min().item():.6f}")
                    
        # Clear memory if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
# universal_tracking.py
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
import json
from datetime import datetime
import psutil
import GPUtil
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

def generate_unified_trial_id(framework, trial_info):
    """Generate a unique, comparable trial ID that works across frameworks."""
    if framework == 'ray':
        trial_id = trial_info.get('trial_id', 'unknown')
        timestamp = trial_info.get('start_time', datetime.now().timestamp())
    else:  # optuna
        trial_id = str(trial_info.number if trial_info else 'unknown')
        timestamp = trial_info.datetime_start.timestamp() if trial_info else datetime.now().timestamp()
    
    # Create a unique hash that's consistent across frameworks
    unique_id = hashlib.md5(f"{framework}_{trial_id}_{timestamp}".encode()).hexdigest()[:8]
    return f"{framework}_{trial_id}_{unique_id}"

class UniversalTrackingCallback(Callback):
    """Universal callback with framework-specific tracking for comparison."""
    def __init__(self, framework="ray", trial=None, results_dir="results", study_name=None):
        super().__init__()
        self.framework = framework.lower()
        self.trial = trial
        self.study_name = study_name or 'default_study'
        
        # Generate unified trial ID and experiment tracking info
        trial_info = {
            'trial_id': os.getenv('TUNE_TRIAL_ID', 'no_trial_id') if framework == 'ray' else trial,
            'start_time': datetime.now().timestamp()
        }
        self.unified_trial_id = generate_unified_trial_id(framework, trial_info)
        
        # Initialize tracking containers
        self.training_data = []
        self.epoch_data = []
        self.layer_activations = defaultdict(list)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.gradient_norms_history = []
        self.param_update_history = []
        self.batch_times = []
        self.memory_usage = []
        self.was_pruned = False
        self.batch_start_time = None
        
        # Setup framework-specific metadata
        self.framework_metadata = {
            'framework': framework,
            'study_name': self.study_name,
            'unified_trial_id': self.unified_trial_id,
            'original_trial_id': (os.getenv('TUNE_TRIAL_ID') if framework == 'ray' 
                                else f"optuna_trial_{trial.number}" if trial else 'no_trial_id'),
            'timestamp': self.timestamp
        }
        
        # Create organized results directory structure
        self.results_dir = os.path.join(results_dir, 'comparison_results', self.study_name)
        self.framework_dir = os.path.join(self.results_dir, framework)
        os.makedirs(self.framework_dir, exist_ok=True)
        
        # Create metadata file for the trial
        self._save_trial_metadata()

    def _get_system_stats(self):
        """Collect system statistics including CPU, memory, and GPU usage."""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024 ** 3),
        }
        
        # Add GPU stats if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                stats.update({
                    'gpu_utilization': gpus[0].load * 100,
                    'gpu_memory_used': gpus[0].memoryUsed,
                    'gpu_memory_total': gpus[0].memoryTotal,
                    'gpu_temperature': gpus[0].temperature
                })
        except Exception as e:
            logger.warning(f"Could not collect GPU stats: {e}")
            
        return stats

    def _save_trial_metadata(self):
        """Save trial metadata for cross-framework comparison."""
        metadata_file = os.path.join(
            self.framework_dir, 
            f"metadata_{self.unified_trial_id}.json"
        )
        with open(metadata_file, 'w') as f:
            json.dump(self.framework_metadata, f, indent=2)

    def _collect_metrics(self, trainer, batch_idx=None):
        """Collect comprehensive metrics with framework identifiers."""
        metrics = {
            'framework': self.framework,
            'unified_trial_id': self.unified_trial_id,
            'study_name': self.study_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add system metrics
        system_stats = self._get_system_stats()
        metrics.update(system_stats)
        
        # Add batch/epoch info if available
        if batch_idx is not None:
            metrics['batch_idx'] = batch_idx
        metrics['epoch'] = trainer.current_epoch
        
        return metrics

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record batch start time."""
        self.batch_start_time = datetime.now()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track batch-level metrics with framework identification."""
        if not self.batch_start_time:
            self.batch_start_time = datetime.now()  # Fallback if start time wasn't set
            
        metrics = self._collect_metrics(trainer, batch_idx)
        
        # Add batch-specific metrics
        batch_metrics = {
            'batch_time': (datetime.now() - self.batch_start_time).total_seconds(),
            'loss': outputs["loss"].item() if isinstance(outputs, dict) else outputs.item(),
            'was_pruned': self.was_pruned
        }
        metrics.update(batch_metrics)
        
        # Add gradient and parameter tracking
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                metrics.update({
                    f"grad/{name}/mean": param.grad.mean().item(),
                    f"grad/{name}/std": param.grad.std().item(),
                    f"grad/{name}/norm": param.grad.norm().item(),
                })
                metrics.update({
                    f"param/{name}/mean": param.mean().item(),
                    f"param/{name}/std": param.std().item(),
                    f"param/{name}/norm": param.norm().item(),
                })
        
        self.training_data.append(metrics)

    def on_train_end(self, trainer, pl_module):
        """Save framework-specific results with cross-reference information."""
        # Save detailed training data
        results_file = os.path.join(
            self.framework_dir,
            f"training_data_{self.unified_trial_id}.csv"
        )
        pd.DataFrame(self.training_data).to_csv(results_file, index=False)
        
        # Save summary statistics
        summary = {
            **self.framework_metadata,
            'total_epochs': trainer.current_epoch + 1,
            'total_batches': len(self.training_data),
            'was_pruned': self.was_pruned,
            'final_loss': self.training_data[-1]['loss'] if self.training_data else None,
            'average_batch_time': np.mean([d['batch_time'] for d in self.training_data]) if self.training_data else None,
            'total_training_time': sum(d['batch_time'] for d in self.training_data) if self.training_data else None
        }
        
        summary_file = os.path.join(
            self.framework_dir,
            f"summary_{self.unified_trial_id}.json"
        )
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save comparison-ready data
        comparison_file = os.path.join(
            self.results_dir,
            f"comparison_data_{self.study_name}.csv"
        )
        comparison_data = pd.DataFrame([summary])
        if os.path.exists(comparison_file):
            existing_data = pd.read_csv(comparison_file)
            comparison_data = pd.concat([existing_data, comparison_data], ignore_index=True)
        comparison_data.to_csv(comparison_file, index=False)
        
        logger.info(f"Saved {self.framework} results with unified ID {self.unified_trial_id}")
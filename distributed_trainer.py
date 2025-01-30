# distributed_trainer.py
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import traceback
from typing import Any, Dict, Optional
from wand_callback import WandbCallback
from universal_tracking import UniversalTrackingCallback

logger = logging.getLogger(__name__)

class RayTuneTrainer:
    def __init__(self, config, data_module, model_class, run_name=None, use_tune=True):
        self.config = config
        self.data_module = data_module
        self.model_class = model_class
        self.use_tune = use_tune
        self.run_name = run_name
        self.project_root = os.path.join(os.getcwd(), "results")
        self.ray_results_dir = os.path.join(self.project_root, "ray_results")
        os.makedirs(self.ray_results_dir, exist_ok=True)

    def setup_callbacks(self, config):
        callbacks = []

        # Add tracking callback
        tracking_callback = UniversalTrackingCallback(
            framework='ray',
            results_dir=self.ray_results_dir,
            study_name=self.run_name
        )
        callbacks.append(tracking_callback)

        # Add early stopping callback if configured
        if config.get('training', {}).get('early_stopping_patience'):
            early_stopping = EarlyStopping(
                monitor=config['validation']['monitor_metric'],
                mode=config['validation']['monitor_mode'],
                patience=config['training']['early_stopping_patience'],
                verbose=True
            )
            callbacks.append(early_stopping)

        # Add WandB callback if configured
        if config.get('wandb'):
            wandb_callback = WandbCallback(
                project_name=config['wandb'].get('project_name', 'default_project'),
                run_name=self.run_name
            )
            callbacks.append(wandb_callback)

        return callbacks

    def tune_function(self, config):
        try:
            # Update the batch size for the data module
            self.data_module.update_batch_size(config['data_module']['batch_size'])

            # Instantiate model
            model = self.model_class(config)

            # Setup callbacks
            callbacks = self.setup_callbacks(config)

            # Initialize trainer
            trainer = pl.Trainer(
                max_epochs=config['scheduler']['max_t'],
                accelerator=config.get('accelerator', 'gpu' if torch.cuda.is_available() else 'cpu'),
                devices=config.get('devices', 1),
                callbacks=callbacks,
                gradient_clip_val=config.get('max_grad_norm', None),
                accumulate_grad_batches=config['training'].get('gradient_accumulation_steps', 1),
                precision=16 if config['training'].get('fp16_training', True) else 32,
                enable_checkpointing=False,
                logger=True
            )

            # Fit the model
            trainer.fit(model, self.data_module)

        except Exception as e:
            logger.error(f"Error in tune_function: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        return None

    def train_with_ray(self, search_space, num_samples=None):
        if not self.use_tune:
            raise ValueError("Ray Tune is disabled, cannot proceed with this configuration.")
        
        try:
            if not search_space:
                logger.warning("Search space is empty. Running with default config.")
                self.tune_function(self.config)
                return

            # Setup Ray Tune reporter
            reporter = CLIReporter(
                metric_columns=[
                    self.config['validation']['monitor_metric'],
                    "training_iteration"
                ]
            )

            # Setup ASHA scheduler
            scheduler = ASHAScheduler(
                metric=self.config['scheduler']['metric'],
                mode=self.config['scheduler']['mode'],
                max_t=self.config['scheduler']['max_t'],
                grace_period=self.config['scheduler']['grace_period'],
                reduction_factor=self.config['scheduler']['reduction_factor']
            )

            # Run optimization
            analysis = tune.run(
                self.tune_function,
                config=search_space,
                num_samples=num_samples or self.config['training']['num_samples'],
                scheduler=scheduler,
                progress_reporter=reporter,
                local_dir=self.ray_results_dir,
                name=self.run_name,
                resources_per_trial={
                    "cpu": self.config['training']['cpus_per_trial'],
                    "gpu": self.config['training']['gpus_per_trial']
                }
            )

            # Get best trial
            best_trial = analysis.get_best_trial(
                self.config['validation']['monitor_metric'],
                self.config['validation']['monitor_mode'],
                "last"
            )
            
            if best_trial:
                best_config = best_trial.config
                logger.info(f"Best trial config: {best_config}")
                return best_config
            
            return None

        except Exception as e:
            logger.error(f"Error in train_with_ray: {str(e)}")
            logger.error(traceback.format_exc())
            return None
# wand_callback.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback  # Import the Callback class
import wandb  # Even though we are not using wandb calls in this minimal version, keep the import for potential later re-enablement
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class WandbCallback(Callback):
    """
    Minimal PyTorch Lightning callback for debugging purposes.
    This version does almost nothing - no WandB initialization or logging.
    It's used to test if the error persists even with a very basic callback.
    """
    def __init__(self, project_name: str, run_name: Optional[str] = None, **kwargs):
        super().__init__()
        self.project_name = project_name
        self.run_name = run_name
        self.wandb_kwargs = kwargs
        self._wandb_init = False
        self.experiment = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        """Intentionally empty setup."""
        super().setup(trainer, pl_module, stage)  # Call parent setup

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int
    ) -> None:
        """Intentionally empty on_train_batch_end."""
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # Call parent method

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int
    ) -> None:
        """Intentionally empty on_validation_batch_end."""
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # Call parent method

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Intentionally empty on_train_epoch_end."""
        super().on_train_epoch_end(trainer, pl_module)  # Call parent method

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        """Intentionally empty teardown."""
        super().teardown(trainer, pl_module, stage)  # Call parent method

    # REMOVE THESE TWO METHODS ENTIRELY:
    # def state_dict(self) -> Dict[str, Any]:
    #     """Return empty state dict for checkpointing."""
    #     parent_state = super().state_dict()  # Call parent's state_dict
    #     return {**parent_state}  # Merge with any additional state if needed

    # def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    #     """Intentionally empty load_state_dict."""
    #     super().load_state_dict(state_dict)  # Call parent method

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: Exception) -> None:
        """Intentionally empty on_exception."""
        super().on_exception(trainer, pl_module, exception)  # Call parent method
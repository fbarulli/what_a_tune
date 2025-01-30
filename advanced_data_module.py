# advanced_data_module.py
import logging
import os
import pandas as pd
import torch
import pytorch_lightning as pl
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm
from advanced_dataset import AdvancedDataset
from config_utils import load_config

logger = logging.getLogger(__name__)

class AdvancedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", config=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = config  # Use passed config instead of loading

        # Initialize attributes
        self.tokenizer_name = None
        self.max_length = None
        self.batch_size = None
        self.num_workers = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Create data directory if it doesn't exist
        data_path = self.data_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)

    def update_batch_size(self, batch_size):
        """Update batch size (called by Ray Tune)"""
        self.batch_size = batch_size
        logger.info(f"Updated batch size to {batch_size}")

    def prepare_data(self):
        """Prepare data - download, tokenize, etc."""
        logger.info("Preparing data...")

        # Get configuration
        data_config = self.config['data_module'] # No default for 'data_module' - assume it's always there
        self.tokenizer_name = data_config['model_name'] # No default for model_name
        self.max_length = data_config['max_length'] # No default for max_length
        self.batch_size = data_config['batch_size'] # No default for batch_size
        self.num_workers = data_config['num_workers'] # Remove default 4 - make num_workers mandatory

        # Load tokenizer
        try:
            model_cache_dir = os.path.join(self.data_dir, "results", "model_cache")
            os.makedirs(model_cache_dir, exist_ok=True)

            tokenizer_cache_path = os.path.join(model_cache_dir, self.tokenizer_name)
            if os.path.exists(tokenizer_cache_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cache_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name,
                    cache_dir=model_cache_dir
                )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise

        # Process data if needed
        if not self._check_processed_data():
            self._process_and_save_data()

    def setup(self, stage=None):
        """Setup datasets for training, validation, and testing"""
        logger.info(f"Setting up data for stage: {stage}")

        data_path = self.data_dir / "data"

        try:
            if stage in ('fit', None):
                # Load train and validation data
                train_df = pd.read_csv(data_path / "train_processed.csv")
                val_df = pd.read_csv(data_path / "val_processed.csv")

                # Create datasets
                self.train_dataset = AdvancedDataset(
                    train_df['text'].tolist(),
                    train_df['label'].tolist(),
                    self.tokenizer,
                    self.max_length
                )

                self.val_dataset = AdvancedDataset(
                    val_df['text'].tolist(),
                    val_df['label'].tolist(),
                    self.tokenizer,
                    self.max_length
                )

                logger.info(f"Created train dataset with {len(self.train_dataset)} samples")
                logger.info(f"Created validation dataset with {len(self.val_dataset)} samples")

            if stage in ('test', None):
                # Load test data
                test_df = pd.read_csv(data_path / "test_processed.csv")

                # Create test dataset
                self.test_dataset = AdvancedDataset(
                    test_df['text'].tolist(),
                    test_df['label'].tolist(),
                    self.tokenizer,
                    self.max_length
                )

                logger.info(f"Created test dataset with {len(self.test_dataset)} samples")

        except Exception as e:
            logger.error(f"Error in setup: {str(e)}")
            raise

    def _create_dataloader(self, dataset, shuffle=False):
        """Helper method to create dataloaders with current settings"""
        if dataset is None:
            return None

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def _check_processed_data(self):
        """Check if processed data exists"""
        data_path = self.data_dir / "data"
        return all(
            os.path.exists(data_path / f"{split}_processed.csv")
            for split in ['train', 'val', 'test']
        )

    def _process_and_save_data(self):
        """Process and save the data"""
        logger.info("Processing data...")
        try:
            # Load raw data from the data path directly
            data_path = self.config['data_module']['data_path'] # No more .get(). Assume 'data_path' is always there
            if not data_path:
                raise ValueError("data_path not specified in config")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at: {data_path}")

            logger.info(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)

            # ADD THE FOLLOWING TO SUPPORT ALTERNATIVE COLUMN NAMES
            text_column = self.config.get('data_module', {}).get('text_column', 'text')
            rating_column = self.config.get('data_module', {}).get('rating_column', 'rating')

            df["label"] = df[rating_column] - 1

            # Split data
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                df[text_column].tolist(),
                df["label"].tolist(),
                test_size=0.2,
                random_state=42
            )

            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts,
                temp_labels,
                test_size=0.5,
                random_state=42
            )

            # Save splits
            data_path = self.data_dir / "data"
            splits = [
                ("train", train_texts, train_labels),
                ("val", val_texts, val_labels),
                ("test", test_texts, test_labels)
            ]

            for split_name, texts, labels in splits:
                df = pd.DataFrame({'text': texts, 'label': labels})
                output_path = data_path / f"{split_name}_processed.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"Saved {split_name} data to {output_path}")

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
import logging
import torch
import traceback

logger = logging.getLogger(__name__)

class AdvancedDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Verify inputs
        if not texts or not labels:
            raise ValueError("Empty texts or labels provided")
        if len(texts) != len(labels):
            raise ValueError(f"Mismatched lengths: texts={len(texts)}, labels={len(labels)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            # Get text and label
            text = str(self.texts[idx])  # Convert to string to handle potential non-string inputs
            label = self.labels[idx]
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Create item dictionary
            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
            return item
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            logger.error(f"Text: {self.texts[idx][:100]}...")
            logger.error(traceback.format_exc())
            raise
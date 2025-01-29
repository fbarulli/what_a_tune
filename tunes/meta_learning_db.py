import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MetaLearningDatabase:
    def __init__(self, project_root=None):
        self.project_root = project_root or os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.project_root, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.storage_path = os.path.join(self.results_dir, "meta_configs.json")
        self.configs = self._load_configs()

    def _load_configs(self):
        return json.load(open(self.storage_path, 'r')) if os.path.exists(self.storage_path) else []

    def add_config(self, config, performance):
        self.configs.append({
            'config': config, 
            'performance': performance, 
            'timestamp': datetime.now().isoformat()
        })
        self._save_configs()

    def _save_configs(self):
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.configs, f, indent=2)
            logger.info(f"Saved meta configs to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving meta configs: {str(e)}")
            raise

    def get_promising_configs(self, top_k=5):
        return [c['config'] for c in sorted(self.configs, 
                key=lambda x: x['performance'], 
                reverse=True)][:top_k] if self.configs else []

    def clear_database(self):
        self.configs = []
        self._save_configs()
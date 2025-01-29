# config_utils.py
import os
import yaml
import logging
from ray import tune
from typing import Dict, Any, Tuple
import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))

def get_config_path() -> str:
    """Get the path to the main config file."""
    return os.path.join(get_project_root(), "config.yaml")

def setup_directories(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """Set up necessary directories from config."""
    project_root = get_project_root()
    dir_config = config['directories'] # Expect 'directories' to be in config

    results_dir = os.path.join(project_root, dir_config['results_dir']) # Expect these to be present
    ray_results_dir = os.path.join(project_root, dir_config['ray_results_dir'])
    model_cache_dir = os.path.join(project_root, dir_config['model_cache_dir'])

    for directory in [results_dir, ray_results_dir, model_cache_dir]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

    return results_dir, ray_results_dir, model_cache_dir

def save_config(config: Dict[str, Any], backup: bool = True) -> None:
    """Save configuration to YAML file."""
    config_path = get_config_path()
    results_dir = os.path.join(get_project_root(), "results")

    if backup and os.path.exists(config_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"config_backup_{timestamp}.yaml"
        backup_path = os.path.join(results_dir, backup_filename)
        os.makedirs(results_dir, exist_ok=True)
        with open(config_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        logger.info(f"Created backup of existing config: {backup_path}")

    try:
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logger.info(f"Successfully saved config to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {str(e)}")
        raise

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file. Fails if config not found."""
    config_path = get_config_path()

    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}") # Explicitly raise FileNotFoundError

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded existing config from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def create_search_space(config: Dict[str, Any]) -> Dict[str, Any]:
    search_space = {}
    space_config = config['search_space'] # Expect 'search_space' to be in config

    for param, spec in space_config.items():
        if not isinstance(spec, dict):
            logger.warning(f"Invalid specification for parameter {param}")
            continue
        param_type = spec['type'] # Expect 'type' to be in spec
        if param_type == 'uniform':
            search_space[param] = tune.uniform(spec['min'], spec['max']) # Expect 'min' and 'max'
        elif param_type == 'loguniform':
            search_space[param] = tune.loguniform(spec['min'], spec['max']) # Expect 'min' and 'max'
        elif param_type == 'categorical':
            search_space[param] = tune.choice(spec['values']) # Expect 'values'
        else:
            logger.warning(f"Unknown parameter type {param_type} for {param}")

    logger.info(f"Created search space with parameters: {list(search_space.keys())}")
    return search_space

def generate_wandb_names(config: Dict[str, Any]) -> Tuple[str, str]:
    """Generate WandB project and run names from config."""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb_config = config['wandb'] # Expect 'wandb' to be in config
    project_name = wandb_config['project_name'] # Expect 'project_name'

    run_name = f"{timestamp}".lower()

    return project_name, run_name
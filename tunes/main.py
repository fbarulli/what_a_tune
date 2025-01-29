# main.py
import logging
import os
import torch
from config_utils import load_config, setup_directories, save_config, create_search_space, get_project_root, generate_wandb_names
from advanced_data_module import AdvancedDataModule
from optimized_model import OptimizedModel
from meta_learning_db import MetaLearningDatabase
from advanced_search_space import AdvancedSearchSpace
from pytorch_lightning import seed_everything

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load and setup configuration
    config = load_config()

    # --- CHECK FOR ESSENTIAL CONFIG SECTIONS ---
    required_sections = ['directories', 'data_module', 'model_name', 'num_labels', 'scheduler', 'search_space', 'training', 'validation', 'wandb', 'advanced_search']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required config section '{section}' is missing in config.yaml")
    # --- END CHECK FOR ESSENTIAL CONFIG SECTIONS ---

    seed = config['seed'] # Expect 'seed' to be in config
    seed_everything(seed)

    # Set up directories and save config
    results_dir, ray_results_dir, model_cache_dir = setup_directories(config)
    save_config(config)

    # Generate wandb names
    project_name, run_name = generate_wandb_names(config)

    # Data Module setup
    data_module = AdvancedDataModule(config=config)
    data_module.prepare_data()
    data_module.setup()

    # Model setup
    model_class = OptimizedModel

     # Meta-learning database and search space
    meta_db = MetaLearningDatabase(project_root=get_project_root())

    # Use previous trials for advanced search space
    previous_trials = []  # In a real implementation, load trials from Optuna
    advanced_search = AdvancedSearchSpace(previous_trials, config=config)
    advanced_search.analyze_parameter_importance()
    advanced_search.fit_gmm_models()

    # Ray Tune Training - Import here to prevent issues if not using ray
    use_tune = config['use_tune'] # Expect 'use_tune' to be in config
    if use_tune:
        from distributed_trainer import RayTuneTrainer
        search_space = create_search_space(config)
        ray_trainer = RayTuneTrainer(
            config=config,
            data_module=data_module,
            model_class=model_class,
            run_name=run_name,
            use_tune=use_tune
        )

        # best_config = ray_trainer.train_with_ray(search_space=search_space, num_samples=config['training']['num_samples']) # Original Ray Tune run - COMMENTED OUT
        print("Running tune_function LOCALLY (no Ray Tune)") # Indicate local run

        # --- ADD BATCH_SIZE TO TUNE_CONFIG FOR LOCAL RUN ---
        best_config = ray_trainer.tune_function(config) # Call tune_function with full config


        if best_config: # This part might not be reached in case of error, but keep it for now
            config.update(best_config) # Update config with best results
            save_config(config, backup=False) # Save updated config
            logger.info(f"Best config updated: {config}")
    else:
         raise ValueError("Ray tune is disabled, cannot run in this config")

    logger.info("Training completed")


if __name__ == "__main__":
    main()
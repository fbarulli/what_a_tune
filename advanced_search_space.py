# advanced_search_space.py
import logging
import os
import hashlib
import pickle
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from optuna.trial import TrialState
import GPyOpt
from meta_learning_db import MetaLearningDatabase
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

class AdvancedSearchSpace:
    def __init__(self, previous_trials, config=None): # Removed default n_components
        self.previous_trials = previous_trials
        self.n_components = None # Will be set from config
        self.param_importance = {}
        self.gmm_models = {}
        self.scalers = {}

        # Setup paths
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.project_root, "results")
        self.gmm_cache_dir = os.path.join(self.results_dir, "gmm_cache")
        os.makedirs(self.gmm_cache_dir, exist_ok=True)

        # Initialize meta learning database
        self.meta_db = MetaLearningDatabase(project_root=self.project_root)
        self.config = config if config else {}

        # Configuration from config (if available), otherwise defaults (or make mandatory in config)
        self.n_components = config['advanced_search']['gmm_n_components'] # Make gmm_n_components mandatory
        self.gmm_covariance_type = config['advanced_search'].get('gmm_covariance_type', 'full') # Keep default 'full' or make mandatory


    def analyze_parameter_importance(self):
        """Analyze the importance of each parameter based on previous trials."""
        if not self.previous_trials:
            return {}

        df = pd.DataFrame([
            t.params | {'value': t.value}
            for t in self.previous_trials
            if t.state == TrialState.COMPLETE
        ])

        importances = {}
        for param in df.columns[:-1]:
            try:
                # Calculate correlation with performance
                corr = abs(df[param].corr(df['value']))

                # Calculate parameter variance
                variance = df[param].var()

                # Calculate gradient-based importance
                sorted_df = df.sort_values('value', ascending=False)
                gradients = np.gradient(sorted_df[param])
                gradient_importance = np.abs(gradients).mean()

                # Calculate quantile-based importance
                top_quantile = df[df['value'] >= df['value'].quantile(0.9)][param].std()
                bottom_quantile = df[df['value'] <= df['value'].quantile(0.1)][param].std()
                quantile_importance = abs(top_quantile - bottom_quantile)

                # Combine metrics
                importances[param] = (corr + variance + gradient_importance + quantile_importance) / self.config['advanced_search']['importance_denominator'] # Configurable denominator
                # Default denominator = 4, can be set in config if needed

            except Exception as e:
                logger.error(f"Error analyzing parameter importance for {param}: {e}")
                importances[param] = 0.0

        self.param_importance = importances
        return importances

    def _get_cache_path(self, prefix, gmm_id):
        """Get the cache path for GMM models and scalers."""
        return os.path.join(self.gmm_cache_dir, f"{prefix}_{gmm_id}.pkl")

    def _generate_gmm_id(self):
        """Generate a unique identifier for GMM models based on trial parameters."""
        if not self.previous_trials:
            return None

        trial_params = [
            sorted(t.params.items())
            for t in self.previous_trials
            if t.state == TrialState.COMPLETE
        ]

        return hashlib.md5(str(trial_params).encode()).hexdigest() if trial_params else None

    def fit_gmm_models(self):
        """Fit Gaussian Mixture Models to the parameter space."""
        gmm_id = self._generate_gmm_id()
        gmm_path = self._get_cache_path("gmm", gmm_id)
        scaler_path = self._get_cache_path("scaler", gmm_id)

        # Try to load cached models
        if gmm_id and os.path.exists(gmm_path) and os.path.exists(scaler_path):
            try:
                with open(gmm_path, 'rb') as f:
                    self.gmm_models['global'] = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scalers['global'] = pickle.load(f)
                logger.info(f"Loaded cached GMM models from {self.gmm_cache_dir}")
                return
            except Exception as e:
                logger.error(f"Error loading cached GMM models: {e}")
                # Continue with fitting new models

        if not self.previous_trials:
            return

        # Prepare training data
        X = np.array([
            list(t.params.values())
            for t in self.previous_trials
            if t.state == TrialState.COMPLETE
        ])

        if not X.size:
            return

        # Fit models
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            gmm = GaussianMixture(
                n_components=min(self.n_components, len(X)), # Use configured n_components
                covariance_type=self.gmm_covariance_type,
                random_state=42
            )

            with tqdm(total=1, desc="Fitting GMM model") as pbar:
                gmm.fit(X_scaled)
                pbar.update(1)

            self.gmm_models['global'] = gmm
            self.scalers['global'] = scaler

            # Cache the models
            if gmm_id:
                with open(gmm_path, 'wb') as f:
                    pickle.dump(gmm, f)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info(f"Saved GMM models to {self.gmm_cache_dir}")

        except Exception as e:
            logger.error(f"Error fitting GMM models: {e}")
            raise

    def suggest_parameter_range(self, param_name, trial):
        """Suggest parameter range based on previous trials and meta-learning."""
        if not self.previous_trials or 'global' not in self.gmm_models:
            return self.config.get('parameter_ranges', {}).get(param_name)

        # Get meta-learning suggestions
        meta_configs = self.meta_db.get_promising_configs()
        values = [
            t.params[param_name]
            for t in self.previous_trials
            if t.state == TrialState.COMPLETE and param_name in t.params
        ]
        meta_values = [
            c.get(param_name)
            for c in meta_configs
            if param_name in c
        ] if meta_configs else []

        if not values:
            return None

        try:
            # Calculate meta statistics
            meta_mean = np.mean(meta_values) if meta_values else None
            meta_std = np.std(meta_values) if meta_values else None

            # Generate samples from GMM
            gmm = self.gmm_models['global']
            scaler = self.scalers['global']
            samples = scaler.inverse_transform(gmm.sample(1000)[0])
            param_idx = list(self.previous_trials[0].params.keys()).index(param_name)
            param_samples = samples[:, param_idx]

            # Calculate statistics
            mean = np.mean(param_samples)
            std = np.std(param_samples)

            # Adjust range based on parameter importance
            width = std * (self.config['advanced_search']['importance_width_factor'] - self.param_importance.get(param_name, self.config['advanced_search']['default_importance'])) # Configurable factors

            return mean - width, mean + width

        except Exception as e:
            logger.error(f"Error suggesting parameter range for {param_name}: {e}")
            return None

    def suggest_gpyopt_parameter(self, param_name, bounds, trial):
        """Suggest parameter value using GPyOpt."""
        try:
            bounds = [{'name': param_name, 'type': 'continuous', 'domain': bounds}]
            optimization = GPyOpt.methods.BayesianOptimization(
                f=lambda x: 0,
                domain=bounds,
                model_type='GP',
                acquisition_type='EI',
                normalize_Y=True
            )
            return optimization.suggest_next_locations()[0][0]
        except Exception as e:
            logger.error(f"Error during GPyOpt suggestion for parameter {param_name}: {e}")
            return None

    def suggest_categorical_parameter(self, param_name, choices, trial):
        """Suggest categorical parameter value."""
        return trial.suggest_categorical(param_name, choices)

    def suggest_int_parameter(self, param_name, range, trial):
        """Suggest integer parameter value."""
        return trial.suggest_int(param_name, *range)

    def suggest_float_parameter(self, param_name, range, trial):
        """Suggest float parameter value."""
        return trial.suggest_float(param_name, *range)
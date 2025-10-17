"""
Surrogate Models Module

This module provides machine learning surrogate models for fast prediction
of material properties and biomechanical behavior from microstructure data.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import logging
from pathlib import Path
import pickle


class SurrogateModels:
    """
    Machine learning surrogate models for material property prediction.
    
    This class provides various surrogate modeling techniques for rapid prediction
    of effective material properties from microstructure descriptors, avoiding
    expensive finite element simulations in biomechanics applications.
    
    Attributes:
        config (dict): Configuration parameters
        logger (logging.Logger): Logger instance
        models (dict): Trained surrogate models
        model_metadata (dict): Model metadata and training information
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SurrogateModels instance.
        
        Args:
            config: Surrogate models configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.model_metadata = {}
        
        # Configuration parameters
        self.model_type = config.get('model_type', 'gaussian_process')
        self.training_data_path = Path(config.get('training_data_path', './training_data'))
        self.model_save_path = Path(config.get('model_save_path', './models'))
        
        # Create directories if they don't exist
        self.training_data_path.mkdir(parents=True, exist_ok=True)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Initialized SurrogateModels")
    
    def extract_microstructure_features(self, microstructure_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from microstructure data for model input.
        
        Args:
            microstructure_data: Microstructure information
            
        Returns:
            Feature vector
        """
        features = []
        
        # Geometric features
        if 'alveolar_density' in microstructure_data:
            features.append(microstructure_data['alveolar_density'])
        
        if 'tissue_density' in microstructure_data:
            features.append(microstructure_data['tissue_density'])
        
        # Fiber orientation features
        if 'fiber_orientation' in microstructure_data:
            fiber_orient = np.array(microstructure_data['fiber_orientation'])
            # Extract invariants from orientation tensor
            features.extend([
                np.trace(fiber_orient),
                np.linalg.norm(fiber_orient, 'fro'),
                np.max(fiber_orient),
                np.min(fiber_orient)
            ])
        
        # Microstructure type encoding
        microstructure_type = microstructure_data.get('microstructure_type', 'alveolar')
        type_encoding = {
            'alveolar': [1, 0, 0],
            'bronchial': [0, 1, 0],
            'vascular': [0, 0, 1]
        }
        features.extend(type_encoding.get(microstructure_type, [0, 0, 0]))
        
        # Resolution features
        if 'resolution' in microstructure_data:
            features.append(microstructure_data['resolution'])
        
        # Ensure consistent feature length (pad with zeros if necessary)
        target_length = 20
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        else:
            features = features[:target_length]
        
        return np.array(features)
    
    def generate_training_data(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate synthetic training data for surrogate models.
        
        Args:
            n_samples: Number of training samples to generate
            
        Returns:
            Training data dictionary
        """
        self.logger.info(f"Generating {n_samples} training samples")
        
        # Generate synthetic microstructure data
        microstructure_data = []
        material_properties = []
        
        for i in range(n_samples):
            # Random microstructure features
            sample_data = {
                'alveolar_density': np.random.uniform(0.1, 0.9),
                'tissue_density': np.random.uniform(0.05, 0.3),
                'fiber_orientation': np.random.randn(3, 3),
                'microstructure_type': np.random.choice(['alveolar', 'bronchial', 'vascular']),
                'resolution': np.random.uniform(10, 100),  # micrometers
                'sample_id': i
            }
            microstructure_data.append(sample_data)
            
            # Generate corresponding material properties
            features = self.extract_microstructure_features(sample_data)
            
            # Simple synthetic relationship between features and properties
            base_modulus = 5.0  # kPa
            
            # Modulus depends on alveolar density and tissue density
            youngs_modulus = base_modulus * (1 + 2 * sample_data['alveolar_density']) * sample_data['tissue_density']
            youngs_modulus += np.random.normal(0, 0.5)  # Add noise
            
            # Poisson's ratio with physical bounds
            poisson_ratio = 0.45 - 0.1 * sample_data['alveolar_density']
            poisson_ratio = np.clip(poisson_ratio + np.random.normal(0, 0.02), 0.3, 0.49)
            
            # Density
            density = 1.0 + 0.2 * sample_data['tissue_density']
            
            properties = {
                'youngs_modulus': youngs_modulus,
                'poisson_ratio': poisson_ratio,
                'density': density,
                'shear_modulus': youngs_modulus / (2 * (1 + poisson_ratio))
            }
            material_properties.append(properties)
        
        training_data = {
            'microstructure_data': microstructure_data,
            'material_properties': material_properties,
            'feature_vectors': [self.extract_microstructure_features(data) for data in microstructure_data],
            'target_properties': np.array([[p['youngs_modulus'], p['poisson_ratio'], p['density']] 
                                          for p in material_properties]),
            'metadata': {
                'n_samples': n_samples,
                'feature_dim': 20,
                'target_dim': 3,
                'generation_method': 'synthetic',
                'timestamp': str(np.datetime64('now'))
            }
        }
        
        # Save training data
        training_file = self.training_data_path / f'training_data_{n_samples}.pkl'
        with open(training_file, 'wb') as f:
            pickle.dump(training_data, f)
        
        self.logger.info(f"Generated training data saved to {training_file}")
        return training_data
    
    def train_gaussian_process(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train Gaussian Process surrogate model.
        
        Args:
            training_data: Training data dictionary
            
        Returns:
            Training results
        """
        self.logger.info("Training Gaussian Process surrogate model")
        
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        X = training_data['feature_vectors']
        y = training_data['target_properties']
        
        # Split training and validation data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define kernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        
        # Train models for each target property
        models = {}
        training_results = {}
        
        property_names = ['youngs_modulus', 'poisson_ratio', 'density']
        
        for i, prop_name in enumerate(property_names):
            # Train GP
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=0.1,
                normalize_y=True
            )
            
            gp.fit(X_train, y_train[:, i])
            models[prop_name] = gp
            
            # Validate
            y_pred = gp.predict(X_val)
            mse = mean_squared_error(y_val[:, i], y_pred)
            r2 = r2_score(y_val[:, i], y_pred)
            
            training_results[prop_name] = {
                'mse': float(mse),
                'r2': float(r2),
                'kernel': str(gp.kernel_),
                'n_train_samples': len(X_train)
            }
        
        # Save models
        for prop_name, model in models.items():
            model_file = self.model_save_path / f'gp_{prop_name}.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        self.models['gaussian_process'] = models
        self.model_metadata['gaussian_process'] = {
            'type': 'GaussianProcess',
            'training_results': training_results,
            'feature_dim': X.shape[1],
            'target_properties': property_names,
            'training_date': str(np.datetime64('now'))
        }
        
        self.logger.info("Gaussian Process training completed")
        return training_results
    
    def train_neural_network(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train Neural Network surrogate model.
        
        Args:
            training_data: Training data dictionary
            
        Returns:
            Training results
        """
        self.logger.info("Training Neural Network surrogate model")
        
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.preprocessing import StandardScaler
        
        X = training_data['feature_vectors']
        y = training_data['target_properties']
        
        # Normalize features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Split training and validation data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        # Train neural network
        nn = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        nn.fit(X_train, y_train)
        
        # Validate
        y_pred_scaled = nn.predict(X_val)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_val_orig = scaler_y.inverse_transform(y_val)
        
        # Calculate metrics for each property
        property_names = ['youngs_modulus', 'poisson_ratio', 'density']
        training_results = {}
        
        for i, prop_name in enumerate(property_names):
            mse = mean_squared_error(y_val_orig[:, i], y_pred[:, i])
            r2 = r2_score(y_val_orig[:, i], y_pred[:, i])
            
            training_results[prop_name] = {
                'mse': float(mse),
                'r2': float(r2),
                'n_train_samples': len(X_train),
                'n_layers': len(nn.hidden_layer_sizes),
                'architecture': str(nn.hidden_layer_sizes)
            }
        
        # Save model and scalers
        model_data = {
            'model': nn,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
        
        model_file = self.model_save_path / 'nn_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.models['neural_network'] = model_data
        self.model_metadata['neural_network'] = {
            'type': 'NeuralNetwork',
            'training_results': training_results,
            'feature_dim': X.shape[1],
            'target_properties': property_names,
            'training_date': str(np.datetime64('now'))
        }
        
        self.logger.info("Neural Network training completed")
        return training_results
    
    def predict_properties(self, microstructure_data: Dict[str, Any],
                         model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict material properties using surrogate models.
        
        Args:
            microstructure_data: Microstructure information
            model_type: Type of model to use ('gaussian_process', 'neural_network')
            
        Returns:
            Predicted material properties
        """
        if model_type is None:
            model_type = self.model_type
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained. Available models: {list(self.models.keys())}")
        
        # Extract features
        features = self.extract_microstructure_features(microstructure_data)
        features = features.reshape(1, -1)  # Reshape for single prediction
        
        property_names = ['youngs_modulus', 'poisson_ratio', 'density']
        predictions = {}
        uncertainties = {}
        
        if model_type == 'gaussian_process':
            models = self.models['gaussian_process']
            
            for prop_name in property_names:
                model = models[prop_name]
                pred, std = model.predict(features, return_std=True)
                predictions[prop_name] = float(pred[0])
                uncertainties[prop_name] = float(std[0])
        
        elif model_type == 'neural_network':
            model_data = self.models['neural_network']
            model = model_data['model']
            scaler_X = model_data['scaler_X']
            scaler_y = model_data['scaler_y']
            
            # Scale features
            features_scaled = scaler_X.transform(features)
            
            # Predict
            pred_scaled = model.predict(features_scaled)
            pred_orig = scaler_y.inverse_transform(pred_scaled)
            
            for i, prop_name in enumerate(property_names):
                predictions[prop_name] = float(pred_orig[0, i])
                uncertainties[prop_name] = 0.1 * abs(predictions[prop_name])  # Estimated uncertainty
        
        # Compute derived properties
        if 'youngs_modulus' in predictions and 'poisson_ratio' in predictions:
            E = predictions['youngs_modulus']
            nu = predictions['poisson_ratio']
            predictions['shear_modulus'] = E / (2 * (1 + nu))
            predictions['bulk_modulus'] = E / (3 * (1 - 2 * nu))
        
        return {
            'predicted_properties': predictions,
            'uncertainties': uncertainties,
            'model_type': model_type,
            'input_features': features.tolist()[0]
        }
    
    def evaluate_model_performance(self, test_data: Dict[str, Any],
                                 model_type: str) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test data dictionary
            model_type: Type of model to evaluate
            
        Returns:
            Performance metrics
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        X_test = test_data['feature_vectors']
        y_test = test_data['target_properties']
        property_names = ['youngs_modulus', 'poisson_ratio', 'density']
        
        performance = {}
        
        for prop_name in property_names:
            predictions = []
            
            for features in X_test:
                micro_data = {'sample_id': len(predictions), 'alveolar_density': features[0]}
                pred = self.predict_properties(micro_data, model_type)
                predictions.append(pred['predicted_properties'][prop_name])
            
            predictions = np.array(predictions)
            true_values = y_test[:, property_names.index(prop_name)]
            
            # Calculate metrics
            mse = np.mean((predictions - true_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - true_values))
            r2 = 1 - np.sum((true_values - predictions) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)
            
            performance[prop_name] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mean_true': float(np.mean(true_values)),
                'mean_pred': float(np.mean(predictions)),
                'std_true': float(np.std(true_values)),
                'std_pred': float(np.std(predictions))
            }
        
        return performance
    
    def load_model(self, model_path: Path, model_type: str) -> None:
        """
        Load pre-trained model from file.
        
        Args:
            model_path: Path to model file
            model_type: Type of model being loaded
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models[model_type] = model_data
        self.logger.info(f"Loaded {model_type} model from {model_path}")
    
    def save_model(self, model_type: str, model_path: Path) -> None:
        """
        Save trained model to file.
        
        Args:
            model_type: Type of model to save
            model_path: Output path for model file
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[model_type], f)
        
        self.logger.info(f"Saved {model_type} model to {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about available models.
        
        Returns:
            Model information dictionary
        """
        return {
            'available_models': list(self.models.keys()),
            'model_metadata': self.model_metadata,
            'config': self.config,
            'model_save_path': str(self.model_save_path),
            'training_data_path': str(self.training_data_path)
        }
    
    def __repr__(self) -> str:
        """String representation of the SurrogateModels instance."""
        return f"SurrogateModels(available_models={list(self.models.keys())})"
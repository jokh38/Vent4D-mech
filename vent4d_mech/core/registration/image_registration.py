"""
Main Image Registration Class

This class provides a unified interface for deformable image registration,
combining classical optimization-based methods with deep learning approaches.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
from pathlib import Path
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import SimpleITK as sitk
    SIMPLEITK_AVAILABLE = True
except ImportError:
    SIMPLEITK_AVAILABLE = False

from .simpleitk_registration import SimpleITKRegistration
from .voxelmorph_registration import VoxelMorphRegistration
from .registration_utils import RegistrationUtils, RegistrationMetrics


class ImageRegistration:
    """
    Unified image registration interface for 4D-CT lung imaging.

    This class provides a high-level interface for deformable image registration,
    supporting both classical optimization-based methods (SimpleITK) and
    deep learning approaches (VoxelMorph). It implements a hybrid strategy
    that combines the accuracy of classical methods with the speed of deep learning.

    Attributes:
        method (str): Registration method ('simpleitk', 'voxelmorph', 'hybrid')
        config (dict): Configuration parameters
        gpu (bool): Whether to use GPU acceleration
        registration (object): Specific registration instance
        utils (RegistrationUtils): Utility functions
        metrics (RegistrationMetrics): Quality metrics
    """

    def __init__(self, method: str = "voxelmorph", config: Optional[Dict[str, Any]] = None,
                 gpu: bool = True):
        """
        Initialize the ImageRegistration instance.

        Args:
            method: Registration method ('simpleitk', 'voxelmorph', 'hybrid')
            config: Configuration parameters
            gpu: Whether to use GPU acceleration
        """
        self.method = method.lower()
        self.config = config or self._get_default_config()
        self.gpu = gpu and TORCH_AVAILABLE

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize registration instance
        self.registration = self._initialize_registration()

        # Initialize utilities and metrics
        self.utils = RegistrationUtils()
        self.metrics = RegistrationMetrics()

        # Performance tracking
        self.performance_stats = {
            'registration_time': 0.0,
            'quality_metrics': {},
            'memory_usage': 0.0
        }

        self.logger.info(f"Initialized ImageRegistration with method: {method}")

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration parameters.

        Returns:
            Default configuration dictionary
        """
        return {
            'simpleitk': {
                'method': 'bspline',
                'grid_spacing': [50.0, 50.0, 50.0],  # mm
                'optimizer': 'lbfgsb',
                'metric': 'mean_squares',
                'multi_resolution': True,
                'shrink_factors': [8, 4, 2, 1],
                'smoothing_sigmas': [4, 2, 1, 0],
                'max_iterations': 100,
                'convergence_tolerance': 1e-6
            },
            'voxelmorph': {
                'model_type': 'vxm',
                'model_path': None,
                'input_shape': [160, 192, 224],
                'batch_size': 1,
                'learning_rate': 1e-4,
                'loss_weights': {
                    'similarity': 1.0,
                    'regularization': 0.01
                }
            },
            'preprocessing': {
                'normalize': True,
                'clip_values': [-1000, 1000],  # HU values
                'resample_spacing': [1.5, 1.5, 3.0],  # mm
                'crop_to_lung': True
            },
            'postprocessing': {
                'smooth_dvf': True,
                'smoothing_sigma': 1.0,
                'invert_transform': False,
                'save_intermediate': False
            },
            'performance': {
                'gpu_memory_fraction': 0.8,
                'num_workers': 4,
                'pin_memory': True
            }
        }

    def _initialize_registration(self) -> Union[SimpleITKRegistration, VoxelMorphRegistration]:
        """
        Initialize the specific registration instance based on method.

        Returns:
            Registration instance

        Raises:
            ValueError: If method is not supported
        """
        if self.method == 'simpleitk':
            if not SIMPLEITK_AVAILABLE:
                raise ImportError("SimpleITK is not available. Install with: pip install SimpleITK")
            return SimpleITKRegistration(self.config['simpleitk'], self.gpu)

        elif self.method == 'voxelmorph':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available. Install with: pip install torch")
            return VoxelMorphRegistration(self.config['voxelmorph'], self.gpu)

        elif self.method == 'hybrid':
            return self._initialize_hybrid_registration()

        else:
            raise ValueError(f"Unsupported registration method: {method}")

    def _initialize_hybrid_registration(self) -> Dict[str, Any]:
        """
        Initialize hybrid registration strategy.

        Returns:
            Dictionary containing both registration instances
        """
        return {
            'simpleitk': SimpleITKRegistration(self.config['simpleitk'], self.gpu),
            'voxelmorph': VoxelMorphRegistration(self.config['voxelmorph'], self.gpu),
            'training_mode': True,
            'silver_standard_data': []
        }

    def register_images(self, fixed_image: Union[np.ndarray, str, Path],
                       moving_image: Union[np.ndarray, str, Path],
                       mask: Optional[Union[np.ndarray, str, Path]] = None,
                       return_transform: bool = True) -> Dict[str, Any]:
        """
        Perform deformable image registration.

        Args:
            fixed_image: Fixed image (target) - array or file path
            moving_image: Moving image (source) - array or file path
            mask: Optional mask for registration region
            return_transform: Whether to return transform parameters

        Returns:
            Dictionary containing registration results
        """
        import time
        start_time = time.time()

        self.logger.info("Starting image registration...")

        # Load and preprocess images
        fixed_data, moving_data = self._load_and_preprocess(fixed_image, moving_image)

        # Perform registration
        if self.method == 'hybrid':
            results = self._hybrid_registration(fixed_data, moving_data, mask)
        else:
            results = self.registration.register_images(fixed_data, moving_data, mask)

        # Postprocess results
        results = self._postprocess_results(results)

        # Calculate performance metrics
        self.performance_stats['registration_time'] = time.time() - start_time
        self.performance_stats['quality_metrics'] = self.metrics.calculate_metrics(
            fixed_data, moving_data, results['dvf']
        )

        self.logger.info(f"Registration completed in {self.performance_stats['registration_time']:.2f}s")

        return results

    def _load_and_preprocess(self, fixed_image: Union[np.ndarray, str, Path],
                           moving_image: Union[np.ndarray, str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess input images.

        Args:
            fixed_image: Fixed image (target)
            moving_image: Moving image (source)

        Returns:
            Preprocessed fixed and moving image arrays
        """
        # Load images based on input type
        if isinstance(fixed_image, (str, Path)):
            fixed_data = self.utils.load_nifti(fixed_image)
        else:
            fixed_data = fixed_image.copy()

        if isinstance(moving_image, (str, Path)):
            moving_data = self.utils.load_nifti(moving_image)
        else:
            moving_data = moving_image.copy()

        # Apply preprocessing
        if self.config['preprocessing']['normalize']:
            fixed_data = self.utils.normalize_image(fixed_data)
            moving_data = self.utils.normalize_image(moving_data)

        if self.config['preprocessing']['clip_values']:
            clip_min, clip_max = self.config['preprocessing']['clip_values']
            fixed_data = np.clip(fixed_data, clip_min, clip_max)
            moving_data = np.clip(moving_data, clip_min, clip_max)

        return fixed_data, moving_data

    def _hybrid_registration(self, fixed_data: np.ndarray, moving_data: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform hybrid registration combining SimpleITK and VoxelMorph.

        Args:
            fixed_data: Fixed image array
            moving_data: Moving image array
            mask: Optional mask

        Returns:
            Registration results dictionary
        """
        if self.registration['training_mode']:
            # Use SimpleITK to generate silver standard training data
            self.logger.info("Generating silver standard training data with SimpleITK...")

            simpleitk_results = self.registration['simpleitk'].register_images(
                fixed_data, moving_data, mask
            )

            # Store results for VoxelMorph training
            self.registration['silver_standard_data'].append({
                'fixed': fixed_data,
                'moving': moving_data,
                'dvf': simpleitk_results['dvf']
            })

            # Train VoxelMorph on accumulated data
            if len(self.registration['silver_standard_data']) >= 10:
                self.logger.info("Training VoxelMorph on silver standard data...")
                self.registration['voxelmorph'].train(
                    self.registration['silver_standard_data']
                )
                self.registration['training_mode'] = False

        # Use trained VoxelMorph for fast inference
        return self.registration['voxelmorph'].register_images(
            fixed_data, moving_data, mask
        )

    def _postprocess_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess registration results.

        Args:
            results: Raw registration results

        Returns:
            Postprocessed results
        """
        # Smooth DVF if requested
        if self.config['postprocessing']['smooth_dvf']:
            results['dvf'] = self.utils.smooth_vector_field(
                results['dvf'],
                sigma=self.config['postprocessing']['smoothing_sigma']
            )

        # Add metadata
        results['method'] = self.method
        results['config'] = self.config
        results['performance_stats'] = self.performance_stats

        return results

    def train_voxelmorph(self, training_data: list, validation_data: Optional[list] = None,
                        epochs: int = 100, batch_size: int = 2) -> Dict[str, Any]:
        """
        Train VoxelMorph model on provided data.

        Args:
            training_data: List of training samples
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training history and metrics
        """
        if self.method not in ['voxelmorph', 'hybrid']:
            raise ValueError("VoxelMorph training is only available for 'voxelmorph' or 'hybrid' methods")

        if self.method == 'voxelmorph':
            return self.registration.train(training_data, validation_data, epochs, batch_size)
        else:
            return self.registration['voxelmorph'].train(training_data, validation_data, epochs, batch_size)

    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save trained model to disk.

        Args:
            save_path: Path to save the model
        """
        if self.method == 'voxelmorph':
            self.registration.save_model(save_path)
        elif self.method == 'hybrid':
            self.registration['voxelmorph'].save_model(save_path)
        else:
            self.logger.warning("Model saving not supported for SimpleITK method")

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load trained model from disk.

        Args:
            model_path: Path to the saved model
        """
        if self.method == 'voxelmorph':
            self.registration.load_model(model_path)
        elif self.method == 'hybrid':
            self.registration['voxelmorph'].load_model(model_path)
        else:
            self.logger.warning("Model loading not supported for SimpleITK method")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Performance statistics dictionary
        """
        return self.performance_stats

    def __repr__(self) -> str:
        """String representation of the ImageRegistration instance."""
        return (f"ImageRegistration(method='{self.method}', "
                f"gpu={self.gpu}, "
                f"config_keys={list(self.config.keys())})")
"""
SimpleITK-based Image Registration

This module implements classical optimization-based deformable image registration
using SimpleITK, providing high-accuracy B-spline FFD and Demons algorithms
for 4D-CT lung image registration.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import time
import logging

try:
    import SimpleITK as sitk
    SIMPLEITK_AVAILABLE = True
except ImportError:
    SIMPLEITK_AVAILABLE = False

from .registration_utils import RegistrationUtils


class SimpleITKRegistration:
    """
    SimpleITK-based deformable image registration.

    This class implements classical optimization-based registration methods
    using SimpleITK, including B-spline Free Form Deformation (FFD) and
    Demons algorithms for non-rigid registration of 4D-CT lung images.

    Attributes:
        config (dict): Configuration parameters
        gpu (bool): Whether to use GPU acceleration (note: SimpleITK runs on CPU)
        utils (RegistrationUtils): Utility functions
        registration_method (sitk.ImageRegistrationMethod): Registration method instance
        logger (logging.Logger): Logger instance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, gpu: bool = False):
        """
        Initialize SimpleITKRegistration instance.

        Args:
            config: Configuration parameters
            gpu: Whether to use GPU acceleration (SimpleITK runs on CPU)

        Raises:
            ImportError: If SimpleITK is not available
        """
        if not SIMPLEITK_AVAILABLE:
            raise ImportError("SimpleITK is not available. Install with: pip install SimpleITK")

        self.config = config or self._get_default_config()
        self.gpu = gpu  # SimpleITK runs on CPU, but we keep this for consistency

        # Initialize utilities
        self.utils = RegistrationUtils()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize registration method
        self.registration_method = None
        if self.config.get('method') == 'bspline':
            self._initialize_registration_method()

        self.logger.info("Initialized SimpleITKRegistration")

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration parameters.

        Returns:
            Default configuration dictionary
        """
        return {
            'method': 'bspline',  # 'bspline' or 'demons'
            'grid_spacing': [50.0, 50.0, 50.0],  # mm for B-spline control points
            'optimizer': 'lbfgsb',  # 'lbfgsb', 'gradient_descent', 'amls'
            'metric': 'mean_squares',  # 'mean_squares', 'mutual_information', 'correlation'
            'multi_resolution': True,
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [4, 2, 1, 0],
            'max_iterations': 100,
            'convergence_tolerance': 1e-6,
            'convergence_window_size': 10,
            'gradient_scale': 1.0,
            'sampling_percentage': 0.2,
            'sampling_strategy': 'random',
            'interpolator': 'linear',  # 'linear', 'bspline', 'nearest_neighbor'
            'demons_type': 'symmetric', # 'symmetric', 'fast', 'standard'
            'demons_update_factor': 1.0,
            'demons_histogram_matching': True
        }

    def _initialize_registration_method(self) -> None:
        """
        Initialize the SimpleITK registration method for B-spline.
        """
        if self.config['method'] != 'bspline':
            return
        self.registration_method = sitk.ImageRegistrationMethod()
        self._setup_bspline_registration()
        self._setup_common_parameters()

    def _setup_bspline_registration(self) -> None:
        """
        Setup B-spline FFD registration parameters.
        """
        # Set metric
        if self.config['metric'] == 'mean_squares':
            self.registration_method.SetMetricAsMeanSquares()
        elif self.config['metric'] == 'mutual_information':
            self.registration_method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50
            )
        elif self.config['metric'] == 'correlation':
            self.registration_method.SetMetricAsCorrelation()
        else:
            raise ValueError(f"Unsupported metric: {self.config['metric']}")

        # Set optimizer
        if self.config['optimizer'] == 'lbfgsb':
            self.registration_method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=self.config['convergence_tolerance'],
                numberOfIterations=self.config['max_iterations'],
                maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=1024,
                costFunctionConvergenceFactor=1e+7
            )
        elif self.config['optimizer'] == 'gradient_descent':
            self.registration_method.SetOptimizerAsGradientDescent(
                learningRate=self.config['gradient_scale'],
                numberOfIterations=self.config['max_iterations'],
                convergenceMinimumValue=self.config['convergence_tolerance'],
                convergenceWindowSize=self.config['convergence_window_size']
            )
        elif self.config['optimizer'] == 'amls':
            self.registration_method.SetOptimizerAsAmoeba(
                simplexDelta=0.1,
                parametersConvergenceTolerance=self.config['convergence_tolerance'],
                functionConvergenceTolerance=1e-6,
                numberOfIterations=self.config['max_iterations']
            )

    def _setup_common_parameters(self) -> None:
        """
        Setup common registration parameters.
        """
        # Set sampling strategy
        if self.config['sampling_strategy'] == 'random':
            self.registration_method.SetMetricSamplingStrategy(
                sitk.ImageRegistrationMethod.RANDOM
            )
        elif self.config['sampling_strategy'] == 'regular':
            self.registration_method.SetMetricSamplingStrategy(
                sitk.ImageRegistrationMethod.REGULAR
            )

        self.registration_method.SetMetricSamplingPercentage(
            self.config['sampling_percentage']
        )

        # Set multi-resolution framework
        if self.config['multi_resolution']:
            self.registration_method.SetShrinkFactorsPerLevel(
                self.config['shrink_factors']
            )
            self.registration_method.SetSmoothingSigmasPerLevel(
                self.config['smoothing_sigmas']
            )
            self.registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Set interpolator
        if self.config['interpolator'] == 'linear':
            self.registration_method.SetInterpolator(sitk.sitkLinear)
        elif self.config['interpolator'] == 'bspline':
            self.registration_method.SetInterpolator(sitk.sitkBSpline)
        elif self.config['interpolator'] == 'nearest_neighbor':
            self.registration_method.SetInterpolator(sitk.sitkNearestNeighbor)

    def register_images(self, fixed_data: np.ndarray, moving_data: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform deformable image registration using SimpleITK.

        Args:
            fixed_data: Fixed image (target) as numpy array
            moving_data: Moving image (source) as numpy array
            mask: Optional mask for registration region

        Returns:
            Dictionary containing registration results
        """
        start_time = time.time()
        self.logger.info(f"Starting {self.config['method']} registration...")

        # Convert numpy arrays to SimpleITK images
        fixed_image = self.utils.numpy_to_sitk(fixed_data)
        moving_image = self.utils.numpy_to_sitk(moving_data)

        # Convert mask if provided
        sitk_mask = None
        if mask is not None:
            sitk_mask = self.utils.numpy_to_sitk(mask.astype(np.uint8))

        try:
            if self.config['method'] == 'bspline':
                transform, final_metric = self._register_bspline(
                    fixed_image, moving_image, sitk_mask
                )
            elif self.config['method'] == 'demons':
                transform, final_metric = self._register_demons(
                    fixed_image, moving_image, sitk_mask
                )

            # Apply transform to get DVF
            dvf = self._compute_dvf(transform, fixed_image)

            # Apply transform to moving image
            resampled_image = sitk.Resample(
                moving_image, fixed_image, transform,
                sitk.sitkLinear, 0.0, moving_image.GetPixelID()
            )

            # Convert back to numpy arrays
            dvf_array = self.utils.sitk_to_numpy(dvf)
            resampled_array = self.utils.sitk_to_numpy(resampled_image)

            registration_time = time.time() - start_time

            results = {
                'dvf': dvf_array,
                'transformed_image': resampled_array,
                'final_metric': final_metric,
                'transform': transform,
                'registration_time': registration_time,
                'method': self.config['method']
            }

            self.logger.info(f"Registration completed in {registration_time:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"Registration failed: {str(e)}")
            raise

    def _register_bspline(self, fixed_image: sitk.Image, moving_image: sitk.Image,
                         mask: Optional[sitk.Image]) -> Tuple[sitk.Transform, float]:
        """
        Perform B-spline FFD registration.

        Args:
            fixed_image: Fixed SimpleITK image
            moving_image: Moving SimpleITK image
            mask: Optional mask image

        Returns:
            Tuple of (transform, final_metric_value)
        """
        # Initialize B-spline transform
        mesh_size = [
            int((fixed_image.GetSize()[i] * fixed_image.GetSpacing()[i]) /
                self.config['grid_spacing'][i]) for i in range(3)
        ]

        initial_transform = sitk.BSplineTransformInitializer(
            fixed_image, mesh_size, order=3
        )
        self.registration_method.SetInitialTransform(initial_transform, inPlace=True)

        # Set mask if provided
        if mask is not None:
            self.registration_method.SetMetricFixedMask(mask)

        # Execute registration
        final_transform = self.registration_method.Execute(fixed_image, moving_image)

        # Get final metric value
        final_metric = self.registration_method.GetMetricValue()

        return final_transform, final_metric

    def _register_demons(self, fixed_image: sitk.Image, moving_image: sitk.Image,
                        mask: Optional[sitk.Image]) -> Tuple[sitk.Transform, float]:
        """
        Perform Demons registration.

        Args:
            fixed_image: Fixed SimpleITK image
            moving_image: Moving SimpleITK image
            mask: Optional mask image

        Returns:
            Tuple of (transform, final_metric_value)
        """
        # Setup demons registration based on type
        demons_type = self.config.get('demons_type', 'symmetric')
        if demons_type == 'symmetric':
            demons_filter = sitk.SymmetricForcesDemonsRegistrationFilter()
        elif demons_type == 'fast':
            demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        elif demons_type == 'standard':
            demons_filter = sitk.DemonsRegistrationFilter()
        else:
            raise ValueError(f"Unsupported demons type: {demons_type}")

        demons_filter.SetNumberOfIterations(self.config['max_iterations'])
        demons_filter.SetStandardDeviations(1.0)

        # Regularization is available in Symmetric and Fast variants
        if hasattr(demons_filter, 'SetUpdateFieldRegularizationType'):
            demons_filter.SetUpdateFieldRegularizationType(sitk.DemonsRegistrationFilter.regularizer_gaussian)
            demons_filter.SetUpdateFieldRegularizationSigma(1.0)

        if mask is not None and hasattr(demons_filter, 'SetUseMask'):
            demons_filter.SetUseMask(True)
            demons_filter.SetFixedImageMask(mask)

        if self.config['demons_histogram_matching']:
            moving_image = sitk.HistogramMatching(moving_image, fixed_image)

        # Execute registration
        displacement_field = demons_filter.Execute(fixed_image, moving_image)

        # Convert to transform
        final_transform = sitk.DisplacementFieldTransform(displacement_field)

        # Get final metric (approximate)
        final_metric = demons_filter.GetMetric()

        return final_transform, final_metric

    def _compute_dvf(self, transform: sitk.Transform, reference_image: sitk.Image) -> sitk.Image:
        """
        Compute displacement vector field from transform.

        Args:
            transform: Transform object
            reference_image: Reference image for DVF computation

        Returns:
            Displacement vector field as SimpleITK image
        """
        # Create displacement field
        displacement_field = sitk.TransformToDisplacementField(
            transform,
            sitk.sitkVectorFloat64,
            reference_image.GetSize(),
            reference_image.GetOrigin(),
            reference_image.GetSpacing(),
            reference_image.GetDirection()
        )

        return displacement_field

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration parameters.

        Args:
            config: New configuration parameters
        """
        self.config.update(config)
        self._initialize_registration_method()
        self.logger.info("Updated configuration and reinitialized registration method")

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration parameters.

        Returns:
            Current configuration dictionary
        """
        return self.config.copy()

    def __repr__(self) -> str:
        """String representation of the SimpleITKRegistration instance."""
        return (f"SimpleITKRegistration(method='{self.config['method']}', "
                f"optimizer='{self.config['optimizer']}', "
                f"metric='{self.config['metric']}')")
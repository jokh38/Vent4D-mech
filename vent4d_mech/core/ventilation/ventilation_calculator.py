"""
Ventilation Calculator

This module implements ventilation calculation methods for lung biomechanics,
supporting both Jacobian-based and strain-based approaches for regional
ventilation analysis.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import logging
from pathlib import Path

from .regional_analysis import RegionalVentilation
from .clinical_metrics import ClinicalMetrics


class VentilationCalculator:
    """
    Calculator for lung ventilation from deformation data.

    This class provides various methods for computing regional lung ventilation
    from deformation gradient or displacement vector fields, supporting both
    Jacobian-based and strain-based approaches.

    Attributes:
        config (dict): Configuration parameters
        regional_analyzer (RegionalVentilation): Regional analysis tools
        clinical_metrics (ClinicalMetrics): Clinical metric calculations
        logger (logging.Logger): Logger instance
        ventilation_results (dict): Computed ventilation results
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize VentilationCalculator instance.

        Args:
            config: Configuration parameters
        """
        self.config = config or self._get_default_config()

        # Initialize components
        self.regional_analyzer = RegionalVentilation(self.config['regional_analysis'])
        self.clinical_metrics = ClinicalMetrics(self.config['clinical_metrics'])

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Results storage
        self.ventilation_results = {}

        self.logger.info("Initialized VentilationCalculator")

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration parameters.

        Returns:
            Default configuration dictionary
        """
        return {
            'ventilation_method': 'jacobian',  # 'jacobian', 'strain_based', 'hybrid'
            'jacobian': {
                'smoothing_sigma': 1.0,
                'normalize_method': 'percentile',  # 'percentile', 'z_score', 'min_max'
                'percentile_range': (5, 95)
            },
            'strain_based': {
                'strain_threshold': 0.1,  # Minimum strain for ventilation calculation
                'weighting_function': 'exponential'
            },
            'regional_analysis': {
                'lung_regions': ['upper_left', 'upper_right', 'middle_left', 'middle_right', 'lower_left', 'lower_right'],
                'lobe_segmentation': True,
                'heterogeneity_metrics': True
            },
            'clinical_metrics': {
                'compute_svi': True,  # Specific ventilation index
                'compute_vt_ratio': True,  # Tidal volume ratio
                'baseline_normalization': True
            },
            'output': {
                'save_components': True,
                'save_statistics': True,
                'visualization_data': True
            }
        }

    def compute_ventilation(self, deformation_gradient: np.ndarray,
                          lung_mask: Optional[np.ndarray] = None,
                          reference_volume: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute regional ventilation from deformation data.

        Args:
            deformation_gradient: Deformation gradient tensor field (D, H, W, 3, 3)
            lung_mask: Optional lung region mask
            reference_volume: Optional reference volume for normalization

        Returns:
            Dictionary containing ventilation results

        Raises:
            ValueError: If input dimensions are invalid
        """
        self.logger.info("Computing regional ventilation...")

        # Validate inputs
        self._validate_inputs(deformation_gradient)

        # Apply mask if provided
        if lung_mask is not None:
            deformation_gradient = deformation_gradient * lung_mask[..., np.newaxis, np.newaxis]

        try:
            # Compute ventilation based on selected method
            if self.config['ventilation_method'] == 'jacobian':
                ventilation_map = self._compute_jacobian_ventilation(deformation_gradient)
            elif self.config['ventilation_method'] == 'strain_based':
                ventilation_map = self._compute_strain_based_ventilation(deformation_gradient)
            elif self.config['ventilation_method'] == 'hybrid':
                ventilation_map = self._compute_hybrid_ventilation(deformation_gradient)
            else:
                raise ValueError(f"Unsupported ventilation method: {self.config['ventilation_method']}")

            # Apply post-processing
            ventilation_map = self._post_process_ventilation(ventilation_map)

            # Compute regional analysis
            regional_results = self.regional_analyzer.analyze_regions(
                ventilation_map, lung_mask
            )

            # Compute clinical metrics
            clinical_results = self.clinical_metrics.compute_metrics(
                ventilation_map, reference_volume
            )

            # Compile results
            self.ventilation_results = {
                'ventilation_map': ventilation_map,
                'regional_analysis': regional_results,
                'clinical_metrics': clinical_results,
                'method': self.config['ventilation_method'],
                'config': self.config
            }

            self.logger.info("Ventilation computation completed successfully")
            return self.ventilation_results

        except Exception as e:
            self.logger.error(f"Ventilation computation failed: {str(e)}")
            raise

    def _validate_inputs(self, deformation_gradient: np.ndarray) -> None:
        """
        Validate input dimensions.

        Args:
            deformation_gradient: Deformation gradient tensor

        Raises:
            ValueError: If input is invalid
        """
        if deformation_gradient.ndim != 5 or deformation_gradient.shape[-2:] != (3, 3):
            raise ValueError(f"Deformation gradient must be 5D with shape (D, H, W, 3, 3), got {deformation_gradient.shape}")

        if np.any(np.isnan(deformation_gradient)):
            raise ValueError("Deformation gradient contains NaN values")

    def _compute_jacobian_ventilation(self, deformation_gradient: np.ndarray) -> np.ndarray:
        """
        Compute ventilation using Jacobian determinant method.

        Args:
            deformation_gradient: Deformation gradient tensor

        Returns:
            Ventilation map (D, H, W)
        """
        # Compute Jacobian determinant J = det(F)
        jacobian = np.linalg.det(deformation_gradient)

        # Ventilation is proportional to volume change: V = J - 1
        ventilation = jacobian - 1.0

        return ventilation

    def _compute_strain_based_ventilation(self, deformation_gradient: np.ndarray) -> np.ndarray:
        """
        Compute ventilation using strain-based method.

        Args:
            deformation_gradient: Deformation gradient tensor

        Returns:
            Ventilation map (D, H, W)
        """
        # Compute Green-Lagrange strain tensor
        F_T = np.transpose(deformation_gradient, (0, 1, 2, 4, 3))
        C = np.matmul(F_T, deformation_gradient)
        E = 0.5 * (C - np.eye(3))

        # Use volumetric strain as ventilation indicator
        volumetric_strain = np.trace(E, axis1=3, axis2=4)

        # Apply weighting function
        strain_threshold = self.config['strain_based']['strain_threshold']
        ventilation = np.where(
            volumetric_strain > strain_threshold,
            volumetric_strain * np.exp(-strain_threshold / volumetric_strain),
            0.0
        )

        return ventilation

    def _compute_hybrid_ventilation(self, deformation_gradient: np.ndarray) -> np.ndarray:
        """
        Compute ventilation using hybrid method.

        Args:
            deformation_gradient: Deformation gradient tensor

        Returns:
            Ventilation map (D, H, W)
        """
        # Compute both methods
        jacobian_vent = self._compute_jacobian_ventilation(deformation_gradient)
        strain_vent = self._compute_strain_based_ventilation(deformation_gradient)

        # Weighted combination
        jacobian_weight = 0.7  # Higher weight for physically-based Jacobian method
        strain_weight = 0.3

        ventilation = jacobian_weight * jacobian_vent + strain_weight * strain_vent

        return ventilation

    def _post_process_ventilation(self, ventilation: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to ventilation map.

        Args:
            ventilation: Raw ventilation map

        Returns:
            Processed ventilation map
        """
        # Apply smoothing
        sigma = self.config['jacobian']['smoothing_sigma']
        if sigma > 0:
            from scipy.ndimage import gaussian_filter
            ventilation = gaussian_filter(ventilation, sigma=sigma)

        # Normalize ventilation
        normalize_method = self.config['jacobian']['normalize_method']
        if normalize_method == 'percentile':
            percentile_range = self.config['jacobian']['percentile_range']
            vmin, vmax = np.percentile(ventilation, percentile_range)
            ventilation = np.clip((ventilation - vmin) / (vmax - vmin), 0, 1)
        elif normalize_method == 'z_score':
            ventilation = (ventilation - np.mean(ventilation)) / (np.std(ventilation) + 1e-8)
        elif normalize_method == 'min_max':
            vmin, vmax = np.min(ventilation), np.max(ventilation)
            ventilation = (ventilation - vmin) / (vmax - vmin + 1e-8)

        return ventilation

    def compute_time_resolved_ventilation(self, deformation_sequence: np.ndarray,
                                       time_points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute time-resolved ventilation from deformation sequence.

        Args:
            deformation_sequence: Sequence of deformation gradients (T, D, H, W, 3, 3)
            time_points: Corresponding time points

        Returns:
            Dictionary with time-resolved ventilation data
        """
        self.logger.info("Computing time-resolved ventilation...")

        n_timepoints = len(time_points)
        ventilation_sequence = np.zeros((n_timepoints,) + deformation_sequence.shape[1:4])

        for t in range(n_timepoints):
            ventilation_map = self._compute_jacobian_ventilation(deformation_sequence[t])
            ventilation_sequence[t] = ventilation_map

        # Compute ventilation rates (derivative)
        ventilation_rates = np.gradient(ventilation_sequence, time_points, axis=0)

        return {
            'ventilation_sequence': ventilation_sequence,
            'ventilation_rates': ventilation_rates,
            'time_points': time_points
        }

    def compare_with_reference(self, ventilation_map: np.ndarray,
                             reference_ventilation: np.ndarray,
                             reference_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compare computed ventilation with reference (e.g., SPECT).

        Args:
            ventilation_map: Computed ventilation map
            reference_ventilation: Reference ventilation map
            reference_mask: Optional mask for comparison region

        Returns:
            Comparison metrics
        """
        if reference_mask is not None:
            ventilation_map = ventilation_map[reference_mask]
            reference_ventilation = reference_ventilation[reference_mask]

        # Compute correlation metrics
        correlation = np.corrcoef(ventilation_map.flatten(), reference_ventilation.flatten())[0, 1]

        # Compute Dice coefficient for binary ventilation maps
        threshold = np.percentile(ventilation_map, 50)
        computed_binary = ventilation_map > threshold
        reference_binary = reference_ventilation > np.percentile(reference_ventilation, 50)

        dice_coefficient = 2 * np.sum(computed_binary & reference_binary) / \
                          (np.sum(computed_binary) + np.sum(reference_binary))

        # Compute mean absolute error
        mae = np.mean(np.abs(ventilation_map - reference_ventilation))

        return {
            'correlation': correlation,
            'dice_coefficient': dice_coefficient,
            'mean_absolute_error': mae,
            'root_mean_squared_error': np.sqrt(np.mean((ventilation_map - reference_ventilation)**2))
        }

    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save ventilation results to files.

        Args:
            output_dir: Output directory
        """
        if not self.ventilation_results:
            raise RuntimeError("No results to save. Compute ventilation first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save ventilation map
        import nibabel as nib
        vent_img = nib.Nifti1Image(self.ventilation_results['ventilation_map'], np.eye(4))
        nib.save(vent_img, output_dir / 'ventilation_map.nii.gz')

        # Save results data
        import json
        results_data = {
            'config': self.config,
            'regional_results': self.ventilation_results['regional_analysis'],
            'clinical_metrics': self.ventilation_results['clinical_metrics'],
            'statistics': self._compute_ventilation_statistics(),
            'timestamp': str(np.datetime64('now'))
        }
        with open(output_dir / 'ventilation_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"Ventilation results saved to {output_dir}")

    def _compute_ventilation_statistics(self) -> Dict[str, float]:
        """
        Compute ventilation statistics.

        Returns:
            Statistics dictionary
        """
        ventilation = self.ventilation_results['ventilation_map']

        return {
            'mean_ventilation': float(np.mean(ventilation)),
            'std_ventilation': float(np.std(ventilation)),
            'min_ventilation': float(np.min(ventilation)),
            'max_ventilation': float(np.max(ventilation)),
            'median_ventilation': float(np.median(ventilation))
        }

    def get_ventilation_map(self) -> Optional[np.ndarray]:
        """
        Get computed ventilation map.

        Returns:
            Ventilation map or None if not computed
        """
        return self.ventilation_results.get('ventilation_map')

    def __repr__(self) -> str:
        """String representation of the VentilationCalculator instance."""
        return (f"VentilationCalculator(method='{self.config['ventilation_method']}', "
                f"has_results={bool(self.ventilation_results)})")
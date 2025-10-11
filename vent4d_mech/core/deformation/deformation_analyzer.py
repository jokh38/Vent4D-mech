"""
Main Deformation Analysis Class

This class provides a comprehensive interface for analyzing lung tissue deformation
from displacement vector fields, implementing continuum mechanics principles for
biomechanical modeling.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import logging
from pathlib import Path

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from ..base_component import BaseComponent
from ..exceptions import ConfigurationError, ValidationError, ComputationError
from .strain_calculator import StrainCalculator
from .deformation_utils import DeformationUtils, StrainInvariants


class DeformationAnalyzer(BaseComponent):
    """
    Main class for deformation analysis of lung tissue.

    This class provides tools for analyzing lung tissue deformation from displacement
    vector fields, implementing continuum mechanics principles for biomechanical
    modeling. It supports both infinitesimal and finite strain theories, with
    emphasis on Green-Lagrange strain tensors for large deformation analysis.

    Attributes:
        config (dict): Configuration parameters
        gpu (bool): Whether to use GPU acceleration
        strain_calculator (StrainCalculator): Strain computation engine
        utils (DeformationUtils): Utility functions
        invariants (StrainInvariants): Strain invariant calculations
        logger (logging.Logger): Logger instance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, gpu: bool = True):
        """
        Initialize DeformationAnalyzer instance.

        Args:
            config: Configuration parameters
            gpu: Whether to use GPU acceleration
        """
        # Initialize using BaseComponent
        super().__init__(config=config or self._get_default_config(), gpu=gpu and CUPY_AVAILABLE)

        # Initialize components
        self.strain_calculator = StrainCalculator(self.config['strain'], self.gpu)
        self.utils = DeformationUtils()
        self.invariants = StrainInvariants()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.analysis_stats = {
            'computation_time': 0.0,
            'memory_usage': 0.0,
            'tensor_dimensions': None
        }

        self.logger.info(f"Initialized DeformationAnalyzer (GPU: {self.gpu})")

    def process(self, dvf: np.ndarray,
                      voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                      mask: Optional[np.ndarray] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Process deformation analysis using the BaseComponent interface.

        This method implements the BaseComponent's process interface and wraps
        the analyze_deformation functionality.

        Args:
            dvf: Displacement vector field (D, H, W, 3)
            voxel_spacing: Physical voxel spacing (mm)
            mask: Optional mask for region of interest
            **kwargs: Additional keyword arguments passed to analyze_deformation

        Returns:
            Dictionary containing deformation analysis results
        """
        return self.analyze_deformation(
                  dvf=dvf,
                  voxel_spacing=voxel_spacing,
                  mask=mask
              )

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration parameters.

        Returns:
            Default configuration dictionary
        """
        return {
            'strain': {
                'theory': 'green_lagrange',  # 'infinitesimal', 'green_lagrange'
                'num_diff_method': 'central',  # 'central', 'forward', 'backward'
                'boundary_treatment': 'zero_padding',  # 'zero_padding', 'nearest', 'reflect'
                'smoothing_sigma': 0.0,  # Gaussian smoothing for strain
            },
            'computation': {
                'gpu_memory_fraction': 0.8,
                'batch_processing': True,
                'batch_size': 64,
                'precision': 'float32',  # 'float32', 'float64'
                'parallel_processing': True
            },
            'analysis': {
                'compute_invariants': True,
                'compute_principal_strains': True,
                'compute_strain_magnitude': True,
                'compute_volumetric_strain': True,
                'output_format': 'tensor'  # 'tensor', 'components', 'full'
            },
            'validation': {
                'check_jacobian_positivity': True,
                'max_strain_threshold': 2.0,  # Maximum allowed strain value
                'min_jacobian_determinant': 0.1
            }
        }

    def analyze_deformation(self, dvf: np.ndarray,
                          voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                          mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive deformation analysis.

        Args:
            dvf: Displacement vector field (D, H, W, 3)
            voxel_spacing: Physical voxel spacing (mm)
            mask: Optional mask for region of interest

        Returns:
            Dictionary containing deformation analysis results

        Raises:
            ValueError: If DVF dimensions are invalid
        """
        import time
        start_time = time.time()

        self.logger.info("Starting deformation analysis...")

        # Validate input
        self._validate_dvf(dvf)

        # Apply mask if provided
        if mask is not None:
            dvf = self.utils.apply_mask_to_dvf(dvf, mask)

        # Convert to GPU if available
        if self.gpu:
            dvf = cp.asarray(dvf)

        try:
            # Calculate deformation gradient tensor
            F = self._compute_deformation_gradient(dvf, voxel_spacing)

            # Calculate strain tensor
            E = self.strain_calculator.compute_strain_tensor(F)

            # Convert back to CPU for further analysis
            if self.gpu:
                F_cpu = cp.asnumpy(F)
                E_cpu = cp.asnumpy(E)
            else:
                F_cpu = F
                E_cpu = E

            # Compute strain invariants
            results = {}
            if self.config['analysis']['compute_invariants']:
                results.update(self._compute_strain_invariants(E_cpu))

            # Compute principal strains
            if self.config['analysis']['compute_principal_strains']:
                results.update(self._compute_principal_strains(E_cpu))

            # Compute strain magnitude
            if self.config['analysis']['compute_strain_magnitude']:
                results.update(self._compute_strain_magnitude(E_cpu))

            # Compute volumetric strain
            if self.config['analysis']['compute_volumetric_strain']:
                results.update(self._compute_volumetric_strain(F_cpu))

            # Perform validation
            if self.config['validation']['check_jacobian_positivity']:
                validation_results = self._validate_deformation(F_cpu)
                results.update(validation_results)

            # Add basic results
            results.update({
                'deformation_gradient': F_cpu,
                'strain_tensor': E_cpu,
                'voxel_spacing': voxel_spacing,
                'theory': self.config['strain']['theory']
            })

            # Update performance stats
            self.analysis_stats['computation_time'] = time.time() - start_time
            self.analysis_stats['tensor_dimensions'] = dvf.shape

            self.logger.info(f"Deformation analysis completed in {self.analysis_stats['computation_time']:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"Deformation analysis failed: {str(e)}")
            raise

    def _validate_dvf(self, dvf: np.ndarray) -> None:
        """
        Validate displacement vector field input.

        Args:
            dvf: Displacement vector field

        Raises:
            ValueError: If DVF dimensions are invalid
        """
        if dvf.ndim != 4 or dvf.shape[-1] != 3:
            raise ValueError(f"DVF must be 4D array with shape (D, H, W, 3), got {dvf.shape}")

        if np.any(np.isnan(dvf)):
            raise ValueError("DVF contains NaN values")

        if np.any(np.isinf(dvf)):
            raise ValueError("DVF contains infinite values")

    def _compute_deformation_gradient(self, dvf: np.ndarray,
                                     voxel_spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Compute deformation gradient tensor F = I + ∇u.

        Args:
            dvf: Displacement vector field
            voxel_spacing: Voxel spacing

        Returns:
            Deformation gradient tensor (D, H, W, 3, 3)
        """
        if self.gpu:
            import cupy as cp
            gradient_dvf = self._compute_gradient_cupy(dvf, voxel_spacing)
        else:
            gradient_dvf = self._compute_gradient_numpy(dvf, voxel_spacing)

        # F = I + ∇u
        identity = np.eye(3)
        F = identity + gradient_dvf

        return F

    def _compute_gradient_numpy(self, dvf: np.ndarray,
                              voxel_spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Compute gradient of DVF using NumPy.

        Args:
            dvf: Displacement vector field
            voxel_spacing: Voxel spacing

        Returns:
            Gradient tensor (D, H, W, 3, 3)
        """
        # Separate components
        u = dvf[..., 0]  # z-component
        v = dvf[..., 1]  # y-component
        w = dvf[..., 2]  # x-component

        # Compute gradients
        grad_u = np.gradient(u, voxel_spacing)
        grad_v = np.gradient(v, voxel_spacing)
        grad_w = np.gradient(w, voxel_spacing)

        # Assemble gradient tensor
        gradient_dvf = np.zeros(dvf.shape[:3] + (3, 3))

        # ∇u = [∂u/∂x, ∂u/∂y, ∂u/∂z]
        gradient_dvf[..., 0, 0] = grad_u[2]  # ∂u/∂x
        gradient_dvf[..., 0, 1] = grad_u[1]  # ∂u/∂y
        gradient_dvf[..., 0, 2] = grad_u[0]  # ∂u/∂z

        # ∇v = [∂v/∂x, ∂v/∂y, ∂v/∂z]
        gradient_dvf[..., 1, 0] = grad_v[2]  # ∂v/∂x
        gradient_dvf[..., 1, 1] = grad_v[1]  # ∂v/∂y
        gradient_dvf[..., 1, 2] = grad_v[0]  # ∂v/∂z

        # ∇w = [∂w/∂x, ∂w/∂y, ∂w/∂z]
        gradient_dvf[..., 2, 0] = grad_w[2]  # ∂w/∂x
        gradient_dvf[..., 2, 1] = grad_w[1]  # ∂w/∂y
        gradient_dvf[..., 2, 2] = grad_w[0]  # ∂w/∂z

        return gradient_dvf

    def _compute_gradient_cupy(self, dvf: np.ndarray,
                              voxel_spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Compute gradient of DVF using CuPy for GPU acceleration.

        Args:
            dvf: Displacement vector field
            voxel_spacing: Voxel spacing

        Returns:
            Gradient tensor (D, H, W, 3, 3)
        """
        import cupy as cp

        # Convert to CuPy array if not already
        if not isinstance(dvf, cp.ndarray):
            dvf_cp = cp.asarray(dvf)
        else:
            dvf_cp = dvf

        # Separate components
        u = dvf_cp[..., 0]
        v = dvf_cp[..., 1]
        w = dvf_cp[..., 2]

        # Compute gradients using CuPy
        grad_u = cp.gradient(u, voxel_spacing)
        grad_v = cp.gradient(v, voxel_spacing)
        grad_w = cp.gradient(w, voxel_spacing)

        # Assemble gradient tensor
        gradient_dvf = cp.zeros(dvf_cp.shape[:3] + (3, 3), dtype=dvf_cp.dtype)

        gradient_dvf[..., 0, 0] = grad_u[2]  # ∂u/∂x
        gradient_dvf[..., 0, 1] = grad_u[1]  # ∂u/∂y
        gradient_dvf[..., 0, 2] = grad_u[0]  # ∂u/∂z

        gradient_dvf[..., 1, 0] = grad_v[2]  # ∂v/∂x
        gradient_dvf[..., 1, 1] = grad_v[1]  # ∂v/∂y
        gradient_dvf[..., 1, 2] = grad_v[0]  # ∂v/∂z

        gradient_dvf[..., 2, 0] = grad_w[2]  # ∂w/∂x
        gradient_dvf[..., 2, 1] = grad_w[1]  # ∂w/∂y
        gradient_dvf[..., 2, 2] = grad_w[0]  # ∂w/∂z

        return gradient_dvf

    def _compute_strain_invariants(self, E: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute strain tensor invariants.

        Args:
            E: Strain tensor (D, H, W, 3, 3)

        Returns:
            Dictionary of strain invariants
        """
        invariants = {}

        if self.config['strain']['theory'] == 'green_lagrange':
            # For Green-Lagrange strain tensor E = 1/2(F^T F - I)
            invariants['I1'], invariants['I2'], invariants['I3'] = self.invariants.green_lagrange_invariants(E)
        else:
            # For infinitesimal strain tensor ε = 1/2(∇u + ∇u^T)
            invariants['I1'], invariants['I2'], invariants['I3'] = self.invariants.infinitesimal_invariants(E)

        return invariants

    def _compute_principal_strains(self, E: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute principal strains and directions.

        Args:
            E: Strain tensor (D, H, W, 3, 3)

        Returns:
            Dictionary of principal strain information
        """
        principal_info = self.invariants.principal_strains(E)

        return {
            'principal_strains': principal_info['eigenvalues'],
            'principal_directions': principal_info['eigenvectors'],
            'max_principal_strain': principal_info['eigenvalues'][..., 0],
            'min_principal_strain': principal_info['eigenvalues'][..., 2]
        }

    def _compute_strain_magnitude(self, E: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute strain magnitude measures.

        Args:
            E: Strain tensor (D, H, W, 3, 3)

        Returns:
            Dictionary of strain magnitude measures
        """
        # Von Mises equivalent strain
        von_mises = self.invariants.von_mises_strain(E)

        # Maximum shear strain
        max_shear = self.invariants.max_shear_strain(E)

        # Effective strain
        effective_strain = self.invariants.effective_strain(E)

        return {
            'von_mises_strain': von_mises,
            'max_shear_strain': max_shear,
            'effective_strain': effective_strain
        }

    def _compute_volumetric_strain(self, F: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute volumetric strain measures.

        Args:
            F: Deformation gradient tensor (D, H, W, 3, 3)

        Returns:
            Dictionary of volumetric strain measures
        """
        # Jacobian determinant
        J = np.linalg.det(F)

        # Volumetric strain
        volumetric_strain = J - 1.0

        # Volume change ratio
        volume_ratio = J

        return {
            'jacobian_determinant': J,
            'volumetric_strain': volumetric_strain,
            'volume_ratio': volume_ratio
        }

    def _validate_deformation(self, F: np.ndarray) -> Dict[str, Any]:
        """
        Validate deformation for physical plausibility.

        Args:
            F: Deformation gradient tensor

        Returns:
            Dictionary of validation results
        """
        validation_results = {}

        # Check Jacobian positivity
        J = np.linalg.det(F)
        negative_jacobian_count = np.sum(J < 0)
        validation_results['negative_jacobian_count'] = negative_jacobian_count
        validation_results['negative_jacobian_percentage'] = negative_jacobian_count / J.size

        # Check minimum Jacobian
        min_jacobian = np.min(J)
        validation_results['min_jacobian'] = min_jacobian

        # Check for excessive deformation
        max_strain = np.max(np.abs(self.strain_calculator.compute_strain_tensor(F)))
        validation_results['max_strain'] = max_strain

        # Generate warnings if validation fails
        if min_jacobian < self.config['validation']['min_jacobian_determinant']:
            self.logger.warning(f"Minimum Jacobian determinant {min_jacobian:.4f} below threshold")

        if max_strain > self.config['validation']['max_strain_threshold']:
            self.logger.warning(f"Maximum strain {max_strain:.4f} above threshold")

        return validation_results

    def save_results(self, results: Dict[str, Any], output_dir: Union[str, Path]) -> None:
        """
        Save deformation analysis results to files.

        Args:
            results: Analysis results dictionary
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save tensor components
        if 'strain_tensor' in results:
            strain_tensor = results['strain_tensor']
            for i in range(3):
                for j in range(3):
                    component_path = output_dir / f'strain_component_{i}{j}.nii.gz'
                    self.utils.save_nifti(strain_tensor[..., i, j], str(component_path))

        # Save scalar results
        scalar_results = ['max_principal_strain', 'volumetric_strain', 'von_mises_strain']
        for result_name in scalar_results:
            if result_name in results:
                result_path = output_dir / f'{result_name}.nii.gz'
                self.utils.save_nifti(results[result_name], str(result_path))

        # Save metadata
        metadata_path = output_dir / 'analysis_metadata.json'
        import json
        metadata = {
            'config': self.config,
            'analysis_stats': self.analysis_stats,
            'timestamp': str(np.datetime64('now'))
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Results saved to {output_dir}")

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration parameters.

        Args:
            config: New configuration parameters
        """
        self.config.update(config)
        self.strain_calculator = StrainCalculator(self.config['strain'], self.gpu)
        self.logger.info("Updated configuration")

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration parameters.

        Returns:
            Current configuration dictionary
        """
        return self.config.copy()

    def get_analysis_stats(self) -> Dict[str, Any]:
        """
        Get analysis statistics.

        Returns:
            Analysis statistics dictionary
        """
        return self.analysis_stats.copy()

    def __repr__(self) -> str:
        """String representation of the DeformationAnalyzer instance."""
        return (f"DeformationAnalyzer(gpu={self.gpu}, "
                f"theory='{self.config['strain']['theory']}', "
                f"analysis_stats_keys={list(self.analysis_stats.keys())})")
"""
Main Mechanical Modeler Class

This class provides a comprehensive interface for constitutive modeling of lung tissue,
implementing various hyperelastic material models and stress-strain calculations
for biomechanical analysis.
"""

from typing import Optional, Dict, Any, Tuple, Union, List
import numpy as np
import logging
from pathlib import Path

from ..base_component import BaseComponent
from ..exceptions import ConfigurationError, ValidationError, ComputationError

from .constitutive_models import (
    ConstitutiveModel,
    NeoHookeanModel,
    MooneyRivlinModel,
    YeohModel,
    OgdenModel,
    LinearElasticModel
)
from .stress_calculator import StressCalculator
from .material_fitting import MaterialFitter


class MechanicalModeler(BaseComponent):
    """
    Main class for mechanical modeling of lung tissue.

    This class provides a comprehensive interface for constitutive modeling of
    lung tissue, implementing various hyperelastic material models and stress-strain
    calculations for biomechanical analysis. It supports both isotropic and
    anisotropic materials, with emphasis on hyperelastic models suitable for
    soft tissue behavior.

    Attributes:
        config (dict): Configuration parameters
        model (ConstitutiveModel): Selected constitutive model
        stress_calculator (StressCalculator): Stress computation engine
        material_fitter (MaterialFitter): Parameter estimation tools
        logger (logging.Logger): Logger instance
        model_parameters (dict): Current model parameters
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, gpu: bool = True):
        """
        Initialize MechanicalModeler instance.

        Args:
            config: Configuration parameters
            gpu: Whether to enable GPU acceleration
        """
        super().__init__(config=config, gpu=gpu)

        # Initialize components
        self.model = None
        self.stress_calculator = StressCalculator()
        self.material_fitter = MaterialFitter()

        # Model parameters
        self.model_parameters = {}

        # Initialize default model
        self._initialize_model()

        self.logger.info(f"Initialized MechanicalModeler with model: {self.config['constitutive_model']}")

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration parameters.

        Returns:
            Default configuration dictionary
        """
        return {
            'constitutive_model': 'mooney_rivlin',  # 'neo_hookean', 'mooney_rivlin', 'yeoh', 'ogden', 'linear_elastic'
            'material_parameters': {
                'neo_hookean': {
                    'C10': 0.135  # kPa
                },
                'mooney_rivlin': {
                    'C10': 0.135,  # kPa
                    'C01': 0.035   # kPa
                },
                'yeoh': {
                    'C10': 0.135,  # kPa
                    'C20': -0.05,  # kPa
                    'C30': 0.01    # kPa
                },
                'ogden': {
                    'mu': [1.0, 0.1],  # kPa
                    'alpha': [2.0, -2.0]
                },
                'linear_elastic': {
                    'youngs_modulus': 5.0,  # kPa
                    'poisson_ratio': 0.45
                }
            },
            'modeling': {
                'incompressible': True,
                'poisson_ratio': 0.45,
                'nearly_incompressible': True,
                'bulk_modulus': 1000.0  # kPa
            },
            'computation': {
                'precision': 'float64',
                'numerical_differentiation': False,
                'analytical_jacobian': True
            },
            'validation': {
                'check_positive_definiteness': True,
                'max_strain_threshold': 2.0,
                'material_stability_check': True
            }
        }

    def _validate_config(self) -> None:
        """
        Validate component configuration.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        super()._validate_config()

        # Check for required sections
        required_sections = ['constitutive_model', 'material_parameters']
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(
                    f"Missing required configuration section: {section}",
                    component=self.component_name,
                    config_key=section
                )

        # Validate constitutive model
        valid_models = ['neo_hookean', 'mooney_rivlin', 'yeoh', 'ogden', 'linear_elastic']
        if self.config['constitutive_model'] not in valid_models:
            raise ConfigurationError(
                f"Invalid constitutive model: {self.config['constitutive_model']}. "
                f"Valid options are: {valid_models}",
                component=self.component_name,
                config_key='constitutive_model',
                config_value=self.config['constitutive_model']
            )

    def process(self, strain_tensor: np.ndarray,
                deformation_gradient: Optional[np.ndarray] = None,
                computation_type: str = 'stress') -> Dict[str, Any]:
        """
        Main processing method for mechanical modeling.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)
            deformation_gradient: Deformation gradient tensor (optional)
            computation_type: Type of computation ('stress', 'energy', 'tangent_modulus')

        Returns:
            Dictionary containing computation results

        Raises:
            ValidationError: If input validation fails
            ComputationError: If computation fails
        """
        self._log_processing_start(
            f"mechanical_modeling_{computation_type}",
            tensor_shape=strain_tensor.shape,
            model=self.config['constitutive_model']
        )

        try:
            # Validate inputs
            self._validate_input_array(strain_tensor, 'strain_tensor', expected_dims=5)
            if strain_tensor.shape[-2:] != (3, 3):
                raise ValidationError(
                    f"Strain tensor must have shape (..., 3, 3), got {strain_tensor.shape}",
                    component=self.component_name,
                    data_type='strain_tensor'
                )

            # Perform computation based on type
            if computation_type == 'stress':
                result = self.compute_stress(strain_tensor, deformation_gradient)
            elif computation_type == 'energy':
                energy_density = self.compute_strain_energy_density(strain_tensor, deformation_gradient)
                result = {'strain_energy_density': energy_density}
            elif computation_type == 'tangent_modulus':
                tangent_modulus = self.compute_tangent_modulus(strain_tensor, deformation_gradient)
                result = {'tangent_modulus': tangent_modulus}
            else:
                raise ValidationError(
                    f"Unknown computation type: {computation_type}",
                    component=self.component_name,
                    validation_rule='computation_type'
                )

            self._log_processing_end(f"mechanical_modeling_{computation_type}", success=True)
            return self._package_results(result)

        except Exception as e:
            self._log_processing_end(f"mechanical_modeling_{computation_type}", success=False)
            if isinstance(e, (ValidationError, ComputationError)):
                raise
            else:
                raise ComputationError(
                    f"Mechanical modeling failed: {str(e)}",
                    component=self.component_name,
                    operation=computation_type
                ) from e

    def compute_stress(self, strain_tensor: np.ndarray,
                      deformation_gradient: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute stress from strain using the selected constitutive model.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)
            deformation_gradient: Deformation gradient tensor (optional for some models)

        Returns:
            Dictionary containing stress tensor and derived quantities
        """
        # Compute stress based on model type
        if self.config['constitutive_model'] == 'linear_elastic':
            stress_tensor = self.model.compute_stress(strain_tensor)
        else:
            # For hyperelastic models, use deformation gradient
            if deformation_gradient is None:
                raise ValidationError(
                    "Deformation gradient is required for hyperelastic models",
                    component=self.component_name
                )
            stress_tensor = self.model.compute_stress(deformation_gradient)

        # Compute derived quantities
        results = {
            'stress_tensor': stress_tensor,
            'von_mises_stress': self._compute_von_mises_stress(stress_tensor),
            'principal_stresses': self._compute_principal_stresses(stress_tensor),
            'hydrostatic_stress': self._compute_hydrostatic_stress(stress_tensor),
            'deviatoric_stress': self._compute_deviatoric_stress(stress_tensor)
        }

        # Validate stress field
        if self.config['validation']['check_positive_definiteness']:
            self._validate_stress_tensor(stress_tensor)

        return results

    def compute_strain_energy_density(self, strain_tensor: np.ndarray,
                                    deformation_gradient: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute strain energy density.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)
            deformation_gradient: Deformation gradient tensor (optional for some models)

        Returns:
            Strain energy density field (D, H, W)
        """
        if self.config['constitutive_model'] == 'linear_elastic':
            return self.model.compute_strain_energy_density(strain_tensor)
        else:
            if deformation_gradient is None:
                raise ValidationError(
                    "Deformation gradient is required for hyperelastic models",
                    component=self.component_name
                )
            return self.model.compute_strain_energy_density(deformation_gradient)

    def compute_tangent_modulus(self, strain_tensor: np.ndarray,
                              deformation_gradient: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute tangent modulus (material stiffness tensor).

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)
            deformation_gradient: Deformation gradient tensor (optional)

        Returns:
            Tangent modulus tensor (D, H, W, 3, 3, 3, 3)
        """
        if self.config['constitutive_model'] == 'linear_elastic':
            return self.model.compute_tangent_modulus()
        else:
            if deformation_gradient is None:
                raise ValidationError(
                    "Deformation gradient is required for hyperelastic models",
                    component=self.component_name
                )
            return self.model.compute_tangent_modulus(deformation_gradient)

    def _initialize_model(self) -> None:
        """
        Initialize the constitutive model based on configuration.
        """
        model_type = self.config['constitutive_model']
        parameters = self.config['material_parameters'].get(model_type, {})

        if model_type == 'neo_hookean':
            self.model = NeoHookeanModel(**parameters)
        elif model_type == 'mooney_rivlin':
            self.model = MooneyRivlinModel(**parameters)
        elif model_type == 'yeoh':
            self.model = YeohModel(**parameters)
        elif model_type == 'ogden':
            self.model = OgdenModel(**parameters)
        elif model_type == 'linear_elastic':
            self.model = LinearElasticModel(**parameters)
        else:
            raise ConfigurationError(
                f"Unsupported constitutive model: {model_type}",
                component=self.component_name,
                config_key='constitutive_model',
                config_value=model_type
            )

        self.model_parameters = parameters.copy()
        self.logger.info(f"Initialized {model_type} model with parameters: {parameters}")

    def _compute_von_mises_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """Compute von Mises stress."""
        hydrostatic = self._compute_hydrostatic_stress(stress_tensor)
        deviatoric = stress_tensor - hydrostatic[..., np.newaxis, np.newaxis] * np.eye(3) / 3
        von_mises = np.sqrt(1.5 * np.sum(deviatoric**2, axis=(3, 4)))
        return von_mises

    def _compute_principal_stresses(self, stress_tensor: np.ndarray) -> np.ndarray:
        """Compute principal stresses."""
        principal_stresses = np.zeros(stress_tensor.shape[:3] + (3,))
        for i in range(stress_tensor.shape[0]):
            for j in range(stress_tensor.shape[1]):
                for k in range(stress_tensor.shape[2]):
                    principal_stresses[i, j, k] = np.linalg.eigvalsh(stress_tensor[i, j, k])
        return principal_stresses

    def _compute_hydrostatic_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """Compute hydrostatic stress (pressure)."""
        return np.trace(stress_tensor, axis1=3, axis2=4) / 3

    def _compute_deviatoric_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """Compute deviatoric stress."""
        hydrostatic = self._compute_hydrostatic_stress(stress_tensor)
        return stress_tensor - hydrostatic[..., np.newaxis, np.newaxis] * np.eye(3) / 3

    def _validate_stress_tensor(self, stress_tensor: np.ndarray) -> None:
        """Validate stress tensor for physical plausibility."""
        if np.any(np.isnan(stress_tensor)):
            raise ValidationError("Stress tensor contains NaN values", component=self.component_name)
        if np.any(np.isinf(stress_tensor)):
            raise ValidationError("Stress tensor contains infinite values", component=self.component_name)

        # Check for reasonable stress magnitudes (in kPa)
        max_stress = np.max(np.abs(stress_tensor))
        if max_stress > 1000:  # 1 MPa
            self.logger.warning(f"Maximum stress {max_stress:.2f} kPa seems unusually high")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_type': self.config['constitutive_model'],
            'parameters': self.model_parameters,
            'model_class': self.model.__class__.__name__ if self.model else None,
            'is_hyperelastic': self.model.is_hyperelastic if hasattr(self.model, 'is_hyperelastic') else False,
            'is_incompressible': self.config['modeling']['incompressible']
        }

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


class MechanicalModeler:
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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MechanicalModeler instance.

        Args:
            config: Configuration parameters
        """
        self.config = config or self._get_default_config()

        # Initialize components
        self.model = None
        self.stress_calculator = StressCalculator()
        self.material_fitter = MaterialFitter()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

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
            raise ValueError(f"Unsupported constitutive model: {model_type}")

        self.model_parameters = parameters.copy()
        self.logger.info(f"Initialized {model_type} model with parameters: {parameters}")

    def set_model(self, model_type: str, parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Set a new constitutive model.

        Args:
            model_type: Type of constitutive model
            parameters: Model parameters
        """
        self.config['constitutive_model'] = model_type

        if parameters is None:
            parameters = self.config['material_parameters'].get(model_type, {})

        self.config['material_parameters'][model_type] = parameters

        # Reinitialize model
        self._initialize_model()

        self.logger.info(f"Set model to {model_type} with parameters: {parameters}")

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
        self.logger.info("Computing stress using constitutive model...")

        try:
            # Compute stress based on model type
            if self.config['constitutive_model'] == 'linear_elastic':
                stress_tensor = self.model.compute_stress(strain_tensor)
            else:
                # For hyperelastic models, use deformation gradient
                if deformation_gradient is None:
                    raise ValueError("Deformation gradient is required for hyperelastic models")
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

            self.logger.info("Stress computation completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Stress computation failed: {str(e)}")
            raise

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
                raise ValueError("Deformation gradient is required for hyperelastic models")
            return self.model.compute_strain_energy_density(deformation_gradient)

    def fit_material_parameters(self, stress_data: np.ndarray, strain_data: np.ndarray,
                               initial_parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Fit material parameters to experimental stress-strain data.

        Args:
            stress_data: Stress tensor data
            strain_data: Strain tensor data
            initial_parameters: Initial parameter estimates

        Returns:
            Fitting results and optimized parameters
        """
        self.logger.info("Fitting material parameters...")

        if initial_parameters is None:
            initial_parameters = self.model_parameters

        fitting_results = self.material_fitter.fit_parameters(
            self.model, stress_data, strain_data, initial_parameters
        )

        # Update model parameters
        self.model_parameters.update(fitting_results['optimized_parameters'])
        self.model.set_parameters(**self.model_parameters)

        self.logger.info(f"Material fitting completed. Parameters: {self.model_parameters}")

        return fitting_results

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
                raise ValueError("Deformation gradient is required for hyperelastic models")
            return self.model.compute_tangent_modulus(deformation_gradient)

    def compute_effective_modulus(self, strain_tensor: np.ndarray) -> np.ndarray:
        """
        Compute effective (tangent) Young's modulus from strain.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)

        Returns:
            Effective Young's modulus field (D, H, W)
        """
        # Compute principal strains
        principal_strains = np.linalg.eigvalsh(strain_tensor)

        # For each voxel, compute effective modulus
        effective_modulus = np.zeros(strain_tensor.shape[:3])

        for i in range(strain_tensor.shape[0]):
            for j in range(strain_tensor.shape[1]):
                for k in range(strain_tensor.shape[2]):
                    # Use maximum principal strain for effective modulus
                    max_strain = np.max(np.abs(principal_strains[i, j, k]))

                    if max_strain > 1e-6:
                        # Compute stress for small perturbation
                        strain_perturbed = strain_tensor[i, j, k].copy()
                        strain_perturbed += 1e-6 * np.eye(3)

                        if self.config['constitutive_model'] == 'linear_elastic':
                            stress_orig = self.model.compute_stress(strain_tensor[i, j, k])
                            stress_perturbed = self.model.compute_stress(strain_perturbed)
                        else:
                            # For hyperelastic models, this is more complex
                            # Use a simplified approach here
                            stress_orig = self.model.compute_stress_from_strain(strain_tensor[i, j, k])
                            stress_perturbed = self.model.compute_stress_from_strain(strain_perturbed)

                        # Effective modulus = Δσ/Δε
                        effective_modulus[i, j, k] = np.max(np.abs(stress_perturbed - stress_orig)) / 1e-6
                    else:
                        # Use material parameter for small strains
                        if self.config['constitutive_model'] == 'linear_elastic':
                            effective_modulus[i, j, k] = self.model_parameters.get('youngs_modulus', 5.0)
                        else:
                            # For hyperelastic models, use initial shear modulus
                            effective_modulus[i, j, k] = 3 * self.model_parameters.get('C10', 0.135)

        return effective_modulus

    def _compute_von_mises_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """
        Compute von Mises stress.

        Args:
            stress_tensor: Stress tensor (D, H, W, 3, 3)

        Returns:
            Von Mises stress field (D, H, W)
        """
        # Von Mises stress: σ_vm = sqrt(3/2 * s_ij * s_ij)
        # where s_ij is the deviatoric stress
        hydrostatic = self._compute_hydrostatic_stress(stress_tensor)
        deviatoric = stress_tensor - hydrostatic[..., np.newaxis, np.newaxis] * np.eye(3) / 3

        von_mises = np.sqrt(1.5 * np.sum(deviatoric**2, axis=(3, 4)))
        return von_mises

    def _compute_principal_stresses(self, stress_tensor: np.ndarray) -> np.ndarray:
        """
        Compute principal stresses.

        Args:
            stress_tensor: Stress tensor (D, H, W, 3, 3)

        Returns:
            Principal stresses (D, H, W, 3)
        """
        principal_stresses = np.zeros(stress_tensor.shape[:3] + (3,))

        for i in range(stress_tensor.shape[0]):
            for j in range(stress_tensor.shape[1]):
                for k in range(stress_tensor.shape[2]):
                    principal_stresses[i, j, k] = np.linalg.eigvalsh(stress_tensor[i, j, k])

        return principal_stresses

    def _compute_hydrostatic_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """
        Compute hydrostatic stress (pressure).

        Args:
            stress_tensor: Stress tensor (D, H, W, 3, 3)

        Returns:
            Hydrostatic stress field (D, H, W)
        """
        return np.trace(stress_tensor, axis1=3, axis2=4) / 3

    def _compute_deviatoric_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """
        Compute deviatoric stress.

        Args:
            stress_tensor: Stress tensor (D, H, W, 3, 3)

        Returns:
            Deviatoric stress tensor (D, H, W, 3, 3)
        """
        hydrostatic = self._compute_hydrostatic_stress(stress_tensor)
        return stress_tensor - hydrostatic[..., np.newaxis, np.newaxis] * np.eye(3) / 3

    def _validate_stress_tensor(self, stress_tensor: np.ndarray) -> None:
        """
        Validate stress tensor for physical plausibility.

        Args:
            stress_tensor: Stress tensor to validate

        Raises:
            ValueError: If stress tensor is physically implausible
        """
        # Check for NaN or infinite values
        if np.any(np.isnan(stress_tensor)):
            raise ValueError("Stress tensor contains NaN values")

        if np.any(np.isinf(stress_tensor)):
            raise ValueError("Stress tensor contains infinite values")

        # Check for reasonable stress magnitudes (in kPa)
        max_stress = np.max(np.abs(stress_tensor))
        if max_stress > 1000:  # 1 MPa
            self.logger.warning(f"Maximum stress {max_stress:.2f} kPa seems unusually high")

    def save_model_parameters(self, file_path: Union[str, Path]) -> None:
        """
        Save model parameters to file.

        Args:
            file_path: Output file path
        """
        import json

        parameters_data = {
            'model_type': self.config['constitutive_model'],
            'parameters': self.model_parameters,
            'config': self.config,
            'timestamp': str(np.datetime64('now'))
        }

        with open(file_path, 'w') as f:
            json.dump(parameters_data, f, indent=2)

        self.logger.info(f"Model parameters saved to {file_path}")

    def load_model_parameters(self, file_path: Union[str, Path]) -> None:
        """
        Load model parameters from file.

        Args:
            file_path: Input file path
        """
        import json

        with open(file_path, 'r') as f:
            parameters_data = json.load(f)

        self.config['constitutive_model'] = parameters_data['model_type']
        self.model_parameters = parameters_data['parameters']
        self.config.update(parameters_data.get('config', {}))

        # Reinitialize model with loaded parameters
        self._initialize_model()

        self.logger.info(f"Model parameters loaded from {file_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Model information dictionary
        """
        return {
            'model_type': self.config['constitutive_model'],
            'parameters': self.model_parameters,
            'model_class': self.model.__class__.__name__,
            'is_hyperelastic': self.model.is_hyperelastic if hasattr(self.model, 'is_hyperelastic') else False,
            'is_incompressible': self.config['modeling']['incompressible']
        }

    def __repr__(self) -> str:
        """String representation of the MechanicalModeler instance."""
        return (f"MechanicalModeler(model='{self.config['constitutive_model']}', "
                f"parameters={list(self.model_parameters.keys())})")
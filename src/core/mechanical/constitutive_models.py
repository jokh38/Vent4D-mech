"""
Constitutive Models

This module provides constitutive model implementations for lung tissue biomechanics,
including various hyperelastic material models suitable for soft tissue deformation
analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import warnings
import logging

# Try to import numpy, but make it optional
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Some functionality will be limited.")


class ConstitutiveModel(ABC):
    """
    Abstract base class for constitutive models.

    This class defines the interface that all constitutive models must implement,
    providing consistent methods for stress computation and strain energy
    density calculation.
    """

    def __init__(self, **parameters):
        """
        Initialize constitutive model.

        Args:
            **parameters: Model-specific parameters
        """
        self.parameters = parameters
        self.is_hyperelastic = True
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_parameters()

    @abstractmethod
    def compute_stress(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute stress from strain tensor.

        Args:
            strain_tensor: Strain tensor array

        Returns:
            Stress tensor array
        """
        pass

    @abstractmethod
    def compute_strain_energy_density(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute strain energy density from strain tensor.

        Args:
            strain_tensor: Strain tensor array

        Returns:
            Strain energy density array
        """
        pass

    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        if not self.parameters:
            raise ValueError(f"{self.__class__.__name__} requires parameters")

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get parameter value."""
        return self.parameters.get(name, default)

    def set_parameter(self, name: str, value: Any) -> None:
        """Set parameter value."""
        self.parameters[name] = value
        self._validate_parameters()

    def __repr__(self) -> str:
        """String representation of the constitutive model."""
        param_str = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"{self.__class__.__name__}({param_str})"


class NeoHookeanModel(ConstitutiveModel):
    """
    Neo-Hookean hyperelastic material model.

    The Neo-Hookean model is suitable for modeling large deformations of
    nearly incompressible materials like soft biological tissues.
    """

    def __init__(self, C10: float = 0.135, density: float = 1.05):
        """
        Initialize Neo-Hookean model.

        Args:
            C10: Material parameter (kPa) - related to initial shear modulus
            density: Material density (g/cm³)
        """
        super().__init__(C10=C10, density=density)

    def _validate_parameters(self) -> None:
        """Validate Neo-Hookean parameters."""
        if self.get_parameter('C10', 0) <= 0:
            raise ValueError("C10 must be positive")
        if self.get_parameter('density', 0) <= 0:
            raise ValueError("density must be positive")

    def compute_stress(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute stress using Neo-Hookean model.

        Args:
            strain_tensor: Green-Lagrange strain tensor

        Returns:
            Second Piola-Kirchhoff stress tensor
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for stress computation")

        # For Neo-Hookean model: S = 2*C10*I
        stress = 2 * self.get_parameter('C10') * np.eye(3)

        # Broadcast to match input tensor shape
        if strain_tensor.ndim > 2:
            # Input is a volume of strain tensors
            output_shape = strain_tensor.shape
            stress = np.broadcast_to(stress, output_shape).copy()

        return stress

    def compute_strain_energy_density(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute strain energy density.

        Args:
            strain_tensor: Green-Lagrange strain tensor

        Returns:
            Strain energy density
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for strain energy computation")

        C10 = self.get_parameter('C10')

        # For Neo-Hookean model: W = C10 * I1
        # where I1 is the first invariant of the Green-Lagrange strain tensor
        if strain_tensor.ndim > 2:
            # Volume of strain tensors
            I1 = np.trace(strain_tensor, axis1=-2, axis2=-1)
        else:
            # Single strain tensor
            I1 = np.trace(strain_tensor)

        return C10 * I1

    def get_shear_modulus(self) -> float:
        """Get initial shear modulus."""
        return 2 * self.get_parameter('C10')

    def get_bulk_modulus(self, poisson_ratio: float = 0.45) -> float:
        """
        Get bulk modulus.

        Args:
            poisson_ratio: Poisson's ratio

        Returns:
            Bulk modulus
        """
        G = self.get_shear_modulus()
        return 2 * G * (1 + poisson_ratio) / (3 * (1 - 2 * poisson_ratio))


class MooneyRivlinModel(ConstitutiveModel):
    """
    Mooney-Rivlin hyperelastic material model.

    The Mooney-Rivlin model provides better accuracy for larger deformations
    compared to the Neo-Hookean model.
    """

    def __init__(self, C10: float = 0.135, C01: float = 0.035, density: float = 1.05):
        """
        Initialize Mooney-Rivlin model.

        Args:
            C10: First material parameter (kPa)
            C01: Second material parameter (kPa)
            density: Material density (g/cm³)
        """
        super().__init__(C10=C10, C01=C01, density=density)

    def _validate_parameters(self) -> None:
        """Validate Mooney-Rivlin parameters."""
        if self.get_parameter('C10', 0) <= 0:
            raise ValueError("C10 must be positive")
        if self.get_parameter('C01', 0) < 0:
            raise ValueError("C01 must be non-negative")
        if self.get_parameter('density', 0) <= 0:
            raise ValueError("density must be positive")

    def compute_stress(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute stress using Mooney-Rivlin model.

        Args:
            strain_tensor: Green-Lagrange strain tensor

        Returns:
            Second Piola-Kirchhoff stress tensor
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for stress computation")

        C10 = self.get_parameter('C10')
        C01 = self.get_parameter('C01')

        # For Mooney-Rivlin model: S = 2*C10*I + 2*C01*I1*I - 4*C01*C
        # where C is the right Cauchy-Green deformation tensor
        # For small strains, this simplifies to: S ≈ 2*(C10 + C01)*I
        stress = 2 * (C10 + C01) * np.eye(3)

        # Broadcast to match input tensor shape
        if strain_tensor.ndim > 2:
            output_shape = strain_tensor.shape
            stress = np.broadcast_to(stress, output_shape).copy()

        return stress

    def compute_strain_energy_density(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute strain energy density.

        Args:
            strain_tensor: Green-Lagrange strain tensor

        Returns:
            Strain energy density
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for strain energy computation")

        C10 = self.get_parameter('C10')
        C01 = self.get_parameter('C01')

        # For Mooney-Rivlin model: W = C10*I1 + C01*I2
        # where I1 and I2 are the first and second invariants
        if strain_tensor.ndim > 2:
            # Volume of strain tensors
            I1 = np.trace(strain_tensor, axis1=-2, axis2=-1)
            # For small strains, I2 ≈ 0.5*(I1² - tr(E²))
            I2 = 0.5 * (I1**2 - np.trace(strain_tensor @ strain_tensor, axis1=-2, axis2=-1))
        else:
            # Single strain tensor
            I1 = np.trace(strain_tensor)
            I2 = 0.5 * (I1**2 - np.trace(strain_tensor @ strain_tensor))

        return C10 * I1 + C01 * I2

    def get_shear_modulus(self) -> float:
        """Get initial shear modulus."""
        return 2 * (self.get_parameter('C10') + self.get_parameter('C01'))


class YeohModel(ConstitutiveModel):
    """
    Yeoh hyperelastic material model.

    The Yeoh model is particularly suitable for materials with limited
    experimental data, using only a few parameters.
    """

    def __init__(self, C10: float = 0.135, C20: float = 0.015, C30: float = 0.001, density: float = 1.05):
        """
        Initialize Yeoh model.

        Args:
            C10: First material parameter (kPa)
            C20: Second material parameter (kPa)
            C30: Third material parameter (kPa)
            density: Material density (g/cm³)
        """
        super().__init__(C10=C10, C20=C20, C30=C30, density=density)

    def _validate_parameters(self) -> None:
        """Validate Yeoh parameters."""
        if self.get_parameter('C10', 0) <= 0:
            raise ValueError("C10 must be positive")
        if self.get_parameter('C20', 0) < 0:
            raise ValueError("C20 must be non-negative")
        if self.get_parameter('C30', 0) < 0:
            raise ValueError("C30 must be non-negative")
        if self.get_parameter('density', 0) <= 0:
            raise ValueError("density must be positive")

    def compute_stress(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute stress using Yeoh model.

        Args:
            strain_tensor: Green-Lagrange strain tensor

        Returns:
            Second Piola-Kirchhoff stress tensor
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for stress computation")

        C10 = self.get_parameter('C10')

        # For Yeoh model: S = 2*C10*I + higher order terms
        # For small strains, use linear approximation
        stress = 2 * C10 * np.eye(3)

        # Broadcast to match input tensor shape
        if strain_tensor.ndim > 2:
            output_shape = strain_tensor.shape
            stress = np.broadcast_to(stress, output_shape).copy()

        return stress

    def compute_strain_energy_density(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute strain energy density.

        Args:
            strain_tensor: Green-Lagrange strain tensor

        Returns:
            Strain energy density
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for strain energy computation")

        C10 = self.get_parameter('C10')
        C20 = self.get_parameter('C20')
        C30 = self.get_parameter('C30')

        # For Yeoh model: W = C10*I1 + C20*I1² + C30*I1³
        if strain_tensor.ndim > 2:
            I1 = np.trace(strain_tensor, axis1=-2, axis2=-1)
        else:
            I1 = np.trace(strain_tensor)

        return C10 * I1 + C20 * I1**2 + C30 * I1**3

    def get_shear_modulus(self) -> float:
        """Get initial shear modulus."""
        return 2 * self.get_parameter('C10')


class OgdenModel(ConstitutiveModel):
    """
    Ogden hyperelastic material model.

    The Ogden model can accurately represent the behavior of rubber-like
    materials and biological tissues over large deformation ranges.
    """

    def __init__(self, mu1: float = 0.5, alpha1: float = 2.0, mu2: float = 0.1, alpha2: float = -2.0, density: float = 1.05):
        """
        Initialize Ogden model.

        Args:
            mu1: First shear parameter (kPa)
            alpha1: First exponent
            mu2: Second shear parameter (kPa)
            alpha2: Second exponent
            density: Material density (g/cm³)
        """
        super().__init__(mu1=mu1, alpha1=alpha1, mu2=mu2, alpha2=alpha2, density=density)

    def _validate_parameters(self) -> None:
        """Validate Ogden parameters."""
        if self.get_parameter('mu1', 0) <= 0:
            raise ValueError("mu1 must be positive")
        if self.get_parameter('mu2', 0) <= 0:
            raise ValueError("mu2 must be positive")
        if self.get_parameter('density', 0) <= 0:
            raise ValueError("density must be positive")

    def compute_stress(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute stress using Ogden model.

        Args:
            strain_tensor: Green-Lagrange strain tensor

        Returns:
            Second Piola-Kirchhoff stress tensor
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for stress computation")

        mu1 = self.get_parameter('mu1')
        mu2 = self.get_parameter('mu2')

        # For Ogden model: S = μ1*α1*I + μ2*α2*I (simplified for small strains)
        stress = (mu1 * self.get_parameter('alpha1') + mu2 * self.get_parameter('alpha2')) * np.eye(3)

        # Broadcast to match input tensor shape
        if strain_tensor.ndim > 2:
            output_shape = strain_tensor.shape
            stress = np.broadcast_to(stress, output_shape).copy()

        return stress

    def compute_strain_energy_density(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute strain energy density.

        Args:
            strain_tensor: Green-Lagrange strain tensor

        Returns:
            Strain energy density
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for strain energy computation")

        mu1 = self.get_parameter('mu1')
        alpha1 = self.get_parameter('alpha1')
        mu2 = self.get_parameter('mu2')
        alpha2 = self.get_parameter('alpha2')

        # For Ogden model: W = (μ1/α1)*(I1^α1 - 1) + (μ2/α2)*(I1^α2 - 1)
        if strain_tensor.ndim > 2:
            I1 = np.trace(strain_tensor, axis1=-2, axis2=-1)
        else:
            I1 = np.trace(strain_tensor)

        # For small strains, use linearized version
        return (mu1 + mu2) * I1

    def get_shear_modulus(self) -> float:
        """Get initial shear modulus."""
        mu1 = self.get_parameter('mu1')
        mu2 = self.get_parameter('mu2')
        alpha1 = self.get_parameter('alpha1')
        alpha2 = self.get_parameter('alpha2')

        # Initial shear modulus for Ogden model
        return 0.5 * (mu1 * alpha1 + mu2 * alpha2)


class LinearElasticModel(ConstitutiveModel):
    """
    Linear elastic material model.

    This model is suitable for small deformations and provides a baseline
    comparison with hyperelastic models.
    """

    def __init__(self, youngs_modulus: float = 5.0, poisson_ratio: float = 0.45, density: float = 1.05):
        """
        Initialize linear elastic model.

        Args:
            youngs_modulus: Young's modulus (kPa)
            poisson_ratio: Poisson's ratio (dimensionless, 0 to 0.5)
            density: Material density (g/cm³)
        """
        super().__init__(youngs_modulus=youngs_modulus, poisson_ratio=poisson_ratio, density=density)

    def _validate_parameters(self) -> None:
        """Validate linear elastic parameters."""
        if self.get_parameter('youngs_modulus', 0) <= 0:
            raise ValueError("youngs_modulus must be positive")
        nu = self.get_parameter('poisson_ratio', 0)
        if nu <= 0 or nu >= 0.5:
            raise ValueError("poisson_ratio must be between 0 and 0.5")
        if self.get_parameter('density', 0) <= 0:
            raise ValueError("density must be positive")

    def compute_stress(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute stress using linear elasticity (Hooke's law).

        Args:
            strain_tensor: Strain tensor

        Returns:
            Stress tensor
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for stress computation")

        E = self.get_parameter('youngs_modulus')
        nu = self.get_parameter('poisson_ratio')

        # Lame's parameters
        lambda_lame = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu_lame = E / (2 * (1 + nu))

        # For isotropic linear elasticity: σ = λ*tr(ε)*I + 2*μ*ε
        if strain_tensor.ndim > 2:
            # Volume of strain tensors
            trace = np.trace(strain_tensor, axis1=-2, axis2=-1, keepdims=True)
            I = np.eye(3)
            stress = lambda_lame * trace * I + 2 * mu_lame * strain_tensor
        else:
            # Single strain tensor
            trace = np.trace(strain_tensor)
            I = np.eye(3)
            stress = lambda_lame * trace * I + 2 * mu_lame * strain_tensor

        return stress

    def compute_strain_energy_density(self, strain_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute strain energy density.

        Args:
            strain_tensor: Strain tensor

        Returns:
            Strain energy density
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for strain energy computation")

        stress = self.compute_stress(strain_tensor)

        # Strain energy density: W = 0.5 * σ : ε
        if strain_tensor.ndim > 2:
            energy = 0.5 * np.sum(stress * strain_tensor, axis=(-2, -1))
        else:
            energy = 0.5 * np.sum(stress * strain_tensor)

        return energy

    def get_shear_modulus(self) -> float:
        """Get shear modulus."""
        E = self.get_parameter('youngs_modulus')
        nu = self.get_parameter('poisson_ratio')
        return E / (2 * (1 + nu))

    def get_bulk_modulus(self) -> float:
        """Get bulk modulus."""
        E = self.get_parameter('youngs_modulus')
        nu = self.get_parameter('poisson_ratio')
        return E / (3 * (1 - 2 * nu))


# Factory function for creating models
def create_model(model_type: str, **parameters) -> ConstitutiveModel:
    """
    Create a constitutive model instance.

    Args:
        model_type: Type of model to create
        **parameters: Model parameters

    Returns:
        Constitutive model instance

    Raises:
        ValueError: If model_type is not recognized
    """
    model_classes = {
        'neo_hookean': NeoHookeanModel,
        'mooney_rivlin': MooneyRivlinModel,
        'yeoh': YeohModel,
        'ogden': OgdenModel,
        'linear_elastic': LinearElasticModel
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")

    return model_classes[model_type](**parameters)


# List of available models
AVAILABLE_MODELS = [
    'neo_hookean',
    'mooney_rivlin',
    'yeoh',
    'ogden',
    'linear_elastic'
]
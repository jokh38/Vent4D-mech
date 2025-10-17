"""
Type Definitions and Protocols for Vent4D-Mech

This module provides comprehensive type definitions, type aliases, and protocols
for the Vent4D-Mech framework to ensure type safety and improve code
documentation and IDE support.
"""

from typing import Protocol, TypeAlias, Union, Optional, Dict, Any, List, Tuple, Callable
from typing_extensions import TypeVar
import numpy as np
from pathlib import Path

# Type aliases for better readability and consistency
# These represent the most commonly used data structures in Vent4D-Mech

# Array types
StrainTensor: TypeAlias = np.ndarray  # Shape: (D, H, W, 3, 3)
StressTensor: TypeAlias = np.ndarray  # Shape: (D, H, W, 3, 3)
DeformationGradient: TypeAlias = np.ndarray  # Shape: (D, H, W, 3, 3)
DVF: TypeAlias = np.ndarray           # Shape: (D, H, W, 3) - Displacement Vector Field
Jacobian: TypeAlias = np.ndarray      # Shape: (D, H, W) - Scalar Jacobian determinant
VentilationField: TypeAlias = np.ndarray  # Shape: (D, H, W) - Ventilation values
Mask: TypeAlias = np.ndarray          # Shape: (D, H, W) - Binary or integer mask
ScalarField: TypeAlias = np.ndarray   # Shape: (D, H, W) - Generic scalar field
VectorField: TypeAlias = np.ndarray   # Shape: (D, H, W, 3) - Generic vector field
TensorField: TypeAlias = np.ndarray   # Shape: (D, H, W, 3, 3) - Generic tensor field

# Medical image types
Image3D: TypeAlias = np.ndarray       # Shape: (D, H, W) - 3D medical image
Image4D: TypeAlias = np.ndarray       # Shape: (T, D, H, W) - 4D time series image
CTImage: TypeAlias = np.ndarray       # Shape: (D, H, W) - CT scan image
MRImage: TypeAlias = np.ndarray       # Shape: (D, H, W) - MRI image

# Configuration and parameter types
ComponentConfig: TypeAlias = Dict[str, Any]
ModelParameters: TypeAlias = Dict[str, float]
MaterialParameters: TypeAlias = Dict[str, float]
SolverConfig: TypeAlias = Dict[str, Any]

# Result types
ProcessingResult: TypeAlias = Dict[str, Any]
ComputationResult: TypeAlias = Dict[str, Union[np.ndarray, float, str]]
AnalysisResult: TypeAlias = Dict[str, Any]

# File path types
FilePath: TypeAlias = Union[str, Path]
DirPath: TypeAlias = Union[str, Path]

# Numerical types
Real: TypeAlias = Union[float, np.floating]
Integer: TypeAlias = Union[int, np.integer]
ArrayLike: TypeAlias = Union[np.ndarray, List, Tuple]

# Generic type variables
T = TypeVar('T')
ComponentType = TypeVar('ComponentType')


class ComponentProtocol(Protocol):
    """
    Protocol defining the interface for all Vent4D-Mech components.

    This protocol ensures that all components implement the required
    methods for consistent behavior throughout the framework.
    """

    def process(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main processing method for the component.

        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments

        Returns:
            Dictionary containing processing results
        """
        ...

    def get_component_info(self) -> Dict[str, Any]:
        """
        Get component information.

        Returns:
            Dictionary containing component metadata
        """
        ...

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update component configuration.

        Args:
            new_config: New configuration parameters
        """
        ...

    @property
    def config(self) -> Dict[str, Any]:
        """Component configuration."""
        ...

    @property
    def component_name(self) -> str:
        """Component name."""
        ...


class RegistrationProtocol(Protocol):
    """Protocol for image registration components."""

    def register(self, fixed_image: Image3D, moving_image: Image3D,
                 initial_transform: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Register moving image to fixed image.

        Args:
            fixed_image: Fixed reference image
            moving_image: Moving image to be registered
            initial_transform: Optional initial transformation

        Returns:
            Registration results including transform and registered image
        """
        ...


class MechanicalModelProtocol(Protocol):
    """Protocol for mechanical modeling components."""

    def compute_stress(self, strain_tensor: StrainTensor,
                      deformation_gradient: Optional[DeformationGradient] = None) -> Dict[str, StressTensor]:
        """
        Compute stress from strain tensor.

        Args:
            strain_tensor: Input strain tensor
            deformation_gradient: Optional deformation gradient

        Returns:
            Dictionary containing stress tensors and derived quantities
        """
        ...

    def compute_strain_energy_density(self, strain_tensor: StrainTensor,
                                    deformation_gradient: Optional[DeformationGradient] = None) -> ScalarField:
        """
        Compute strain energy density.

        Args:
            strain_tensor: Input strain tensor
            deformation_gradient: Optional deformation gradient

        Returns:
            Strain energy density field
        """
        ...


class DeformationProtocol(Protocol):
    """Protocol for deformation analysis components."""

    def compute_strain_tensor(self, deformation_gradient: DeformationGradient,
                             strain_type: str = "green_lagrange") -> StrainTensor:
        """
        Compute strain tensor from deformation gradient.

        Args:
            deformation_gradient: Deformation gradient tensor
            strain_type: Type of strain measure

        Returns:
            Strain tensor
        """
        ...

    def compute_deformation_gradient(self, displacement_field: DVF) -> DeformationGradient:
        """
        Compute deformation gradient from displacement field.

        Args:
            displacement_field: Displacement vector field

        Returns:
            Deformation gradient tensor
        """
        ...


class VentilationProtocol(Protocol):
    """Protocol for ventilation analysis components."""

    def compute_ventilation(self, jacobian_determinant: Jacobian,
                           lung_mask: Optional[Mask] = None) -> Dict[str, VentilationField]:
        """
        Compute ventilation from Jacobian determinant.

        Args:
            jacobian_determinant: Jacobian determinant field
            lung_mask: Optional lung mask

        Returns:
            Dictionary containing ventilation fields
        """
        ...


class InverseProblemProtocol(Protocol):
    """Protocol for inverse problem solving components."""

    def estimate_parameters(self, observed_data: np.ndarray,
                          model_predictions: np.ndarray,
                          initial_parameters: ModelParameters) -> Dict[str, Any]:
        """
        Estimate model parameters from observed data.

        Args:
            observed_data: Observed measurements
            model_predictions: Current model predictions
            initial_parameters: Initial parameter estimates

        Returns:
            Parameter estimation results
        """
        ...


class FEMProtocol(Protocol):
    """Protocol for finite element method components."""

    def solve_mechanical_problem(self, mesh_data: Dict[str, Any],
                               material_properties: MaterialParameters,
                               boundary_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve mechanical problem using finite element method.

        Args:
            mesh_data: Mesh geometry and topology
            material_properties: Material property definitions
            boundary_conditions: Boundary condition definitions

        Returns:
            FEM solution results
        """
        ...


# Function type aliases
ImageProcessor: TypeAlias = Callable[[Image3D], Image3D]
TensorProcessor: TypeAlias = Callable[[TensorField], TensorField]
MetricFunction: TypeAlias = Callable[[np.ndarray, np.ndarray], float]
OptimizationFunction: TypeAlias = Callable[[ModelParameters], float]

# Configuration schema types
class ConfigSchema:
    """Base class for configuration schemas."""

    def validate(self, config: ComponentConfig) -> bool:
        """Validate configuration against schema."""
        raise NotImplementedError

    def get_defaults(self) -> ComponentConfig:
        """Get default configuration values."""
        raise NotImplementedError


# Common data structures
class BoundingBox:
    """Bounding box representation for spatial regions."""

    def __init__(self, origin: Tuple[int, int, int], size: Tuple[int, int, int]):
        self.origin = origin  # (x, y, z) origin coordinates
        self.size = size      # (width, height, depth) sizes

    @property
    def corners(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get opposite corners of the bounding box."""
        end = tuple(o + s for o, s in zip(self.origin, self.size))
        return self.origin, end


class SpatialMetadata:
    """Metadata for spatial data."""

    def __init__(self, spacing: Tuple[float, float, float],
                 origin: Tuple[float, float, float],
                 direction: Optional[np.ndarray] = None):
        self.spacing = spacing      # Voxel spacing in mm
        self.origin = origin        # Origin in patient coordinates
        self.direction = direction  # Direction cosines matrix


class ImageMetadata(SpatialMetadata):
    """Metadata for medical images."""

    def __init__(self, spacing: Tuple[float, float, float],
                 origin: Tuple[float, float, float],
                 direction: Optional[np.ndarray] = None,
                 modality: Optional[str] = None,
                 acquisition_date: Optional[str] = None):
        super().__init__(spacing, origin, direction)
        self.modality = modality        # Image modality (CT, MR, etc.)
        self.acquisition_date = acquisition_date


# Validation function types
ValidationFunction: TypeAlias = Callable[[np.ndarray], bool]
ArrayValidator: TypeAlias = Callable[[np.ndarray, Optional[str]], None]

# Performance metrics types
PerformanceMetrics: TypeAlias = Dict[str, Union[float, str, Dict[str, float]]]
TimingInfo: TypeAlias = Dict[str, float]

# Export type checking function
def validate_tensor_array(array: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None,
                         expected_dims: Optional[int] = None,
                         allow_nan: bool = False, allow_inf: bool = False) -> bool:
    """
    Validate numpy array properties for Vent4D-Mech tensors.

    Args:
        array: Array to validate
        expected_shape: Expected shape (None for flexible)
        expected_dims: Expected number of dimensions
        allow_nan: Whether NaN values are allowed
        allow_inf: Whether infinite values are allowed

    Returns:
        True if array passes all validations
    """
    if not isinstance(array, np.ndarray):
        return False

    if expected_dims is not None and array.ndim != expected_dims:
        return False

    if expected_shape is not None:
        for i, expected_size in enumerate(expected_shape):
            if expected_size is not None and array.shape[i] != expected_size:
                return False

    if not allow_nan and np.any(np.isnan(array)):
        return False

    if not allow_inf and np.any(np.isinf(array)):
        return False

    return True


# Type checking utilities
def is_strain_tensor(array: np.ndarray) -> bool:
    """Check if array is a valid strain tensor."""
    return validate_tensor_array(array, expected_dims=5, expected_shape=(None, None, None, 3, 3))


def is_stress_tensor(array: np.ndarray) -> bool:
    """Check if array is a valid stress tensor."""
    return validate_tensor_array(array, expected_dims=5, expected_shape=(None, None, None, 3, 3))


def is_deformation_gradient(array: np.ndarray) -> bool:
    """Check if array is a valid deformation gradient."""
    return validate_tensor_array(array, expected_dims=5, expected_shape=(None, None, None, 3, 3))


def is_dvf(array: np.ndarray) -> bool:
    """Check if array is a valid displacement vector field."""
    return validate_tensor_array(array, expected_dims=4, expected_shape=(None, None, None, 3))


def is_3d_image(array: np.ndarray) -> bool:
    """Check if array is a valid 3D image."""
    return validate_tensor_array(array, expected_dims=3)


def is_4d_image(array: np.ndarray) -> bool:
    """Check if array is a valid 4D image."""
    return validate_tensor_array(array, expected_dims=4)


def is_scalar_field(array: np.ndarray) -> bool:
    """Check if array is a valid scalar field."""
    return validate_tensor_array(array, expected_dims=3)


def is_mask(array: np.ndarray) -> bool:
    """Check if array is a valid mask."""
    if not validate_tensor_array(array, expected_dims=3):
        return False
    # Masks should contain integers or boolean values
    return np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)
"""
Configuration Schemas with Pydantic Validation

This module provides Pydantic-based configuration schemas for the Vent4D-Mech framework,
offering strong type safety, automatic validation, and detailed error messages
for all configuration parameters.

Key Features:
- Type-safe configuration parameters
- Automatic validation with detailed error messages
- IDE autocompletion and type checking support
- Self-documenting configuration schemas
- JSON/YAML serialization support
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Dict, List, Any, Optional, Union, Literal
from pathlib import Path
import logging


# Base Configuration Models
class BaseConfigModel(BaseModel):
    """Base configuration model with common functionality."""

    model_config = ConfigDict(
        extra="forbid",  # Prevent unknown fields
        validate_assignment=True,  # Validate on field assignment
        use_enum_values=True,  # Use enum values instead of objects
        validate_by_name=True,  # Allow both alias and field names
    )


# Performance Configuration
class MemoryManagementConfig(BaseConfigModel):
    """Memory management configuration."""
    chunk_size: int = Field(
        default=64,
        gt=0,
        description="Size of processing chunks in voxels"
    )
    cache_size: int = Field(
        default=1024,
        gt=0,
        description="Cache size in MB"
    )
    memory_limit: int = Field(
        default=8192,
        gt=0,
        description="Memory limit in MB"
    )


class OptimizationConfig(BaseConfigModel):
    """Optimization settings."""
    vectorized_operations: bool = Field(
        default=True,
        description="Enable vectorized operations"
    )
    sparse_matrices: bool = Field(
        default=True,
        description="Use sparse matrices when beneficial"
    )
    adaptive_solvers: bool = Field(
        default=True,
        description="Use adaptive solvers when available"
    )


class PerformanceConfig(BaseConfigModel):
    """Performance configuration settings."""
    gpu_acceleration: bool = Field(
        default=True,
        description="Enable GPU acceleration when available"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing"
    )
    num_processes: Union[int, Literal["auto"]] = Field(
        default="auto",
        description="Number of processes or 'auto' for automatic detection"
    )
    memory_management: MemoryManagementConfig = Field(
        default_factory=MemoryManagementConfig,
        description="Memory management settings"
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization settings"
    )

    @field_validator('num_processes')
    @classmethod
    def validate_num_processes(cls, v):
        if v != "auto" and (not isinstance(v, int) or v <= 0):
            raise ValueError("num_processes must be 'auto' or a positive integer")
        return v


# Registration Configuration
class VoxelMorphConfig(BaseConfigModel):
    """VoxelMorph registration configuration."""
    model_type: Literal["unsupervised", "supervised"] = Field(
        default="unsupervised",
        description="Type of VoxelMorph model"
    )
    input_shape: List[int] = Field(
        default=[128, 128, 128],
        description="Input image shape [D, H, W]"
    )
    batch_size: int = Field(
        default=1,
        gt=0,
        description="Training batch size"
    )
    learning_rate: float = Field(
        default=0.0001,
        gt=0,
        description="Learning rate for training"
    )
    epochs: int = Field(
        default=1000,
        gt=0,
        description="Number of training epochs"
    )
    loss_weights: Dict[str, float] = Field(
        default={"image_similarity": 1.0, "regularization": 0.01},
        description="Loss function weights"
    )

    @field_validator('input_shape')
    @classmethod
    def validate_input_shape(cls, v):
        if len(v) != 3 or any(not isinstance(dim, int) or dim <= 0 for dim in v):
            raise ValueError("input_shape must be a list of 3 positive integers")
        return v


class SimpleITKConfig(BaseConfigModel):
    """SimpleITK registration configuration."""
    interpolator: Literal["Linear", "NearestNeighbor", "BSpline", "Gaussian"] = Field(
        default="Linear",
        description="Image interpolation method"
    )
    metric: Literal["MeanSquares", "MutualInformation", "Correlation", "Demons"] = Field(
        default="MeanSquares",
        description="Registration metric"
    )
    optimizer: Literal["LBFGS", "GradientDescent", "Amoeba"] = Field(
        default="LBFGS",
        description="Optimization algorithm"
    )
    shrink_factors: List[int] = Field(
        default=[8, 4, 2, 1],
        description="Multi-resolution shrink factors"
    )
    smooth_sigmas: List[float] = Field(
        default=[4, 2, 1, 0],
        description="Multi-resolution smoothing sigmas"
    )
    sampling_rates: List[float] = Field(
        default=[0.25, 0.5, 0.75, 1.0],
        description="Multi-resolution sampling rates"
    )


class DeformableConfig(BaseConfigModel):
    """Deformable registration configuration."""
    grid_spacing: List[int] = Field(
        default=[10, 10, 10],
        description="Deformation field grid spacing"
    )
    regularization_weight: float = Field(
        default=0.01,
        ge=0,
        description="Regularization weight for deformation field"
    )
    max_iterations: int = Field(
        default=100,
        gt=0,
        description="Maximum number of iterations"
    )
    convergence_threshold: float = Field(
        default=1e-6,
        gt=0,
        description="Convergence threshold"
    )

    @field_validator('grid_spacing')
    @classmethod
    def validate_grid_spacing(cls, v):
        if len(v) != 3 or any(not isinstance(val, int) or val <= 0 for val in v):
            raise ValueError("grid_spacing must be a list of 3 positive integers")
        return v


class RegistrationParametersConfig(BaseConfigModel):
    """Registration method parameters."""
    voxelmorph: Optional[VoxelMorphConfig] = Field(
        default=None,
        description="VoxelMorph configuration"
    )
    simpleitk: Optional[SimpleITKConfig] = Field(
        default=None,
        description="SimpleITK configuration"
    )
    deformable: Optional[DeformableConfig] = Field(
        default=None,
        description="Deformable registration configuration"
    )


class PreprocessingConfig(BaseConfigModel):
    """Image preprocessing configuration."""
    normalize_intensity: bool = Field(
        default=True,
        description="Normalize image intensity values"
    )
    clip_outliers: bool = Field(
        default=True,
        description="Clip intensity outliers"
    )
    resample_to_iso: bool = Field(
        default=True,
        description="Resample to isotropic spacing"
    )
    target_spacing: List[float] = Field(
        default=[1.0, 1.0, 1.0],
        description="Target isotropic spacing"
    )

    @field_validator('target_spacing')
    @classmethod
    def validate_target_spacing(cls, v):
        if len(v) != 3 or any(not isinstance(val, (int, float)) or val <= 0 for val in v):
            raise ValueError("target_spacing must be a list of 3 positive numbers")
        return v


class RegistrationConfig(BaseConfigModel):
    """Registration configuration settings."""
    method: Literal["voxelmorph", "simpleitk", "deformable"] = Field(
        default="voxelmorph",
        description="Registration method"
    )
    gpu_acceleration: bool = Field(
        default=True,
        description="Enable GPU acceleration for registration"
    )
    parameters: RegistrationParametersConfig = Field(
        default_factory=RegistrationParametersConfig,
        description="Registration method parameters"
    )
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Image preprocessing settings"
    )


# Material Parameters
class MaterialParameters(BaseConfigModel):
    """Base material parameters with density."""
    density: float = Field(
        default=1.05,
        gt=0,
        description="Material density in g/cm³"
    )


class NeoHookeanParams(MaterialParameters):
    """Neo-Hookean material parameters."""
    C10: float = Field(
        default=0.135,
        gt=0,
        description="Neo-Hookean material parameter C10 in kPa"
    )

    @field_validator('C10')
    @classmethod
    def validate_c10(cls, v):
        if v > 100:  # Very high for lung tissue
            logging.warning("C10 parameter seems very high for lung tissue")
        return v


class MooneyRivlinParams(MaterialParameters):
    """Mooney-Rivlin material parameters."""
    C10: float = Field(
        default=0.135,
        ge=0,
        description="Mooney-Rivlin material parameter C10 in kPa"
    )
    C01: float = Field(
        default=0.035,
        ge=0,
        description="Mooney-Rivlin material parameter C01 in kPa"
    )

    @field_validator('C10', 'C01')
    @classmethod
    def validate_mooney_rivlin_params(cls, v):
        if v > 100:  # Very high for lung tissue
            logging.warning("Mooney-Rivlin parameter seems very high for lung tissue")
        return v


class YeohParams(MaterialParameters):
    """Yeoh material parameters."""
    C10: float = Field(
        default=0.135,
        gt=0,
        description="Yeoh material parameter C10 in kPa"
    )
    C20: float = Field(
        default=0.015,
        description="Yeoh material parameter C20 in kPa"
    )
    C30: float = Field(
        default=0.001,
        description="Yeoh material parameter C30 in kPa"
    )


class OgdenParams(MaterialParameters):
    """Ogden material parameters."""
    mu1: float = Field(
        default=0.5,
        gt=0,
        description="Ogden material parameter μ1 in kPa"
    )
    alpha1: float = Field(
        default=2.0,
        description="Ogden material parameter α1 (dimensionless)"
    )
    mu2: float = Field(
        default=0.1,
        gt=0,
        description="Ogden material parameter μ2 in kPa"
    )
    alpha2: float = Field(
        default=-2.0,
        description="Ogden material parameter α2 (dimensionless)"
    )


class LinearElasticParams(MaterialParameters):
    """Linear elastic material parameters."""
    youngs_modulus: float = Field(
        default=5.0,
        gt=0,
        le=1000,
        description="Young's modulus in kPa"
    )
    poisson_ratio: float = Field(
        default=0.45,
        gt=0,
        lt=0.5,
        description="Poisson's ratio (dimensionless)"
    )


class MaterialParametersConfig(BaseConfigModel):
    """Configuration for all material parameters."""
    neo_hookean: Optional[NeoHookeanParams] = Field(
        default=None,
        description="Neo-Hookean material parameters"
    )
    mooney_rivlin: Optional[MooneyRivlinParams] = Field(
        default=None,
        description="Mooney-Rivlin material parameters"
    )
    yeoh: Optional[YeohParams] = Field(
        default=None,
        description="Yeoh material parameters"
    )
    ogden: Optional[OgdenParams] = Field(
        default=None,
        description="Ogden material parameters"
    )
    linear_elastic: Optional[LinearElasticParams] = Field(
        default=None,
        description="Linear elastic material parameters"
    )


# Mechanical Configuration
class BoundaryConditionsConfig(BaseConfigModel):
    """Boundary conditions configuration."""
    type: Literal["displacement_controlled", "force_controlled", "mixed"] = Field(
        default="displacement_controlled",
        description="Type of boundary conditions"
    )
    fixed_surfaces: List[str] = Field(
        default=["chest_wall", "mediastinum"],
        description="Fixed surface names"
    )
    loaded_surfaces: List[str] = Field(
        default=["pleural_surface"],
        description="Loaded surface names"
    )
    load_magnitude: float = Field(
        default=0.1,
        description="Load magnitude in kPa"
    )


class SolverConfig(BaseConfigModel):
    """Solver configuration."""
    type: Literal["linear", "nonlinear"] = Field(
        default="nonlinear",
        description="Solver type"
    )
    tolerance: float = Field(
        default=1e-6,
        gt=0,
        description="Convergence tolerance"
    )
    max_iterations: int = Field(
        default=50,
        gt=0,
        description="Maximum number of iterations"
    )
    line_search: bool = Field(
        default=True,
        description="Enable line search"
    )


class MechanicalConfig(BaseConfigModel):
    """Mechanical modeling configuration."""
    constitutive_model: Literal["neo_hookean", "mooney_rivlin", "yeoh", "ogden", "linear_elastic"] = Field(
        default="mooney_rivlin",
        description="Constitutive model type"
    )
    material_parameters: MaterialParametersConfig = Field(
        default_factory=MaterialParametersConfig,
        description="Material parameters for different models"
    )
    boundary_conditions: BoundaryConditionsConfig = Field(
        default_factory=BoundaryConditionsConfig,
        description="Boundary conditions"
    )
    solver: SolverConfig = Field(
        default_factory=SolverConfig,
        description="Solver configuration"
    )


# Logging Configuration
class FileLoggingConfig(BaseConfigModel):
    """File logging configuration."""
    enabled: bool = Field(
        default=True,
        description="Enable file logging"
    )
    file_path: str = Field(
        default="./logs/vent4d_mech.log",
        description="Log file path"
    )
    max_file_size_mb: int = Field(
        default=100,
        gt=0,
        description="Maximum log file size in MB"
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        description="Number of backup files to keep"
    )


class ConsoleLoggingConfig(BaseConfigModel):
    """Console logging configuration."""
    enabled: bool = Field(
        default=True,
        description="Enable console logging"
    )
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Console logging level"
    )


class PerformanceLoggingConfig(BaseConfigModel):
    """Performance logging configuration."""
    enabled: bool = Field(
        default=True,
        description="Enable performance logging"
    )
    log_memory_usage: bool = Field(
        default=True,
        description="Log memory usage"
    )
    log_timing: bool = Field(
        default=True,
        description="Log timing information"
    )


class LoggingConfig(BaseConfigModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Default logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file_logging: FileLoggingConfig = Field(
        default_factory=FileLoggingConfig,
        description="File logging settings"
    )
    console_logging: ConsoleLoggingConfig = Field(
        default_factory=ConsoleLoggingConfig,
        description="Console logging settings"
    )
    performance_logging: PerformanceLoggingConfig = Field(
        default_factory=PerformanceLoggingConfig,
        description="Performance logging settings"
    )


# Validation Configuration
class ValidationChecksConfig(BaseConfigModel):
    """Validation checks configuration."""
    input_validation: bool = Field(
        default=True,
        description="Enable input validation"
    )
    output_validation: bool = Field(
        default=True,
        description="Enable output validation"
    )
    parameter_validation: bool = Field(
        default=True,
        description="Enable parameter validation"
    )
    convergence_validation: bool = Field(
        default=True,
        description="Enable convergence validation"
    )


class ErrorHandlingConfig(BaseConfigModel):
    """Error handling configuration."""
    raise_on_error: bool = Field(
        default=False,
        description="Raise exception on validation error"
    )
    log_errors: bool = Field(
        default=True,
        description="Log validation errors"
    )
    continue_on_warning: bool = Field(
        default=True,
        description="Continue processing on warnings"
    )


class ValidationConfig(BaseConfigModel):
    """Validation configuration."""
    enabled: bool = Field(
        default=True,
        description="Enable validation"
    )
    strict_mode: bool = Field(
        default=False,
        description="Enable strict validation mode"
    )
    checks: ValidationChecksConfig = Field(
        default_factory=ValidationChecksConfig,
        description="Validation checks"
    )
    error_handling: ErrorHandlingConfig = Field(
        default_factory=ErrorHandlingConfig,
        description="Error handling settings"
    )


# Main Vent4D-Mech Configuration
class FrameworkConfig(BaseConfigModel):
    """Framework information."""
    name: str = Field(
        default="Vent4D-Mech",
        description="Framework name"
    )
    description: str = Field(
        default="Python-based Lung Tissue Dynamics Modeling",
        description="Framework description"
    )


class Vent4DMechConfig(BaseConfigModel):
    """Main Vent4D-Mech configuration schema."""
    version: str = Field(
        default="0.1.0",
        description="Configuration version"
    )
    framework: FrameworkConfig = Field(
        default_factory=FrameworkConfig,
        description="Framework information"
    )
    registration: RegistrationConfig = Field(
        default_factory=RegistrationConfig,
        description="Registration configuration"
    )
    mechanical: MechanicalConfig = Field(
        default_factory=MechanicalConfig,
        description="Mechanical modeling configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Validation configuration"
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional sections for extensibility
        validate_assignment=True
    )

    @model_validator(mode='after')
    def validate_model_consistency(self):
        """Validate consistency between configuration sections."""
        # Check if registration GPU setting matches performance GPU setting
        reg_gpu = self.registration.gpu_acceleration
        perf_gpu = self.performance.gpu_acceleration
        if reg_gpu and not perf_gpu:
            logging.warning(
                "Registration GPU acceleration enabled but performance GPU disabled. "
                "Performance GPU setting will override."
            )

        return self

    def get_material_parameters(self, model_name: str) -> Optional[MaterialParameters]:
        """
        Get material parameters for a specific constitutive model.

        Args:
            model_name: Name of constitutive model

        Returns:
            Material parameters for the specified model or None if not found
        """
        material_params = self.mechanical.material_parameters
        model_map = {
            'neo_hookean': material_params.neo_hookean,
            'mooney_rivlin': material_params.mooney_rivlin,
            'yeoh': material_params.yeoh,
            'ogden': material_params.ogden,
            'linear_elastic': material_params.linear_elastic
        }
        return model_map.get(model_name)

    def validate_for_model(self, model_name: str) -> bool:
        """
        Validate configuration for a specific constitutive model.

        Args:
            model_name: Name of constitutive model to validate for

        Returns:
            True if configuration is valid for the specified model
        """
        # Check if material parameters exist for the model
        material_params = self.get_material_parameters(model_name)
        if material_params is None:
            raise ValueError(f"No material parameters defined for model: {model_name}")

        # Check if mechanical model is set to the same model
        if self.mechanical.constitutive_model != model_name:
            logging.warning(
                f"Constitutive model mismatch: mechanical.config has '{self.mechanical.constitutive_model}' "
                f"but validation requested for '{model_name}'"
            )

        return True


# Configuration Factory Functions
def create_default_config() -> Vent4DMechConfig:
    """
    Create default Vent4D-Mech configuration.

    Returns:
        Default configuration object
    """
    return Vent4DMechConfig()


def create_minimal_config() -> Vent4DMechConfig:
    """
    Create minimal configuration for basic functionality.

    Returns:
        Minimal configuration object
    """
    return Vent4DMechConfig(
        registration=RegistrationConfig(
            method="simpleitk",
            gpu_acceleration=False
        ),
        mechanical=MechanicalConfig(
            constitutive_model="neo_hookean",
            material_parameters=MaterialParametersConfig(
                neo_hookean=NeoHookeanParams()
            )
        ),
        performance=PerformanceConfig(
            gpu_acceleration=False,
            parallel_processing=False
        ),
        logging=LoggingConfig(
            level="INFO",
            console_logging=ConsoleLoggingConfig(enabled=True)
        )
    )


def validate_config_dict(config_dict: Dict[str, Any]) -> Vent4DMechConfig:
    """
    Validate a configuration dictionary against the schema.

    Args:
        config_dict: Configuration dictionary to validate

    Returns:
        Validated configuration object

    Raises:
        pydantic.ValidationError: If configuration is invalid
    """
    return Vent4DMechConfig(**config_dict)


# Utility Functions
def config_to_dict(config: Vent4DMechConfig) -> Dict[str, Any]:
    """
    Convert configuration object to dictionary.

    Args:
        config: Configuration object

    Returns:
        Configuration dictionary
    """
    return config.dict()


def config_to_json(config: Vent4DMechConfig) -> str:
    """
    Convert configuration object to JSON string.

    Args:
        config: Configuration object

    Returns:
        JSON string representation
    """
    return config.json(indent=2)


def config_from_json(json_str: str) -> Vent4DMechConfig:
    """
    Create configuration object from JSON string.

    Args:
        json_str: JSON string representation

    Returns:
        Configuration object
    """
    return Vent4DMechConfig.parse_raw(json_str)
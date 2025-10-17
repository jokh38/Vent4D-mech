"""
Configuration Validation

This module provides validation capabilities for configuration parameters
in the Vent4D-Mech framework, ensuring that all settings are valid
and consistent before they are used.

Features both legacy validation and Pydantic-based validation for
enhanced type safety and detailed error messages.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Try to import Pydantic - optional for backward compatibility
try:
    from pydantic import ValidationError as PydanticValidationError
    from .schemas import Vent4DMechConfig, validate_config_dict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    Vent4DMechConfig = None
    validate_config_dict = None
    PydanticValidationError = None


class ValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class ConfigValidation:
    """
    Configuration validator for Vent4D-Mech.

    This class provides comprehensive validation capabilities for all configuration
    parameters, ensuring type safety, value ranges, and logical consistency
    across the framework.

    Supports both legacy validation and enhanced Pydantic-based validation.
    """

    def __init__(self, strict_mode: bool = False, use_pydantic: Optional[bool] = None):
        """
        Initialize ConfigValidation.

        Args:
            strict_mode: Whether to raise exceptions for all validation errors
            use_pydantic: Whether to use Pydantic validation.
                         If None, will use Pydantic if available.
        """
        self.logger = logging.getLogger(__name__)
        self.strict_mode = strict_mode
        self.validation_errors = []
        self.validation_warnings = []

        # Determine whether to use Pydantic
        if use_pydantic is None:
            self.use_pydantic = PYDANTIC_AVAILABLE
        else:
            if use_pydantic and not PYDANTIC_AVAILABLE:
                self.logger.warning(
                    "Pydantic validation requested but Pydantic is not available. "
                    "Falling back to legacy validation."
                )
                self.use_pydantic = False
            else:
                self.use_pydantic = use_pydantic

        if self.use_pydantic:
            self.logger.info("Using Pydantic-based configuration validation")
        else:
            self.logger.info("Using legacy configuration validation")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate complete configuration dictionary.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            ValidationError: If strict_mode is enabled and validation fails
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()

        self.logger.info("Starting configuration validation")

        if self.use_pydantic and PYDANTIC_AVAILABLE:
            return self._validate_with_pydantic(config)
        else:
            return self._validate_legacy(config)

    def _validate_with_pydantic(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration using Pydantic schemas.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            ValidationError: If strict_mode is enabled and validation fails
        """
        try:
            # Validate with Pydantic
            validated_config = validate_config_dict(config)
            self.logger.info("Pydantic validation passed")
            return True

        except PydanticValidationError as e:
            # Convert Pydantic errors to legacy format for consistency
            self._convert_pydantic_errors(e)

            # Log validation results
            self._log_validation_results()

            # Determine success
            is_valid = len(self.validation_errors) == 0

            if self.strict_mode and not is_valid:
                error_msg = f"Configuration validation failed: {self.validation_errors}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg)

            return is_valid

    def _validate_legacy(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration using legacy validation logic.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            ValidationError: If strict_mode is enabled and validation fails
        """
        # Validate structure
        self._validate_structure(config)

        # Validate each section
        if 'registration' in config:
            self._validate_registration_config(config['registration'])

        if 'mechanical' in config:
            self._validate_mechanical_config(config['mechanical'])

        if 'performance' in config:
            self._validate_performance_config(config['performance'])

        if 'logging' in config:
            self._validate_logging_config(config['logging'])

        if 'validation' in config:
            self._validate_validation_section(config['validation'])

        if 'data_handling' in config:
            self._validate_data_handling_config(config['data_handling'])

        # Log validation results
        self._log_validation_results()

        # Determine success
        is_valid = len(self.validation_errors) == 0

        if self.strict_mode and not is_valid:
            error_msg = f"Configuration validation failed: {self.validation_errors}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)

        return is_valid

    def _convert_pydantic_errors(self, pydantic_error: PydanticValidationError) -> None:
        """
        Convert Pydantic validation errors to legacy format.

        Args:
            pydantic_error: Pydantic ValidationError exception
        """
        for error in pydantic_error.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            message = error['msg']
            error_type = error['type']

            # Format error message
            if error_type == 'value_error.missing':
                formatted_error = f"Missing required field: {field_path}"
            elif error_type == 'value_error.extra':
                formatted_error = f"Unknown field: {field_path}"
            elif error_type == 'type_error':
                formatted_error = f"Invalid type for {field_path}: {message}"
            elif error_type == 'value_error.const':
                formatted_error = f"Invalid value for {field_path}: {message}"
            else:
                formatted_error = f"Validation error for {field_path}: {message}"

            # Add to appropriate list based on severity
            if error_type in ['value_error.extra']:
                self.validation_warnings.append(formatted_error)
            else:
                self.validation_errors.append(formatted_error)

    def get_pydantic_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get the Pydantic schema for configuration.

        Returns:
            Pydantic schema dictionary or None if Pydantic is not available
        """
        if not PYDANTIC_AVAILABLE:
            return None

        return Vent4DMechConfig.schema()

    def validate_with_schema(self, config: Dict[str, Any], schema_section: Optional[str] = None) -> bool:
        """
        Validate configuration using Pydantic schema validation.

        Args:
            config: Configuration dictionary to validate
            schema_section: Optional section name to validate specific section

        Returns:
            True if configuration is valid

        Raises:
            ValidationError: If validation fails
        """
        if not PYDANTIC_AVAILABLE:
            self.logger.warning("Pydantic not available, falling back to legacy validation")
            return self.validate_config(config)

        try:
            if schema_section:
                # Validate specific section
                if schema_section not in config:
                    self.validation_errors.append(f"Section '{schema_section}' not found in configuration")
                    return False

                section_config = {schema_section: config[schema_section]}
                temp_config = Vent4DMechConfig(**section_config)
            else:
                # Validate entire configuration
                temp_config = Vent4DMechConfig(**config)

            self.logger.info(f"Schema validation passed for section: {schema_section or 'full config'}")
            return True

        except PydanticValidationError as e:
            self._convert_pydantic_errors(e)
            self._log_validation_results()

            if self.strict_mode and self.validation_errors:
                error_msg = f"Schema validation failed: {self.validation_errors}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg)

            return False

    def _validate_structure(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure."""
        if not isinstance(config, dict):
            self.validation_errors.append("Configuration must be a dictionary")
            return

        # Check for required top-level sections
        recommended_sections = ['registration', 'mechanical', 'performance', 'logging']
        for section in recommended_sections:
            if section not in config:
                self.validation_warnings.append(f"Missing recommended section: {section}")

        # Check for unknown top-level sections
        known_sections = [
            'version', 'framework', 'registration', 'mechanical', 'deformation',
            'inverse', 'ventilation', 'microstructure', 'fem', 'performance',
            'data_handling', 'logging', 'validation', 'output', 'quality_assurance'
        ]

        for section in config.keys():
            if section not in known_sections:
                self.validation_warnings.append(f"Unknown configuration section: {section}")

    def _validate_registration_config(self, registration_config: Dict[str, Any]) -> None:
        """Validate registration configuration."""
        # Validate method
        if 'method' in registration_config:
            method = registration_config['method']
            valid_methods = ['voxelmorph', 'simpleitk', 'deformable']
            if method not in valid_methods:
                self.validation_errors.append(
                    f"Invalid registration method: {method}. Valid options: {valid_methods}"
                )

        # Validate GPU acceleration
        if 'gpu_acceleration' in registration_config:
            if not isinstance(registration_config['gpu_acceleration'], bool):
                self.validation_errors.append("gpu_acceleration must be a boolean")

        # Validate parameters section
        if 'parameters' in registration_config:
            self._validate_registration_parameters(registration_config['parameters'])

        # Validate preprocessing
        if 'preprocessing' in registration_config:
            self._validate_preprocessing_config(registration_config['preprocessing'])

    def _validate_registration_parameters(self, params: Dict[str, Any]) -> None:
        """Validate registration method parameters."""
        if not isinstance(params, dict):
            self.validation_errors.append("Registration parameters must be a dictionary")
            return

        # Validate voxelmorph parameters
        if 'voxelmorph' in params:
            vm_config = params['voxelmorph']
            self._validate_voxelmorph_config(vm_config)

        # Validate SimpleITK parameters
        if 'simpleitk' in params:
            sitk_config = params['simpleitk']
            self._validate_simpleitk_config(sitk_config)

    def _validate_voxelmorph_config(self, vm_config: Dict[str, Any]) -> None:
        """Validate VoxelMorph configuration."""
        required_floats = ['learning_rate']
        required_ints = ['batch_size', 'epochs']
        required_lists = ['input_shape']

        for param in required_floats:
            if param in vm_config:
                if not isinstance(vm_config[param], (int, float)) or vm_config[param] <= 0:
                    self.validation_errors.append(f"voxelmorph.{param} must be a positive number")

        for param in required_ints:
            if param in vm_config:
                if not isinstance(vm_config[param], int) or vm_config[param] <= 0:
                    self.validation_errors.append(f"voxelmorph.{param} must be a positive integer")

        if 'input_shape' in vm_config:
            shape = vm_config['input_shape']
            if not isinstance(shape, list) or len(shape) != 3:
                self.validation_errors.append("voxelmorph.input_shape must be a list of 3 integers")
            elif not all(isinstance(s, int) and s > 0 for s in shape):
                self.validation_errors.append("voxelmorph.input_shape values must be positive integers")

    def _validate_simpleitk_config(self, sitk_config: Dict[str, Any]) -> None:
        """Validate SimpleITK configuration."""
        valid_interpolators = ['Linear', 'NearestNeighbor', 'BSpline', 'Gaussian']
        valid_metrics = ['MeanSquares', 'MutualInformation', 'Correlation', 'Demons']
        valid_optimizers = ['LBFGS', 'GradientDescent', 'Amoeba']

        if 'interpolator' in sitk_config:
            if sitk_config['interpolator'] not in valid_interpolators:
                self.validation_errors.append(
                    f"Invalid interpolator: {sitk_config['interpolator']}. "
                    f"Valid options: {valid_interpolators}"
                )

        if 'metric' in sitk_config:
            if sitk_config['metric'] not in valid_metrics:
                self.validation_errors.append(
                    f"Invalid metric: {sitk_config['metric']}. Valid options: {valid_metrics}"
                )

        if 'optimizer' in sitk_config:
            if sitk_config['optimizer'] not in valid_optimizers:
                self.validation_errors.append(
                    f"Invalid optimizer: {sitk_config['optimizer']}. Valid options: {valid_optimizers}"
                )

    def _validate_preprocessing_config(self, preprocessing_config: Dict[str, Any]) -> None:
        """Validate preprocessing configuration."""
        boolean_params = ['normalize_intensity', 'clip_outliers', 'resample_to_iso']
        for param in boolean_params:
            if param in preprocessing_config:
                if not isinstance(preprocessing_config[param], bool):
                    self.validation_errors.append(f"preprocessing.{param} must be a boolean")

        if 'target_spacing' in preprocessing_config:
            spacing = preprocessing_config['target_spacing']
            if not isinstance(spacing, list) or len(spacing) != 3:
                self.validation_errors.append("preprocessing.target_spacing must be a list of 3 positive numbers")
            elif not all(isinstance(s, (int, float)) and s > 0 for s in spacing):
                self.validation_errors.append("preprocessing.target_spacing values must be positive numbers")

    def _validate_mechanical_config(self, mechanical_config: Dict[str, Any]) -> None:
        """Validate mechanical configuration."""
        # Validate constitutive model
        if 'constitutive_model' in mechanical_config:
            model = mechanical_config['constitutive_model']
            valid_models = ['neo_hookean', 'mooney_rivlin', 'yeoh', 'ogden', 'linear_elastic']
            if model not in valid_models:
                self.validation_errors.append(
                    f"Invalid constitutive model: {model}. Valid options: {valid_models}"
                )

        # Validate material parameters
        if 'material_parameters' in mechanical_config:
            self._validate_material_parameters(mechanical_config['material_parameters'])

        # Validate boundary conditions
        if 'boundary_conditions' in mechanical_config:
            self._validate_boundary_conditions(mechanical_config['boundary_conditions'])

        # Validate solver
        if 'solver' in mechanical_config:
            self._validate_solver_config(mechanical_config['solver'])

    def _validate_material_parameters(self, material_params: Dict[str, Any]) -> None:
        """Validate material parameters."""
        if not isinstance(material_params, dict):
            self.validation_errors.append("material_parameters must be a dictionary")
            return

        valid_models = ['neo_hookean', 'mooney_rivlin', 'yeoh', 'ogden', 'linear_elastic']

        for model_name, params in material_params.items():
            if model_name not in valid_models:
                self.validation_warnings.append(f"Unknown material model: {model_name}")
                continue

            if not isinstance(params, dict):
                self.validation_errors.append(f"material_parameters.{model_name} must be a dictionary")
                continue

            # Validate specific model parameters
            if model_name == 'linear_elastic':
                self._validate_linear_elastic_params(params)
            elif model_name == 'neo_hookean':
                self._validate_neo_hookean_params(params)
            elif model_name == 'mooney_rivlin':
                self._validate_mooney_rivlin_params(params)

    def _validate_linear_elastic_params(self, params: Dict[str, Any]) -> None:
        """Validate linear elastic material parameters."""
        if 'youngs_modulus' in params:
            E = params['youngs_modulus']
            if not isinstance(E, (int, float)) or E <= 0:
                self.validation_errors.append("youngs_modulus must be a positive number")
            elif E > 1000:  # kPa - very high for lung tissue
                self.validation_warnings.append("youngs_modulus seems very high for lung tissue")

        if 'poisson_ratio' in params:
            nu = params['poisson_ratio']
            if not isinstance(nu, (int, float)) or nu <= 0 or nu >= 0.5:
                self.validation_errors.append("poisson_ratio must be between 0 and 0.5")

    def _validate_neo_hookean_params(self, params: Dict[str, Any]) -> None:
        """Validate Neo-Hookean material parameters."""
        if 'C10' in params:
            C10 = params['C10']
            if not isinstance(C10, (int, float)) or C10 <= 0:
                self.validation_errors.append("C10 parameter must be positive")
            elif C10 > 100:  # kPa - very high for lung tissue
                self.validation_warnings.append("C10 parameter seems very high for lung tissue")

    def _validate_mooney_rivlin_params(self, params: Dict[str, Any]) -> None:
        """Validate Mooney-Rivlin material parameters."""
        for param in ['C10', 'C01']:
            if param in params:
                value = params[param]
                if not isinstance(value, (int, float)) or value < 0:
                    self.validation_errors.append(f"{param} parameter must be non-negative")
                elif value > 100:  # kPa - very high for lung tissue
                    self.validation_warnings.append(f"{param} parameter seems very high for lung tissue")

    def _validate_boundary_conditions(self, bc_config: Dict[str, Any]) -> None:
        """Validate boundary conditions configuration."""
        if 'type' in bc_config:
            bc_type = bc_config['type']
            valid_types = ['displacement_controlled', 'force_controlled', 'mixed']
            if bc_type not in valid_types:
                self.validation_errors.append(
                    f"Invalid boundary condition type: {bc_type}. Valid options: {valid_types}"
                )

        if 'load_magnitude' in bc_config:
            load = bc_config['load_magnitude']
            if not isinstance(load, (int, float)):
                self.validation_errors.append("load_magnitude must be a number")

    def _validate_solver_config(self, solver_config: Dict[str, Any]) -> None:
        """Validate solver configuration."""
        if 'type' in solver_config:
            solver_type = solver_config['type']
            valid_types = ['linear', 'nonlinear']
            if solver_type not in valid_types:
                self.validation_errors.append(
                    f"Invalid solver type: {solver_type}. Valid options: {valid_types}"
                )

        if 'tolerance' in solver_config:
            tolerance = solver_config['tolerance']
            if not isinstance(tolerance, (int, float)) or tolerance <= 0:
                self.validation_errors.append("solver tolerance must be a positive number")

        if 'max_iterations' in solver_config:
            max_iter = solver_config['max_iterations']
            if not isinstance(max_iter, int) or max_iter <= 0:
                self.validation_errors.append("max_iterations must be a positive integer")

    def _validate_performance_config(self, perf_config: Dict[str, Any]) -> None:
        """Validate performance configuration."""
        boolean_params = ['gpu_acceleration', 'parallel_processing']
        for param in boolean_params:
            if param in perf_config:
                if not isinstance(perf_config[param], bool):
                    self.validation_errors.append(f"performance.{param} must be a boolean")

        if 'num_processes' in perf_config:
            num_proc = perf_config['num_processes']
            if num_proc != 'auto' and (not isinstance(num_proc, int) or num_proc <= 0):
                self.validation_errors.append("num_processes must be 'auto' or a positive integer")

        # Validate memory management
        if 'memory_management' in perf_config:
            self._validate_memory_config(perf_config['memory_management'])

    def _validate_memory_config(self, memory_config: Dict[str, Any]) -> None:
        """Validate memory management configuration."""
        size_params = ['chunk_size', 'cache_size', 'memory_limit']
        for param in size_params:
            if param in memory_config:
                value = memory_config[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    self.validation_errors.append(f"memory_management.{param} must be a positive number")

    def _validate_logging_config(self, logging_config: Dict[str, Any]) -> None:
        """Validate logging configuration."""
        if 'level' in logging_config:
            level = logging_config['level']
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if level not in valid_levels:
                self.validation_errors.append(
                    f"Invalid logging level: {level}. Valid options: {valid_levels}"
                )

        # Validate file logging
        if 'file_logging' in logging_config:
            self._validate_file_logging_config(logging_config['file_logging'])

        # Validate console logging
        if 'console_logging' in logging_config:
            self._validate_console_logging_config(logging_config['console_logging'])

    def _validate_file_logging_config(self, file_config: Dict[str, Any]) -> None:
        """Validate file logging configuration."""
        if 'enabled' in file_config:
            if not isinstance(file_config['enabled'], bool):
                self.validation_errors.append("file_logging.enabled must be a boolean")

        if 'file_path' in file_config:
            file_path = file_config['file_path']
            if not isinstance(file_path, str):
                self.validation_errors.append("file_logging.file_path must be a string")

        if 'max_file_size_mb' in file_config:
            size = file_config['max_file_size_mb']
            if not isinstance(size, (int, float)) or size <= 0:
                self.validation_errors.append("max_file_size_mb must be a positive number")

    def _validate_console_logging_config(self, console_config: Dict[str, Any]) -> None:
        """Validate console logging configuration."""
        if 'enabled' in console_config:
            if not isinstance(console_config['enabled'], bool):
                self.validation_errors.append("console_logging.enabled must be a boolean")

        if 'level' in console_config:
            level = console_config['level']
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if level not in valid_levels:
                self.validation_errors.append(
                    f"Invalid console logging level: {level}. Valid options: {valid_levels}"
                )

    def _validate_validation_section(self, validation_config: Dict[str, Any]) -> None:
        """Validate validation configuration section."""
        boolean_params = ['enabled', 'strict_mode']
        for param in boolean_params:
            if param in validation_config:
                if not isinstance(validation_config[param], bool):
                    self.validation_errors.append(f"validation.{param} must be a boolean")

        if 'checks' in validation_config:
            checks = validation_config['checks']
            if not isinstance(checks, dict):
                self.validation_errors.append("validation.checks must be a dictionary")
            else:
                boolean_check_params = [
                    'input_validation', 'output_validation',
                    'parameter_validation', 'convergence_validation'
                ]
                for param in boolean_check_params:
                    if param in checks and not isinstance(checks[param], bool):
                        self.validation_errors.append(f"validation.checks.{param} must be a boolean")

    def _validate_data_handling_config(self, data_config: Dict[str, Any]) -> None:
        """Validate data handling configuration."""
        # Validate input formats
        if 'input_formats' in data_config:
            formats = data_config['input_formats']
            if not isinstance(formats, list) or not all(isinstance(f, str) for f in formats):
                self.validation_errors.append("input_formats must be a list of strings")

        # Validate output formats
        if 'output_formats' in data_config:
            formats = data_config['output_formats']
            if not isinstance(formats, list) or not all(isinstance(f, str) for f in formats):
                self.validation_errors.append("output_formats must be a list of strings")

        # Validate temp files configuration
        if 'temp_files' in data_config:
            self._validate_temp_files_config(data_config['temp_files'])

    def _validate_temp_files_config(self, temp_config: Dict[str, Any]) -> None:
        """Validate temporary files configuration."""
        if 'directory' in temp_config:
            if not isinstance(temp_config['directory'], str):
                self.validation_errors.append("temp_files.directory must be a string")

        boolean_params = ['cleanup_on_exit']
        for param in boolean_params:
            if param in temp_config:
                if not isinstance(temp_config[param], bool):
                    self.validation_errors.append(f"temp_files.{param} must be a boolean")

        if 'max_age_hours' in temp_config:
            age = temp_config['max_age_hours']
            if not isinstance(age, (int, float)) or age <= 0:
                self.validation_errors.append("max_age_hours must be a positive number")

    def _log_validation_results(self) -> None:
        """Log validation results."""
        if self.validation_errors:
            self.logger.error(f"Configuration validation found {len(self.validation_errors)} errors:")
            for error in self.validation_errors:
                self.logger.error(f"  - {error}")

        if self.validation_warnings:
            self.logger.warning(f"Configuration validation found {len(self.validation_warnings)} warnings:")
            for warning in self.validation_warnings:
                self.logger.warning(f"  - {warning}")

        if not self.validation_errors and not self.validation_warnings:
            self.logger.info("Configuration validation passed successfully")

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get detailed validation report.

        Returns:
            Validation report dictionary
        """
        return {
            'is_valid': len(self.validation_errors) == 0,
            'errors': self.validation_errors.copy(),
            'warnings': self.validation_warnings.copy(),
            'error_count': len(self.validation_errors),
            'warning_count': len(self.validation_warnings),
            'strict_mode': self.strict_mode
        }

    def clear_validation_results(self) -> None:
        """Clear validation errors and warnings."""
        self.validation_errors.clear()
        self.validation_warnings.clear()

    def __repr__(self) -> str:
        """String representation of the ConfigValidation instance."""
        return f"ConfigValidation(strict_mode={self.strict_mode})"
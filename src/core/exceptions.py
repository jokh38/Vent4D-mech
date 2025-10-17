"""
Custom Exception Hierarchy for Vent4D-Mech

This module defines a comprehensive exception hierarchy for the Vent4D-Mech
framework to provide consistent error handling and meaningful error messages
throughout all components.
"""

from typing import Optional, Any, Dict


class Vent4DMechError(Exception):
    """
    Base exception for all Vent4D-Mech errors.

    This is the root exception class that all other Vent4D-Mech exceptions
    should inherit from. It provides consistent error formatting and
    additional context information.
    """

    def __init__(self, message: str, component: Optional[str] = None,
                 error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """
        Initialize Vent4D-Mech base exception.

        Args:
            message: Error message describing the issue
            component: Name of the component where the error occurred
            error_code: Optional error code for programmatic handling
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.component = component
        self.error_code = error_code
        self.context = context or {}

        # Build full error message
        self.full_message = self._build_full_message()

    def _build_full_message(self) -> str:
        """Build the full error message with component and context information."""
        parts = [self.message]

        if self.component:
            parts.insert(0, f"[{self.component}]")

        if self.error_code:
            parts.append(f"(Error Code: {self.error_code})")

        if self.context:
            context_parts = [f"{k}={v}" for k, v in self.context.items()]
            parts.append(f"Context: {', '.join(context_parts)}")

        return " ".join(parts)

    def __str__(self) -> str:
        """String representation of the exception."""
        return self.full_message

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for serialization.

        Returns:
            Dictionary representation of the exception
        """
        return {
            'exception_type': self.__class__.__name__,
            'message': self.message,
            'component': self.component,
            'error_code': self.error_code,
            'context': self.context,
            'full_message': self.full_message
        }


class ConfigurationError(Vent4DMechError):
    """
    Exception raised for configuration-related errors.

    This includes invalid configuration parameters, missing required
    settings, incompatible configuration combinations, etc.
    """

    def __init__(self, message: str, component: Optional[str] = None,
                 config_key: Optional[str] = None, config_value: Optional[Any] = None):
        """
        Initialize configuration error.

        Args:
            message: Error message describing the configuration issue
            component: Name of the component where the error occurred
            config_key: The configuration key that caused the issue
            config_value: The problematic configuration value
        """
        context = {}
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = config_value

        super().__init__(
            message=message,
            component=component,
            error_code="CONFIG_ERROR",
            context=context
        )
        self.config_key = config_key
        self.config_value = config_value


class ValidationError(Vent4DMechError):
    """
    Exception raised for data validation errors.

    This includes invalid input data, failed data integrity checks,
    incompatible data formats, etc.
    """

    def __init__(self, message: str, component: Optional[str] = None,
                 data_type: Optional[str] = None, validation_rule: Optional[str] = None):
        """
        Initialize validation error.

        Args:
            message: Error message describing the validation issue
            component: Name of the component where the error occurred
            data_type: Type of data that failed validation
            validation_rule: The specific validation rule that failed
        """
        context = {}
        if data_type:
            context['data_type'] = data_type
        if validation_rule:
            context['validation_rule'] = validation_rule

        super().__init__(
            message=message,
            component=component,
            error_code="VALIDATION_ERROR",
            context=context
        )
        self.data_type = data_type
        self.validation_rule = validation_rule


class ComputationError(Vent4DMechError):
    """
    Exception raised for computation-related errors.

    This includes numerical issues, convergence failures, memory errors,
    algorithmic failures, etc.
    """

    def __init__(self, message: str, component: Optional[str] = None,
                 operation: Optional[str] = None, stage: Optional[str] = None):
        """
        Initialize computation error.

        Args:
            message: Error message describing the computation issue
            component: Name of the component where the error occurred
            operation: The operation that failed
            stage: The stage of computation where the error occurred
        """
        context = {}
        if operation:
            context['operation'] = operation
        if stage:
            context['stage'] = stage

        super().__init__(
            message=message,
            component=component,
            error_code="COMPUTATION_ERROR",
            context=context
        )
        self.operation = operation
        self.stage = stage


class ModelError(Vent4DMechError):
    """
    Exception raised for model-related errors.

    This includes model initialization failures, parameter issues,
    model execution errors, etc.
    """

    def __init__(self, message: str, component: Optional[str] = None,
                 model_type: Optional[str] = None, model_parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize model error.

        Args:
            message: Error message describing the model issue
            component: Name of the component where the error occurred
            model_type: Type of model that caused the error
            model_parameters: Model parameters that caused the issue
        """
        context = {}
        if model_type:
            context['model_type'] = model_type
        if model_parameters:
            context['model_parameters'] = model_parameters

        super().__init__(
            message=message,
            component=component,
            error_code="MODEL_ERROR",
            context=context
        )
        self.model_type = model_type
        self.model_parameters = model_parameters


class DataError(Vent4DMechError):
    """
    Exception raised for data-related errors.

    This includes data loading failures, format issues, corruption,
    missing files, etc.
    """

    def __init__(self, message: str, component: Optional[str] = None,
                 file_path: Optional[str] = None, data_format: Optional[str] = None):
        """
        Initialize data error.

        Args:
            message: Error message describing the data issue
            component: Name of the component where the error occurred
            file_path: Path to the file that caused the error
            data_format: Format of the problematic data
        """
        context = {}
        if file_path:
            context['file_path'] = file_path
        if data_format:
            context['data_format'] = data_format

        super().__init__(
            message=message,
            component=component,
            error_code="DATA_ERROR",
            context=context
        )
        self.file_path = file_path
        self.data_format = data_format


class ResourceError(Vent4DMechError):
    """
    Exception raised for resource-related errors.

    This includes memory issues, GPU availability, file system errors,
    network problems, etc.
    """

    def __init__(self, message: str, component: Optional[str] = None,
                 resource_type: Optional[str] = None, resource_info: Optional[str] = None):
        """
        Initialize resource error.

        Args:
            message: Error message describing the resource issue
            component: Name of the component where the error occurred
            resource_type: Type of resource (memory, gpu, disk, etc.)
            resource_info: Additional information about the resource
        """
        context = {}
        if resource_type:
            context['resource_type'] = resource_type
        if resource_info:
            context['resource_info'] = resource_info

        super().__init__(
            message=message,
            component=component,
            error_code="RESOURCE_ERROR",
            context=context
        )
        self.resource_type = resource_type
        self.resource_info = resource_info


class PipelineError(Vent4DMechError):
    """
    Exception raised for pipeline-related errors.

    This includes stage ordering issues, dependency failures, integration
    problems between components, etc.
    """

    def __init__(self, message: str, component: Optional[str] = None,
                 stage: Optional[str] = None, pipeline_state: Optional[str] = None):
        """
        Initialize pipeline error.

        Args:
            message: Error message describing the pipeline issue
            component: Name of the component where the error occurred
            stage: Pipeline stage where the error occurred
            pipeline_state: State of the pipeline when error occurred
        """
        context = {}
        if stage:
            context['stage'] = stage
        if pipeline_state:
            context['pipeline_state'] = pipeline_state

        super().__init__(
            message=message,
            component=component,
            error_code="PIPELINE_ERROR",
            context=context
        )
        self.stage = stage
        self.pipeline_state = pipeline_state


# Utility functions for error handling
def handle_component_error(func):
    """
    Decorator for standardized error handling in component methods.

    This decorator wraps component methods to provide consistent error
    handling, logging, and exception formatting.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with standardized error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Vent4DMechError:
            # Re-raise Vent4D-Mech exceptions as-is
            raise
        except ValueError as e:
            # Convert ValueError to ValidationError
            component = args[0].component_name if hasattr(args[0], 'component_name') else None
            raise ValidationError(str(e), component=component)
        except TypeError as e:
            # Convert TypeError to ValidationError
            component = args[0].component_name if hasattr(args[0], 'component_name') else None
            raise ValidationError(f"Type error: {str(e)}", component=component)
        except MemoryError as e:
            # Convert MemoryError to ResourceError
            component = args[0].component_name if hasattr(args[0], 'component_name') else None
            raise ResourceError(f"Memory error: {str(e)}", component=component, resource_type="memory")
        except FileNotFoundError as e:
            # Convert FileNotFoundError to DataError
            component = args[0].component_name if hasattr(args[0], 'component_name') else None
            raise DataError(f"File not found: {str(e)}", component=component, file_path=str(e.filename))
        except Exception as e:
            # Convert any other exception to ComputationError
            component = args[0].component_name if hasattr(args[0], 'component_name') else None
            raise ComputationError(f"Unexpected error: {str(e)}", component=component)

    return wrapper


def create_error_context(**kwargs) -> Dict[str, Any]:
    """
    Create error context dictionary from keyword arguments.

    Args:
        **kwargs: Context key-value pairs

    Returns:
        Dictionary containing only non-None context values
    """
    return {k: v for k, v in kwargs.items() if v is not None}
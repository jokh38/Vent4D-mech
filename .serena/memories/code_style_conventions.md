# Vent4D-Mech Code Style and Conventions

## General Style
- **Python Version**: 3.9+ support required (as specified in setup.py)
- **Code Formatting**: Black formatter with default settings (configured in dev dependencies)
- **Linting**: flake8 for code quality checks (version 6.0.0+)
- **Type Checking**: mypy for static type analysis (version 1.5.0+)
- **Docstrings**: Comprehensive docstrings following Google/NumPy style

## Naming Conventions
- **Classes**: PascalCase (e.g., `Vent4DMechPipeline`, `ImageRegistration`, `DeformationAnalyzer`)
- **Functions/Methods**: snake_case (e.g., `analyze_deformation`, `compute_stress`, `register_images`)
- **Variables**: snake_case (e.g., `voxel_spacing`, `strain_tensor`, `displacement_field`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_TOLERANCE`, `GPU_MEMORY_FRACTION`, `MAX_ITERATIONS`)
- **Private members**: Leading underscore (e.g., `_compute_gradient`, `_validate_input`, `_initialize_gpu`)

## Type Hints
- **Strict Typing**: All public APIs require type hints
- **Complex Types**: Use `typing.Dict`, `typing.List`, `typing.Optional`, `typing.Union`
- **NumPy Arrays**: Use `npt.NDArray` from `numpy.typing`
- **Path Handling**: Use `pathlib.Path` for file paths
- **Configuration**: Use `Dict[str, Any]` for config parameters
- **Medical Images**: Use appropriate SimpleITK or nibabel types

## Module Structure Standards
- **Init Files**: All modules have proper `__init__.py` with clean exports
- **Dependencies**: Import order: standard library → third-party → local imports
- **Circular Imports**: Avoid circular dependencies between core modules
- **Module Size**: Keep modules focused on single responsibilities
- **Interface Consistency**: All similar classes follow the same interface patterns

## Documentation Style
- **Module Headers**: Comprehensive module description with purpose and key features
- **Class Docstrings**: Include attributes, parameters, and usage examples
- **Method Docstrings**: Include Args, Returns, Raises sections
- **Type Documentation**: Document complex parameter types and return values
- **Examples**: Provide usage examples in docstrings where appropriate
- **Mathematical Notation**: Use LaTeX-style notation for mathematical expressions

## Error Handling Patterns
- **Logging**: Use Python's logging module with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Custom Exceptions**: Create specific exception classes for different error types
- **Validation**: Input validation with descriptive error messages
- **Graceful Degradation**: Fallback to CPU when GPU unavailable
- **Resource Management**: Use context managers for file handles and GPU memory
- **Exception Hierarchy**: Base exception class for Vent4D-Mech specific errors

## Performance Patterns
- **GPU Acceleration**: Use CuPy for NumPy-compatible GPU operations when available
- **Memory Management**: Use context managers and proper resource cleanup
- **Batch Processing**: Implement batched processing for large volumes
- **Lazy Loading**: Import heavy modules only when needed
- **Caching**: Cache expensive computations and frequently accessed data
- **Parallel Processing**: Use joblib for CPU-bound parallelization

## Configuration Patterns
- **YAML Configuration**: Hierarchical config system with sensible defaults
- **Environment Variables**: Support for environment-based configuration overrides
- **Validation**: Config validation with clear error messages
- **Flexibility**: Support for multiple methods and approaches in each module
- **Default Values**: Always provide reasonable defaults for all parameters

## Testing Patterns
- **pytest**: Use pytest framework for all tests (version 7.4.0+)
- **Fixtures**: Use pytest fixtures for test data and setup
- **Coverage**: Aim for high test coverage with pytest-cov (version 4.1.0+)
- **Synthetic Data**: Use synthetic data for reproducible testing
- **Integration Tests**: Test pipeline workflows end-to-end
- **Mock Testing**: Mock external dependencies (GPU, network resources)

## Code Organization Patterns

### Class Organization
```python
class ExampleClass:
    """Class docstring with comprehensive description."""
    
    # Class attributes first
    DEFAULT_PARAM = "value"
    
    def __init__(self, param: str):
        """Initialize with parameter."""
        self.param = param
    
    # Public methods next
    def public_method(self) -> ReturnType:
        """Public method with comprehensive docstring."""
        pass
    
    # Private methods last
    def _private_method(self) -> None:
        """Private method for internal use."""
        pass
```

### Function Organization
```python
def example_function(
    param1: Type1,
    param2: Type2,
    optional_param: Optional[Type3] = None
) -> ReturnType:
    """Function with comprehensive docstring.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        optional_param: Description of optional parameter
    
    Returns:
        Description of return value
    
    Raises:
        SpecificException: When something goes wrong
    """
    pass
```

## Import Organization
```python
# Standard library imports
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import SimpleITK as sitk
import torch
from cupy import cuda

# Local imports
from vent4d_mech.core.base import BaseClass
from vent4d_mech.utils.helpers import helper_function
```

## GPU/CPU Compatibility Patterns
```python
def gpu_compatible_function(data: npt.NDArray) -> npt.NDArray:
    """Function that works on both CPU and GPU."""
    try:
        import cupy as cp
        gpu_data = cp.asarray(data)
        # GPU operations
        result = gpu_operation(gpu_data)
        return cp.asnumpy(result)
    except ImportError:
        # Fallback to CPU
        result = cpu_operation(data)
        return result
```

## Configuration Integration Patterns
```python
def configurable_function(
    data: npt.NDArray,
    config: Optional[Dict[str, Any]] = None
) -> ReturnType:
    """Function with configuration support."""
    # Merge with defaults
    final_config = {**DEFAULT_CONFIG, **(config or {})}
    
    # Validate configuration
    _validate_config(final_config)
    
    # Use configuration
    param = final_config.get("parameter_name", DEFAULT_VALUE)
    return process_data(data, param)
```

## Clinical Data Patterns
- **DICOM Handling**: Proper handling of DICOM metadata and pixel data
- **NIfTI Support**: Standard NIfTI file format support
- **Metadata Preservation**: Maintain clinical metadata throughout processing
- **Units and Standards**: Use appropriate medical imaging units and coordinate systems
- **Patient Privacy**: Ensure no PHI is logged or stored inappropriately

## Memory Management Patterns
- **Large Arrays**: Use memory mapping for very large datasets
- **GPU Memory**: Explicit GPU memory management and cleanup
- **Streaming**: Process data in chunks when possible
- **Resource Monitoring**: Monitor memory usage and provide warnings
- **Cleanup**: Ensure proper cleanup of temporary files and arrays

## Code Quality Standards
- **Black Formatting**: Run black on all Python files before commits
- **Flake8 Linting**: Pass all flake8 checks
- **Type Checking**: Pass mypy static analysis
- **Documentation**: All public APIs must have comprehensive docstrings
- **Testing**: New features must include appropriate tests
- **Performance**: Monitor performance impact of new features
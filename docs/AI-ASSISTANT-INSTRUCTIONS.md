# AI Code Assistant Guidelines for Vent4D-Mech Project

## Overview

This document provides specific guidelines for AI code assistants working on the Vent4D-Mech lung tissue dynamics modeling framework. These instructions are based on analysis of common development patterns and challenges specific to this medical imaging and mechanical modeling codebase.

## Project Context

**Project Type**: Medical image processing and mechanical modeling framework
**Domain**: Computational biomechanics for lung tissue analysis
**Primary Usage**: Research and clinical application development
**Key Technologies**: Python, PyTorch, SimpleITK, CuPy, SfePy, medical imaging formats (NIfTI, DICOM)
**Performance Requirements**: GPU acceleration support, large 4D-CT dataset processing

## Critical Domain-Specific Considerations

### 1. Medical Data Handling
- **Validation**: Always validate medical image formats (NIfTI, DICOM) before processing
- **Patient Safety**: Maintain data integrity - never modify original medical data
- **Metadata Preservation**: Preserve all DICOM headers and imaging metadata
- **Coordinate Systems**: Be explicit about coordinate system conventions (RAS vs LPS)
- **Unit Consistency**: Ensure consistent units (mm for spatial, HU for intensity)

### 2. GPU/CPU Fallback Requirements
- **Default to CPU**: Always provide CPU fallback for GPU operations
- **Memory Management**: Implement batch processing for large 3D/4D volumes
- **Device Detection**: Check GPU availability before attempting GPU operations
- **Graceful Degradation**: Provide warnings when falling back to CPU

### 3. Scientific Computing Standards
- **Numerical Precision**: Use appropriate precision (float32 for GPU, float64 for validation)
- **Reproducibility**: Set random seeds for stochastic operations
- **Validation**: Include synthetic data validation for algorithms
- **Error Metrics**: Use standard biomechanics error metrics (RMSE, MAE, correlation coefficients)

## Development Patterns

### 1. Module Structure
```python
# Standard module pattern for Vent4D-Mech
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

# Configure module-specific logger
logger = logging.getLogger(__name__)

class Vent4DComponent:
    """Base class for Vent4D-Mech components"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""
        pass

    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Main processing method with timing and logging"""
        start_time = time.time()
        try:
            self.logger.info(f"[{self.__class__.__name__}] Starting processing")
            result = self._process_data(data, **kwargs)
            duration = time.time() - start_time
            self.logger.info(f"[{self.__class__.__name__}] Completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"[{self.__class__.__name__}] Failed after {duration:.2f}s: {str(e)}")
            raise
```

### 2. GPU/CPU Operations Pattern
```python
def process_with_gpu_fallback(data: np.ndarray, gpu_function, cpu_function):
    """Execute GPU function with CPU fallback"""
    try:
        import cupy as cp
        if cp.cuda.is_available():
            logger.debug("[GPU] Using GPU acceleration")
            gpu_data = cp.asarray(data)
            result = gpu_function(gpu_data)
            return cp.asnumpy(result)
        else:
            logger.warning("[GPU] CUDA not available, using CPU")
    except ImportError:
        logger.warning("[GPU] CuPy not installed, using CPU")
    except Exception as e:
        logger.error(f"[GPU] GPU operation failed: {str(e)}, falling back to CPU")

    return cpu_function(data)
```

### 3. Medical Image Validation Pattern
```python
def validate_medical_image(image_path: Path, expected_shape: Optional[Tuple] = None) -> bool:
    """Validate medical image format and properties"""
    try:
        import nibabel as nib
        import SimpleITK as sitk

        # Try NIfTI first
        if image_path.suffix.lower() in ['.nii', '.nii.gz']:
            img = nib.load(str(image_path))
            data = img.get_fdata()
            affine = img.affine

            # Validate shape if specified
            if expected_shape and data.shape != expected_shape:
                raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")

            # Check for valid data
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                raise ValueError("Image contains invalid values (NaN/Inf)")

            logger.debug(f"[VALIDATION] NIfTI image valid: shape={data.shape}, dtype={data.dtype}")
            return True

        # Try DICOM
        elif image_path.suffix.lower() in ['.dcm', '.dicom'] or image_path.is_dir():
            reader = sitk.ImageSeriesReader()
            if image_path.is_dir():
                series_ids = reader.GetGDCMSeriesIDs(str(image_path))
                if not series_ids:
                    raise ValueError("No DICOM series found")
                series_file_names = reader.GetGDCMSeriesFileNames(str(image_path), series_ids[0])
            else:
                series_file_names = [str(image_path)]

            reader.SetFileNames(series_file_names)
            image = reader.Execute()

            # Validate image properties
            size = image.GetSize()
            spacing = image.GetSpacing()
            origin = image.GetOrigin()

            logger.debug(f"[VALIDATION] DICOM series valid: size={size}, spacing={spacing}")
            return True

        else:
            raise ValueError(f"Unsupported medical image format: {image_path.suffix}")

    except Exception as e:
        logger.error(f"[VALIDATION] Image validation failed: {str(e)}")
        return False
```

### 4. Configuration-Driven Development
```python
# Always use configuration-driven parameters
def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration with defaults"""
    import yaml

    # Default configuration
    default_config = {
        'registration': {
            'method': 'hybrid',
            'simpleitk_params': {'learning_rate': 0.1, 'number_of_iterations': 100},
            'voxelmorph_params': {'model_type': 'vm2'}
        },
        'deformation': {
            'strain_theory': 'green_lagrange',
            'batch_size': 64,
            'precision': 'float32'
        },
        'performance': {
            'gpu_acceleration': True,
            'gpu_memory_fraction': 0.8,
            'parallel_processing': True
        }
    }

    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)

        # Merge with defaults
        config = {**default_config, **user_config}
        logger.info(f"[CONFIG] Loaded configuration from {config_path}")
    else:
        config = default_config
        logger.info("[CONFIG] Using default configuration")

    return config
```

## Specific Implementation Guidelines

### 1. Registration Module Development
- **Hybrid Approach**: Always support both SimpleITK and VoxelMorph methods
- **Memory Management**: Process large 3D volumes in batches
- **Validation**: Include landmark validation for registration accuracy
- **Parameter Tuning**: Provide sensible defaults for registration parameters

### 2. Deformation Field Processing
```python
def compute_deformation_field(moving_image, fixed_image, config):
    """Compute deformation field with validation"""
    start_time = time.time()

    try:
        logger.info("[DEFORMATION] Computing displacement field")

        # Validate input images
        if not validate_medical_image(moving_image) or not validate_medical_image(fixed_image):
            raise ValueError("Invalid medical images")

        # Compute deformation field based on configuration
        if config['registration']['method'] == 'simpleitk':
            displacement_field = compute_sitk_deformation(moving_image, fixed_image)
        elif config['registration']['method'] == 'voxelmorph':
            displacement_field = compute_voxelmorph_deformation(moving_image, fixed_image)
        else:
            displacement_field = compute_hybrid_deformation(moving_image, fixed_image)

        # Validate deformation field
        if np.any(np.isnan(displacement_field)):
            raise ValueError("Deformation field contains NaN values")

        duration = time.time() - start_time
        logger.info(f"[DEFORMATION] Computation completed in {duration:.2f}s")
        return displacement_field

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"[DEFORMATION] Failed after {duration:.2f}s: {str(e)}")
        raise
```

### 3. Mechanical Model Implementation
- **Constitutive Models**: Implement standard hyperelastic models (Neo-Hookean, Mooney-Rivlin, Yeoh)
- **Energy Minimization**: Use robust numerical optimization methods
- **Material Parameters**: Provide literature-based default values for lung tissue
- **Validation**: Compare against analytical solutions for simple geometries

### 4. FEM Integration Pattern
```python
def run_fem_simulation(mesh, material_properties, boundary_conditions, config):
    """Run finite element simulation with error handling"""
    try:
        import sfepy

        logger.info("[FEM] Starting finite element simulation")

        # Validate mesh quality
        if not validate_mesh_quality(mesh):
            raise ValueError("Poor mesh quality detected")

        # Create problem definition
        problem = create_sfepy_problem(mesh, material_properties, boundary_conditions)

        # Solve with appropriate solver
        solver_config = config.get('fem', {}).get('solver', 'auto')
        if solver_config == 'auto':
            solver = sfepy.solvers.auto_select_solver(problem)
        else:
            solver = sfepy.solvers.get_solver(solver_config)

        # Run simulation
        solution = solver(problem)

        # Validate solution
        if not validate_solution(solution):
            logger.warning("[FEM] Solution validation warnings detected")

        logger.info("[FEM] Simulation completed successfully")
        return solution

    except Exception as e:
        logger.error(f"[FEM] Simulation failed: {str(e)}")
        raise
```

## Testing Requirements

### 1. Unit Tests
- Test individual components with synthetic data
- Validate numerical accuracy against analytical solutions
- Test GPU/CPU fallback mechanisms
- Include boundary condition tests

### 2. Integration Tests
- End-to-end pipeline validation
- Medical image format compatibility
- Configuration system validation
- Performance benchmarking

### 3. Validation Tests
- Compare against phantom studies
- Validate strain computation accuracy
- Test material parameter estimation
- Cross-validate with clinical data when available

## Common Pitfalls to Avoid

1. **Don't assume GPU availability** - always provide CPU fallback
2. **Don't ignore medical image metadata** - preserve all patient information
3. **Don't hardcode medical parameters** - use literature-based defaults with citations
4. **Don't skip mesh quality validation** - poor meshes lead to simulation failures
5. **Don't ignore numerical stability** - check for NaN/Inf values in computations
6. **Don't mix coordinate systems** - be explicit about spatial coordinate conventions
7. **Don't forget memory management** - process large volumes in batches
8. **Don't skip configuration validation** - validate all user-provided parameters

## Environment Variables and Configuration

Always use these for configuration:
- `VENT4D_CONFIG_PATH`: Path to configuration YAML file
- `VENT4D_GPU_MEMORY_FRACTION`: GPU memory allocation (default: 0.8)
- `VENT4D_DATA_DIR`: Root directory for medical data
- `VENT4D_OUTPUT_DIR`: Output directory for results
- `VENT4D_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Documentation Requirements

- Include mathematical formulations in docstrings
- Provide parameter descriptions with units
- Include example usage with synthetic data
- Add performance characteristics documentation
- Include relevant scientific references

## Scientific Validation

When implementing new algorithms:
1. **Validate against known solutions** - use analytical problems when possible
2. **Include convergence studies** - show mesh/parameter independence
3. **Compare with literature** - benchmark against published results
4. **Provide uncertainty quantification** - estimate numerical errors
5. **Include reproducibility information** - random seeds, software versions

## Conclusion

The Vent4D-Mech project requires careful attention to scientific computing standards, medical data handling, and numerical stability. Always prioritize data integrity, provide fallback mechanisms, and include comprehensive validation. The framework is used in medical research, so reliability and reproducibility are paramount.
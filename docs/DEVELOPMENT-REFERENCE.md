# Vent4D-Mech Development Reference Guide

## Project Overview

**Vent4D-Mech**: Comprehensive Python framework for lung tissue dynamics modeling that transitions from empirical HU-based analysis to physics-based mechanical modeling.

**Primary Technologies**: Python, PyTorch, SimpleITK, CuPy, SfePy, NIfTI/DICOM processing
**Domain**: Computational biomechanics, medical imaging, finite element analysis
**Performance Requirements**: GPU acceleration, large 4D-CT dataset processing

## Architecture and Codebase Structure

### Core Pipeline Components
1. **Image Registration** (`vent4d_mech.core.registration`) - Hybrid SimpleITK + VoxelMorph
2. **Deformation Analysis** (`vent4d_mech.core.deformation`) - Green-Lagrange strain tensors
3. **Mechanical Modeling** (`vent4d_mech.core.mechanical`) - Hyperelastic constitutive models
4. **Inverse Problem Solving** (`vent4d_mech.core.inverse`) - Young's modulus estimation
5. **Microstructure Integration** (`vent4d_mech.core.microstructure`) - Human Organ Atlas
6. **FEM Workflow** (`vent4d_mech.core.fem`) - SfePy integration
7. **Ventilation Analysis** (`vent4d_mech.core.ventilation`) - Jacobian-based calculation

### Configuration System
- **Location**: `config/default.yaml`
- **Structure**: Hierarchical YAML with module-specific sections
- **Key Sections**: registration, deformation, mechanical, inverse, microstructure, fem, ventilation, performance

## Development Workflow

### Environment Setup
```bash
# Virtual environment
python -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt
pip install -e .

# GPU support (optional)
pip install cupy-cuda11x  # or cupy-cuda12x
```

### Testing Commands
```bash
# All tests
pytest

# Specific modules
pytest tests/test_registration.py
pytest tests/test_deformation.py

# Coverage
pytest --cov=vent4d_mech

# Verbose
pytest -v
```

### Code Quality
```bash
# Format
black vent4d_mech/ tests/

# Lint
flake8 vent4d_mech/ tests/

# Type checking
mypy vent4d_mech/
```

## Critical Development Patterns

### 1. Module Template
```python
"""
Module docstring with mathematical formulations and parameter descriptions.
"""

import logging
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class ComponentClass:
    """Component description with scientific background."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        required_params = ['param1', 'param2']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")

    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Main processing method with comprehensive logging."""
        start_time = time.time()
        try:
            self.logger.info(f"[{self.__class__.__name__}] Starting processing")

            # Validate input
            self._validate_input(data)

            # Process data
            result = self._process_data(data, **kwargs)

            # Validate output
            self._validate_output(result)

            duration = time.time() - start_time
            self.logger.info(f"[{self.__class__.__name__}] Completed in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"[{self.__class__.__name__}] Failed after {duration:.2f}s: {str(e)}")
            raise
```

### 2. GPU/CPU Fallback Pattern
```python
def process_with_gpu_fallback(data: np.ndarray,
                            gpu_operation: callable,
                            cpu_operation: callable,
                            config: Dict[str, Any]) -> np.ndarray:
    """Execute operation with GPU acceleration and CPU fallback."""

    if not config.get('performance', {}).get('gpu_acceleration', True):
        logger.debug("[GPU] GPU acceleration disabled, using CPU")
        return cpu_operation(data)

    try:
        import cupy as cp
        if not cp.cuda.is_available():
            raise ImportError("CUDA not available")

        # Check GPU memory
        gpu_memory_fraction = config.get('performance', {}).get('gpu_memory_fraction', 0.8)
        with cp.cuda.Device(0):
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=int(cp.cuda.Device(0).mem_info[0] * gpu_memory_fraction))

        logger.debug("[GPU] Using GPU acceleration")
        gpu_data = cp.asarray(data)
        gpu_result = gpu_operation(gpu_data)
        return cp.asnumpy(gpu_result)

    except ImportError:
        logger.warning("[GPU] CuPy not available, using CPU")
    except Exception as e:
        logger.error(f"[GPU] GPU operation failed: {str(e)}, falling back to CPU")

    return cpu_operation(data)
```

### 3. Medical Image Processing Pattern
```python
def load_medical_image(image_path: Path,
                      validate: bool = True,
                      expected_shape: Optional[Tuple] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load medical image with comprehensive validation."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        # Try NIfTI format
        if image_path.suffix.lower() in ['.nii', '.nii.gz']:
            import nibabel as nib
            img = nib.load(str(image_path))
            data = img.get_fdata()
            metadata = {
                'affine': img.affine,
                'header': dict(img.header),
                'shape': data.shape,
                'dtype': data.dtype,
                'voxel_size': img.header.get_zooms()[:3]
            }

        # Try DICOM format
        elif image_path.suffix.lower() in ['.dcm', '.dicom'] or image_path.is_dir():
            import SimpleITK as sitk
            if image_path.is_dir():
                reader = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(str(image_path))
                if not series_ids:
                    raise ValueError("No DICOM series found")
                series_file_names = reader.GetGDCMSeriesFileNames(str(image_path), series_ids[0])
                reader.SetFileNames(series_file_names)
            else:
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(image_path))

            sitk_image = reader.Execute()
            data = sitk.GetArrayFromImage(sitk_image)
            metadata = {
                'spacing': sitk_image.GetSpacing(),
                'origin': sitk_image.GetOrigin(),
                'direction': sitk_image.GetDirection(),
                'size': sitk_image.GetSize(),
                'shape': data.shape,
                'dtype': data.dtype
            }

        else:
            raise ValueError(f"Unsupported medical image format: {image_path.suffix}")

        # Validate data
        if validate:
            _validate_medical_data(data, metadata, expected_shape)

        logger.info(f"[IMAGE] Loaded medical image: shape={data.shape}, dtype={data.dtype}")
        return data, metadata

    except Exception as e:
        logger.error(f"[IMAGE] Failed to load medical image: {str(e)}")
        raise

def _validate_medical_data(data: np.ndarray, metadata: Dict[str, Any],
                          expected_shape: Optional[Tuple] = None):
    """Validate medical data integrity."""

    # Check for invalid values
    if np.any(np.isnan(data)):
        raise ValueError("Image data contains NaN values")

    if np.any(np.isinf(data)):
        raise ValueError("Image data contains infinite values")

    # Check expected shape
    if expected_shape and data.shape != expected_shape:
        logger.warning(f"[IMAGE] Unexpected shape: expected {expected_shape}, got {data.shape}")

    # Check voxel spacing
    if 'voxel_size' in metadata:
        voxel_size = metadata['voxel_size']
        if any(vs <= 0 for vs in voxel_size):
            raise ValueError(f"Invalid voxel spacing: {voxel_size}")

    logger.debug(f"[IMAGE] Data validation passed")
```

### 4. Configuration Management
```python
def load_configuration(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration with defaults and validation."""
    import yaml

    # Default configuration
    default_config = {
        'registration': {
            'method': 'hybrid',
            'simpleitk_params': {
                'learning_rate': 0.1,
                'number_of_iterations': 100,
                'metric': 'mutual_information'
            },
            'voxelmorph_params': {
                'model_type': 'vm2',
                'batch_size': 1
            }
        },
        'deformation': {
            'strain_theory': 'green_lagrange',
            'computation': {
                'batch_processing': True,
                'batch_size': 64,
                'precision': 'float32'
            }
        },
        'mechanical': {
            'constitutive_model': 'neo_hookean',
            'material_params': {
                'young_modulus': 1000.0,  # Pa
                'poisson_ratio': 0.45
            }
        },
        'performance': {
            'gpu_acceleration': True,
            'gpu_memory_fraction': 0.8,
            'parallel_processing': True,
            'num_workers': 4
        }
    }

    # Load user configuration if provided
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)

            # Deep merge configurations
            config = _deep_merge(default_config, user_config)
            logger.info(f"[CONFIG] Loaded configuration from {config_path}")

        except Exception as e:
            logger.error(f"[CONFIG] Failed to load user config: {str(e)}, using defaults")
            config = default_config
    else:
        config = default_config
        logger.info("[CONFIG] Using default configuration")

    # Validate configuration
    _validate_configuration(config)
    return config

def _deep_merge(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = default.copy()
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def _validate_configuration(config: Dict[str, Any]):
    """Validate configuration parameters."""

    # Validate performance settings
    if config.get('performance', {}).get('gpu_memory_fraction', 0.8) > 1.0:
        raise ValueError("GPU memory fraction must be <= 1.0")

    if config.get('performance', {}).get('num_workers', 1) < 1:
        raise ValueError("Number of workers must be >= 1")

    # Validate mechanical parameters
    material_params = config.get('mechanical', {}).get('material_params', {})
    if material_params.get('poisson_ratio', 0.45) >= 0.5:
        raise ValueError("Poisson ratio must be < 0.5 for incompressible materials")

    logger.debug("[CONFIG] Configuration validation passed")
```

## Module-Specific Guidelines

### Registration Module
- **Hybrid Approach**: Combine SimpleITK (classical) and VoxelMorph (deep learning)
- **Validation**: Landmark-based validation for registration accuracy
- **Memory**: Process 3D volumes in batches for GPU memory management
- **Parameters**: Provide literature-based defaults for lung registration

### Deformation Analysis
- **Strain Theory**: Green-Lagrange for large deformations (preferred for lung tissue)
- **Numerical Methods**: Central finite differences for gradient computation
- **Validation**: Compare against analytical deformation fields
- **Output**: 6-component symmetric strain tensor fields

### Mechanical Modeling
- **Constitutive Models**: Neo-Hookean, Mooney-Rivlin, Yeoh (hyperelastic)
- **Parameter Estimation**: Regularized inverse problem solving
- **Material Properties**: Lung tissue typical values (E: 0.5-5 kPa, ν: 0.45-0.49)
- **Validation**: Phantom studies and analytical solutions

### FEM Integration
- **Mesh Generation**: Quality metrics (aspect ratio, skewness)
- **Solver Selection**: Auto-select based on problem characteristics
- **Boundary Conditions**: Physiological constraints for lung mechanics
- **Convergence**: Monitor residual norms and energy balance

## Testing Strategy

### Test Data Generation
```python
def generate_synthetic_lung_data(shape=(128, 128, 64),
                                noise_level=0.01,
                                deformation_type='breathing') -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic lung CT data for testing."""

    # Create synthetic lung geometry
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    z = np.linspace(-1, 1, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Lung shape (ellipsoid with bronchial tree)
    lung_mask = (X**2/0.8**2 + Y**2/0.6**2 + Z**2/1.0**2) < 1.0

    # Add bronchial tree structure
    bronchi = (np.abs(X) < 0.1) & (np.abs(Y) < 0.1) & (Z > 0)
    lung_mask = lung_mask | bronchi

    # Generate CT intensities (HU values)
    tissue_intensity = -600  # HU for lung tissue
    air_intensity = -1000   # HU for air

    fixed_image = np.where(lung_mask,
                          tissue_intensity + np.random.normal(0, 50, shape),
                          air_intensity)

    # Generate deformation field
    if deformation_type == 'breathing':
        # Simulate breathing deformation
        deformation_scale = 0.1
        dx = deformation_scale * X * (1 + 0.2 * Z)
        dy = deformation_scale * Y * (1 + 0.1 * Z)
        dz = deformation_scale * Z * 0.5
    else:
        raise ValueError(f"Unknown deformation type: {deformation_type}")

    # Apply deformation
    from scipy.ndimage import map_coordinates
    coordinates = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    deformed_coords = coordinates + np.stack([dx, dy, dz])

    moving_image = map_coordinates(fixed_image, deformed_coords, order=1, mode='nearest')

    # Add noise
    noise = np.random.normal(0, noise_level * np.abs(fixed_image).max(), shape)
    fixed_image += noise
    moving_image += noise

    return fixed_image, moving_image
```

### Validation Metrics
```python
def compute_registration_metrics(fixed_image: np.ndarray,
                               moving_image: np.ndarray,
                               deformation_field: np.ndarray,
                               ground_truth: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute comprehensive registration validation metrics."""

    metrics = {}

    # Intensity-based metrics
    mse = np.mean((fixed_image - moving_image)**2)
    metrics['mse'] = float(mse)

    # Mutual information
    from sklearn.metrics import mutual_info_score
    hist_2d, _, _ = np.histogram2d(fixed_image.ravel(), moving_image.ravel(), bins=50)
    mi = mutual_info_score(None, None, contingency=hist_2d)
    metrics['mutual_information'] = float(mi)

    # Deformation field metrics
    if ground_truth is not None:
        deformation_error = np.sqrt(np.sum((deformation_field - ground_truth)**2, axis=-1))
        metrics['deformation_rmse'] = float(np.mean(deformation_error))
        metrics['deformation_max_error'] = float(np.max(deformation_error))

    # Jacobian determinant (volume change)
    jacobian = compute_jacobian_determinant(deformation_field)
    metrics['jacobian_mean'] = float(np.mean(jacobian))
    metrics['jacobian_std'] = float(np.std(jacobian))
    metrics['negative_jacobian_fraction'] = float(np.mean(jacobian < 0))

    return metrics
```

## Performance Optimization

### Memory Management
- **Batch Processing**: Process large volumes in chunks
- **GPU Memory**: Use CuPy memory pools and limits
- **Sparse Operations**: Use sparse matrices for large FEM problems
- **Data Types**: Use float32 for GPU, float64 for validation

### Parallel Processing
```python
from joblib import Parallel, delayed
import multiprocessing

def parallel_patient_processing(patient_dirs: List[Path],
                              processing_function: callable,
                              config: Dict[str, Any]) -> List[Any]:
    """Process multiple patients in parallel."""

    num_workers = config.get('performance', {}).get('num_workers', multiprocessing.cpu_count())

    def process_single_patient(patient_dir):
        try:
            logger.info(f"[PARALLEL] Processing patient: {patient_dir.name}")
            result = processing_function(patient_dir, config)
            logger.info(f"[PARALLEL] Completed patient: {patient_dir.name}")
            return result
        except Exception as e:
            logger.error(f"[PARALLEL] Failed patient {patient_dir.name}: {str(e)}")
            return None

    results = Parallel(n_jobs=num_workers)(
        delayed(process_single_patient)(patient_dir)
        for patient_dir in patient_dirs
    )

    return [r for r in results if r is not None]
```

## Debugging and Troubleshooting

### Common Issues and Solutions

1. **GPU Memory Errors**
   - Reduce batch size in configuration
   - Use CPU fallback for large volumes
   - Implement gradient checkpointing

2. **Registration Convergence**
   - Check image preprocessing and normalization
   - Adjust optimization parameters (learning rate, iterations)
   - Validate initial alignment

3. **FEM Solver Failures**
   - Check mesh quality metrics
   - Validate boundary conditions
   - Use appropriate solver for problem size

4. **Medical Data Issues**
   - Verify coordinate system conventions (RAS vs LPS)
   - Check DICOM series ordering
   - Validate image orientation matrices

### Logging Configuration
```python
def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup comprehensive logging configuration."""

    import logging
    import sys

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
```

## Environment Variables

```bash
# Configuration
VENT4D_CONFIG_PATH=/path/to/config.yaml
VENT4D_DATA_DIR=/path/to/data
VENT4D_OUTPUT_DIR=/path/to/output

# Performance
VENT4D_GPU_MEMORY_FRACTION=0.8
VENT4D_NUM_WORKERS=4

# Logging
VENT4D_LOG_LEVEL=INFO
VENT4D_LOG_FILE=/path/to/vent4d.log

# External services
VENT4D_OLLAMA_URL=http://localhost:11434
VENT4D_HUMAN_ORGAN_ATLAS_URL=https://api.humanorganatlas.org
```

## Documentation Requirements

### Code Documentation
- **Module docstrings**: Include mathematical formulations
- **Function docstrings**: Parameter descriptions with units and ranges
- **Class docstrings**: Scientific background and usage examples
- **Inline comments**: Explain complex numerical operations

### Scientific References
- Include relevant literature citations in docstrings
- Provide parameter values with source references
- Document validation methods and accuracy metrics

## Future Development Considerations

### Scalability
- Distributed computing for large patient cohorts
- Cloud integration for GPU resources
- Database integration for metadata management

### New Features
- Deep learning constitutive models
- Real-time ventilation prediction
- Patient-specific model calibration
- Clinical outcome prediction integration

### Code Quality
- Increase test coverage (>90%)
- Add continuous integration pipelines
- Performance benchmarking suite
- Documentation generation automation

---

**Key Principles for Development:**
1. **Scientific rigor** - Validate all numerical methods
2. **Medical safety** - Preserve data integrity and patient information
3. **Performance** - Efficient GPU/CPU utilization with fallbacks
4. **Reproducibility** - Consistent results across platforms
5. **Maintainability** - Clean, well-documented, modular code
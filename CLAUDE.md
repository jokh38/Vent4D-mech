# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Install GPU support (optional but recommended)
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
```

### Testing
```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_registration.py
pytest tests/test_deformation.py

# Run with coverage
pytest --cov=vent4d_mech

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Format code
black vent4d_mech/ tests/

# Lint code
flake8 vent4d_mech/ tests/

# Type checking
mypy vent4d_mech/
```

### Documentation
```bash
# Build documentation
cd docs/
make html

# Clean build
make clean
```

## Architecture Overview

Vent4D-Mech is a comprehensive Python framework for lung tissue dynamics modeling that transitions from empirical HU-based analysis to physics-based mechanical modeling. The architecture follows a modular pipeline design:

### Core Pipeline Components
The main pipeline follows this sequence:
1. **Image Registration** (`vent4d_mech.core.registration`) - Hybrid approach combining SimpleITK (classical optimization) and VoxelMorph (deep learning)
2. **Deformation Analysis** (`vent4d_mech.core.deformation`) - Large deformation theory using Green-Lagrange strain tensors
3. **Mechanical Modeling** (`vent4d_mech.core.mechanical`) - Hyperelastic constitutive models (Neo-Hookean, Mooney-Rivlin, Yeoh)
4. **Inverse Problem Solving** (`vent4d_mech.core.inverse`) - Regularized optimization for Young's modulus estimation
5. **Microstructure Integration** (`vent4d_mech.core.microstructure`) - Human Organ Atlas integration for multi-scale modeling
6. **FEM Workflow** (`vent4d_mech.core.fem`) - Python-native finite element simulation using SfePy
7. **Ventilation Analysis** (`vent4d_mech.core.ventilation`) - Jacobian-based ventilation calculation

### Key Design Patterns
- **Modular Pipeline**: Each stage can be run independently or as part of the complete pipeline
- **GPU Acceleration**: CuPy and PyTorch integration for high-performance computing with fallback to CPU
- **Configuration-Driven**: YAML-based configuration system with sensible defaults in `config/default.yaml`
- **Medical Image Format Support**: NIfTI and DICOM input/output through SimpleITK and nibabel

### Critical Dependencies
- **Medical Imaging**: SimpleITK, nibabel, pydicom for image processing
- **Deep Learning**: PyTorch, VoxelMorph for registration
- **GPU Computing**: CuPy for NumPy-compatible GPU operations
- **Finite Element**: SfePy for mechanical simulation
- **Scientific Computing**: NumPy, SciPy, scikit-learn

## Configuration System

The framework uses a hierarchical YAML configuration system:

### Main Configuration Structure
- `registration`: Method selection (simpleitk, voxelmorph, hybrid) and parameters
- `deformation`: Strain theory (green_lagrange vs infinitesimal) and computation settings
- `mechanical`: Constitutive model selection and material parameters
- `inverse`: Solver configuration for Young's modulus estimation
- `microstructure`: Human Organ Atlas integration settings
- `fem`: Mesh generation and solver configuration
- `ventilation`: Analysis method and regional segmentation
- `performance`: GPU acceleration and parallel processing settings

### GPU Configuration
GPU acceleration is configurable per-module:
```yaml
performance:
  gpu_acceleration: true
  gpu_memory_fraction: 0.8
  parallel_processing: true
  num_workers: 4
```

## Data Flow and Processing

### Input Data Requirements
- **4D-CT**: Paired inhale/exhale phases (NIfTI or DICOM)
- **Segmentation**: Lung region masks (optional, auto-generated if not provided)
- **Voxel Spacing**: Typically 1.5×1.5×3.0mm for clinical CT

### Output Data Types
- **Deformation Fields**: 3D displacement vector fields
- **Strain Tensors**: 6-component Green-Lagrange strain fields
- **Material Properties**: Young's modulus distributions
- **Ventilation Maps**: Regional ventilation indices

### Memory Management
The framework implements batch processing for large volumes:
```yaml
deformation:
  computation:
    batch_processing: true
    batch_size: 64
    precision: "float32"
```

## Performance Considerations

### GPU Acceleration
- CuPy provides NumPy-compatible GPU arrays for numerical computations
- PyTorch handles deep learning operations for registration
- Zero-copy interoperability between CuPy and PyTorch when possible

### Parallel Processing
- Joblib for patient-level parallelization
- Multiprocessing for CPU-bound tasks
- Dask integration for out-of-core computation on large datasets

## Testing Strategy

### Test Categories
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Pipeline workflow validation
- **Performance Tests**: GPU acceleration and memory usage
- **Validation Tests**: Comparison against clinical SPECT/CT data

### Test Data
- Synthetic data generation for reproducible testing
- Phantom studies for algorithm validation
- Clinical data comparison (when available)

## Common Development Patterns

### Adding New Constitutive Models
1. Implement in `vent4d_mech/core/mechanical/mechanical_modeler.py`
2. Add configuration parameters in `config/default.yaml`
3. Update unit tests in `tests/test_mechanical.py`

### Extending Registration Methods
1. Create new registration class in `vent4d_mech/core/registration/`
2. Inherit from base `ImageRegistration` class
3. Register in `vent4d_mech/core/registration/__init__.py`

### Integration with External Data Sources
1. Extend `vent4d_mech/core/microstructure/microstructure_db.py`
2. Add configuration options for API endpoints
3. Implement data caching and validation

## Clinical Validation Framework

The framework includes tools for validation against clinical standards:
- **SPECT/CT Comparison**: Quantitative validation against clinical gold standard
- **Reproducibility Analysis**: Test-retest reliability metrics
- **Phantom Studies**: Synthetic data validation pipeline
- **Clinical Correlation**: Patient outcome analysis tools
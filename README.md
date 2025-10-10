# Vent4D-Mech: Python-based Lung Tissue Dynamics Modeling

A comprehensive framework for patient-specific lung biomechanical modeling from 4D-CT to high-fidelity ventilation analysis.

## Overview

Vent4D-Mech is a pure Python framework that advances beyond empirical Hounsfield Unit (HU) based models to create physics-based, patient-specific mechanical models of lung tissue deformation. This project implements the transition from correlation-based analysis (HU values) to causation-based analysis (tissue mechanical properties) for improved accuracy in ventilation prediction.

## Key Features

- **Deformable Image Registration**: Hybrid approach combining SimpleITK (classical optimization) and VoxelMorph (deep learning)
- **Strain Tensor Analysis**: Large deformation theory using Green-Lagrange strain tensors
- **Constitutive Modeling**: Hyperelastic material models (Neo-Hookean, Mooney-Rivlin, Yeoh)
- **Inverse Problem Solving**: Regularized optimization for Young's modulus estimation
- **Multi-scale Modeling**: Integration with Human Organ Atlas for microstructure-based property estimation
- **Finite Element Workflow**: Python-native FEM simulation using SfePy/EasyFEA
- **GPU Acceleration**: CuPy and PyTorch integration for high-performance computing
- **Clinical Validation**: SPECT/CT comparison capabilities

## Architecture

```
Vent4D-Mech Pipeline
├── Image Registration (SimpleITK + VoxelMorph)
├── Deformation Analysis (Strain Tensor Calculation)
├── Mechanical Modeling (Hyperelastic Constitutive Models)
├── Inverse Problem (Young's Modulus Estimation)
├── Microstructure DB (Human Organ Atlas Integration)
├── FEM Workflow (Lung Deformation Simulation)
└── Ventilation Analysis (Jacobian-based Calculation)
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for performance)
- 16GB+ RAM (for 3D medical image processing)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/your-org/vent4d-mech.git
cd vent4d-mech
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package**
```bash
pip install -e .
```

### GPU Support (Optional but Recommended)

For GPU acceleration with CuPy:
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

## Quick Start

### Basic Usage

```python
import vent4d_mech as v4d

# Initialize the pipeline
pipeline = v4d.Vent4DMechPipeline(config_path="config/default.yaml")

# Load 4D-CT data
pipeline.load_ct_data(
    inhale_path="data/inhale.nii.gz",
    exhale_path="data/exhale.nii.gz"
)

# Run complete analysis
results = pipeline.run_analysis()

# Access results
ventilation_map = results.ventilation_map
youngs_modulus_map = results.youngs_modulus
strain_tensors = results.strain_tensors
```

### Configuration

Create a configuration file `config.yaml`:

```yaml
image_registration:
  method: "voxelmorph"  # or "simpleitk"
  gpu: true

deformation_analysis:
  strain_theory: "green_lagrange"  # Large deformation theory
  voxel_spacing: [1.5, 1.5, 3.0]  # mm

mechanical_modeling:
  constitutive_model: "mooney_rivlin"
  initial_parameters:
    C10: 0.135  # kPa
    C01: 0.035  # kPa

inverse_problem:
  regularization_method: "tikhonov"
  optimization_method: "least_squares"

fem_workflow:
  solver: "sfepy"
  mesh_resolution: 2.0  # mm

performance:
  gpu_acceleration: true
  parallel_processing: true
  memory_limit: "8GB"
```

## Core Modules

### 1. Image Registration (`vent4d_mech.core.registration`)

- **SimpleITK**: Classical B-spline FFD and Demons algorithms
- **VoxelMorph**: Deep learning-based unsupervised registration
- **Hybrid Strategy**: Use SimpleITK for training data, VoxelMorph for fast inference

### 2. Deformation Analysis (`vent4d_mech.core.deformation`)

- Displacement Vector Field (DVF) processing
- Deformation gradient tensor calculation
- Green-Lagrange strain tensor computation
- Strain invariants analysis

### 3. Mechanical Modeler (`vent4d_mech.core.mechanical`)

- Hyperelastic constitutive models
- Material parameter estimation
- Stress-strain relationship modeling

### 4. Young's Modulus Estimator (`vent4d_mech.core.inverse`)

- Inverse problem formulation
- Regularized optimization
- Physical constraints integration

### 5. Microstructure Database (`vent4d_mech.core.microstructure`)

- Human Organ Atlas integration
- Multi-scale modeling (RVE analysis)
- Structure-property relationships

### 6. FEM Workflow (`vent4d_mech.core.fem`)

- Mesh generation from CT segmentation
- Boundary condition application
- Nonlinear solver integration

### 7. Ventilation Calculator (`vent4d_mech.core.ventilation`)

- Jacobian-based ventilation calculation
- Regional ventilation analysis
- Clinical metrics computation

## Performance Optimization

### GPU Acceleration

The framework leverages GPU acceleration through:
- **CuPy**: NumPy-compatible GPU arrays
- **PyTorch**: Deep learning operations
- **Zero-copy interoperability**: Direct GPU memory sharing

### Parallel Processing

- **Multiprocessing**: Patient-level parallelization
- **Joblib**: Task-level parallelization
- **Dask**: Out-of-core computation for large datasets

## Data Requirements

### Input Data

- **4D-CT**: Inhale and exhale phases (NIfTI or DICOM)
- **Segmentation**: Lung region masks (optional)
- **Clinical Data**: SPECT/CT for validation (optional)

### Output Data

- **Deformation Fields**: 3D displacement vectors
- **Strain Maps**: 6-component strain tensor fields
- **Material Properties**: Young's modulus distributions
- **Ventilation Maps**: Regional ventilation indices

## Validation

The framework includes validation tools for:

- **SPECT/CT Comparison**: Quantitative validation against clinical gold standard
- **Reproducibility**: Test-retest reliability analysis
- **Phantom Studies**: Synthetic data validation
- **Clinical Correlation**: Patient outcome analysis

## Examples

See the `examples/` directory for:

- `basic_pipeline.py`: Complete workflow example
- `registration_comparison.py`: SimpleITK vs VoxelMorph comparison
- `strain_analysis.py`: Deformation analysis example
- `fem_simulation.py`: Finite element simulation example
- `validation_study.py`: Clinical validation example

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_registration.py

# Run with coverage
pytest --cov=vent4d_mech
```

## Documentation

Comprehensive documentation is available at:

- **API Reference**: Complete module documentation
- **Tutorials**: Step-by-step guides
- **Theory Background**: Mathematical formulations
- **Clinical Guide**: Practical usage recommendations

## Citation

If you use Vent4D-Mech in your research, please cite:

```bibtex
@software{vent4d_mech_2024,
  title={Vent4D-Mech: Python-based Lung Tissue Dynamics Modeling},
  author={Vent4D-Mech Development Team},
  year={2024},
  url={https://github.com/your-org/vent4d-mech}
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## Acknowledgments

- Human Organ Atlas (ESRF) for microstructure data
- VoxelMorph team for deep learning registration framework
- SfePy developers for finite element tools
- Clinical partners for validation data

## Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Community forum for questions
- **Documentation**: Detailed guides and API reference
- **Email**: vent4d-mech@example.com

---

**Note**: This framework is designed for research and clinical investigation purposes. Clinical use requires appropriate validation and regulatory approval.
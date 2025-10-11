# Vent4D-Mech Testing Strategy

## Testing Framework Configuration

### Core Testing Stack
- **pytest**: Primary testing framework (v7.4.0+)
- **pytest-cov**: Coverage reporting (v4.1.0+)
- **pytest-mock**: Mocking and patching utilities
- **pytest-xdist**: Parallel test execution
- **hypothesis**: Property-based testing

### Test Data Management
- **Synthetic Data**: Generated programmatically for reproducibility
- **Test Fixtures**: Reusable test data and setup
- **Data Versioning**: Versioned test datasets for regression testing
- **Memory Management**: Efficient handling of large 3D test datasets

## Test Categories and Organization

### 1. Unit Tests (`tests/unit/`)
**Purpose**: Test individual components in isolation
**Coverage Target**: >90% line coverage

#### Module Structure
```
tests/unit/
├── test_registration/
│   ├── test_simpleitk_registration.py
│   ├── test_voxelmorph_registration.py
│   └── test_registration_utils.py
├── test_deformation/
│   ├── test_deformation_analyzer.py
│   ├── test_strain_calculator.py
│   ├── test_strain_tensor.py
│   └── test_deformation_utils.py
├── test_mechanical/
│   ├── test_constitutive_models.py
│   ├── test_hyperelastic_models.py
│   ├── test_mechanical_modeler.py
│   └── test_tissue_mechanics.py
├── test_inverse/
│   ├── test_youngs_modulus_estimator.py
│   ├── test_parameter_estimation.py
│   └── test_regularization.py
├── test_microstructure/
│   ├── test_microstructure_db.py
│   ├── test_homogenization.py
│   └── test_atlas_integration.py
├── test_fem/
│   ├── test_fem_workflow.py
│   ├── test_finite_element_solver.py
│   ├── test_mesh_generation.py
│   ├── test_boundary_conditions.py
│   └── test_material_models.py
├── test_ventilation/
│   ├── test_ventilation_calculator.py
│   └── test_ventilation_calculation.py
├── test_config/
│   └── test_config_manager.py
├── test_utils/
│   └── test_io_utils.py
└── test_pipeline.py
```

#### Unit Test Patterns
```python
import pytest
import numpy as np
from unittest.mock import Mock, patch

class TestConstitutiveModel:
    """Test suite for constitutive models."""
    
    @pytest.fixture
    def sample_parameters(self):
        """Sample material parameters."""
        return {"C10": 0.135, "C01": 0.035}
    
    @pytest.fixture
    def sample_deformation_gradient(self):
        """Sample deformation gradient tensor."""
        # Identity tensor with small perturbation
        F = np.eye(3)
        F[0, 0] = 1.1
        F[1, 1] = 0.95
        return F
    
    def test_stress_computation(self, sample_parameters, sample_deformation_gradient):
        """Test stress tensor computation."""
        model = MooneyRivlinModel(sample_parameters)
        stress = model.compute_stress(sample_deformation_gradient)
        
        # Stress tensor should be symmetric
        assert np.allclose(stress, stress.T)
        
        # Stress should be positive for this deformation
        assert np.all(stress > 0)
    
    def test_tangent_modulus_computation(self, sample_parameters, sample_deformation_gradient):
        """Test tangent modulus computation."""
        model = MooneyRivlinModel(sample_parameters)
        tangent = model.compute_tangent_modulus(sample_deformation_gradient)
        
        # Tangent modulus should have correct shape
        assert tangent.shape == (3, 3, 3, 3)
        
        # Should have major and minor symmetries
        assert np.allclose(tangent, np.transpose(tangent, (1, 0, 2, 3)))  # Major symmetry
        assert np.allclose(tangent, np.transpose(tangent, (0, 1, 3, 2)))  # Minor symmetry
    
    @pytest.mark.parametrize("strain_value", [0.8, 0.9, 1.0, 1.1, 1.2])
    def test_uniaxial_loading(self, sample_parameters, strain_value):
        """Test model behavior under uniaxial loading."""
        # Create uniaxial deformation gradient
        F = np.diag([strain_value, 1/np.sqrt(strain_value), 1/np.sqrt(strain_value)])
        
        model = MooneyRivlinModel(sample_parameters)
        stress = model.compute_stress(F)
        
        # First principal stress should be dominant
        assert abs(stress[0, 0]) > abs(stress[1, 1])
        assert abs(stress[0, 0]) > abs(stress[2, 2])
```

### 2. Integration Tests (`tests/integration/`)
**Purpose**: Test module interactions and end-to-end workflows
**Coverage Target**: All major workflows

#### Test Scenarios
```python
class TestPipelineIntegration:
    """Test suite for pipeline integration."""
    
    @pytest.fixture
    def synthetic_4dct_data(self):
        """Generate synthetic 4D-CT data."""
        # Create synthetic inhale/exhale volumes
        inhale = generate_synthetic_lung_volume(shape=(128, 128, 64))
        exhale = apply_synthetic_deformation(inhale, deformation_type="breathing")
        return {"inhale": inhale, "exhale": exhale}
    
    def test_full_pipeline_workflow(self, synthetic_4dct_data):
        """Test complete pipeline workflow."""
        pipeline = Vent4DMechPipeline(config_path="tests/data/test_config.yaml")
        
        # Load data
        pipeline.load_ct_data(
            inhale_path="tests/data/synthetic_inhale.nii.gz",
            exhale_path="tests/data/synthetic_exhale.nii.gz"
        )
        
        # Run full analysis
        results = pipeline.run_analysis()
        
        # Validate results
        assert results.ventilation_map is not None
        assert results.youngs_modulus is not None
        assert results.strain_tensors is not None
        
        # Check physical constraints
        assert np.all(results.ventilation_map > 0)
        assert np.all(results.youngs_modulus > 0)
    
    def test_gpu_cpu_consistency(self, synthetic_4dct_data):
        """Test GPU and CPU computation consistency."""
        config_cpu = {"performance": {"gpu_acceleration": False}}
        config_gpu = {"performance": {"gpu_acceleration": True}}
        
        pipeline_cpu = Vent4DMechPipeline(config=config_cpu)
        pipeline_gpu = Vent4DMechPipeline(config=config_gpu)
        
        # Run with both configurations
        results_cpu = pipeline_cpu.run_analysis()
        results_gpu = pipeline_gpu.run_analysis()
        
        # Results should be similar (within numerical tolerance)
        assert np.allclose(results_cpu.ventilation_map, results_gpu.ventilation_map, rtol=1e-6)
```

### 3. Performance Tests (`tests/performance/`)
**Purpose**: Validate performance requirements and benchmarks
**Coverage Target**: Critical performance paths

#### Performance Test Patterns
```python
class TestPerformance:
    """Test suite for performance validation."""
    
    @pytest.mark.performance
    def test_gpu_speedup(self, large_3d_data):
        """Test GPU acceleration provides speedup."""
        import time
        
        # CPU version
        start_time = time.time()
        result_cpu = run_registration_cpu(large_3d_data)
        cpu_time = time.time() - start_time
        
        # GPU version
        start_time = time.time()
        result_gpu = run_registration_gpu(large_3d_data)
        gpu_time = time.time() - start_time
        
        # GPU should be faster
        speedup_ratio = cpu_time / gpu_time
        assert speedup_ratio > 2.0  # At least 2x speedup
        
        # Results should be similar
        assert np.allclose(result_cpu, result_gpu, rtol=1e-4)
    
    @pytest.mark.performance
    def test_memory_usage(self, large_3d_data):
        """Test memory usage stays within bounds."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run memory-intensive operation
        result = run_memory_intensive_operation(large_3d_data)
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 4 * 1024**3  # Less than 4GB increase
        
        # Cleanup should reduce memory
        del result
        gc.collect()
        final_memory = process.memory_info().rss
        assert final_memory - initial_memory < 500 * 1024**2  # Less than 500MB residual
```

### 4. Validation Tests (`tests/validation/`)
**Purpose**: Validate against known results and clinical data
**Coverage Target**: Key validation scenarios

#### Validation Test Patterns
```python
class TestValidation:
    """Test suite for clinical validation."""
    
    @pytest.mark.validation
    def test_spect_correlation(self, ventilation_data, spect_reference):
        """Test ventilation map correlation with SPECT reference."""
        correlation = compute_correlation(ventilation_data, spect_reference)
        
        # Should have reasonable correlation with SPECT
        assert correlation > 0.7  # Pearson correlation > 0.7
        
        # Regional analysis
        regional_correlations = compute_regional_correlations(
            ventilation_data, spect_reference
        )
        for region, corr in regional_correlations.items():
            assert corr > 0.6  # Regional correlations > 0.6
    
    @pytest.mark.validation
    def test_phantom_validation(self, phantom_data):
        """Test validation using phantom data with known properties."""
        results = run_phantom_analysis(phantom_data)
        
        # Known phantom properties should be recovered
        true_modulus = phantom_data["true_youngs_modulus"]
        estimated_modulus = results.youngs_modulus
        
        relative_error = np.abs(estimated_modulus - true_modulus) / true_modulus
        assert np.mean(relative_error) < 0.1  # Mean error < 10%
```

## Test Data Generation

### Synthetic Data Utilities
```python
class SyntheticDataGenerator:
    """Generate synthetic medical image data for testing."""
    
    @staticmethod
    def generate_lung_volume(
        shape: Tuple[int, int, int],
        noise_level: float = 0.01
    ) -> npt.NDArray:
        """Generate synthetic lung CT volume."""
        # Create lung shape
        volume = np.zeros(shape)
        
        # Add lung lobes
        volume = add_lung_lobes(volume, shape)
        
        # Add vascular structures
        volume = add_vascular_structures(volume)
        
        # Add noise
        volume += np.random.normal(0, noise_level, shape)
        
        return volume
    
    @staticmethod
    def apply_breathing_deformation(
        volume: npt.NDArray,
        deformation_amplitude: float = 0.1
    ) -> npt.NDArray:
        """Apply breathing deformation to volume."""
        # Generate realistic breathing deformation field
        deformation_field = generate_breathing_field(volume.shape, deformation_amplitude)
        
        # Apply deformation
        deformed_volume = apply_deformation_field(volume, deformation_field)
        
        return deformed_volume
    
    @staticmethod
    def generate_spect_reference(
        ventilation_map: npt.NDArray,
        noise_level: float = 0.05
    ) -> npt.NDArray:
        """Generate synthetic SPECT reference from ventilation."""
        # Add SPECT-specific noise and resolution effects
        spect = apply_spect_resolution(ventilation_map)
        spect += np.random.normal(0, noise_level, spect.shape)
        spect = np.clip(spect, 0, None)
        
        return spect
```

## Continuous Integration Configuration

### GitHub Actions Workflow
```yaml
name: Vent4D-Mech CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .[dev,test]
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=vent4d_mech --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v -m performance
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## Quality Gates

### Test Coverage Requirements
- **Unit Tests**: >90% line coverage
- **Integration Tests**: 100% major workflow coverage
- **Overall Coverage**: >85% combined coverage

### Performance Requirements
- **GPU Speedup**: >2x speedup for large datasets
- **Memory Usage**: <4GB for standard datasets
- **Processing Time**: <30 minutes for typical 4D-CT

### Validation Requirements
- **SPECT Correlation**: >0.7 Pearson correlation
- **Phantom Accuracy**: <10% relative error
- **Reproducibility**: >0.95 test-retest correlation

### Code Quality Requirements
- **Black Formatting**: 100% compliance
- **Flake8 Linting**: Zero violations
- **MyPy Type Checking**: Zero type errors
- **Documentation**: 100% public API coverage

## Test Execution Guidelines

### Local Development
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=vent4d_mech --cov-report=html

# Run performance tests
pytest -m performance

# Run validation tests
pytest -m validation
```

### Pre-commit Testing
```bash
# Quick smoke test
pytest tests/unit/test_core.py -v

# Full test suite before commits
pytest tests/ --cov=vent4d_mech
```

### Release Testing
```bash
# Complete test suite
pytest tests/ -v --cov=vent4d_mech --cov-report=html

# Performance benchmarking
pytest tests/performance/ -v --benchmark-only

# Validation suite
pytest tests/validation/ -v
```
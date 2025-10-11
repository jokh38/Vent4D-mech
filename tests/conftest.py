"""
Pytest Configuration and Fixtures

This module provides common fixtures and configuration for pytest
tests in the Vent4D-Mech test suite.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock numpy and scipy for testing since they may not be available
class MockNumPy:
    """Mock NumPy module for testing without actual NumPy."""

    def array(self, data):
        """Mock array creation."""
        return data if isinstance(data, list) else [[data]]

    def eye(self, n):
        """Mock identity matrix."""
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def zeros(self, shape):
        """Mock zeros array."""
        if isinstance(shape, int):
            return [[0] * shape]
        elif isinstance(shape, tuple):
            if len(shape) == 2:
                return [[0 for _ in range(shape[1])] for _ in range(shape[0])]
            elif len(shape) == 3:
                return [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
        return [[0]]

    def random(self):
        """Mock random module."""
        class MockRandom:
            def rand(self, *args):
                if len(args) == 0:
                    return 0.5
                elif len(args) == 1:
                    return [0.5] * args[0]
                elif len(args) == 2:
                    return [[0.5 for _ in range(args[1])] for _ in range(args[0])]
                elif len(args) == 3:
                    return [[[0.5 for _ in range(args[2])] for _ in range(args[1])] for _ in range(args[0])]
                return [0.5]
        return MockRandom()

    # Add other commonly used numpy functions as needed
    def linspace(self, start, stop, num):
        """Mock linspace."""
        if num <= 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]

    def mean(self, data):
        """Mock mean."""
        if isinstance(data, (list, tuple)):
            return sum(data) / len(data) if data else 0
        return 0.5

    def std(self, data):
        """Mock standard deviation."""
        return 0.1  # Mock value


# Mock numpy and scipy if not available
try:
    import numpy
except ImportError:
    sys.modules['numpy'] = MockNumPy()
    numpy = MockNumPy()

try:
    import scipy
except ImportError:
    # Create minimal scipy mock
    sys.modules['scipy'] = type('MockScipy', (), {})()
    sys.modules['scipy.optimize'] = type('MockOptimize', (), {})()


@pytest.fixture
def sample_strain_data():
    """Sample strain data for testing."""
    return [0.0, 0.05, 0.1, 0.15, 0.2]


@pytest.fixture
def sample_stress_data():
    """Sample stress data for testing."""
    return [0.0, 1.2, 2.5, 3.8, 5.1]


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'registration': {
            'method': 'simpleitk',
            'gpu_acceleration': False
        },
        'mechanical': {
            'constitutive_model': 'neo_hookean',
            'material_parameters': {
                'C10': 0.135,
                'density': 1.05
            }
        },
        'performance': {
            'gpu_acceleration': False,
            'parallel_processing': False
        },
        'logging': {
            'level': 'INFO',
            'console_logging': {'enabled': True}
        }
    }


@pytest.fixture
def mock_3d_image():
    """Mock 3D medical image data."""
    # Create a simple 32x32x32 test image
    size = 32
    return [[[i + j + k for k in range(size)] for j in range(size)] for i in range(size)]


@pytest.fixture
def mock_deformation_gradient():
    """Mock deformation gradient tensor."""
    # Identity tensor with small perturbation
    return [[[1.01 if i == j else 0.001 for j in range(3)] for i in range(3)]]


@pytest.fixture
def sample_material_parameters():
    """Sample material parameters for different models."""
    return {
        'neo_hookean': {'C10': 0.135, 'density': 1.05},
        'mooney_rivlin': {'C10': 0.135, 'C01': 0.035, 'density': 1.05},
        'yeoh': {'C10': 0.135, 'C20': 0.015, 'C30': 0.001, 'density': 1.05},
        'linear_elastic': {'youngs_modulus': 5.0, 'poisson_ratio': 0.45, 'density': 1.05}
    }


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Setup test environment for all tests."""
    # Create temporary directories for test outputs
    temp_dir = tmp_path / "vent4d_mech_tests"
    temp_dir.mkdir()

    # Set environment variables for testing
    os.environ['VENT4D_MECH_TEST_MODE'] = '1'
    os.environ['VENT4D_MECH_TEMP_DIR'] = str(temp_dir)

    yield

    # Cleanup
    os.environ.pop('VENT4D_MECH_TEST_MODE', None)
    os.environ.pop('VENT4D_MECH_TEMP_DIR', None)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for test files."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# Skip markers for tests requiring external dependencies
def requires_numpy():
    """Mark test as requiring NumPy."""
    try:
        import numpy
        return pytest.mark.skipif(False, reason="NumPy is available")
    except ImportError:
        return pytest.mark.skipif(True, reason="NumPy not available")


def requires_scipy():
    """Mark test as requiring SciPy."""
    try:
        import scipy
        return pytest.mark.skipif(False, reason="SciPy is available")
    except ImportError:
        return pytest.mark.skipif(True, reason="SciPy not available")


def requires_matplotlib():
    """Mark test as requiring Matplotlib."""
    try:
        import matplotlib
        return pytest.mark.skipif(False, reason="Matplotlib is available")
    except ImportError:
        return pytest.mark.skipif(True, reason="Matplotlib not available")


# Custom assertion helpers
def assert_valid_material_model(model, model_type):
    """Assert that a material model is valid."""
    assert model is not None
    assert model.__class__.__name__.lower().replace('_', '') == model_type.replace('_', '')
    assert hasattr(model, 'parameters')
    assert hasattr(model, 'compute_stress')
    assert hasattr(model, 'compute_strain_energy_density')


def assert_valid_config(config):
    """Assert that configuration is valid."""
    assert isinstance(config, dict)
    assert 'registration' in config
    assert 'mechanical' in config
    assert 'performance' in config
    assert 'logging' in config


def assert_valid_tensor(tensor, expected_shape=None):
    """Assert that a tensor has valid properties."""
    assert tensor is not None
    if expected_shape:
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


# Test data generators
def generate_test_strain_stress_data(num_points=10, noise_level=0.0):
    """Generate test strain-stress data."""
    strain = [i * 0.05 for i in range(num_points)]
    # Simple linear relationship with optional noise
    base_stress = [s * 25 for s in strain]  # 25 kPa at 100% strain

    if noise_level > 0:
        import random
        stress = [s * (1 + random.uniform(-noise_level, noise_level)) for s in base_stress]
    else:
        stress = base_stress

    return strain, stress


# Performance testing utilities
def time_function_call(func, *args, **kwargs):
    """Time a function call and return result and duration."""
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return result, duration


# Memory usage utilities (mock implementation)
def get_memory_usage():
    """Get current memory usage (mock implementation)."""
    return 100.0  # Mock memory usage in MB


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
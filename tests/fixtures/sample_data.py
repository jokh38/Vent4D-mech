"""
Sample Test Data

This module provides sample data for testing purposes, including
synthetic medical images, stress-strain data, and configuration examples.
"""

import random
import math
from typing import List, Dict, Any, Tuple


def generate_synthetic_ct_volume(size: Tuple[int, int, int] = (64, 64, 64),
                               noise_level: float = 0.01) -> List[List[List[float]]]:
    """
    Generate synthetic CT volume data for testing.

    Args:
        size: Volume dimensions (depth, height, width)
        noise_level: Level of noise to add (0.0 to 1.0)

    Returns:
        3D list representing CT volume in Hounsfield units
    """
    depth, height, width = size
    volume = []

    # Create base structure (lung-like)
    for z in range(depth):
        slice_data = []
        for y in range(height):
            row_data = []
            for x in range(width):
                # Create lung-like intensity distribution
                # Background (outside body)
                if (x < width * 0.1 or x > width * 0.9 or
                    y < height * 0.1 or y > height * 0.9):
                    intensity = -1000  # Air
                # Lung tissue
                elif (0.2 < x/width < 0.8 and 0.2 < y/height < 0.8):
                    # Create some variation within lung
                    base_intensity = -700
                    variation = math.sin(x * 0.1) * math.cos(y * 0.1) * 50
                    intensity = base_intensity + variation
                # Body tissue
                else:
                    intensity = 50  # Soft tissue

                # Add noise
                if noise_level > 0:
                    noise = random.gauss(0, noise_level * 100)
                    intensity += noise

                row_data.append(intensity)
            slice_data.append(row_data)
        volume.append(slice_data)

    return volume


def generate_strain_stress_data(material_type: str = 'neo_hookean',
                               num_points: int = 20,
                               max_strain: float = 0.3,
                               noise_level: float = 0.02) -> Tuple[List[float], List[float]]:
    """
    Generate synthetic stress-strain data for material testing.

    Args:
        material_type: Type of material to simulate
        num_points: Number of data points
        max_strain: Maximum strain value
        noise_level: Noise level to add (0.0 to 1.0)

    Returns:
        Tuple of (strain_data, stress_data)
    """
    strain_data = []
    stress_data = []

    # Material parameters for different types
    material_params = {
        'neo_hookean': {'C10': 0.135, 'nonlinearity': 1.2},
        'mooney_rivlin': {'C10': 0.135, 'C01': 0.035, 'nonlinearity': 1.5},
        'linear_elastic': {'E': 5.0, 'nonlinearity': 1.0},
        'yeoh': {'C10': 0.135, 'C20': 0.015, 'nonlinearity': 1.8}
    }

    params = material_params.get(material_type, material_params['neo_hookean'])

    for i in range(num_points):
        strain = (i / (num_points - 1)) * max_strain if num_points > 1 else 0
        strain_data.append(strain)

        # Generate stress based on material type
        if material_type == 'neo_hookean':
            # Neo-Hookean: σ = 2*C10*λ (simplified for uniaxial)
            stress = 2 * params['C10'] * (1 + strain) ** params['nonlinearity']
        elif material_type == 'mooney_rivlin':
            # Mooney-Rivlin (simplified)
            stress = 2 * (params['C10'] + params['C01']) * (1 + strain) ** params['nonlinearity']
        elif material_type == 'linear_elastic':
            # Linear elastic: σ = E*ε
            stress = params['E'] * strain
        else:
            # Generic nonlinear behavior
            stress = params['C10'] * strain * (1 + strain) ** params['nonlinearity']

        # Add noise
        if noise_level > 0:
            noise = random.gauss(0, noise_level * stress)
            stress += noise

        stress_data.append(max(0, stress))  # Ensure non-negative stress

    return strain_data, stress_data


def generate_deformation_field(size: Tuple[int, int, int] = (32, 32, 32),
                              max_displacement: float = 5.0) -> List[List[List[List[float]]]]:
    """
    Generate synthetic deformation field for testing.

    Args:
        size: Field dimensions (depth, height, width)
        max_displacement: Maximum displacement magnitude

    Returns:
        4D list representing deformation field
    """
    depth, height, width = size
    field = []

    for z in range(depth):
        slice_data = []
        for y in range(height):
            row_data = []
            for x in range(width):
                # Create smooth deformation field
                # Using sinusoidal functions for smoothness
                dx = max_displacement * math.sin(2 * math.pi * x / width) * math.cos(2 * math.pi * y / height)
                dy = max_displacement * math.cos(2 * math.pi * x / width) * math.sin(2 * math.pi * y / height)
                dz = max_displacement * math.sin(2 * math.pi * z / depth) * 0.5

                row_data.append([dx, dy, dz])
            slice_data.append(row_data)
        field.append(slice_data)

    return field


def generate_test_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Generate various test configurations.

    Returns:
        Dictionary of test configurations
    """
    configs = {}

    # Minimal configuration
    configs['minimal'] = {
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
            'level': 'INFO'
        }
    }

    # Complete configuration
    configs['complete'] = {
        'registration': {
            'method': 'voxelmorph',
            'gpu_acceleration': True,
            'parameters': {
                'voxelmorph': {
                    'model_type': 'unsupervised',
                    'input_shape': [128, 128, 128],
                    'batch_size': 1,
                    'learning_rate': 0.0001,
                    'epochs': 100
                }
            },
            'preprocessing': {
                'normalize_intensity': True,
                'clip_outliers': True,
                'resample_to_iso': True,
                'target_spacing': [1.0, 1.0, 1.0]
            }
        },
        'mechanical': {
            'constitutive_model': 'mooney_rivlin',
            'material_parameters': {
                'C10': 0.135,
                'C01': 0.035,
                'density': 1.05
            },
            'boundary_conditions': {
                'type': 'displacement_controlled',
                'load_magnitude': 0.1
            }
        },
        'performance': {
            'gpu_acceleration': True,
            'parallel_processing': True,
            'num_processes': 4,
            'memory_management': {
                'chunk_size': 64,
                'cache_size': 1024
            }
        },
        'logging': {
            'level': 'DEBUG',
            'file_logging': {
                'enabled': True,
                'file_path': './logs/test.log'
            },
            'performance_logging': {
                'enabled': True
            }
        }
    }

    # Testing configuration
    configs['testing'] = {
        'registration': {
            'method': 'simpleitk',
            'gpu_acceleration': False
        },
        'mechanical': {
            'constitutive_model': 'neo_hookean',
            'material_parameters': {
                'C10': 0.1,
                'density': 1.0
            }
        },
        'performance': {
            'gpu_acceleration': False,
            'parallel_processing': False
        },
        'logging': {
            'level': 'WARNING',  # Less verbose for testing
            'console_logging': {
                'enabled': True
            }
        },
        'validation': {
            'enabled': True,
            'strict_mode': False
        }
    }

    return configs


def generate_material_parameter_sets() -> Dict[str, Dict[str, float]]:
    """
    Generate material parameter sets for different models.

    Returns:
        Dictionary of material parameters
    """
    return {
        'neo_hookean_soft': {
            'C10': 0.05,
            'density': 1.0
        },
        'neo_hookean_normal': {
            'C10': 0.135,
            'density': 1.05
        },
        'neo_hookean_stiff': {
            'C10': 0.5,
            'density': 1.1
        },
        'mooney_rivlin_soft': {
            'C10': 0.08,
            'C01': 0.02,
            'density': 1.0
        },
        'mooney_rivlin_normal': {
            'C10': 0.135,
            'C01': 0.035,
            'density': 1.05
        },
        'mooney_rivlin_stiff': {
            'C10': 0.3,
            'C01': 0.1,
            'density': 1.1
        },
        'yeoh_soft': {
            'C10': 0.08,
            'C20': 0.01,
            'C30': 0.001,
            'density': 1.0
        },
        'yeoh_normal': {
            'C10': 0.135,
            'C20': 0.015,
            'C30': 0.001,
            'density': 1.05
        },
        'linear_elastic_soft': {
            'youngs_modulus': 1.0,
            'poisson_ratio': 0.45,
            'density': 1.0
        },
        'linear_elastic_normal': {
            'youngs_modulus': 5.0,
            'poisson_ratio': 0.45,
            'density': 1.05
        },
        'linear_elastic_stiff': {
            'youngs_modulus': 20.0,
            'poisson_ratio': 0.45,
            'density': 1.1
        }
    }


def generate_test_scenarios() -> List[Dict[str, Any]]:
    """
    Generate test scenarios for integration testing.

    Returns:
        List of test scenario dictionaries
    """
    scenarios = []

    # Basic scenario
    scenarios.append({
        'name': 'basic_neo_hookean',
        'description': 'Basic test with Neo-Hookean model',
        'config': 'minimal',
        'material_model': 'neo_hookean',
        'material_params': 'neo_hookean_normal',
        'expected_stress_range': (0.1, 10.0)
    })

    # Advanced scenario
    scenarios.append({
        'name': 'advanced_mooney_rivlin',
        'description': 'Advanced test with Mooney-Rivlin model',
        'config': 'complete',
        'material_model': 'mooney_rivlin',
        'material_params': 'mooney_rivlin_normal',
        'expected_stress_range': (0.1, 15.0)
    })

    # Linear elastic scenario
    scenarios.append({
        'name': 'linear_elastic_test',
        'description': 'Test with linear elastic model',
        'config': 'testing',
        'material_model': 'linear_elastic',
        'material_params': 'linear_elastic_normal',
        'expected_stress_range': (0.1, 2.0)
    })

    # Performance scenario
    scenarios.append({
        'name': 'performance_test',
        'description': 'Performance test with larger dataset',
        'config': 'complete',
        'material_model': 'neo_hookean',
        'material_params': 'neo_hookean_normal',
        'data_size': (128, 128, 128),
        'expected_stress_range': (0.1, 10.0)
    })

    return scenarios


def create_synthetic_lung_mask(size: Tuple[int, int, int] = (64, 64, 64)) -> List[List[List[bool]]]:
    """
    Create synthetic lung mask for testing.

    Args:
        size: Volume dimensions (depth, height, width)

    Returns:
        3D list representing lung mask
    """
    depth, height, width = size
    mask = []

    for z in range(depth):
        slice_mask = []
        for y in range(height):
            row_mask = []
            for x in range(width):
                # Create elliptical lung regions
                # Left lung
                center_x_left = width * 0.3
                center_y = height * 0.5
                radius_x = width * 0.15
                radius_y = height * 0.3

                dist_left = ((x - center_x_left) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2

                # Right lung
                center_x_right = width * 0.7
                dist_right = ((x - center_x_right) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2

                # Point is in lung if within either ellipse
                is_lung = (dist_left < 1.0) or (dist_right < 1.0)
                row_mask.append(is_lung)
            slice_mask.append(row_mask)
        mask.append(slice_mask)

    return mask


# Utility functions for test data
def save_test_data_to_file(data: Any, filename: str, format: str = 'json') -> None:
    """
    Save test data to file for debugging.

    Args:
        data: Data to save
        filename: Output filename
        format: File format ('json', 'pickle')
    """
    import json
    import pickle

    if format == 'json':
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_test_data_from_file(filename: str, format: str = 'json') -> Any:
    """
    Load test data from file.

    Args:
        filename: Input filename
        format: File format ('json', 'pickle')

    Returns:
        Loaded data
    """
    import json
    import pickle

    if format == 'json':
        with open(filename, 'r') as f:
            return json.load(f)
    elif format == 'pickle':
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


# Data validation functions
def validate_strain_stress_data(strain: List[float], stress: List[float]) -> bool:
    """
    Validate strain-stress data format.

    Args:
        strain: Strain data
        stress: Stress data

    Returns:
        True if data is valid
    """
    if not isinstance(strain, list) or not isinstance(stress, list):
        return False

    if len(strain) != len(stress):
        return False

    if len(strain) < 2:
        return False

    # Check for monotonic increasing strain
    for i in range(1, len(strain)):
        if strain[i] <= strain[i-1]:
            return False

    # Check for non-negative stress
    for s in stress:
        if s < 0:
            return False

    return True


def validate_volume_data(volume: List[List[List[float]]], expected_shape: Tuple[int, int, int] = None) -> bool:
    """
    Validate 3D volume data format.

    Args:
        volume: 3D volume data
        expected_shape: Expected shape (depth, height, width)

    Returns:
        True if data is valid
    """
    if not isinstance(volume, list) or len(volume) == 0:
        return False

    depth = len(volume)
    height = len(volume[0]) if volume[0] else 0
    width = len(volume[0][0]) if volume[0] and volume[0][0] else 0

    if expected_shape:
        if (depth, height, width) != expected_shape:
            return False

    # Check that all slices have consistent dimensions
    for slice_data in volume:
        if len(slice_data) != height:
            return False
        for row in slice_data:
            if len(row) != width:
                return False

    return True
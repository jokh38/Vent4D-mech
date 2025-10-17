"""
Default Configuration

This module provides default configuration parameters for the Vent4D-Mech framework,
including all necessary settings for registration, mechanical modeling, and
performance optimization.
"""

from typing import Dict, Any
import logging


class DefaultConfig:
    """
    Default configuration parameters for Vent4D-Mech.

    This class provides comprehensive default settings for all framework components,
    ensuring consistent behavior and sensible starting points for users.
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get complete default configuration.

        Returns:
            Dictionary containing all default configuration parameters
        """
        return {
            'version': '0.1.0',
            'framework': {
                'name': 'Vent4D-Mech',
                'description': 'Python-based Lung Tissue Dynamics Modeling'
            },

            'registration': {
                'method': 'voxelmorph',
                'gpu_acceleration': True,
                'parameters': {
                    'voxelmorph': {
                        'model_type': 'unsupervised',
                        'input_shape': [128, 128, 128],
                        'batch_size': 1,
                        'learning_rate': 0.0001,
                        'epochs': 1000,
                        'loss_weights': {
                            'image_similarity': 1.0,
                            'regularization': 0.01
                        }
                    },
                    'simpleitk': {
                        'interpolator': 'Linear',
                        'metric': 'MeanSquares',
                        'optimizer': 'LBFGS',
                        'shrink_factors': [8, 4, 2, 1],
                        'smooth_sigmas': [4, 2, 1, 0],
                        'sampling_rates': [0.25, 0.5, 0.75, 1.0]
                    },
                    'deformable': {
                        'grid_spacing': [10, 10, 10],
                        'regularization_weight': 0.01,
                        'max_iterations': 100,
                        'convergence_threshold': 1e-6
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
                    'neo_hookean': {
                        'C10': 0.135,  # kPa - typical for lung tissue
                        'density': 1.05  # g/cm³ - tissue density
                    },
                    'mooney_rivlin': {
                        'C10': 0.135,  # kPa
                        'C01': 0.035,  # kPa
                        'density': 1.05  # g/cm³
                    },
                    'yeoh': {
                        'C10': 0.135,  # kPa
                        'C20': 0.015,  # kPa
                        'C30': 0.001,  # kPa
                        'density': 1.05  # g/cm³
                    },
                    'ogden': {
                        'mu1': 0.5,     # kPa
                        'alpha1': 2.0,  # dimensionless
                        'mu2': 0.1,     # kPa
                        'alpha2': -2.0, # dimensionless
                        'density': 1.05 # g/cm³
                    },
                    'linear_elastic': {
                        'youngs_modulus': 5.0,  # kPa
                        'poisson_ratio': 0.45,   # nearly incompressible
                        'density': 1.05          # g/cm³
                    }
                },
                'boundary_conditions': {
                    'type': 'displacement_controlled',
                    'fixed_surfaces': ['chest_wall', 'mediastinum'],
                    'loaded_surfaces': ['pleural_surface'],
                    'load_magnitude': 0.1  # kPa - typical pleural pressure
                },
                'solver': {
                    'type': 'nonlinear',
                    'tolerance': 1e-6,
                    'max_iterations': 50,
                    'line_search': True
                }
            },

            'deformation': {
                'strain_calculation': {
                    'method': 'finite_differences',
                    'smoothing_sigma': 1.0,
                    'regularization_weight': 0.01
                },
                'deformation_gradient': {
                    'compute_determinant': True,
                    'compute_inverse': True,
                    'validation_enabled': True
                }
            },

            'inverse': {
                'youngs_modulus_estimation': {
                    'method': 'variational',
                    'regularization_type': 'tikhonov',
                    'regularization_weight': 0.1,
                    'optimization_method': 'L-BFGS-B',
                    'bounds': {
                        'youngs_modulus': [0.1, 50.0]  # kPa
                    }
                },
                'parameter_estimation': {
                    'cost_function': 'weighted_least_squares',
                    'weight_displacement': 1.0,
                    'weight_volume': 0.1,
                    'max_iterations': 100
                }
            },

            'ventilation': {
                'calculation_method': 'jacobian_determinant',
                'regional_analysis': {
                    'lung_segments': 18,  # number of lung segments
                    'lobe_segmentation': True
                },
                'metrics': {
                    'tidal_volume': True,
                    'functional_residual_capacity': True,
                    'ventilation_perfusion': False
                }
            },

            'microstructure': {
                'atlas_integration': {
                    'enabled': False,
                    'atlas_path': None,
                    'registration_method': 'affine'
                },
                'homogenization': {
                    'method': 'periodic_boundary_conditions',
                    'representative_volume_elements': {
                        'alveolar_sac': 20,    # μm
                        'alveolar_duct': 200,  # μm
                        'acinus': 5000        # μm
                    }
                }
            },

            'fem': {
                'mesh_generation': {
                    'element_type': 'tetrahedral',
                    'mesh_size': 5.0,  # mm
                    'refinement_regions': ['airways', 'vessels'],
                    'quality_metrics': {
                        'min_angle': 10.0,  # degrees
                        'max_angle': 140.0, # degrees
                        'min_jacobian': 0.1
                    }
                },
                'finite_element_solver': {
                    'software': 'fenics',
                    'linear_solver': 'mumps',
                    'preconditioner': 'ilu',
                    'convergence_tolerance': 1e-8
                }
            },

            'performance': {
                'gpu_acceleration': True,
                'parallel_processing': True,
                'num_processes': 'auto',
                'memory_management': {
                    'chunk_size': 64,  # voxels
                    'cache_size': 1024,  # MB
                    'memory_limit': 8192  # MB
                },
                'optimization': {
                    'vectorized_operations': True,
                    'sparse_matrices': True,
                    'adaptive_solvers': True
                }
            },

            'data_handling': {
                'input_formats': ['.nii', '.nii.gz', '.h5', '.mhd'],
                'output_formats': ['.nii.gz', '.h5', '.vtk'],
                'compression': {
                    'enabled': True,
                    'algorithm': 'gzip',
                    'compression_level': 6
                },
                'temp_files': {
                    'directory': './temp',
                    'cleanup_on_exit': True,
                    'max_age_hours': 24
                }
            },

            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_logging': {
                    'enabled': True,
                    'file_path': './logs/vent4d_mech.log',
                    'max_file_size_mb': 100,
                    'backup_count': 5
                },
                'console_logging': {
                    'enabled': True,
                    'level': 'INFO'
                },
                'performance_logging': {
                    'enabled': True,
                    'log_memory_usage': True,
                    'log_timing': True
                }
            },

            'validation': {
                'enabled': True,
                'strict_mode': False,
                'checks': {
                    'input_validation': True,
                    'output_validation': True,
                    'parameter_validation': True,
                    'convergence_validation': True
                },
                'error_handling': {
                    'raise_on_error': False,
                    'log_errors': True,
                    'continue_on_warning': True
                }
            },

            'output': {
                'save_intermediate_results': False,
                'final_results': {
                    'format': 'hdf5',
                    'compression': True,
                    'metadata': True
                },
                'visualization': {
                    'generate_plots': True,
                    'save_plots': True,
                    'plot_format': 'png',
                    'dpi': 300
                },
                'reports': {
                    'generate_html': True,
                    'generate_pdf': False,
                    'include_performance_metrics': True
                }
            },

            'quality_assurance': {
                'test_data': {
                    'synthetic_data_enabled': True,
                    'synthetic_data_size': [64, 64, 64],
                    'noise_level': 0.01
                },
                'benchmarks': {
                    'enabled': False,
                    'reference_results_path': None,
                    'tolerance': 0.05
                }
            }
        }

    @staticmethod
    def get_registration_config() -> Dict[str, Any]:
        """Get registration-specific configuration."""
        full_config = DefaultConfig.get_default_config()
        return full_config['registration']

    @staticmethod
    def get_mechanical_config() -> Dict[str, Any]:
        """Get mechanical modeling configuration."""
        full_config = DefaultConfig.get_default_config()
        return full_config['mechanical']

    @staticmethod
    def get_performance_config() -> Dict[str, Any]:
        """Get performance configuration."""
        full_config = DefaultConfig.get_default_config()
        return full_config['performance']

    @staticmethod
    def get_logging_config() -> Dict[str, Any]:
        """Get logging configuration."""
        full_config = DefaultConfig.get_default_config()
        return full_config['logging']

    @staticmethod
    def get_material_parameters(model_name: str) -> Dict[str, Any]:
        """
        Get material parameters for a specific constitutive model.

        Args:
            model_name: Name of constitutive model

        Returns:
            Material parameters dictionary

        Raises:
            ValueError: If model name is not recognized
        """
        full_config = DefaultConfig.get_default_config()
        material_params = full_config['mechanical']['material_parameters']

        if model_name not in material_params:
            available_models = list(material_params.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")

        return material_params[model_name].copy()

    @staticmethod
    def validate_config_structure(config: Dict[str, Any]) -> bool:
        """
        Validate that config has the expected structure.

        Args:
            config: Configuration dictionary

        Returns:
            True if structure is valid
        """
        required_sections = [
            'registration', 'mechanical', 'performance',
            'logging', 'validation'
        ]

        for section in required_sections:
            if section not in config:
                logging.warning(f"Missing required configuration section: {section}")
                return False

        return True

    @staticmethod
    def get_minimal_config() -> Dict[str, Any]:
        """
        Get minimal configuration for basic functionality.

        Returns:
            Minimal configuration dictionary
        """
        return {
            'registration': {
                'method': 'simpleitk',
                'gpu_acceleration': False
            },
            'mechanical': {
                'constitutive_model': 'neo_hookean',
                'material_parameters': DefaultConfig.get_material_parameters('neo_hookean')
            },
            'performance': {
                'gpu_acceleration': False,
                'parallel_processing': False
            },
            'logging': {
                'level': 'INFO',
                'console_logging': {'enabled': True}
            },
            'validation': {
                'enabled': True,
                'strict_mode': False
            }
        }


# Convenience function to get configuration
def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return DefaultConfig.get_default_config()


def get_minimal_config() -> Dict[str, Any]:
    """
    Get minimal configuration.

    Returns:
        Minimal configuration dictionary
    """
    return DefaultConfig.get_minimal_config()
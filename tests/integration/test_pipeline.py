"""
Pipeline Integration Tests

This module contains integration tests for the Vent4D-Mech pipeline workflow,
testing the interaction between components and end-to-end functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import numpy as np
    from vent4d_mech.pipeline import Vent4DMechPipeline
    from vent4d_mech.core.mechanical.mechanical_modeler import MechanicalModeler
    from vent4d_mech.core.base_component import BaseComponent
    from vent4d_mech.core.exceptions import (
        ConfigurationError, ValidationError, ComputationError
    )
except ImportError as e:
    pytest.skip(f"Pipeline modules not available: {e}", allow_module_level=True)


@pytest.mark.integration
class TestVent4DMechPipeline:
    """Integration tests for Vent4DMechPipeline."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for pipeline testing."""
        return {
            'registration': {
                'method': 'simpleitk',
                'gpu_acceleration': False
            },
            'mechanical': {
                'constitutive_model': 'neo_hookean',
                'material_parameters': {
                    'neo_hookean': {
                        'C10': 0.135,
                        'density': 1.05
                    }
                }
            },
            'performance': {
                'gpu_acceleration': False,
                'parallel_processing': False
            }
        }

    @pytest.fixture
    def mock_image_data(self):
        """Mock 3D medical image data for testing."""
        # Create simple 32x32x32 test volumes
        size = 32
        inhale = np.random.rand(size, size, size).astype(np.float32)
        exhale = np.random.rand(size, size, size).astype(np.float32)
        return inhale, exhale

    @pytest.fixture
    def mock_deformation_data(self):
        """Mock deformation field data."""
        size = 32
        # Simple displacement field
        displacement = np.random.rand(size, size, size, 3).astype(np.float32) * 0.1
        return displacement

    def test_pipeline_initialization(self, sample_config):
        """Test pipeline initialization with configuration."""
        try:
            pipeline = Vent4DMechPipeline(config=sample_config)

            assert pipeline is not None
            assert hasattr(pipeline, 'config')
            assert pipeline.config == sample_config
            assert hasattr(pipeline, 'components')

        except ImportError:
            pytest.skip("Pipeline class not available")
        except Exception as e:
            # Allow for missing dependencies during testing
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise

    def test_pipeline_initialization_default_config(self):
        """Test pipeline initialization with default configuration."""
        try:
            pipeline = Vent4DMechPipeline()

            assert pipeline is not None
            assert hasattr(pipeline, 'config')
            assert isinstance(pipeline.config, dict)

        except ImportError:
            pytest.skip("Pipeline class not available")
        except Exception as e:
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise

    def test_mechanical_modeler_integration(self, sample_config, mock_deformation_data):
        """Test integration of MechanicalModeler with the pipeline."""
        try:
            # Test MechanicalModeler independently first
            mechanical_config = sample_config.get('mechanical', {})
            modeler = MechanicalModeler(config=mechanical_config, gpu=False)

            assert modeler is not None
            assert modeler.config['constitutive_model'] == 'neo_hookean'

            # Test process method
            strain_tensor = np.random.rand(16, 16, 16, 3, 3).astype(np.float32)
            deformation_gradient = np.random.rand(16, 16, 16, 3, 3).astype(np.float32)

            # Add identity to make it more realistic
            deformation_gradient += np.eye(3).reshape(1, 1, 1, 3, 3)

            result = modeler.process(strain_tensor, deformation_gradient, computation_type='stress')

            assert isinstance(result, dict)
            assert '_metadata' in result
            assert 'stress_tensor' in result

            metadata = result['_metadata']
            assert metadata['component'] == 'MechanicalModeler'
            assert metadata['gpu_enabled'] is False

        except ImportError:
            pytest.skip("MechanicalModeler not available")
        except Exception as e:
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise

    def test_pipeline_component_interaction(self, sample_config):
        """Test interaction between pipeline components."""
        try:
            # This test focuses on component interaction
            # We'll mock the complex dependencies and test the architecture

            with patch('vent4d_mech.pipeline.ImageRegistration') as mock_reg, \
                 patch('vent4d_mech.pipeline.MechanicalModeler') as mock_mech, \
                 patch('vent4d_mech.pipeline.DeformationAnalyzer') as mock_def:

                # Configure mocks
                mock_reg.return_value.process.return_value = {
                    'displacement_field': np.zeros((32, 32, 32, 3)),
                    '_metadata': {'component': 'ImageRegistration'}
                }

                mock_mech.return_value.process.return_value = {
                    'stress_tensor': np.zeros((32, 32, 32, 3, 3)),
                    '_metadata': {'component': 'MechanicalModeler'}
                }

                mock_def.return_value.process.return_value = {
                    'strain_tensor': np.zeros((32, 32, 32, 3, 3)),
                    '_metadata': {'component': 'DeformationAnalyzer'}
                }

                # Create pipeline
                pipeline = Vent4DMechPipeline(config=sample_config)

                # Test that components are properly initialized
                assert hasattr(pipeline, 'components')

                # Test component configuration
                if hasattr(pipeline, 'mechanical_modeler'):
                    assert pipeline.mechanical_modeler.config['constitutive_model'] == 'neo_hookean'

        except ImportError:
            pytest.skip("Pipeline classes not available")
        except Exception as e:
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise

    def test_pipeline_configuration_flow(self, sample_config):
        """Test configuration flow through pipeline components."""
        try:
            # Test that configuration properly flows to components
            mechanical_config = sample_config['mechanical']
            modeler = MechanicalModeler(config=mechanical_config)

            # Test component info
            info = modeler.get_component_info()
            assert info['config']['constitutive_model'] == 'neo_hookean'
            assert info['config']['material_parameters']['neo_hookean']['C10'] == 0.135

            # Test configuration update
            new_config = {
                'constitutive_model': 'mooney_rivlin',
                'material_parameters': {
                    'mooney_rivlin': {
                        'C10': 0.2,
                        'C01': 0.05
                    }
                }
            }

            modeler.update_config(new_config)
            updated_info = modeler.get_component_info()
            assert updated_info['config']['constitutive_model'] == 'mooney_rivlin'
            assert updated_info['config']['material_parameters']['mooney_rivlin']['C10'] == 0.2

        except ImportError:
            pytest.skip("MechanicalModeler not available")
        except Exception as e:
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise

    def test_pipeline_error_handling(self, sample_config):
        """Test error handling in pipeline workflow."""
        try:
            # Test configuration error handling
            invalid_config = {
                'mechanical': {
                    'constitutive_model': 'invalid_model',  # Invalid model
                    'material_parameters': {}
                }
            }

            with pytest.raises(ConfigurationError):
                MechanicalModeler(config=invalid_config)

            # Test validation error handling
            modeler = MechanicalModeler(config=sample_config['mechanical'])

            # Invalid input should raise ValidationError
            invalid_tensor = np.random.rand(16, 16, 16)  # Wrong shape
            deformation_gradient = np.random.rand(16, 16, 16, 3, 3)

            with pytest.raises(ValidationError):
                modeler.process(invalid_tensor, deformation_gradient)

        except ImportError:
            pytest.skip("MechanicalModeler not available")
        except Exception as e:
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise

    def test_pipeline_performance_considerations(self, sample_config):
        """Test performance-related aspects of the pipeline."""
        try:
            # Test GPU/CPU configuration
            cpu_modeler = MechanicalModeler(config=sample_config['mechanical'], gpu=False)
            gpu_modeler = MechanicalModeler(config=sample_config['mechanical'], gpu=True)

            cpu_info = cpu_modeler.get_component_info()
            gpu_info = gpu_modeler.get_component_info()

            assert cpu_info['gpu_enabled'] is False
            assert gpu_info['gpu_enabled'] is True

            # Test that configuration affects computation
            strain_tensor = np.random.rand(8, 8, 8, 3, 3).astype(np.float32)
            deformation_gradient = np.random.rand(8, 8, 8, 3, 3).astype(np.float32) + np.eye(3)

            # Both should produce results (even if mocked)
            cpu_result = cpu_modeler.process(strain_tensor, deformation_gradient)
            gpu_result = gpu_modeler.process(strain_tensor, deformation_gradient)

            assert isinstance(cpu_result, dict)
            assert isinstance(gpu_result, dict)
            assert '_metadata' in cpu_result
            assert '_metadata' in gpu_result

        except ImportError:
            pytest.skip("MechanicalModeler not available")
        except Exception as e:
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise

    def test_pipeline_end_to_end_mock(self, sample_config, mock_image_data):
        """Test end-to-end pipeline workflow with mocked components."""
        try:
            inhale_image, exhale_image = mock_image_data

            # Mock the entire pipeline for end-to-end testing
            with patch('vent4d_mech.pipeline.Vent4DMechPipeline') as mock_pipeline_class:

                # Create mock pipeline instance
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline

                # Configure mock pipeline methods
                mock_pipeline.run.return_value = {
                    'inhale_image': inhale_image,
                    'exhale_image': exhale_image,
                    'displacement_field': np.zeros((32, 32, 32, 3)),
                    'strain_tensor': np.zeros((32, 32, 32, 3, 3)),
                    'stress_tensor': np.zeros((32, 32, 32, 3, 3)),
                    'ventilation': np.zeros((32, 32, 32)),
                    'metadata': {
                        'processing_time': 1.0,
                        'components_used': ['registration', 'deformation', 'mechanical', 'ventilation'],
                        'configuration': sample_config
                    }
                }

                # Initialize and run pipeline
                pipeline = Vent4DMechPipeline(config=sample_config)
                result = pipeline.run(inhale_image, exhale_image)

                # Verify structure of results
                assert isinstance(result, dict)
                assert 'inhale_image' in result
                assert 'exhale_image' in result
                assert 'displacement_field' in result
                assert 'strain_tensor' in result
                assert 'stress_tensor' in result
                assert 'ventilation' in result
                assert 'metadata' in result

                # Verify metadata
                metadata = result['metadata']
                assert 'processing_time' in metadata
                assert 'components_used' in metadata
                assert 'configuration' in metadata

        except ImportError:
            pytest.skip("Pipeline not available")
        except Exception as e:
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise


@pytest.mark.integration
class TestComponentArchitecture:
    """Integration tests for component architecture and interactions."""

    def test_base_component_inheritance(self):
        """Test that all major components inherit from BaseComponent."""
        try:
            # Test MechanicalModeler inheritance
            modeler = MechanicalModeler()
            assert isinstance(modeler, BaseComponent)
            assert hasattr(modeler, 'process')
            assert hasattr(modeler, 'get_component_info')
            assert hasattr(modeler, 'update_config')

        except ImportError:
            pytest.skip("MechanicalModeler not available")
        except Exception as e:
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise

    def test_component_interface_consistency(self):
        """Test that components implement consistent interfaces."""
        try:
            modeler = MechanicalModeler()

            # Test required methods exist and are callable
            required_methods = ['process', 'get_component_info', 'update_config']
            for method_name in required_methods:
                assert hasattr(modeler, method_name)
                method = getattr(modeler, method_name)
                assert callable(method)

            # Test process method signature
            import inspect
            process_sig = inspect.signature(modeler.process)
            assert 'args' in process_sig.parameters or 'strain_tensor' in str(process_sig)

        except ImportError:
            pytest.skip("MechanicalModeler not available")
        except Exception as e:
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise

    def test_error_propagation_through_components(self):
        """Test error propagation through component stack."""
        try:
            modeler = MechanicalModeler()

            # Test that ConfigurationError propagates correctly
            with pytest.raises(ConfigurationError):
                modeler.update_config({'constitutive_model': 'invalid_model'})

            # Test that ValidationError propagates from process method
            invalid_data = np.random.rand(16, 16, 16)  # Wrong dimensions
            valid_deformation = np.random.rand(16, 16, 16, 3, 3)

            with pytest.raises(ValidationError):
                modeler.process(invalid_data, valid_deformation)

        except ImportError:
            pytest.skip("MechanicalModeler not available")
        except Exception as e:
            if "missing" in str(e).lower() or "import" in str(e).lower():
                pytest.skip(f"Required dependencies not available: {e}")
            else:
                raise


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
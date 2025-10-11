"""
Base Component Unit Tests

This module contains unit tests for the BaseComponent class and related
architecture improvements in the Vent4D-Mech framework.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add project to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from vent4d_mech.core.base_component import BaseComponent
    from vent4d_mech.core.exceptions import (
        ConfigurationError, ValidationError, ComputationError
    )
    from vent4d_mech.core.types import is_strain_tensor, is_stress_tensor
except ImportError as e:
    # Skip tests if modules not available
    pytest.skip(f"Modules not available: {e}", allow_module_level=True)


class MockComponent(BaseComponent):
    """Mock component for testing BaseComponent functionality."""

    def _get_default_config(self) -> dict:
        return {
            'test_param': 'default_value',
            'numeric_param': 42,
            'nested': {
                'inner_param': 'inner_value'
            }
        }

    def process(self, *args, **kwargs) -> dict:
        """Mock process method."""
        self._log_processing_start('mock_operation', test_arg='test_value')

        try:
            # Mock some processing
            result = {'processed': True, 'input_count': len(args)}

            self._log_processing_end('mock_operation', success=True)
            return self._package_results(result)

        except Exception as e:
            self._log_processing_end('mock_operation', success=False)
            raise ComputationError(
                f"Mock processing failed: {str(e)}",
                component=self.component_name
            ) from e


class TestBaseComponent:
    """Test cases for BaseComponent class."""

    def test_initialization_with_default_config(self):
        """Test component initialization with default configuration."""
        component = MockComponent()

        assert component.config is not None
        assert component.config['test_param'] == 'default_value'
        assert component.config['numeric_param'] == 42
        assert component.gpu is True  # Default GPU setting
        assert component.component_name == 'MockComponent'
        assert component.logger is not None

    def test_initialization_with_custom_config(self):
        """Test component initialization with custom configuration."""
        custom_config = {
            'test_param': 'custom_value',
            'new_param': 'new_value'
        }
        component = MockComponent(config=custom_config, gpu=False)

        assert component.config['test_param'] == 'custom_value'
        assert component.config['new_param'] == 'new_value'
        assert component.config['numeric_param'] == 42  # Default value preserved
        assert component.gpu is False

    def test_initialization_with_empty_config(self):
        """Test component initialization with empty configuration."""
        component = MockComponent(config={})

        # Should merge with default config
        assert component.config['test_param'] == 'default_value'
        assert component.config['numeric_param'] == 42

    def test_config_validation_basic(self):
        """Test basic configuration validation."""
        component = MockComponent()

        # Valid config should work
        component._validate_config()

        # Invalid config type should raise ValueError
        with pytest.raises(ValueError):
            component.config = "not_a_dict"
            component._validate_config()

    def test_get_component_info(self):
        """Test getting component information."""
        component = MockComponent(config={'custom': 'value'})

        info = component.get_component_info()

        assert isinstance(info, dict)
        assert info['name'] == 'MockComponent'
        assert info['gpu_enabled'] is True
        assert info['config']['custom'] == 'value'
        assert 'logger_name' in info

    def test_update_config_success(self):
        """Test successful configuration update."""
        component = MockComponent()
        original_config = component.config.copy()

        new_config = {'test_param': 'updated_value', 'new_param': 'new_value'}
        component.update_config(new_config)

        assert component.config['test_param'] == 'updated_value'
        assert component.config['new_param'] == 'new_value'
        assert component.config['numeric_param'] == original_config['numeric_param']

    def test_update_config_validation_failure(self):
        """Test configuration update with validation failure."""
        component = MockComponent()
        original_config = component.config.copy()

        # Try to update with invalid config (non-dict)
        with pytest.raises(ValueError):
            component.config = 'invalid'
            component._validate_config()

        # Config should be restored to original
        assert component.config == original_config

    def test_validate_input_array_success(self):
        """Test successful input array validation."""
        component = MockComponent()

        # Mock valid numpy array
        mock_array = Mock()
        mock_array.ndim = 3
        mock_array.shape = (32, 32, 32)
        mock_array.__contains__ = Mock(return_value=False)  # No NaN
        mock_array.any = Mock(return_value=False)  # No infinite values

        # Should not raise exception
        component._validate_input_array(mock_array, 'test_array', expected_dims=3)

    def test_validate_input_array_failures(self):
        """Test input array validation failures."""
        component = MockComponent()

        # Test wrong dimensions
        mock_array = Mock()
        mock_array.ndim = 2
        with pytest.raises(ValueError, match="test_array must have 3 dimensions"):
            component._validate_input_array(mock_array, 'test_array', expected_dims=3)

        # Test NaN values
        mock_array.ndim = 3
        mock_array.__contains__ = Mock(return_value=True)  # Contains NaN
        with pytest.raises(ValueError, match="test_array contains NaN values"):
            component._validate_input_array(mock_array, 'test_array')

    def test_process_method(self):
        """Test the process method with logging and error handling."""
        component = MockComponent()

        result = component.process('arg1', 'arg2', test_kwarg='test_value')

        assert isinstance(result, dict)
        assert result['processed'] is True
        assert result['input_count'] == 2
        assert '_metadata' in result

        metadata = result['_metadata']
        assert metadata['component'] == 'MockComponent'
        assert metadata['gpu_enabled'] is True

    def test_process_method_error_handling(self):
        """Test process method error handling."""
        component = MockComponent()

        # Mock the process to raise an exception
        original_process = component.process
        def failing_process(*args, **kwargs):
            raise ValueError("Test error")

        component.process = failing_process

        with pytest.raises(ComputationError, match="Mock processing failed"):
            original_process()

    def test_logging_methods(self):
        """Test logging methods."""
        component = MockComponent()

        # These should not raise exceptions
        component._log_processing_start('test_operation', param1='value1')
        component._log_processing_end('test_operation', success=True)
        component._log_processing_end('test_operation', success=False)

    def test_package_results(self):
        """Test results packaging with metadata."""
        component = MockComponent()

        raw_results = {'output': 'test_value', 'number': 42}
        packaged = component._package_results(raw_results)

        assert 'output' in packaged
        assert 'number' in packaged
        assert '_metadata' in packaged

        metadata = packaged['_metadata']
        assert metadata['component'] == 'MockComponent'
        assert metadata['gpu_enabled'] is True
        assert 'config' in metadata

    def test_string_representation(self):
        """Test string representation of component."""
        component = MockComponent(config={'param1': 'value1'})

        repr_str = repr(component)
        assert 'MockComponent' in repr_str
        assert 'gpu=True' in repr_str
        assert 'param1' in repr_str


class TestConfigurationError:
    """Test cases for ConfigurationError exception."""

    def test_configuration_error_creation(self):
        """Test ConfigurationError creation and properties."""
        error = ConfigurationError(
            "Invalid configuration",
            component="TestComponent",
            config_key="invalid_key",
            config_value="invalid_value"
        )

        assert "Invalid configuration" in str(error)
        assert error.component == "TestComponent"
        assert error.config_key == "invalid_key"
        assert error.config_value == "invalid_value"
        assert error.error_code == "CONFIG_ERROR"

    def test_configuration_error_to_dict(self):
        """Test ConfigurationError serialization."""
        error = ConfigurationError(
            "Test error",
            component="TestComponent",
            config_key="test_key"
        )

        error_dict = error.to_dict()

        assert error_dict['exception_type'] == 'ConfigurationError'
        assert error_dict['message'] == 'Test error'
        assert error_dict['component'] == 'TestComponent'
        assert error_dict['config_key'] == 'test_key'
        assert error_dict['error_code'] == 'CONFIG_ERROR'


class TestValidationError:
    """Test cases for ValidationError exception."""

    def test_validation_error_creation(self):
        """Test ValidationError creation and properties."""
        error = ValidationError(
            "Invalid data format",
            component="TestComponent",
            data_type="strain_tensor",
            validation_rule="shape_check"
        )

        assert "Invalid data format" in str(error)
        assert error.component == "TestComponent"
        assert error.data_type == "strain_tensor"
        assert error.validation_rule == "shape_check"
        assert error.error_code == "VALIDATION_ERROR"


class TestComputationError:
    """Test cases for ComputationError exception."""

    def test_computation_error_creation(self):
        """Test ComputationError creation and properties."""
        error = ComputationError(
            "Computation failed",
            component="TestComponent",
            operation="stress_computation",
            stage="validation"
        )

        assert "Computation failed" in str(error)
        assert error.component == "TestComponent"
        assert error.operation == "stress_computation"
        assert error.stage == "validation"
        assert error.error_code == "COMPUTATION_ERROR"


class TestTypeSafety:
    """Test cases for type safety utilities."""

    def test_strain_tensor_validation(self):
        """Test strain tensor type validation."""
        if 'is_strain_tensor' not in globals():
            pytest.skip("Type utilities not available")

        # Mock valid strain tensor
        mock_tensor = Mock()
        mock_tensor.ndim = 5
        mock_tensor.shape = (32, 32, 32, 3, 3)
        mock_tensor.__contains__ = Mock(return_value=False)  # No NaN
        mock_tensor.any = Mock(return_value=False)  # No infinite values

        assert is_strain_tensor(mock_tensor) is True

        # Test invalid dimensions
        mock_tensor.ndim = 4
        assert is_strain_tensor(mock_tensor) is False

        # Test invalid shape
        mock_tensor.ndim = 5
        mock_tensor.shape = (32, 32, 32, 3, 4)  # Wrong last dimension
        assert is_strain_tensor(mock_tensor) is False

    def test_stress_tensor_validation(self):
        """Test stress tensor type validation."""
        if 'is_stress_tensor' not in globals():
            pytest.skip("Type utilities not available")

        # Mock valid stress tensor
        mock_tensor = Mock()
        mock_tensor.ndim = 5
        mock_tensor.shape = (32, 32, 32, 3, 3)
        mock_tensor.__contains__ = Mock(return_value=False)
        mock_tensor.any = Mock(return_value=False)

        assert is_stress_tensor(mock_tensor) is True

        # Test invalid dimensions
        mock_tensor.ndim = 4
        assert is_stress_tensor(mock_tensor) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
"""
Configuration Unit Tests

This module contains unit tests for configuration management functionality.
"""

import pytest
import sys
import os

# Add project to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import config modules directly to avoid package import issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vent4d_mech', 'config'))
import default_config
import config_validation


class TestDefaultConfig:
    """Test cases for DefaultConfig class."""

    def test_get_default_config(self):
        """Test that default configuration can be retrieved."""
        config = default_config.DefaultConfig.get_default_config()

        assert isinstance(config, dict)
        assert len(config) > 0

        # Check for required sections
        required_sections = ['registration', 'mechanical', 'performance', 'logging']
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"

    def test_get_registration_config(self):
        """Test registration configuration retrieval."""
        reg_config = default_config.DefaultConfig.get_registration_config()

        assert isinstance(reg_config, dict)
        assert 'method' in reg_config
        assert 'gpu_acceleration' in reg_config
        assert 'parameters' in reg_config

    def test_get_mechanical_config(self):
        """Test mechanical configuration retrieval."""
        mech_config = default_config.DefaultConfig.get_mechanical_config()

        assert isinstance(mech_config, dict)
        assert 'constitutive_model' in mech_config
        assert 'material_parameters' in mech_config

    def test_get_performance_config(self):
        """Test performance configuration retrieval."""
        perf_config = default_config.DefaultConfig.get_performance_config()

        assert isinstance(perf_config, dict)
        assert 'gpu_acceleration' in perf_config
        assert 'parallel_processing' in perf_config

    def test_get_logging_config(self):
        """Test logging configuration retrieval."""
        log_config = default_config.DefaultConfig.get_logging_config()

        assert isinstance(log_config, dict)
        assert 'level' in log_config
        assert 'console_logging' in log_config

    def test_get_material_parameters(self):
        """Test material parameter retrieval for different models."""
        # Test Neo-Hookean
        neo_params = default_config.DefaultConfig.get_material_parameters('neo_hookean')
        assert isinstance(neo_params, dict)
        assert 'C10' in neo_params
        assert neo_params['C10'] > 0

        # Test Mooney-Rivlin
        mr_params = default_config.DefaultConfig.get_material_parameters('mooney_rivlin')
        assert isinstance(mr_params, dict)
        assert 'C10' in mr_params
        assert 'C01' in mr_params

        # Test invalid model
        with pytest.raises(ValueError):
            default_config.DefaultConfig.get_material_parameters('invalid_model')

    def test_get_minimal_config(self):
        """Test minimal configuration retrieval."""
        minimal_config = default_config.DefaultConfig.get_minimal_config()

        assert isinstance(minimal_config, dict)
        assert 'registration' in minimal_config
        assert 'mechanical' in minimal_config
        assert 'performance' in minimal_config
        assert 'logging' in minimal_config

    def test_validate_config_structure(self):
        """Test configuration structure validation."""
        # Valid config
        valid_config = {
            'registration': {},
            'mechanical': {},
            'performance': {},
            'logging': {},
            'validation': {}
        }
        assert default_config.DefaultConfig.validate_config_structure(valid_config) is True

        # Invalid config (missing sections)
        invalid_config = {
            'registration': {},
            'mechanical': {}
        }
        assert default_config.DefaultConfig.validate_config_structure(invalid_config) is False


class TestConfigValidation:
    """Test cases for ConfigValidation class."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = config_validation.ConfigValidation()
        assert validator is not None
        assert hasattr(validator, 'validate_config')
        assert validator.strict_mode is False

        # Test with strict mode
        strict_validator = config_validation.ConfigValidation(strict_mode=True)
        assert strict_validator.strict_mode is True

    def test_validate_config_basic(self):
        """Test basic configuration validation."""
        validator = config_validation.ConfigValidation()

        # Get a valid config
        config = default_config.DefaultConfig.get_minimal_config()

        # Should validate successfully
        is_valid = validator.validate_config(config)
        assert is_valid is True

    def test_validate_registration_config(self):
        """Test registration configuration validation."""
        validator = config_validation.ConfigValidation()

        # Valid registration config
        valid_reg = {
            'method': 'simpleitk',
            'gpu_acceleration': True,
            'parameters': {
                'simpleitk': {
                    'interpolator': 'Linear',
                    'metric': 'MeanSquares'
                }
            }
        }

        # Should not raise exceptions
        try:
            validator._validate_registration_config(valid_reg)
        except Exception as e:
            pytest.fail(f"Valid registration config validation failed: {e}")

        # Invalid registration config
        invalid_reg = {
            'method': 'invalid_method',
            'gpu_acceleration': 'not_boolean'
        }

        # Should have validation errors
        validator._validate_registration_config(invalid_reg)
        assert len(validator.validation_errors) > 0

    def test_validate_mechanical_config(self):
        """Test mechanical configuration validation."""
        validator = config_validation.ConfigValidation()

        # Valid mechanical config
        valid_mech = {
            'constitutive_model': 'neo_hookean',
            'material_parameters': {
                'neo_hookean': {
                    'C10': 0.135,
                    'density': 1.05
                }
            }
        }

        # Should not raise exceptions
        try:
            validator._validate_mechanical_config(valid_mech)
        except Exception as e:
            pytest.fail(f"Valid mechanical config validation failed: {e}")

    def test_validate_logging_config(self):
        """Test logging configuration validation."""
        validator = config_validation.ConfigValidation()

        # Valid logging config
        valid_log = {
            'level': 'INFO',
            'console_logging': {
                'enabled': True,
                'level': 'INFO'
            },
            'file_logging': {
                'enabled': False
            }
        }

        # Should not raise exceptions
        try:
            validator._validate_logging_config(valid_log)
        except Exception as e:
            pytest.fail(f"Valid logging config validation failed: {e}")

        # Invalid logging config
        invalid_log = {
            'level': 'INVALID_LEVEL',
            'console_logging': {
                'enabled': 'not_boolean'
            }
        }

        # Should have validation errors
        validator._validate_logging_config(invalid_log)
        assert len(validator.validation_errors) > 0

    def test_get_validation_report(self):
        """Test validation report generation."""
        validator = config_validation.ConfigValidation()

        # Validate a config with some issues
        config_with_issues = {
            'registration': {
                'method': 'invalid_method',
                'gpu_acceleration': 'not_boolean'
            },
            'mechanical': {
                'constitutive_model': 'invalid_model'
            }
        }

        validator.validate_config(config_with_issues)
        report = validator.get_validation_report()

        assert isinstance(report, dict)
        assert 'is_valid' in report
        assert 'errors' in report
        assert 'warnings' in report
        assert 'error_count' in report
        assert 'warning_count' in report
        assert report['error_count'] > 0

    def test_clear_validation_results(self):
        """Test clearing validation results."""
        validator = config_validation.ConfigValidation()

        # Generate some validation errors
        invalid_config = {'registration': {'method': 'invalid'}}
        validator.validate_config(invalid_config)

        assert len(validator.validation_errors) > 0

        # Clear results
        validator.clear_validation_results()
        assert len(validator.validation_errors) == 0
        assert len(validator.validation_warnings) == 0

    def test_validate_material_parameters(self):
        """Test material parameter validation."""
        validator = config_validation.ConfigValidation()

        # Valid Neo-Hookean parameters
        valid_neo = {
            'neo_hookean': {
                'C10': 0.135,
                'density': 1.05
            }
        }

        validator._validate_material_parameters(valid_neo)
        assert len(validator.validation_errors) == 0

        # Invalid parameters
        invalid_neo = {
            'neo_hookean': {
                'C10': -1.0,  # Negative value
                'density': 0     # Zero density
            }
        }

        validator._validate_material_parameters(invalid_neo)
        assert len(validator.validation_errors) > 0


class TestConfigIntegration:
    """Integration tests for configuration components."""

    def test_config_validation_with_default_config(self):
        """Test validating the default configuration."""
        validator = config_validation.ConfigValidation()
        config = default_config.DefaultConfig.get_default_config()

        is_valid = validator.validate_config(config)
        assert is_valid is True

    def test_config_validation_strict_mode(self):
        """Test strict mode validation."""
        validator = config_validation.ConfigValidation(strict_mode=True)

        # Valid config should pass
        valid_config = default_config.DefaultConfig.get_minimal_config()
        is_valid = validator.validate_config(valid_config)
        assert is_valid is True

        # Invalid config should raise exception in strict mode
        invalid_config = {
            'registration': {
                'method': 'invalid_method'
            }
        }

        with pytest.raises(config_validation.ValidationError):
            validator.validate_config(invalid_config)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
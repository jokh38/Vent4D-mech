"""
Mechanical Modeling Unit Tests

This module contains unit tests for mechanical modeling functionality.
"""

import pytest
import sys
import os
import numpy as np

from src.core.mechanical import constitutive_models


class TestConstitutiveModels:
    """Test cases for constitutive models."""

    def test_create_neo_hookean_model(self):
        """Test Neo-Hookean model creation and basic functionality."""
        model = constitutive_models.NeoHookeanModel(C10=0.135, density=1.05)

        assert model is not None
        assert model.get_parameter('C10') == 0.135
        assert model.get_parameter('density') == 1.05
        assert model.is_hyperelastic is True

    def test_create_mooney_rivlin_model(self):
        """Test Mooney-Rivlin model creation."""
        model = constitutive_models.MooneyRivlinModel(C10=0.135, C01=0.035, density=1.05)

        assert model is not None
        assert model.get_parameter('C10') == 0.135
        assert model.get_parameter('C01') == 0.035
        assert model.get_parameter('density') == 1.05

    def test_create_yeoh_model(self):
        """Test Yeoh model creation."""
        model = constitutive_models.YeohModel(C10=0.135, C20=0.015, C30=0.001, density=1.05)

        assert model is not None
        assert model.get_parameter('C10') == 0.135
        assert model.get_parameter('C20') == 0.015
        assert model.get_parameter('C30') == 0.001

    def test_create_linear_elastic_model(self):
        """Test linear elastic model creation."""
        model = constitutive_models.LinearElasticModel(
            youngs_modulus=5.0, poisson_ratio=0.45, density=1.05
        )

        assert model is not None
        assert model.get_parameter('youngs_modulus') == 5.0
        assert model.get_parameter('poisson_ratio') == 0.45
        assert model.get_parameter('density') == 1.05

    def test_parameter_validation(self):
        """Test parameter validation for models."""
        # Valid parameters should work
        model = constitutive_models.NeoHookeanModel(C10=0.135)
        assert model is not None

        # Invalid parameters should raise ValueError
        with pytest.raises(ValueError):
            constitutive_models.NeoHookeanModel(C10=-1.0)  # Negative C10

        with pytest.raises(ValueError):
            constitutive_models.LinearElasticModel(
                youngs_modulus=5.0, poisson_ratio=0.6  # Invalid poisson_ratio
            )

    def test_parameter_get_set(self):
        """Test parameter getting and setting."""
        model = constitutive_models.NeoHookeanModel(C10=0.1)

        # Test getting parameters
        assert model.get_parameter('C10') == 0.1
        assert model.get_parameter('nonexistent', 'default') == 'default'

        # Test setting parameters
        model.set_parameter('C10', 0.2)
        assert model.get_parameter('C10') == 0.2

    def test_model_repr(self):
        """Test model string representation."""
        model = constitutive_models.NeoHookeanModel(C10=0.135, density=1.05)
        repr_str = repr(model)

        assert 'NeoHookeanModel' in repr_str
        assert 'C10=0.135' in repr_str
        assert 'density=1.05' in repr_str

    def test_create_model_factory_function(self):
        """Test the create_model factory function."""
        # Test valid model creation
        model = constitutive_models.create_model('neo_hookean', C10=0.135)
        assert isinstance(model, constitutive_models.NeoHookeanModel)

        model = constitutive_models.create_model('linear_elastic', youngs_modulus=5.0)
        assert isinstance(model, constitutive_models.LinearElasticModel)

        # Test invalid model type
        with pytest.raises(ValueError):
            constitutive_models.create_model('invalid_model')

    def test_available_models_list(self):
        """Test the available models list."""
        models = constitutive_models.AVAILABLE_MODELS

        assert isinstance(models, list)
        assert len(models) > 0
        assert 'neo_hookean' in models
        assert 'mooney_rivlin' in models
        assert 'linear_elastic' in models

    def test_shear_modulus_calculation(self):
        """Test shear modulus calculation for different models."""
        # Neo-Hookean
        neo_model = constitutive_models.NeoHookeanModel(C10=0.1)
        assert neo_model.get_shear_modulus() == pytest.approx(0.2)  # 2 * C10

        # Mooney-Rivlin
        mr_model = constitutive_models.MooneyRivlinModel(C10=0.1, C01=0.05)
        assert mr_model.get_shear_modulus() == pytest.approx(0.3)  # 2 * (C10 + C01)

        # Linear Elastic
        le_model = constitutive_models.LinearElasticModel(youngs_modulus=6.0, poisson_ratio=0.4)
        expected_shear = 6.0 / (2 * (1 + 0.4))  # E / (2 * (1 + ν))
        assert abs(le_model.get_shear_modulus() - expected_shear) < 1e-10

    def test_bulk_modulus_calculation(self):
        """Test bulk modulus calculation for different models."""
        # Neo-Hookean
        neo_model = constitutive_models.NeoHookeanModel(C10=0.1)
        bulk_modulus = neo_model.get_bulk_modulus(poisson_ratio=0.45)
        assert bulk_modulus > 0

        # Linear Elastic
        le_model = constitutive_models.LinearElasticModel(youngs_modulus=6.0, poisson_ratio=0.45)
        expected_bulk = 6.0 / (3 * (1 - 2 * 0.45))  # E / (3 * (1 - 2ν))
        assert abs(le_model.get_bulk_modulus() - expected_bulk) < 1e-10


class TestStressComputation:
    """Test cases for stress computation (with mocked NumPy)."""

    def test_stress_computation_with_mock_numpy(self):
        """Test stress computation using mocked numpy functionality."""
        # Mock numpy dependency check
        original_available = constitutive_models.NUMPY_AVAILABLE
        constitutive_models.NUMPY_AVAILABLE = True

        try:
            model = constitutive_models.NeoHookeanModel(C10=0.135)

            # Create a mock strain tensor
            mock_strain = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

            # This should not crash (even with mocked numpy)
            try:
                stress = model.compute_stress(mock_strain)
                # The computation should return something (mocked)
                assert stress is not None
            except ImportError:
                # Expected if numpy is truly not available
                pass

        finally:
            constitutive_models.NUMPY_AVAILABLE = original_available

    def test_strain_energy_computation_with_mock_numpy(self):
        """Test strain energy density computation."""
        # Mock numpy dependency check
        original_available = constitutive_models.NUMPY_AVAILABLE
        constitutive_models.NUMPY_AVAILABLE = True

        try:
            model = constitutive_models.NeoHookeanModel(C10=0.135)

            # Create a mock strain tensor
            mock_strain = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

            # This should not crash (even with mocked numpy)
            try:
                energy = model.compute_strain_energy_density(mock_strain)
                # The computation should return something (mocked)
                assert energy is not None
            except ImportError:
                # Expected if numpy is truly not available
                pass

        finally:
            constitutive_models.NUMPY_AVAILABLE = original_available


class TestModelSpecificBehavior:
    """Test cases for model-specific behavior."""

    def test_neo_hookean_specific_behavior(self):
        """Test Neo-Hookean model specific behavior."""
        model = constitutive_models.NeoHookeanModel(C10=0.135)

        # Test that it's marked as hyperelastic
        assert model.is_hyperelastic is True

        # Test parameter defaults
        default_model = constitutive_models.NeoHookeanModel()
        assert default_model.get_parameter('C10') == 0.135  # Default value
        assert default_model.get_parameter('density') == 1.05  # Default value

    def test_mooney_rivlin_specific_behavior(self):
        """Test Mooney-Rivlin model specific behavior."""
        model = constitutive_models.MooneyRivlinModel(C10=0.1, C01=0.05)

        # Test that it's marked as hyperelastic
        assert model.is_hyperelastic is True

        # Test parameter validation
        assert model.get_parameter('C10') > 0
        assert model.get_parameter('C01') >= 0  # Can be zero

    def test_linear_elastic_specific_behavior(self):
        """Test linear elastic model specific behavior."""
        model = constitutive_models.LinearElasticModel(youngs_modulus=5.0, poisson_ratio=0.45)

        # Test that it's marked as hyperelastic (for consistency)
        assert model.is_hyperelastic is True

        # Test parameter bounds
        with pytest.raises(ValueError):
            constitutive_models.LinearElasticModel(youngs_modulus=-1.0)

        with pytest.raises(ValueError):
            constitutive_models.LinearElasticModel(poisson_ratio=0.0)  # Too low

        with pytest.raises(ValueError):
            constitutive_models.LinearElasticModel(poisson_ratio=0.5)  # Too high


class TestModelInheritance:
    """Test cases for model inheritance and base class functionality."""

    def test_consentitive_model_abstract_methods(self):
        """Test that ConstitutiveModel is abstract."""
        # Should not be able to instantiate abstract base class
        with pytest.raises(TypeError):
            constitutive_models.ConstitutiveModel()

    def test_model_inheritance_structure(self):
        """Test that all models inherit from ConstitutiveModel."""
        models_to_test = [
            constitutive_models.NeoHookeanModel,
            constitutive_models.MooneyRivlinModel,
            constitutive_models.YeohModel,
            constitutive_models.OgdenModel,
            constitutive_models.LinearElasticModel
        ]

        for model_class in models_to_test:
            # Create instance
            if model_class == constitutive_models.LinearElasticModel:
                model = model_class(youngs_modulus=5.0)
            elif model_class == constitutive_models.OgdenModel:
                model = model_class(mu1=0.5, alpha1=2.0, mu2=0.1, alpha2=-2.0)
            else:
                model = model_class(C10=0.1)

            # Check inheritance
            assert isinstance(model, constitutive_models.ConstitutiveModel)

            # Check required methods exist
            assert hasattr(model, 'compute_stress')
            assert hasattr(model, 'compute_strain_energy_density')
            assert callable(getattr(model, 'compute_stress'))
            assert callable(getattr(model, 'compute_strain_energy_density'))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
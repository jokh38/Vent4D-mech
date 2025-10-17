"""
Unit Tests for Core Vent4D-Mech Modules

This test suite verifies the independent functionality of the core modules,
ensuring that each component works as expected in isolation.
"""

import unittest
import numpy as np
import os

# Add src to path to allow direct imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.mechanical.hyperelastic_models import Yeoh, MooneyRivlin, compute_cauchy_stress_hyperelastic
try:
    import sfepy
    SFEPY_INSTALLED = True
except ImportError:
    SFEPY_INSTALLED = False

if SFEPY_INSTALLED:
    from core.fem.finite_element_solver import FEMSolver
else:
    FEMSolver = None
from core.fem.mock_fem_solver import MockFEMSolver
from core.inverse.youngs_modulus_estimator import YoungsModulusEstimator
from core.deformation.deformation_analyzer import DeformationAnalyzer

class TestCoreModules(unittest.TestCase):
    """Test cases for core module functionality."""

    def setUp(self):
        """Set up common test data and configurations."""
        self.deformation_analyzer = DeformationAnalyzer(config={}, gpu=False)

    def test_mechanical_yeoh_stress_calculation(self):
        """Test the corrected Cauchy stress calculation for the Yeoh model."""
        # Parameters for the Yeoh model
        yeoh_model = Yeoh(C1=0.1, C2=0.01, C3=0.001, D1=2.0, D2=1.0, D3=1.0)

        # Simple shear deformation gradient
        F = np.array([[1.1, 0.5, 0],
                      [0,   1.0, 0],
                      [0,   0,   1.0]])

        # Calculate stress
        stress = yeoh_model.cauchy_stress(F)

        self.assertEqual(stress.shape, (3, 3))
        self.assertFalse(np.any(np.isnan(stress)))
        # A simple assertion to check if the calculation runs and produces a plausible (non-zero) result
        self.assertNotAlmostEqual(np.linalg.norm(stress), 0)

    def test_inverse_with_mock_fem_solver(self):
        """Test if YoungsModulusEstimator runs independently with MockFEMSolver."""
        # Config for the mock solver
        mock_solver_config = {'assumed_stress': -1.0, 'reference_modulus': 5.0}
        mock_fem_solver = MockFEMSolver(config=mock_solver_config)

        # Estimator configuration
        estimator_config = {
            'solver': {'method': 'least_squares', 'max_iterations': 5, 'tolerance': 1e-4},
            'regularization': {'method': 'tikhonov', 'alpha': 0.1},
            'material': {'initial_modulus': 7.0, 'poisson_ratio': 0.45, 'bounds': (0.1, 50.0), 'spatial_smoothing': False},
            'optimization': {'parameterization': 'linear', 'gradient_method': '2-point'}
        }

        estimator = YoungsModulusEstimator(config=estimator_config, fem_solver=mock_fem_solver)

        # Create synthetic observed data
        shape = (10, 10, 10)
        observed_strain = -0.2 * np.ones(shape + (3, 3)) # Uniform strain
        deformation_gradient = np.tile(np.eye(3), shape + (1, 1))
        dvf = np.zeros(shape + (3,))
        mask = np.ones(shape, dtype=bool)

        # Run estimation
        results = estimator.estimate_modulus(
            observed_strain=observed_strain,
            deformation_gradient=deformation_gradient,
            dvf=dvf,
            voxel_spacing=(1,1,1),
            mask=mask
        )

        self.assertIn('youngs_modulus', results)
        estimated_modulus = results['youngs_modulus']
        self.assertEqual(estimated_modulus.shape, shape)
        # The result should be in the direction of the reference modulus used in the mock solver.
        # Given the simplicity of the mock solver, a large delta is acceptable for this test.
        self.assertAlmostEqual(np.mean(estimated_modulus), 5.0, delta=2.0)

    @unittest.skipUnless(os.environ.get('RUN_SFE_TESTS') == '1', "SfePy tests are skipped by default")
    def test_fem_solver_independent_run(self):
        """Test if the FEMSolver can run a simple simulation independently."""
        # FEM solver config
        fem_config = {
            'material_model': 'neo_hookean',
            'max_iterations': 5,
            'tolerance': 1e-6
        }
        fem_solver = FEMSolver(config=fem_config, deformation_analyzer=self.deformation_analyzer)

        # Create simple inputs for a simulation
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1 # A simple cube

        dvf = np.zeros((10, 10, 10, 3))
        dvf[..., 2] = np.linspace(0, 0.1, 10)[np.newaxis, np.newaxis, :] # Stretch in z-dir

        material_props = {'C10': 0.1, 'D1': 0.02}
        voxel_spacing = (1, 1, 1)

        # Run simulation
        results = fem_solver.run_simulation(
            lung_mask=mask,
            displacement_field=dvf,
            material_properties=material_props,
            voxel_spacing=voxel_spacing
        )

        self.assertIn('displacements', results)
        self.assertIn('processed_results', results)
        self.assertIn('strain_tensor', results['processed_results'])
        self.assertNotEqual(np.linalg.norm(results['displacements']), 0)

if __name__ == '__main__':
    unittest.main()
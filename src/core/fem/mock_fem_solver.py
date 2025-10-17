"""
Mock FEM Solver for Testing

This module provides a mock FEM solver for testing the inverse problem
solver without the need for a full-blown, time-consuming FEM simulation.
"""

from typing import Dict, Any, Tuple
import numpy as np
import logging

class MockFEMSolver:
    """
    A mock FEM solver that simulates the output of a real FEM solver.

    This class provides a simplified, analytical forward model to quickly
    generate a predicted strain field based on a given Young's modulus,
    bypassing the need for a complex SfePy simulation. It is intended for
    use in testing and debugging the `YoungsModulusEstimator`.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized MockFEMSolver.")

    def run_simulation(self, lung_mask: np.ndarray, displacement_field: np.ndarray,
                       material_properties: Dict[str, Any], voxel_spacing: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Run the mock simulation.

        This method mimics the interface of the real FEMSolver but returns
        a result based on a simple analytical model (linear elasticity).

        Args:
            lung_mask: Not used in mock, for interface compatibility.
            displacement_field: The DVF used to derive an approximate strain.
            material_properties: Dictionary containing 'youngs_modulus' and 'poisson_ratio'.
            voxel_spacing: Not used in mock, for interface compatibility.

        Returns:
            A dictionary containing the mock simulation results, including the
            analytically calculated 'strain_tensor'.
        """
        self.logger.info("Running mock FEM simulation...")

        modulus_field = material_properties.get('youngs_modulus')
        poisson_ratio = material_properties.get('poisson_ratio')

        if modulus_field is None or poisson_ratio is None:
            raise ValueError("MockFEMSolver requires 'youngs_modulus' and 'poisson_ratio' in material_properties.")

        # Simple analytical model: Assume observed strain is inversely proportional to stiffness (modulus)
        # This is a gross simplification but serves the purpose of testing the inverse loop.
        # Let's assume a reference modulus and calculate the expected strain.
        reference_modulus = self.config.get('reference_modulus', 5.0) # kPa

        # Use the DVF to calculate an initial strain field (e.g., infinitesimal strain)
        # Approximate diagonal components of the strain tensor du/dx, dv/dy, dw/dz
        _, _, du_dx = np.gradient(displacement_field[..., 0], axis=(0,1,2))
        _, dv_dy, _ = np.gradient(displacement_field[..., 1], axis=(0,1,2))
        dw_dz, _, _ = np.gradient(displacement_field[..., 2], axis=(0,1,2))

        # This is a simplification for mock purposes.
        initial_strain = np.zeros(displacement_field.shape[:-1] + (3,3))
        initial_strain[..., 0, 0] = du_dx
        initial_strain[..., 1, 1] = dv_dy
        initial_strain[..., 2, 2] = dw_dz

        # Predicted strain is scaled by the ratio of reference modulus to local modulus
        # predicted_strain = initial_strain * (reference_modulus / modulus_field[..., np.newaxis, np.newaxis])

        # A simpler model: Stress is proportional to strain via Hooke's Law.
        # Let's assume a uniform stress and calculate strain: strain = stress / E
        # This is not physically accurate but creates a clear dependency for the test.
        assumed_stress = self.config.get('assumed_stress', -1.0) # Uniaxial stress in z-direction

        # Create a placeholder for predicted strain
        predicted_strain = np.zeros_like(initial_strain)

        # Calculate strain based on a simplified Hooke's law (uniaxial)
        # E_zz = sigma_zz / E
        # E_xx = E_yy = -nu * E_zz
        strain_zz = assumed_stress / modulus_field
        strain_xx = -poisson_ratio * strain_zz
        strain_yy = -poisson_ratio * strain_zz

        predicted_strain[..., 0, 0] = strain_xx
        predicted_strain[..., 1, 1] = strain_yy
        predicted_strain[..., 2, 2] = strain_zz

        self.logger.info("Mock simulation completed.")

        return {
            'displacements': np.zeros_like(displacement_field),
            'processed_results': {
                'strain_tensor': predicted_strain,
                'von_mises_stress': np.zeros_like(modulus_field)
            }
        }
"""
Stress Calculator

This module provides stress computation capabilities for biomechanical analysis,
including various stress measures and tensor operations for lung tissue mechanics.
"""

from typing import Dict, Any, Optional, Union
import warnings
import logging

# Try to import numpy, but make it optional
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Some functionality will be limited.")

from .constitutive_models import ConstitutiveModel


class StressCalculator:
    """
    Stress calculator for biomechanical analysis.

    This class provides comprehensive stress computation capabilities including
    different stress measures, principal stresses, and stress invariants for
    biomechanical analysis of lung tissue.
    """

    def __init__(self, constitutive_model: Optional[ConstitutiveModel] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize StressCalculator.

        Args:
            constitutive_model: Constitutive model for stress computation
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.constitutive_model = constitutive_model
        self.stress_history = []

    def set_constitutive_model(self, model: ConstitutiveModel) -> None:
        """
        Set the constitutive model.

        Args:
            model: Constitutive model instance
        """
        self.constitutive_model = model
        self.logger.info(f"Set constitutive model: {model.__class__.__name__}")

    def compute_cauchy_stress(self, deformation_gradient: 'np.ndarray',
                             second_piola_kirchhoff: Optional['np.ndarray'] = None) -> 'np.ndarray':
        """
        Compute Cauchy stress from deformation gradient.

        Args:
            deformation_gradient: Deformation gradient tensor F
            second_piola_kirchhoff: Optional second Piola-Kirchhoff stress

        Returns:
            Cauchy stress tensor σ

        Raises:
            ValueError: If constitutive model is not set
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for stress computation")

        if self.constitutive_model is None:
            raise ValueError("Constitutive model must be set before stress computation")

        if second_piola_kirchhoff is None:
            # Compute Green-Lagrange strain
            strain = self.compute_green_lagrange_strain(deformation_gradient)
            second_piola_kirchhoff = self.constitutive_model.compute_stress(strain)

        # Compute Cauchy stress: σ = (1/J) * F * S * F^T
        J = self.compute_jacobian(deformation_gradient)
        cauchy_stress = np.einsum('...ik,...jk,...jl->...il',
                                  deformation_gradient,
                                  second_piola_kirchhoff,
                                  deformation_gradient) / J

        return cauchy_stress

    def compute_green_lagrange_strain(self, deformation_gradient: 'np.ndarray') -> 'np.ndarray':
        """
        Compute Green-Lagrange strain tensor.

        Args:
            deformation_gradient: Deformation gradient tensor F

        Returns:
            Green-Lagrange strain tensor E = 0.5 * (F^T * F - I)
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for strain computation")

        # Compute right Cauchy-Green tensor: C = F^T * F
        C = np.einsum('...ki,...kj->...ij', deformation_gradient, deformation_gradient)

        # Compute Green-Lagrange strain: E = 0.5 * (C - I)
        I = np.eye(3)
        if deformation_gradient.ndim > 2:
            # Volume of deformation gradients
            I = np.broadcast_to(I, C.shape)

        E = 0.5 * (C - I)

        return E

    def compute_jacobian(self, deformation_gradient: 'np.ndarray') -> 'np.ndarray':
        """
        Compute Jacobian determinant.

        Args:
            deformation_gradient: Deformation gradient tensor F

        Returns:
            Jacobian determinant J = det(F)
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for Jacobian computation")

        if deformation_gradient.ndim == 2:
            # Single tensor
            return np.linalg.det(deformation_gradient)
        elif deformation_gradient.ndim == 3:
            # Vector of 2D tensors
            return np.linalg.det(deformation_gradient)
        elif deformation_gradient.ndim == 5:
            # Volume of 3D tensors (D, H, W, 3, 3)
            D, H, W = deformation_gradient.shape[:3]
            jacobians = np.zeros((D, H, W))

            for i in range(D):
                for j in range(H):
                    for k in range(W):
                        jacobians[i, j, k] = np.linalg.det(deformation_gradient[i, j, k])

            return jacobians
        else:
            raise ValueError(f"Unsupported tensor dimensions: {deformation_gradient.ndim}")

    def compute_principal_stresses(self, stress_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute principal stresses.

        Args:
            stress_tensor: Stress tensor

        Returns:
            Principal stresses (sorted)
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for principal stress computation")

        if stress_tensor.ndim == 2:
            # Single stress tensor
            eigenvalues = np.linalg.eigvalsh(stress_tensor)
            return np.sort(eigenvalues)
        elif stress_tensor.ndim == 3:
            # Vector of 2D stress tensors
            eigenvalues = np.linalg.eigvalsh(stress_tensor)
            return np.sort(eigenvalues, axis=-1)
        elif stress_tensor.ndim == 5:
            # Volume of 3D stress tensors (D, H, W, 3, 3)
            D, H, W = stress_tensor.shape[:3]
            principal_stresses = np.zeros((D, H, W, 3))

            for i in range(D):
                for j in range(H):
                    for k in range(W):
                        eigenvalues = np.linalg.eigvalsh(stress_tensor[i, j, k])
                        principal_stresses[i, j, k] = np.sort(eigenvalues)

            return principal_stresses
        else:
            raise ValueError(f"Unsupported tensor dimensions: {stress_tensor.ndim}")

    def compute_von_mises_stress(self, stress_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute von Mises equivalent stress.

        Args:
            stress_tensor: Stress tensor

        Returns:
            von Mises stress
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for von Mises stress computation")

        if stress_tensor.ndim == 2:
            # Single stress tensor
            s11, s22, s33 = stress_tensor[0, 0], stress_tensor[1, 1], stress_tensor[2, 2]
            s12, s13, s23 = stress_tensor[0, 1], stress_tensor[0, 2], stress_tensor[1, 1]
            von_mises = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 +
                                      6 * (s12**2 + s13**2 + s23**2)))
            return von_mises
        elif stress_tensor.ndim == 5:
            # Volume of 3D stress tensors
            s11 = stress_tensor[..., 0, 0]
            s22 = stress_tensor[..., 1, 1]
            s33 = stress_tensor[..., 2, 2]
            s12 = stress_tensor[..., 0, 1]
            s13 = stress_tensor[..., 0, 2]
            s23 = stress_tensor[..., 1, 2]

            von_mises = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 +
                                      6 * (s12**2 + s13**2 + s23**2)))
            return von_mises
        else:
            raise ValueError(f"Unsupported tensor dimensions: {stress_tensor.ndim}")

    def compute_stress_invariants(self, stress_tensor: 'np.ndarray') -> Dict[str, 'np.ndarray']:
        """
        Compute stress invariants.

        Args:
            stress_tensor: Stress tensor

        Returns:
            Dictionary containing stress invariants
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for stress invariant computation")

        principal_stresses = self.compute_principal_stresses(stress_tensor)

        if stress_tensor.ndim == 2:
            # Single stress tensor
            I1 = np.sum(principal_stresses)
            I2 = principal_stresses[0] * principal_stresses[1] + \
                 principal_stresses[1] * principal_stresses[2] + \
                 principal_stresses[0] * principal_stresses[2]
            I3 = np.prod(principal_stresses)
        else:
            # Volume of stress tensors
            I1 = np.sum(principal_stresses, axis=-1)
            I2 = (principal_stresses[..., 0] * principal_stresses[..., 1] +
                  principal_stresses[..., 1] * principal_stresses[..., 2] +
                  principal_stresses[..., 0] * principal_stresses[..., 2])
            I3 = np.prod(principal_stresses, axis=-1)

        return {
            'I1': I1,  # First invariant (trace)
            'I2': I2,  # Second invariant
            'I3': I3,  # Third invariant (determinant)
            'principal_stresses': principal_stresses
        }

    def compute_deviatoric_stress(self, stress_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute deviatoric stress tensor.

        Args:
            stress_tensor: Stress tensor

        Returns:
            Deviatoric stress tensor s = σ - (1/3) * tr(σ) * I
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for deviatoric stress computation")

        # Compute mean stress
        if stress_tensor.ndim == 2:
            mean_stress = np.trace(stress_tensor) / 3.0
            I = np.eye(3)
        elif stress_tensor.ndim == 5:
            mean_stress = np.trace(stress_tensor, axis1=-2, axis2=-1) / 3.0
            I = np.eye(3)
            # Broadcast identity tensor
            I = np.broadcast_to(I, stress_tensor.shape).copy()
        else:
            raise ValueError(f"Unsupported tensor dimensions: {stress_tensor.ndim}")

        # Compute deviatoric stress
        if stress_tensor.ndim == 5:
            mean_stress = mean_stress[..., np.newaxis, np.newaxis]

        deviatoric_stress = stress_tensor - mean_stress * I

        return deviatoric_stress

    def compute_hydrostatic_stress(self, stress_tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Compute hydrostatic stress (pressure).

        Args:
            stress_tensor: Stress tensor

        Returns:
            Hydrostatic stress p = (1/3) * tr(σ)
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for hydrostatic stress computation")

        return np.trace(stress_tensor, axis1=-2, axis2=-1) / 3.0

    def analyze_stress_state(self, stress_tensor: 'np.ndarray') -> Dict[str, Any]:
        """
        Perform comprehensive stress analysis.

        Args:
            stress_tensor: Stress tensor

        Returns:
            Dictionary containing stress analysis results
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for stress analysis")

        self.logger.info("Performing comprehensive stress analysis")

        analysis = {}

        # Basic stress measures
        analysis['principal_stresses'] = self.compute_principal_stresses(stress_tensor)
        analysis['von_mises'] = self.compute_von_mises_stress(stress_tensor)
        analysis['invariants'] = self.compute_stress_invariants(stress_tensor)
        analysis['deviatoric'] = self.compute_deviatoric_stress(stress_tensor)
        analysis['hydrostatic'] = self.compute_hydrostatic_stress(stress_tensor)

        # Stress state classification
        if stress_tensor.ndim == 5:
            # For volume data, compute statistics
            analysis['statistics'] = {
                'max_principal': {
                    'max': np.max(analysis['principal_stresses']),
                    'min': np.min(analysis['principal_stresses']),
                    'mean': np.mean(analysis['principal_stresses'])
                },
                'von_mises': {
                    'max': np.max(analysis['von_mises']),
                    'min': np.min(analysis['von_mises']),
                    'mean': np.mean(analysis['von_mises'])
                },
                'hydrostatic': {
                    'max': np.max(analysis['hydrostatic']),
                    'min': np.min(analysis['hydrostatic']),
                    'mean': np.mean(analysis['hydrostatic'])
                }
            }

        # Store in history
        self.stress_history.append(analysis)
        self.logger.debug("Stress analysis completed")

        return analysis

    def validate_stress_tensor(self, stress_tensor: 'np.ndarray') -> bool:
        """
        Validate stress tensor format and values.

        Args:
            stress_tensor: Stress tensor to validate

        Returns:
            True if stress tensor is valid
        """
        if not NUMPY_AVAILABLE:
            return False

        # Check shape
        if stress_tensor.ndim not in [2, 3, 5]:
            self.logger.error(f"Invalid stress tensor dimensions: {stress_tensor.ndim}")
            return False

        if stress_tensor.shape[-2:] != (3, 3):
            self.logger.error(f"Invalid stress tensor shape: {stress_tensor.shape}")
            return False

        # Check for invalid values
        if np.any(np.isnan(stress_tensor)):
            self.logger.error("Stress tensor contains NaN values")
            return False

        if np.any(np.isinf(stress_tensor)):
            self.logger.error("Stress tensor contains infinite values")
            return False

        return True

    def get_stress_summary(self) -> Dict[str, Any]:
        """
        Get summary of stress computations.

        Returns:
            Stress computation summary
        """
        if not self.stress_history:
            return {'message': 'No stress computations performed'}

        return {
            'total_computations': len(self.stress_history),
            'constitutive_model': self.constitutive_model.__class__.__name__ if self.constitutive_model else None,
            'latest_computation': self.stress_history[-1] if self.stress_history else None
        }

    def clear_history(self) -> None:
        """Clear stress computation history."""
        self.stress_history.clear()
        self.logger.debug("Stress computation history cleared")

    def __repr__(self) -> str:
        """String representation of the StressCalculator instance."""
        model_name = self.constitutive_model.__class__.__name__ if self.constitutive_model else None
        return f"StressCalculator(model={model_name}, computations={len(self.stress_history)})"
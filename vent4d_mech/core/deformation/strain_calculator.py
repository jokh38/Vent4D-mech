"""
Strain Tensor Calculator

This module implements strain tensor calculation for both infinitesimal and
finite deformation theories, with emphasis on Green-Lagrange strain tensors
for large deformation analysis in lung tissue.
"""

from typing import Tuple, Union
import numpy as np
import logging

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class StrainCalculator:
    """
    Calculator for strain tensors in continuum mechanics.

    This class implements various strain tensor calculation methods, supporting
    both infinitesimal strain theory (small deformations) and Green-Lagrange
    strain theory (large deformations) for comprehensive lung tissue analysis.

    Attributes:
        config (dict): Configuration parameters
        gpu (bool): Whether to use GPU acceleration
        xp (module): NumPy or CuPy module based on GPU availability
        logger (logging.Logger): Logger instance
    """

    def __init__(self, config: dict, gpu: bool = True):
        """
        Initialize StrainCalculator instance.

        Args:
            config: Configuration parameters for strain calculation
            gpu: Whether to use GPU acceleration
        """
        self.config = config
        self.gpu = gpu and CUPY_AVAILABLE

        # Set computation backend
        self.xp = cp if self.gpu else np

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initialized StrainCalculator (GPU: {self.gpu}, backend: {self.xp.__name__})")

    def compute_strain_tensor(self, F: np.ndarray) -> np.ndarray:
        """
        Compute strain tensor from deformation gradient.

        Args:
            F: Deformation gradient tensor (D, H, W, 3, 3)

        Returns:
            Strain tensor (D, H, W, 3, 3)

        Raises:
            ValueError: If strain theory is not supported
        """
        theory = self.config.get('theory', 'green_lagrange').lower()

        if theory == 'infinitesimal':
            return self.compute_infinitesimal_strain(F)
        elif theory == 'green_lagrange':
            return self.compute_green_lagrange_strain(F)
        elif theory == 'almansi':
            return self.compute_almansi_strain(F)
        elif theory == 'logarithmic':
            return self.compute_logarithmic_strain(F)
        else:
            raise ValueError(f"Unsupported strain theory: {theory}")

    def compute_infinitesimal_strain(self, F: np.ndarray) -> np.ndarray:
        """
        Compute infinitesimal strain tensor ε = 1/2(∇u + ∇u^T).

        Args:
            F: Deformation gradient tensor (D, H, W, 3, 3)

        Returns:
            Infinitesimal strain tensor ε (D, H, W, 3, 3)
        """
        # For infinitesimal strain: ε = 1/2(F + F^T) - I
        # where F = I + ∇u for small deformations

        # Convert to backend array
        if self.gpu and not isinstance(F, cp.ndarray):
            F = cp.asarray(F)

        # Compute strain: ε = 1/2(F + F^T) - I
        F_T = self.xp.transpose(F, (0, 1, 2, 4, 3))
        strain = 0.5 * (F + F_T)

        # Subtract identity
        identity = self.xp.eye(3)
        strain = strain - identity

        # Convert back to numpy if needed
        if self.gpu:
            return cp.asnumpy(strain)
        return strain

    def compute_green_lagrange_strain(self, F: np.ndarray) -> np.ndarray:
        """
        Compute Green-Lagrange strain tensor E = 1/2(F^T F - I).

        This is the preferred strain measure for large deformations in lung tissue.

        Args:
            F: Deformation gradient tensor (D, H, W, 3, 3)

        Returns:
            Green-Lagrange strain tensor E (D, H, W, 3, 3)
        """
        # Convert to backend array
        if self.gpu and not isinstance(F, cp.ndarray):
            F = cp.asarray(F)

        # Compute C = F^T * F (Right Cauchy-Green deformation tensor)
        F_T = self.xp.transpose(F, (0, 1, 2, 4, 3))
        C = self.xp.matmul(F_T, F)

        # Compute Green-Lagrange strain: E = 1/2(C - I)
        identity = self.xp.eye(3)
        strain = 0.5 * (C - identity)

        # Convert back to numpy if needed
        if self.gpu:
            return cp.asnumpy(strain)
        return strain

    def compute_almansi_strain(self, F: np.ndarray) -> np.ndarray:
        """
        Compute Almansi strain tensor e = 1/2(I - (F F^T)^(-1)).

        Args:
            F: Deformation gradient tensor (D, H, W, 3, 3)

        Returns:
            Almansi strain tensor e (D, H, W, 3, 3)
        """
        # Convert to backend array
        if self.gpu and not isinstance(F, cp.ndarray):
            F = cp.asarray(F)

        # Compute B = F * F^T (Left Cauchy-Green deformation tensor)
        F_T = self.xp.transpose(F, (0, 1, 2, 4, 3))
        B = self.xp.matmul(F, F_T)

        # Compute B^(-1)
        B_inv = self.xp.zeros_like(B)
        for i in range(F.shape[0]):
            for j in range(F.shape[1]):
                for k in range(F.shape[2]):
                    B_inv[i, j, k] = self.xp.linalg.inv(B[i, j, k])

        # Compute Almansi strain: e = 1/2(I - B^(-1))
        identity = self.xp.eye(3)
        strain = 0.5 * (identity - B_inv)

        # Convert back to numpy if needed
        if self.gpu:
            return cp.asnumpy(strain)
        return strain

    def compute_logarithmic_strain(self, F: np.ndarray) -> np.ndarray:
        """
        Compute logarithmic (Hencky) strain tensor.

        Args:
            F: Deformation gradient tensor (D, H, W, 3, 3)

        Returns:
            Logarithmic strain tensor (D, H, W, 3, 3)
        """
        # Convert to backend array
        if self.gpu and not isinstance(F, cp.ndarray):
            F = cp.asarray(F)

        # Compute right Cauchy-Green tensor C = F^T * F
        F_T = self.xp.transpose(F, (0, 1, 2, 4, 3))
        C = self.xp.matmul(F_T, F)

        # Compute eigenvalues and eigenvectors
        strain = self.xp.zeros_like(F)
        for i in range(F.shape[0]):
            for j in range(F.shape[1]):
                for k in range(F.shape[2]):
                    # Eigen decomposition
                    eigenvalues, eigenvectors = self.xp.linalg.eigh(C[i, j, k])

                    # Logarithmic strain in principal directions
                    log_eigenvalues = 0.5 * self.xp.log(eigenvalues)

                    # Reconstruct strain tensor
                    log_strain = eigenvectors @ self.xp.diag(log_eigenvalues) @ eigenvectors.T
                    strain[i, j, k] = log_strain

        # Convert back to numpy if needed
        if self.gpu:
            return cp.asnumpy(strain)
        return strain

    def compute_deformation_rate(self, F_dot: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Compute deformation rate tensor D = 1/2(L + L^T), where L = F_dot * F^(-1).

        Args:
            F_dot: Time derivative of deformation gradient
            F: Deformation gradient tensor

        Returns:
            Deformation rate tensor D (D, H, W, 3, 3)
        """
        # Convert to backend array
        if self.gpu:
            if not isinstance(F_dot, cp.ndarray):
                F_dot = cp.asarray(F_dot)
            if not isinstance(F, cp.ndarray):
                F = cp.asarray(F)

        # Compute velocity gradient L = F_dot * F^(-1)
        F_inv = self.xp.zeros_like(F)
        for i in range(F.shape[0]):
            for j in range(F.shape[1]):
                for k in range(F.shape[2]):
                    F_inv[i, j, k] = self.xp.linalg.inv(F[i, j, k])

        L = self.xp.matmul(F_dot, F_inv)

        # Compute deformation rate D = 1/2(L + L^T)
        L_T = self.xp.transpose(L, (0, 1, 2, 4, 3))
        D = 0.5 * (L + L_T)

        # Convert back to numpy if needed
        if self.gpu:
            return cp.asnumpy(D)
        return D

    def compute_stretch_ratios(self, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute principal stretch ratios λ_i from deformation gradient.

        Args:
            F: Deformation gradient tensor (D, H, W, 3, 3)

        Returns:
            Tuple of principal stretch ratios (λ1, λ2, λ3)
        """
        # Convert to backend array
        if self.gpu and not isinstance(F, cp.ndarray):
            F = cp.asarray(F)

        # Compute right Cauchy-Green tensor C = F^T * F
        F_T = self.xp.transpose(F, (0, 1, 2, 4, 3))
        C = self.xp.matmul(F_T, F)

        # Compute eigenvalues (λ_i^2)
        eigenvalues = self.xp.zeros(F.shape[:3] + (3,))
        for i in range(F.shape[0]):
            for j in range(F.shape[1]):
                for k in range(F.shape[2]):
                    eigenvalues[i, j, k] = self.xp.linalg.eigvalsh(C[i, j, k])

        # Stretch ratios λ_i = sqrt(eigenvalues)
        stretch_ratios = self.xp.sqrt(eigenvalues)

        # Convert back to numpy if needed
        if self.gpu:
            stretch_ratios = cp.asnumpy(stretch_ratios)

        return (stretch_ratios[..., 0], stretch_ratios[..., 1], stretch_ratios[..., 2])

    def apply_boundary_conditions(self, strain: np.ndarray,
                                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply boundary conditions to strain tensor.

        Args:
            strain: Strain tensor (D, H, W, 3, 3)
            mask: Optional mask for boundary regions

        Returns:
            Strain tensor with boundary conditions applied
        """
        if mask is None:
            return strain

        # Apply zero boundary condition
        strain_masked = strain.copy()
        strain_masked[~mask] = 0.0

        return strain_masked

    def smooth_strain_field(self, strain: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian smoothing to strain tensor components.

        Args:
            strain: Strain tensor (D, H, W, 3, 3)
            sigma: Gaussian smoothing sigma

        Returns:
            Smoothed strain tensor
        """
        from scipy.ndimage import gaussian_filter

        smoothed_strain = np.zeros_like(strain)

        # Apply smoothing to each component
        for i in range(3):
            for j in range(3):
                smoothed_strain[..., i, j] = gaussian_filter(strain[..., i, j], sigma=sigma)

        return smoothed_strain

    def __repr__(self) -> str:
        """String representation of the StrainCalculator instance."""
        return (f"StrainCalculator(gpu={self.gpu}, "
                f"theory='{self.config.get('theory', 'green_lagrange')}', "
                f"backend='{self.xp.__name__}')")
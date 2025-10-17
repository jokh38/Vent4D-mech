"""
Deformation Analysis Utilities

This module provides utility functions for deformation analysis, including
tensor operations, coordinate transformations, and biomechanical calculations.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import logging
from pathlib import Path

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

import nibabel as nib


class DeformationUtils:
    """
    Utility functions for deformation analysis.

    This class provides various utility functions for deformation analysis,
    including tensor operations, coordinate transformations, and biomechanical
    calculations.
    """

    def __init__(self):
        """Initialize DeformationUtils."""
        self.logger = logging.getLogger(__name__)

    def apply_mask_to_dvf(self, dvf: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to displacement vector field.

        Args:
            dvf: Displacement vector field (D, H, W, 3)
            mask: Boolean mask (D, H, W)

        Returns:
            Masked displacement vector field
        """
        dvf_masked = dvf.copy()
        dvf_masked[~mask] = 0.0
        return dvf_masked

    def save_nifti(self, data: np.ndarray, file_path: str,
                   reference_image: Optional[nib.Nifti1Image] = None) -> None:
        """
        Save numpy array as NIfTI file.

        Args:
            data: Data to save
            file_path: Output file path
            reference_image: Reference image for header information
        """
        if reference_image is None:
            # Create simple affine matrix
            affine = np.eye(4)
            img = nib.Nifti1Image(data, affine)
        else:
            img = nib.Nifti1Image(data, reference_image.affine, reference_image.header)

        nib.save(img, file_path)
        self.logger.info(f"Saved NIfTI file to {file_path}")

    def transform_coordinates(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Transform coordinates using transformation matrix.

        Args:
            points: Points to transform (N, 3)
            transform: Transformation matrix (4, 4)

        Returns:
            Transformed points (N, 3)
        """
        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])

        # Apply transformation
        transformed = (transform @ points_homogeneous.T).T

        # Convert back to 3D coordinates
        return transformed[:, :3]

    def compute_principal_stretches(self, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute principal stretches and directions from deformation gradient.

        Args:
            F: Deformation gradient tensor (D, H, W, 3, 3)

        Returns:
            Tuple of (stretches, directions)
        """
        # Compute right Cauchy-Green tensor C = F^T * F
        F_T = np.transpose(F, (0, 1, 2, 4, 3))
        C = np.matmul(F_T, F)

        # Compute eigenvalues and eigenvectors
        stretches = np.zeros(F.shape[:3] + (3,))
        directions = np.zeros(F.shape[:3] + (3, 3))

        for i in range(F.shape[0]):
            for j in range(F.shape[1]):
                for k in range(F.shape[2]):
                    eigenvalues, eigenvectors = np.linalg.eigh(C[i, j, k])
                    # Principal stretches λ_i = sqrt(eigenvalues)
                    stretches[i, j, k] = np.sqrt(eigenvalues)
                    directions[i, j, k] = eigenvectors

        return stretches, directions


class StrainInvariants:
    """
    Calculator for strain tensor invariants and derived measures.

    This class provides methods for computing various strain invariants,
    principal strains, and derived biomechanical measures.
    """

    def __init__(self):
        """Initialize StrainInvariants."""
        self.logger = logging.getLogger(__name__)

    def green_lagrange_invariants(self, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute invariants of Green-Lagrange strain tensor.

        Args:
            E: Green-Lagrange strain tensor (D, H, W, 3, 3)

        Returns:
            Tuple of (I1, I2, I3) invariants
        """
        # First invariant: I1 = tr(E)
        I1 = np.trace(E, axis1=3, axis2=4)

        # Second invariant: I2 = 1/2[(tr(E))^2 - tr(E^2)]
        E_squared = np.matmul(E, E)
        tr_E_squared = np.trace(E_squared, axis1=3, axis2=4)
        I2 = 0.5 * (I1**2 - tr_E_squared)

        # Third invariant: I3 = det(E)
        I3 = np.linalg.det(E)

        return I1, I2, I3

    def infinitesimal_invariants(self, epsilon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute invariants of infinitesimal strain tensor.

        Args:
            epsilon: Infinitesimal strain tensor (D, H, W, 3, 3)

        Returns:
            Tuple of (I1, I2, I3) invariants
        """
        # For infinitesimal strain, invariants are similar to Green-Lagrange
        return self.green_lagrange_invariants(epsilon)

    def principal_strains(self, strain_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute principal strains and directions.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)

        Returns:
            Dictionary with eigenvalues and eigenvectors
        """
        eigenvalues = np.zeros(strain_tensor.shape[:3] + (3,))
        eigenvectors = np.zeros(strain_tensor.shape[:3] + (3, 3))

        for i in range(strain_tensor.shape[0]):
            for j in range(strain_tensor.shape[1]):
                for k in range(strain_tensor.shape[2]):
                    # Compute eigenvalues and eigenvectors
                    vals, vecs = np.linalg.eigh(strain_tensor[i, j, k])
                    eigenvalues[i, j, k] = vals
                    eigenvectors[i, j, k] = vecs

        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors
        }

    def von_mises_strain(self, strain_tensor: np.ndarray) -> np.ndarray:
        """
        Compute von Mises equivalent strain.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)

        Returns:
            Von Mises strain (D, H, W)
        """
        # Von Mises strain: ε_vm = sqrt(2/3 * ε_ij * ε_ij)
        strain_deviator = strain_tensor - (1/3) * np.trace(strain_tensor, axis1=3, axis2=4)[..., np.newaxis, np.newaxis] * np.eye(3)
        von_mises = np.sqrt((2/3) * np.sum(strain_deviator**2, axis=(3, 4)))

        return von_mises

    def max_shear_strain(self, strain_tensor: np.ndarray) -> np.ndarray:
        """
        Compute maximum shear strain.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)

        Returns:
            Maximum shear strain (D, H, W)
        """
        # Maximum shear strain: γ_max = ε_max - ε_min
        principal_info = self.principal_strains(strain_tensor)
        eigenvalues = principal_info['eigenvalues']

        max_shear = eigenvalues[..., 0] - eigenvalues[..., 2]
        return max_shear

    def effective_strain(self, strain_tensor: np.ndarray) -> np.ndarray:
        """
        Compute effective strain measure.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)

        Returns:
            Effective strain (D, H, W)
        """
        # Effective strain: ε_eff = sqrt(2/3 * ε_ij * ε_ij_deviator)
        trace_strain = np.trace(strain_tensor, axis1=3, axis2=4)
        strain_deviator = strain_tensor - (1/3) * trace_strain[..., np.newaxis, np.newaxis] * np.eye(3)
        effective = np.sqrt((2/3) * np.sum(strain_deviator**2, axis=(3, 4)))

        return effective

    def volumetric_strain_from_tensor(self, strain_tensor: np.ndarray) -> np.ndarray:
        """
        Compute volumetric strain from strain tensor.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)

        Returns:
            Volumetric strain (D, H, W)
        """
        # Volumetric strain: ε_vol = tr(ε)
        return np.trace(strain_tensor, axis1=3, axis2=4)

    def dilatational_strain(self, strain_tensor: np.ndarray) -> np.ndarray:
        """
        Compute dilatational (hydrostatic) strain component.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)

        Returns:
            Dilatational strain (D, H, W, 3, 3)
        """
        # Dilatational strain: ε_dil = 1/3 * tr(ε) * I
        trace_strain = np.trace(strain_tensor, axis1=3, axis2=4)
        dilatational = (1/3) * trace_strain[..., np.newaxis, np.newaxis] * np.eye(3)

        return dilatational

    def deviatoric_strain(self, strain_tensor: np.ndarray) -> np.ndarray:
        """
        Compute deviatoric strain component.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)

        Returns:
            Deviatoric strain (D, H, W, 3, 3)
        """
        # Deviatoric strain: ε_dev = ε - 1/3 * tr(ε) * I
        trace_strain = np.trace(strain_tensor, axis1=3, axis2=4)
        dilatational = (1/3) * trace_strain[..., np.newaxis, np.newaxis] * np.eye(3)
        deviatoric = strain_tensor - dilatational

        return deviatoric

    def strain_energy_density(self, strain_tensor: np.ndarray,
                            young_modulus: np.ndarray,
                            poisson_ratio: float = 0.45) -> np.ndarray:
        """
        Compute strain energy density.

        Args:
            strain_tensor: Strain tensor (D, H, W, 3, 3)
            young_modulus: Young's modulus field (D, H, W)
            poisson_ratio: Poisson's ratio

        Returns:
            Strain energy density (D, H, W)
        """
        # Lamé parameters
        lambda_lame = (young_modulus * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        mu_lame = young_modulus / (2 * (1 + poisson_ratio))

        # Strain energy density: W = 1/2 * λ * (tr(ε))^2 + μ * tr(ε^2)
        trace_strain = np.trace(strain_tensor, axis1=3, axis2=4)
        strain_squared = np.matmul(strain_tensor, strain_tensor)
        tr_strain_squared = np.trace(strain_squared, axis1=3, axis2=4)

        energy_density = 0.5 * (lambda_lame * trace_strain**2 + 2 * mu_lame * tr_strain_squared)

        return energy_density
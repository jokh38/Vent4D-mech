"""
Registration Utilities

This module provides utility functions for image registration preprocessing,
postprocessing, quality metrics, and data handling.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import logging

try:
    import SimpleITK as sitk
    SIMPLEITK_AVAILABLE = True
except ImportError:
    SIMPLEITK_AVAILABLE = False

try:
    import torch
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import nibabel as nib


class RegistrationUtils:
    """
    Utility functions for image registration.

    This class provides various utility functions for image preprocessing,
    postprocessing, format conversion, and data handling in the context
    of medical image registration.
    """

    def __init__(self):
        """Initialize RegistrationUtils."""
        self.logger = logging.getLogger(__name__)

    def load_nifti(self, file_path: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load NIfTI file and return as numpy array.

        Args:
            file_path: Path to NIfTI file or numpy array

        Returns:
            Image data as numpy array

        Raises:
            FileNotFoundError: If file does not exist
        """
        if isinstance(file_path, np.ndarray):
            return file_path

        try:
            img = nib.load(str(file_path))
            return img.get_fdata().astype(np.float32)
        except Exception as e:
            self.logger.error(f"Failed to load NIfTI file {file_path}: {str(e)}")
            raise

    def save_nifti(self, data: np.ndarray, file_path: str,
                   reference_image: Optional[Union[str, nib.Nifti1Image]] = None) -> None:
        """
        Save numpy array as NIfTI file.

        Args:
            data: Image data to save
            file_path: Output file path
            reference_image: Reference image for header information
        """
        if reference_image is None:
            # Create simple affine matrix
            affine = np.eye(4)
            img = nib.Nifti1Image(data, affine)
        elif isinstance(reference_image, str):
            ref_img = nib.load(reference_image)
            img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
        else:
            img = nib.Nifti1Image(data, reference_image.affine, reference_image.header)

        nib.save(img, file_path)
        self.logger.info(f"Saved NIfTI file to {file_path}")

    def normalize_image(self, image: np.ndarray, method: str = 'z_score') -> np.ndarray:
        """
        Normalize image using specified method.

        Args:
            image: Input image
            method: Normalization method ('z_score', 'min_max', 'histogram_equalization')

        Returns:
            Normalized image
        """
        if method == 'z_score':
            mean = np.mean(image)
            std = np.std(image)
            return (image - mean) / (std + 1e-8)

        elif method == 'min_max':
            min_val = np.min(image)
            max_val = np.max(image)
            return (image - min_val) / (max_val - min_val + 1e-8)

        elif method == 'histogram_equalization':
            # Simple histogram equalization
            hist, bins = np.histogram(image.flatten(), bins=256, density=True)
            cdf = hist.cumsum()
            cdf = 255 * cdf / cdf[-1]
            image_eq = np.interp(image.flatten(), bins[:-1], cdf)
            return image_eq.reshape(image.shape)

        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def resize_image(self, image: np.ndarray, target_shape: Tuple[int, int, int],
                     order: int = 1) -> np.ndarray:
        """
        Resize image to target shape.

        Args:
            image: Input image
            target_shape: Target shape (D, H, W)
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)

        Returns:
            Resized image
        """
        if image.shape == target_shape:
            return image

        # Calculate zoom factors
        zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]

        # Resize using scipy.ndimage.zoom
        resized = ndimage.zoom(image, zoom_factors, order=order)

        return resized.astype(image.dtype)

    def resize_vector_field(self, dvf: np.ndarray, target_shape: Tuple[int, int, int],
                           order: int = 1) -> np.ndarray:
        """
        Resize displacement vector field.

        Args:
            dvf: Displacement vector field (D, H, W, 3)
            target_shape: Target shape (D, H, W)
            order: Interpolation order

        Returns:
            Resized vector field
        """
        if dvf.shape[:3] == target_shape:
            return dvf

        # Resize each component separately
        zoom_factors = [t / s for t, s in zip(target_shape, dvf.shape[:3])]
        resized_dvf = np.zeros(target_shape + (3,))

        for i in range(3):
            resized_dvf[..., i] = ndimage.zoom(dvf[..., i], zoom_factors, order=order)

        return resized_dvf

    def interpolate_vector_field(self, dvf: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Interpolate vector field to target coordinates.

        Args:
            dvf: Displacement vector field
            target_shape: Target shape

        Returns:
            Interpolated vector field
        """
        # Create coordinate grids for target shape
        z, y, x = np.mgrid[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]]

        # Normalize coordinates to [0, 1]
        z_norm = z / (target_shape[0] - 1) * (dvf.shape[0] - 1)
        y_norm = y / (target_shape[1] - 1) * (dvf.shape[1] - 1)
        x_norm = x / (target_shape[2] - 1) * (dvf.shape[2] - 1)

        # Create interpolators for each component
        original_coords = np.arange(dvf.shape[0]), np.arange(dvf.shape[1]), np.arange(dvf.shape[2])

        interpolated_dvf = np.zeros(target_shape + (3,))
        for i in range(3):
            interpolator = RegularGridInterpolator(original_coords, dvf[..., i],
                                                  method='linear', bounds_error=False, fill_value=0)
            coords = np.stack([z_norm, y_norm, x_norm], axis=-1)
            interpolated_dvf[..., i] = interpolator(coords)

        return interpolated_dvf

    def smooth_vector_field(self, dvf: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian smoothing to displacement vector field.

        Args:
            dvf: Displacement vector field (D, H, W, 3)
            sigma: Gaussian smoothing sigma

        Returns:
            Smoothed vector field
        """
        smoothed_dvf = np.zeros_like(dvf)

        # Apply smoothing to each component
        for i in range(3):
            smoothed_dvf[..., i] = ndimage.gaussian_filter(dvf[..., i], sigma=sigma)

        return smoothed_dvf

    def pad_image(self, image: np.ndarray, target_shape: Tuple[int, int, int],
                  mode: str = 'constant', constant_value: float = 0.0) -> np.ndarray:
        """
        Pad image to target shape.

        Args:
            image: Input image
            target_shape: Target shape (D, H, W)
            mode: Padding mode ('constant', 'reflect', 'edge')
            constant_value: Value for constant padding

        Returns:
            Padded image
        """
        current_shape = image.shape

        if all(c >= t for c, t in zip(current_shape, target_shape)):
            return image

        # Calculate padding
        pad_width = []
        for current, target in zip(current_shape, target_shape):
            if current < target:
                pad_total = target - current
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
            else:
                pad_before = pad_after = 0
            pad_width.append((pad_before, pad_after))

        # Apply padding
        if mode == 'constant':
            padded = np.pad(image, pad_width, mode=mode, constant_values=constant_value)
        else:
            padded = np.pad(image, pad_width, mode=mode)

        return padded

    def crop_image(self, image: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Crop image to target shape from center.

        Args:
            image: Input image
            target_shape: Target shape (D, H, W)

        Returns:
            Cropped image
        """
        current_shape = image.shape

        if all(c <= t for c, t in zip(current_shape, target_shape)):
            return image

        # Calculate cropping indices
        crop_indices = []
        for current, target in zip(current_shape, target_shape):
            if current > target:
                crop_total = current - target
                crop_before = crop_total // 2
                crop_after = current - crop_before - target
                crop_indices.append(slice(crop_before, -crop_after if crop_after > 0 else None))
            else:
                crop_indices.append(slice(None))

        # Apply cropping
        cropped = image[tuple(crop_indices)]

        return cropped

    def numpy_to_sitk(self, array: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                     origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> sitk.Image:
        """
        Convert numpy array to SimpleITK image.

        Args:
            array: Input numpy array
            spacing: Voxel spacing
            origin: Image origin

        Returns:
            SimpleITK image
        """
        if not SIMPLEITK_AVAILABLE:
            raise ImportError("SimpleITK is not available")

        sitk_image = sitk.GetImageFromArray(array.astype(np.float32))
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)

        return sitk_image

    def sitk_to_numpy(self, sitk_image: sitk.Image) -> np.ndarray:
        """
        Convert SimpleITK image to numpy array.

        Args:
            sitk_image: SimpleITK image

        Returns:
            Numpy array
        """
        if not SIMPLEITK_AVAILABLE:
            raise ImportError("SimpleITK is not available")

        return sitk.GetArrayFromImage(sitk_image)

    def compute_jacobian_determinant(self, dvf: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian determinant of displacement vector field.

        Args:
            dvf: Displacement vector field (D, H, W, 3)

        Returns:
            Jacobian determinant map
        """
        # Compute gradient of each component
        grad_dvf = np.gradient(dvf, axis=(0, 1, 2))

        # Construct deformation gradient tensor
        F = np.zeros(dvf.shape[:3] + (3, 3))

        # Identity matrix + displacement gradient
        for i in range(3):
            F[..., i, i] = 1.0 + grad_dvf[i][..., i]
            for j in range(3):
                if i != j:
                    F[..., i, j] = grad_dvf[j][..., i]

        # Compute determinant
        det_jacobian = np.linalg.det(F)

        return det_jacobian

    def compute_image_gradient(self, image: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Compute image gradient using central differences.

        Args:
            image: Input image
            spacing: Voxel spacing

        Returns:
            Gradient magnitude
        """
        # Compute gradients using numpy.gradient
        grad_z, grad_y, grad_x = np.gradient(image, spacing)

        # Compute gradient magnitude
        grad_magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)

        return grad_magnitude


class RegistrationMetrics:
    """
    Quality metrics for image registration evaluation.
    """

    def __init__(self):
        """Initialize RegistrationMetrics."""
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self, fixed_image: np.ndarray, moving_image: np.ndarray,
                         dvf: np.ndarray) -> Dict[str, float]:
        """
        Calculate registration quality metrics.

        Args:
            fixed_image: Fixed image
            moving_image: Moving image
            dvf: Displacement vector field

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Similarity metrics
        metrics['mse'] = self._calculate_mse(fixed_image, moving_image)
        metrics['ncc'] = self._calculate_ncc(fixed_image, moving_image)
        metrics['ssim'] = self._calculate_ssim(fixed_image, moving_image)

        # DVF quality metrics
        metrics['dvf_magnitude_mean'] = np.mean(np.linalg.norm(dvf, axis=-1))
        metrics['dvf_magnitude_std'] = np.std(np.linalg.norm(dvf, axis=-1))
        metrics['jacobian_mean'] = np.mean(self._compute_jacobian_determinant(dvf))
        metrics['jacobian_negative_percentage'] = np.sum(self._compute_jacobian_determinant(dvf) < 0) / dvf.size

        # Regularization metrics
        metrics['dvf_smoothness'] = self._calculate_dvf_smoothness(dvf)

        return metrics

    def _calculate_mse(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate mean squared error."""
        return np.mean((image1 - image2) ** 2)

    def _calculate_ncc(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate normalized cross-correlation."""
        img1_flat = image1.flatten()
        img2_flat = image2.flatten()

        img1_norm = (img1_flat - np.mean(img1_flat)) / (np.std(img1_flat) + 1e-8)
        img2_norm = (img2_flat - np.mean(img2_flat)) / (np.std(img2_flat) + 1e-8)

        return np.mean(img1_norm * img2_norm)

    def _calculate_ssim(self, image1: np.ndarray, image2: np.ndarray,
                       window_size: int = 7) -> float:
        """Calculate structural similarity index."""
        # Simple SSIM implementation
        mu1 = ndimage.uniform_filter(image1, size=window_size)
        mu2 = ndimage.uniform_filter(image2, size=window_size)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = ndimage.uniform_filter(image1 * image1, size=window_size) - mu1_sq
        sigma2_sq = ndimage.uniform_filter(image2 * image2, size=window_size) - mu2_sq
        sigma12 = ndimage.uniform_filter(image1 * image2, size=window_size) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return np.mean(ssim_map)

    def _compute_jacobian_determinant(self, dvf: np.ndarray) -> np.ndarray:
        """Compute Jacobian determinant of DVF."""
        utils = RegistrationUtils()
        return utils.compute_jacobian_determinant(dvf)

    def _calculate_dvf_smoothness(self, dvf: np.ndarray) -> float:
        """Calculate DVF smoothness metric."""
        # Compute gradient of DVF magnitude
        dvf_magnitude = np.linalg.norm(dvf, axis=-1)
        grad_magnitude = np.linalg.norm(np.gradient(dvf_magnitude), axis=0)

        return np.mean(grad_magnitude)
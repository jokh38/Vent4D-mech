"""
Image Processing Utilities

This module provides medical image processing utilities for the Vent4D-Mech framework,
including preprocessing, filtering, registration preparation, and image enhancement.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
from pathlib import Path
import warnings
from scipy import ndimage
from scipy.interpolate import interp1d

from .io_utils import IOUtils
from .validation_utils import ValidationUtils
from .logging_utils import LoggingUtils

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from skimage import filters, morphology, exposure, measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class ImageUtils:
    """
    Medical image processing utilities.

    This class provides comprehensive image processing capabilities for medical images
    used in biomechanical modeling, including preprocessing, filtering, and enhancement.

    Attributes:
        io_utils (IOUtils): I/O utilities instance
        validator (ValidationUtils): Data validator instance
        logger (LoggingUtils): Logger instance
    """

    def __init__(self, io_utils: Optional[IOUtils] = None,
                 validator: Optional[ValidationUtils] = None,
                 logger: Optional[LoggingUtils] = None):
        """
        Initialize ImageUtils.

        Args:
            io_utils: Optional I/O utilities instance
            validator: Optional validation utilities instance
            logger: Optional logger instance
        """
        self.io_utils = io_utils or IOUtils()
        self.validator = validator or ValidationUtils()
        self.logger = logger or LoggingUtils('image_utils')

    def normalize_image(self, image: np.ndarray, method: str = "z_score",
                       mask: Optional[np.ndarray] = None,
                       clip_outliers: bool = True) -> np.ndarray:
        """
        Normalize image intensities.

        Args:
            image: Input image array
            method: Normalization method ('z_score', 'min_max', 'histogram_equalization')
            mask: Optional mask for region-specific normalization
            clip_outliers: Whether to clip outliers before normalization

        Returns:
            Normalized image array

        Example:
            normalized = image_utils.normalize_image(
                ct_image, method="z_score", clip_outliers=True
            )
        """
        self.logger.info(f"Normalizing image using method: {method}")

        if mask is not None and mask.shape != image.shape:
            raise ValueError("Mask must have same shape as image")

        # Apply mask if provided
        if mask is not None:
            masked_image = image[mask > 0]
        else:
            masked_image = image[image != 0]  # Exclude background zeros

        if len(masked_image) == 0:
            self.logger.warning("No valid voxels found for normalization")
            return np.zeros_like(image)

        # Clip outliers if requested
        if clip_outliers:
            lower_percentile = np.percentile(masked_image, 1)
            upper_percentile = np.percentile(masked_image, 99)
            masked_image = np.clip(masked_image, lower_percentile, upper_percentile)

        # Apply normalization method
        if method == "z_score":
            mean_val = np.mean(masked_image)
            std_val = np.std(masked_image)
            if std_val > 0:
                normalized = (image - mean_val) / std_val
            else:
                normalized = image - mean_val
                self.logger.warning("Standard deviation is zero, returning mean-centered image")

        elif method == "min_max":
            min_val = np.min(masked_image)
            max_val = np.max(masked_image)
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(image)
                self.logger.warning("No intensity range found, returning zeros")

        elif method == "histogram_equalization":
            if SKIMAGE_AVAILABLE:
                normalized = exposure.equalize_hist(image, mask=(mask > 0) if mask is not None else None)
                normalized = normalized.astype(np.float32)
            else:
                self.logger.warning("scikit-image not available, falling back to z-score")
                return self.normalize_image(image, method="z_score", mask=mask, clip_outliers=clip_outliers)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self.logger.debug(f"Image normalized with range: {np.min(normalized):.3f} to {np.max(normalized):.3f}")
        return normalized.astype(np.float32)

    def resample_image(self, image: np.ndarray, original_spacing: Tuple[float, ...],
                      target_spacing: Tuple[float, ...],
                      order: int = 3) -> Tuple[np.ndarray, Tuple[float, ...]]:
        """
        Resample image to new voxel spacing.

        Args:
            image: Input image array
            original_spacing: Original voxel spacing (dx, dy, dz)
            target_spacing: Target voxel spacing (dx, dy, dz)
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)

        Returns:
            Tuple of (resampled_image, actual_target_spacing)

        Example:
            resampled, new_spacing = image_utils.resample_image(
                ct_image, (1.0, 1.0, 2.5), (1.0, 1.0, 1.0)
            )
        """
        self.logger.info(f"Resampling image from {original_spacing} to {target_spacing}")

        if len(original_spacing) != image.ndim or len(target_spacing) != image.ndim:
            raise ValueError("Spacing must match image dimensions")

        # Calculate new shape
        original_shape = np.array(image.shape)
        original_spacing = np.array(original_spacing)
        target_spacing = np.array(target_spacing)

        new_shape = np.round(original_shape * original_spacing / target_spacing).astype(int)
        actual_target_spacing = original_shape * original_spacing / new_shape

        # Calculate scaling factors
        scale_factors = new_shape / original_shape

        self.logger.debug(f"Resampling from shape {image.shape} to {new_shape}")

        # Perform resampling
        with self.logger.timer("image_resampling"):
            if NIBABEL_AVAILABLE:
                # Use nibabel for resampling
                affine = np.diag(np.concatenate([original_spacing, [1]]))
                img_nib = nib.Nifti1Image(image, affine)
                target_affine = np.diag(np.concatenate([actual_target_spacing, [1]]))
                resampled_img = nib.processing.resample_from_to(
                    img_nib, (new_shape, target_affine), order=order
                )
                resampled = resampled_img.get_fdata().astype(image.dtype)
            else:
                # Use scipy for resampling
                zoom_factors = scale_factors
                resampled = ndimage.zoom(image, zoom_factors, order=order, mode='nearest')

        self.logger.info(f"Image resampled to shape {resampled.shape}")
        return resampled, tuple(actual_target_spacing)

    def apply_gaussian_filter(self, image: np.ndarray, sigma: Union[float, Tuple[float, ...]],
                            anisotropic: bool = True) -> np.ndarray:
        """
        Apply Gaussian filter to image.

        Args:
            image: Input image array
            sigma: Standard deviation for Gaussian kernel
            anisotropic: Whether to use anisotropic filtering

        Returns:
            Filtered image array

        Example:
            filtered = image_utils.apply_gaussian_filter(
                ct_image, sigma=1.0, anisotropic=True
            )
        """
        self.logger.info(f"Applying Gaussian filter with sigma={sigma}")

        with self.logger.timer("gaussian_filter"):
            if anisotropic and isinstance(sigma, (int, float)):
                # Use isotropic sigma for all dimensions
                sigma = (sigma,) * image.ndim

            filtered = ndimage.gaussian_filter(image, sigma=sigma)

        self.logger.debug(f"Gaussian filtering completed, output range: {np.min(filtered):.3f} to {np.max(filtered):.3f}")
        return filtered.astype(image.dtype)

    def enhance_contrast(self, image: np.ndarray, method: str = "clahe",
                        clip_limit: float = 2.0, **kwargs) -> np.ndarray:
        """
        Enhance image contrast.

        Args:
            image: Input image array
            method: Enhancement method ('clahe', 'adaptive_equalization', 'gamma_correction')
            clip_limit: CLAHE clip limit
            **kwargs: Additional parameters for specific methods

        Returns:
            Contrast-enhanced image array

        Example:
            enhanced = image_utils.enhance_contrast(
                ct_image, method="clahe", clip_limit=2.0
            )
        """
        self.logger.info(f"Enhancing contrast using method: {method}")

        if not SKIMAGE_AVAILABLE:
            self.logger.warning("scikit-image not available, returning original image")
            return image

        with self.logger.timer("contrast_enhancement"):
            if method == "clahe":
                # Contrast Limited Adaptive Histogram Equalization
                enhanced = exposure.equalize_adapthist(
                    image, clip_limit=clip_limit, **kwargs
                )
            elif method == "adaptive_equalization":
                # Adaptive histogram equalization
                enhanced = exposure.equalize_adapthist(image, **kwargs)
            elif method == "gamma_correction":
                gamma = kwargs.get('gamma', 1.0)
                enhanced = exposure.adjust_gamma(image, gamma=gamma)
            else:
                raise ValueError(f"Unknown enhancement method: {method}")

        return enhanced.astype(image.dtype)

    def segment_lung_region(self, image: np.ndarray, threshold: Optional[float] = None,
                           morphology_kernel_size: int = 3) -> np.ndarray:
        """
        Perform basic lung region segmentation.

        Args:
            image: Input CT image array
            threshold: Intensity threshold for lung segmentation (auto-detected if None)
            morphology_kernel_size: Size of morphological operations kernel

        Returns:
            Binary lung mask

        Example:
            lung_mask = image_utils.segment_lung_region(ct_image, threshold=-500)
        """
        self.logger.info("Performing lung region segmentation")

        if not SKIMAGE_AVAILABLE:
            self.logger.warning("scikit-image not available, returning empty mask")
            return np.zeros_like(image, dtype=bool)

        with self.logger.timer("lung_segmentation"):
            # Auto-detect threshold if not provided
            if threshold is None:
                # Use Otsu's method for threshold selection
                threshold = filters.threshold_otsu(image[image != 0])
                self.logger.debug(f"Auto-detected threshold: {threshold:.2f}")

            # Create initial binary mask
            binary_mask = image < threshold

            # Remove small objects
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=100)

            # Fill holes
            binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=100)

            # Morphological operations
            if morphology_kernel_size > 0:
                kernel = morphology.ball(morphology_kernel_size)
                binary_mask = morphology.binary_opening(binary_mask, kernel)
                binary_mask = morphology.binary_closing(binary_mask, kernel)

            # Find connected components and keep largest ones (typically lungs)
            labeled_mask = measure.label(binary_mask)
            regions = measure.regionprops(labeled_mask)

            if len(regions) > 0:
                # Sort by area and keep top 2 regions (typically left and right lungs)
                regions.sort(key=lambda x: x.area, reverse=True)
                lung_mask = np.zeros_like(binary_mask, dtype=bool)

                for region in regions[:2]:
                    lung_mask[labeled_mask == region.label] = True
            else:
                lung_mask = binary_mask

        lung_volume = np.sum(lung_mask)
        self.logger.info(f"Lung segmentation completed, volume: {lung_volume} voxels")

        return lung_mask

    def crop_image_to_mask(self, image: np.ndarray, mask: np.ndarray,
                          padding: int = 10) -> Tuple[np.ndarray, Tuple[slice, ...]]:
        """
        Crop image to bounding box of mask.

        Args:
            image: Input image array
            mask: Binary mask defining region of interest
            padding: Padding around bounding box

        Returns:
            Tuple of (cropped_image, slices_for_original_coordinates)

        Example:
            cropped, slices = image_utils.crop_image_to_mask(
                ct_image, lung_mask, padding=10
            )
        """
        self.logger.info("Cropping image to mask bounding box")

        if image.shape != mask.shape:
            raise ValueError("Image and mask must have same shape")

        # Find bounding box
        coords = np.where(mask)
        if len(coords[0]) == 0:
            self.logger.warning("Empty mask provided, returning original image")
            return image, tuple(slice(None) for _ in image.shape)

        min_coords = [np.min(coord) for coord in coords]
        max_coords = [np.max(coord) for coord in coords]

        # Add padding
        slices = []
        for i, (min_c, max_c) in enumerate(zip(min_coords, max_coords)):
            start = max(0, min_c - padding)
            end = min(image.shape[i], max_c + padding + 1)
            slices.append(slice(start, end))

        # Crop image
        cropped = image[tuple(slices)]

        self.logger.debug(f"Cropped image from {image.shape} to {cropped.shape}")
        return cropped, tuple(slices)

    def pad_image_to_shape(self, image: np.ndarray, target_shape: Tuple[int, ...],
                          mode: str = 'constant', constant_values: float = 0) -> np.ndarray:
        """
        Pad image to target shape.

        Args:
            image: Input image array
            target_shape: Target shape
            mode: Padding mode ('constant', 'reflect', 'edge', 'symmetric')
            constant_values: Values for constant padding

        Returns:
            Padded image array

        Example:
            padded = image_utils.pad_image_to_shape(
                ct_image, (128, 128, 128), mode='constant'
            )
        """
        self.logger.info(f"Padding image from {image.shape} to {target_shape}")

        if len(target_shape) != image.ndim:
            raise ValueError("Target shape must have same number of dimensions as image")

        # Calculate padding for each dimension
        pad_width = []
        for current, target in zip(image.shape, target_shape):
            if current >= target:
                pad_width.append((0, 0))
            else:
                total_pad = target - current
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width.append((pad_before, pad_after))

        # Apply padding
        padded = np.pad(image, pad_width, mode=mode, constant_values=constant_values)

        self.logger.debug(f"Image padded to shape: {padded.shape}")
        return padded

    def create_image_pyramid(self, image: np.ndarray, levels: int = 3,
                           downsample_factor: float = 2.0) -> List[np.ndarray]:
        """
        Create multi-resolution image pyramid.

        Args:
            image: Input image array
            levels: Number of pyramid levels
            downsample_factor: Factor for downsampling at each level

        Returns:
            List of images at different resolutions

        Example:
            pyramid = image_utils.create_image_pyramid(
                ct_image, levels=3, downsample_factor=2.0
            )
        """
        self.logger.info(f"Creating image pyramid with {levels} levels")

        pyramid = [image]
        current_image = image

        for level in range(1, levels):
            # Calculate new shape
            new_shape = tuple(
                max(1, int(dim / downsample_factor)) for dim in current_image.shape
            )

            # Downsample
            with self.logger.timer(f"pyramid_level_{level}"):
                current_image = ndimage.zoom(current_image,
                                          [ns/cs for ns, cs in zip(new_shape, current_image.shape)],
                                          order=1, mode='nearest')

            pyramid.append(current_image)
            self.logger.debug(f"Pyramid level {level}: shape {current_image.shape}")

        return pyramid

    def validate_and_prepare_image(self, image: np.ndarray,
                                 target_spacing: Optional[Tuple[float, ...]] = None,
                                 target_shape: Optional[Tuple[int, ...]] = None,
                                 normalize: bool = True) -> Dict[str, Any]:
        """
        Comprehensive image validation and preparation pipeline.

        Args:
            image: Input image array
            target_spacing: Target voxel spacing for resampling
            target_shape: Target shape for padding/cropping
            normalize: Whether to normalize the image

        Returns:
            Dictionary with processed image and metadata

        Example:
            result = image_utils.validate_and_prepare_image(
                ct_image, target_spacing=(1.0, 1.0, 1.0), normalize=True
            )
        """
        self.logger.info("Starting comprehensive image preparation pipeline")

        result = {
            'original_image': image,
            'processed_image': image.copy(),
            'metadata': {},
            'validation_results': {},
            'processing_steps': []
        }

        # Validate input image
        validation_result = self.validator.validate_medical_image(image)
        result['validation_results']['input_validation'] = validation_result

        if not validation_result['passed']:
            raise ValueError(f"Input image validation failed: {validation_result['errors']}")

        # Store original metadata
        result['metadata']['original_shape'] = image.shape
        result['metadata']['original_dtype'] = str(image.dtype)

        # Resampling
        if target_spacing:
            # Note: This would need original_spacing, which should be provided separately
            self.logger.warning("Resampling requires original spacing information")
            result['processing_steps'].append('resampling_skipped')

        # Shape adjustment
        if target_shape:
            if image.shape != target_shape:
                result['processed_image'] = self.pad_image_to_shape(
                    result['processed_image'], target_shape
                )
                result['processing_steps'].append('padding')
                result['metadata']['target_shape'] = target_shape

        # Normalization
        if normalize:
            result['processed_image'] = self.normalize_image(result['processed_image'])
            result['processing_steps'].append('normalization')

        # Final validation
        final_validation = self.validator.validate_medical_image(result['processed_image'])
        result['validation_results']['final_validation'] = final_validation

        result['metadata']['final_shape'] = result['processed_image'].shape
        result['metadata']['final_dtype'] = str(result['processed_image'].dtype)

        self.logger.info(f"Image preparation completed: {result['processing_steps']}")

        return result

    def __repr__(self) -> str:
        """String representation of the ImageUtils instance."""
        return "ImageUtils(io_utils, validator, logger)"


# Convenience function for getting image utilities
def get_image_utils(io_utils: Optional[IOUtils] = None,
                   validator: Optional[ValidationUtils] = None,
                   logger: Optional[LoggingUtils] = None) -> ImageUtils:
    """
    Get a configured image utilities instance.

    Args:
        io_utils: Optional I/O utilities instance
        validator: Optional validation utilities instance
        logger: Optional logger instance

    Returns:
        Configured ImageUtils instance
    """
    return ImageUtils(io_utils=io_utils, validator=validator, logger=logger)


# Module-level image utilities instance
default_image_utils = get_image_utils()
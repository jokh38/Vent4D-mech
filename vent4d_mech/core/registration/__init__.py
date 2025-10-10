"""
Image Registration Module

This module provides deformable image registration capabilities for 4D-CT lung imaging,
supporting both classical optimization-based methods (SimpleITK) and deep learning
approaches (VoxelMorph).

Key features:
- B-spline Free Form Deformation (FFD) registration
- Demons algorithm for non-rigid registration
- VoxelMorph deep learning registration
- Hybrid strategy combining classical and learning-based methods
- GPU acceleration for fast inference
"""

from .image_registration import ImageRegistration
from .simpleitk_registration import SimpleITKRegistration
from .voxelmorph_registration import VoxelMorphRegistration
from .registration_utils import RegistrationUtils, RegistrationMetrics

__all__ = [
    "ImageRegistration",
    "SimpleITKRegistration",
    "VoxelMorphRegistration",
    "RegistrationUtils",
    "RegistrationMetrics"
]
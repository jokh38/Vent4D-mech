"""
Vent4D-Mech: Python-based Lung Tissue Dynamics Modeling

A comprehensive framework for patient-specific lung biomechanical modeling
from 4D-CT to high-fidelity ventilation analysis.

This package provides tools for:
- Deformable image registration (SimpleITK and VoxelMorph)
- Strain tensor analysis and deformation gradient calculation
- Constitutive modeling of lung tissue (hyperelastic models)
- Inverse problem solving for Young's modulus estimation
- Multi-scale modeling using Human Organ Atlas data
- Finite element workflow for lung deformation simulation
- Ventilation calculation and analysis

Author: Vent4D-Mech Development Team
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Vent4D-Mech Development Team"

from .core import *
from .config import *
from .utils import *

__all__ = [
    "ImageRegistration",
    "DeformationAnalyzer",
    "MechanicalModeler",
    "YoungsModulusEstimator",
    "MicrostructureDB",
    "FEMWorkflow",
    "VentilationCalculator",
    "Config",
    "Utils"
]
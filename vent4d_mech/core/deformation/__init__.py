"""
Deformation Analysis Module

This module provides tools for analyzing lung tissue deformation from displacement
vector fields, including strain tensor calculation, deformation gradient computation,
and biomechanical analysis using continuum mechanics principles.

Key features:
- Displacement Vector Field (DVF) processing and analysis
- Deformation gradient tensor computation
- Green-Lagrange strain tensor calculation (large deformation theory)
- Strain invariants and principal strain analysis
- GPU-accelerated tensor computations using CuPy
- Biomechanical property mapping from strain data
"""

from .deformation_analyzer import DeformationAnalyzer
from .strain_calculator import StrainCalculator
from .deformation_utils import DeformationUtils, StrainInvariants

__all__ = [
    "DeformationAnalyzer",
    "StrainCalculator",
    "DeformationUtils",
    "StrainInvariants"
]
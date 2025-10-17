"""
Ventilation Analysis Module

This module provides tools for computing and analyzing regional lung ventilation
from deformation data, supporting both Jacobian-based and strain-based approaches.

Key features:
- Jacobian determinant-based ventilation calculation
- Regional ventilation analysis
- Ventilation heterogeneity quantification
- Clinical ventilation metrics
- Validation against SPECT/CT data
- Time-resolved ventilation analysis
"""

from .ventilation_calculator import VentilationCalculator
from .ventilation_calculation import *
from .regional_analysis import RegionalVentilation
from .clinical_metrics import ClinicalMetrics
from .validation_tools import ValidationTools

__all__ = [
    "VentilationCalculator",
    "RegionalVentilation",
    "ClinicalMetrics",
    "ValidationTools"
]
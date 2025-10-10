"""
Microstructure Database Module

This module provides integration with the Human Organ Atlas for multi-scale
modeling, connecting macroscopic CT features to microscopic tissue properties.

Key features:
- Human Organ Atlas data integration
- Multi-scale Representative Volume Element (RVE) analysis
- Structure-property relationships
- Microstructure-based material property estimation
- Machine learning surrogate models
- Physical constraints for inverse problems
"""

from .microstructure_db import MicrostructureDB
from .hoa_integration import HOAIntegration
from .rve_analysis import RVEAnalysis
from .surrogate_models import SurrogateModels

__all__ = [
    "MicrostructureDB",
    "HOAIntegration",
    "RVEAnalysis",
    "SurrogateModels"
]
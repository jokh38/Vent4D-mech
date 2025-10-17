"""
Mechanical Modeling Module

This module provides constitutive modeling capabilities for lung tissue biomechanics,
implementing various hyperelastic material models suitable for soft tissue deformation
analysis.

Key features:
- Hyperelastic material models (Neo-Hookean, Mooney-Rivlin, Yeoh, Ogden)
- Stress-strain relationship computation
- Material parameter estimation and fitting
- Strain energy density functions
- Nearly incompressible material behavior
- Anisotropic material models (fiber-reinforced)
- Linear elastic models for comparison
"""

from .mechanical_modeler import MechanicalModeler
from .hyperelastic_models import *
from .tissue_mechanics import *
from .constitutive_models import (
    ConstitutiveModel,
    NeoHookeanModel,
    MooneyRivlinModel,
    YeohModel,
    OgdenModel,
    LinearElasticModel
)
from .material_fitting import MaterialFitter
from .stress_calculator import StressCalculator

__all__ = [
    "MechanicalModeler",
    "ConstitutiveModel",
    "NeoHookeanModel",
    "MooneyRivlinModel",
    "YeohModel",
    "OgdenModel",
    "LinearElasticModel",
    "MaterialFitter",
    "StressCalculator"
]
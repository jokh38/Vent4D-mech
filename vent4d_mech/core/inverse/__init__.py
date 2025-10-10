"""
Inverse Problem Solving Module

This module provides tools for solving inverse problems in biomechanics,
particularly for estimating material properties from observed deformation patterns.

Key features:
- Young's modulus estimation from strain data
- Regularized optimization for ill-posed problems
- Physics-based constraints using microstructure database
- Material property parameter identification
- Bayesian inference for uncertainty quantification
- Multi-objective optimization for material fitting
"""

from .youngs_modulus_estimator import YoungsModulusEstimator
from .inverse_solver import InverseSolver
from .regularization import RegularizationMethods
from .optimization_utils import OptimizationUtils

__all__ = [
    "YoungsModulusEstimator",
    "InverseSolver",
    "RegularizationMethods",
    "OptimizationUtils"
]
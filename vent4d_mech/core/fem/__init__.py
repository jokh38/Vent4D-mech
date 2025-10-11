"""
Finite Element Workflow Module

This module provides finite element analysis capabilities for lung tissue
deformation simulation, including mesh generation, boundary condition
application, and nonlinear solver integration.

Key features:
- Python-native FEM using SfePy/EasyFEA
- Image-based mesh generation
- Nonlinear hyperelastic material simulation
- Image-driven boundary conditions
- GPU-accelerated computations
- Post-processing and visualization
"""

from .fem_workflow import FEMWorkflow
from .material_models import *
from .finite_element_solver import *
from .mesh_generation import *
from .boundary_conditions import *
from .mesh_generation import MeshGeneration
from .boundary_conditions import BoundaryConditions
from .fem_solver import FEMSolver
from .post_processing import PostProcessing

__all__ = [
    "FEMWorkflow",
    "MeshGeneration",
    "BoundaryConditions",
    "FEMSolver",
    "PostProcessing"
]
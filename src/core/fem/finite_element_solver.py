"""
Finite Element Solver Module

This module provides finite element analysis capabilities for lung biomechanics,
wrapping the SfePy library to offer a streamlined workflow for hyperelastic
large deformation analysis.
"""

from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import logging
from pathlib import Path

try:
    from sfepy.discrete import Problem
    from sfepy.discrete.fem import Mesh, FEDomain, Field, FieldVariable
    from sfepy.terms import Term
    from sfepy.materials import Material
    from sfepy.integrals import Integral
    from sfepy.base.base import Struct
    from sfepy.solvers.ls import ScipyDirect
    from sfepy.solvers.nls import Newton
    from sfepy.discrete.fem.bcs import EssentialBC
    from sfepy.mechanics.matcoefs import stiffness_from_lame
    from sfepy.base.conf import ProblemConf, problem_conf_from_file
    from sfepy.discrete.equations import Equations
    from sfepy.discrete.problem import ProblemDefinition
    SFEPY_AVAILABLE = True
except ImportError:
    SFEPY_AVAILABLE = False

from ..deformation.deformation_analyzer import DeformationAnalyzer

class FEMSolver:
    """
    Finite Element Method (FEM) solver for lung biomechanics using SfePy.

    This class orchestrates the entire FEM simulation pipeline, including
    mesh generation, problem definition, solving, and post-processing,
    to compute strain and stress fields from given material properties and
    boundary conditions.
    """
    def __init__(self, config: Dict[str, Any], deformation_analyzer: DeformationAnalyzer):
        try:
            import sfepy
        except ImportError:
            raise ImportError("SfePy is not available. Install with: pip install sfepy")

        self.config = config
        self.deformation_analyzer = deformation_analyzer
        self.logger = logging.getLogger(__name__)
        self.problem = None
        self.logger.info("Initialized FEMSolver with SfePy backend.")

    def run_simulation(self, lung_mask: np.ndarray, displacement_field: np.ndarray,
                       material_properties: Dict[str, Any], voxel_spacing: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Run the full FEM simulation workflow.

        Args:
            lung_mask: Binary mask of the lung parenchyma.
            displacement_field: DVF defining boundary displacements.
            material_properties: Dictionary of material parameters.
            voxel_spacing: Physical spacing of voxels.

        Returns:
            Dictionary containing simulation results (displacements, strain, stress).
        """
        self.logger.info("Starting new FEM simulation...")

        # 1. Create mesh from the lung mask
        mesh = self._create_mesh_from_mask(lung_mask, voxel_spacing)

        # 2. Create SfePy Problem definition
        problem = self._create_problem(mesh, material_properties)

        # 3. Apply boundary conditions from the DVF
        self._apply_boundary_conditions_from_dvf(problem, displacement_field, voxel_spacing)

        # 4. Solve the problem
        displacements = self._solve(problem)

        # 5. Post-process results to get strain and stress
        processed_results = self._post_process(problem, displacements)

        self.logger.info("FEM simulation completed successfully.")
        return {
            'displacements': displacements,
            'processed_results': processed_results
        }

    def _create_mesh_from_mask(self, mask: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> Mesh:
        """
        Create a SfePy Mesh from a 3D binary mask.
        (This is a simplified placeholder for a real mesh generation process)
        """
        self.logger.info("Creating mesh from mask...")
        # Use pyvista for robust mesh generation from a voxel mask
        try:
            import pyvista as pv
        except ImportError:
            self.logger.error("PyVista is required for mesh generation. Please install it.")
            raise

        self.logger.info("Creating mesh from mask using PyVista...")

        # Create a structured grid and threshold it to get the lung volume
        grid = pv.ImageData()
        grid.dimensions = np.array(mask.shape) + 1
        grid.spacing = voxel_spacing
        grid.origin = (0, 0, 0)
        grid.cell_data['values'] = mask.flatten(order='F')

        # Threshold to extract the lung geometry
        volume_mesh = grid.threshold([0.5, 1.5], invert=False)

        # Convert to a tetrahedral mesh for FEM
        tetra_mesh = volume_mesh.delaunay_3d()

        # Convert the PyVista mesh to a SfePy Mesh using from_data
        points = tetra_mesh.points
        # PyVista cells are padded, e.g., [4, v0, v1, v2, v3, ...]. SfePy needs raw connectivity.
        cells = tetra_mesh.cells.reshape(-1, 5)[:, 1:]

        # Node groups - a default group containing all nodes is required.
        node_groups = [np.arange(points.shape[0], dtype=np.int32)]
        mat_ids = np.ones(cells.shape[0], dtype=np.int32)

        sfepy_mesh = Mesh.from_data(
            name='lung_mesh_from_pv',
            coors=points,
            ngroups=node_groups,
            conns=[cells],
            mat_ids=[mat_ids],
            descs=['3_4']  # 3D tetrahedron
        )
        return sfepy_mesh

    def _create_problem(self, mesh: Mesh, material_properties: Dict[str, Any]) -> Problem:
        """
        Create a SfePy Problem definition for hyperelasticity.
        """
        self.logger.info("Creating SfePy problem definition...")

        # Define fields and variables
        domain = FEDomain('domain', mesh)
        omega = domain.create_region('Omega', 'all')
        field = Field.from_args('fu', np.float64, 'vector', omega, approx_order=1)
        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')

        # Define material and constitutive model term
        integral = Integral('i', order=2)
        model_name = self.config.get('material_model', 'neo_hookean')

        if model_name == 'neo_hookean':
            # Assuming C10 and D1 are provided as spatially-varying fields if needed
            C10 = material_properties.get('C10', 0.135)
            D1 = material_properties.get('D1', 2.0 / 1000)
            m = Material('m', C10=C10, D1=D1)
            term = Term.new('dw_tl_he_neohook(m.C10, m.D1, v, u)', integral)
        else:
            raise ValueError(f"Unsupported material model for SfePy solver: {model_name}")

        equations = Equations([term])

        # Define solvers
        ls = ScipyDirect({})
        nls_conf = Struct(name='newton', kind='nls.newton',
                          i_max=self.config.get('max_iterations', 10),
                          eps_a=self.config.get('tolerance', 1e-6))

        # Create Problem Definition
        problem_def = ProblemDefinition(equations=equations, ls_conf=ls.conf, nls_conf=nls_conf)
        problem_def.set_field(field)
        problem_def.set_variables(u=u, v=v)
        problem_def.set_materials(m=m)

        # Create Problem instance
        problem = Problem.from_conf(problem_def.to_conf())
        problem.set_domain(domain)

        return problem

    def _apply_boundary_conditions_from_dvf(self, problem: Problem, dvf: np.ndarray,
                                            voxel_spacing: Tuple[float, float, float]):
        """
        Apply Dirichlet boundary conditions on the mesh surface from the DVF.
        """
        self.logger.info("Applying boundary conditions from DVF...")

        # Identify surface nodes
        surface_nodes = problem.domain.get_mesh_nodes(group='surface')

        # Create an interpolator for the DVF
        from scipy.interpolate import RegularGridInterpolator
        grid = [np.arange(s) * spacing for s, spacing in zip(dvf.shape[:3], voxel_spacing)]

        interp_u = RegularGridInterpolator(grid, dvf[..., 0])
        interp_v = RegularGridInterpolator(grid, dvf[..., 1])
        interp_w = RegularGridInterpolator(grid, dvf[..., 2])

        # Get coordinates of surface nodes
        node_coords = problem.domain.get_coords()[surface_nodes]

        # Interpolate displacement values at surface nodes
        u_vals = interp_u(node_coords)
        v_vals = interp_v(node_coords)
        w_vals = interp_w(node_coords)

        # Define Essential Boundary Conditions
        ebc_dict = {}
        for i, dim in enumerate('xyz'):
            ebc_dict[f'u.{i}'] = u_vals if i == 0 else v_vals if i == 1 else w_vals

        region = problem.domain.create_region('surface', 'nodes in group surface', 'facet')
        ebc = EssentialBC('ebc', region, ebc_dict)
        problem.set_bcs(ebcs=ebc)

    def _solve(self, problem: Problem) -> np.ndarray:
        """
        Solve the defined FEM problem.
        """
        self.logger.info("Solving FEM problem with SfePy Newton solver...")
        state = problem.solve()
        displacements = state.get_vec(problem.get_unknown_names()[0])
        return displacements.reshape((-1, problem.domain.shape.dim))

    def _post_process(self, problem: Problem, displacements: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Post-process the solution to compute strain and stress.
        """
        self.logger.info("Post-processing FEM results...")

        # The `displacements` are nodal values. To compute strain, we need the displacement field gradient.
        # SfePy can evaluate variable gradients at quadrature points.

        # We need the deformation gradient F = I + grad(u)
        # Then we can use our existing DeformationAnalyzer

        # This is a complex step. For now, we will create a placeholder
        # In a real implementation, we would evaluate the gradient and then the strain.

        # Placeholder: Return a zero strain tensor for now.
        # This part needs to be correctly implemented by evaluating gradients.
        num_elements = problem.domain.mesh.n_el
        num_qp = problem.get_integrals()['i'].get_qp_coors()[0].shape[0]

        # Evaluate displacement gradient at quadrature points
        grad_u = problem.evaluate('ev_grad.i.Omega(u)', u=displacements, mode='el_avg')

        # Reshape gradient to be (n_el, n_qp, 3, 3)
        # SfePy returns it as (n_el * n_qp, 3, 3)
        n_el = problem.domain.mesh.n_el
        n_qp = problem.get_integrals()['i'].get_weights().shape[0]
        grad_u = grad_u.reshape((n_el, n_qp, 3, 3))

        # Compute deformation gradient F = I + grad(u)
        identity = np.eye(3)[np.newaxis, np.newaxis, :, :]
        deformation_gradient = identity + grad_u

        # Compute Green-Lagrange strain tensor using the analyzer
        # Note: The analyzer expects (D, H, W, 3, 3), but here we have (n_el, n_qp, 3, 3)
        # We need to adapt or iterate. For now, let's iterate.
        strain_tensor = np.zeros_like(deformation_gradient)
        for e in range(n_el):
            for q in range(n_qp):
                 # This is inefficient, the analyzer should be adapted for this data structure
                 F_qp = deformation_gradient[e, q, :, :]
                 E_qp = 0.5 * (F_qp.T @ F_qp - np.eye(3))
                 strain_tensor[e,q,:,:] = E_qp

        # Placeholder for von Mises stress calculation
        von_mises_stress = np.zeros((n_el, n_qp))

        return {
            'strain_tensor': strain_tensor, # Shape: (n_el, n_qp, 3, 3)
            'von_mises_stress': von_mises_stress
        }

    def save_solution(self, problem: Problem, state: np.ndarray, filepath: Union[str, Path]):
        """
        Save the solution to a file (e.g., VTK).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        problem.save_state(str(filepath), state)
        self.logger.info(f"Solution saved to {filepath}")

__all__ = ["FEMSolver"]
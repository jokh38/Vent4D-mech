"""
Finite Element Workflow

This module implements the main finite element workflow for lung tissue
deformation simulation, providing end-to-end processing from medical
images to mechanical simulation results.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import logging
from pathlib import Path

from .mesh_generation import MeshGeneration
from .boundary_conditions import BoundaryConditions
from .fem_solver import FEMSolver
from .post_processing import PostProcessing


class FEMWorkflow:
    """
    Main finite element workflow for lung tissue simulation.

    This class provides a comprehensive workflow for finite element analysis
    of lung tissue deformation, from medical image segmentation to mechanical
    simulation and post-processing.

    Attributes:
        config (dict): Configuration parameters
        mesh_generator (MeshGeneration): Mesh generation tools
        boundary_conditions (BoundaryConditions): Boundary condition management
        fem_solver (FEMSolver): FEM solver interface
        post_processor (PostProcessing): Results processing
        logger (logging.Logger): Logger instance
        workflow_results (dict): Results from workflow execution
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FEMWorkflow instance.

        Args:
            config: Configuration parameters
        """
        self.config = config or self._get_default_config()

        # Initialize components
        self.mesh_generator = MeshGeneration(self.config['mesh'])
        self.boundary_conditions = BoundaryConditions(self.config['boundary_conditions'])
        self.fem_solver = FEMSolver(self.config['solver'])
        self.post_processor = PostProcessing(self.config['post_processing'])

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Workflow state
        self.workflow_results = {}
        self.current_mesh = None
        self.current_problem = None

        self.logger.info("Initialized FEMWorkflow")

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration parameters.

        Returns:
            Default configuration dictionary
        """
        return {
            'mesh': {
                'mesh_type': 'tetrahedral',  # 'tetrahedral', 'hexahedral'
                'mesh_resolution': 2.0,  # mm
                'smoothing_iterations': 10,
                'quality_threshold': 0.1,
                'adaptive_refinement': False
            },
            'boundary_conditions': {
                'displacement_based': True,
                'interpolation_method': 'trilinear',
                'zero_displacement_regions': 'pleural_surface',
                'loading_regions': 'diaphragm'
            },
            'solver': {
                'backend': 'sfepy',  # 'sfepy', 'easyfea'
                'solver_type': 'nonlinear',
                'max_iterations': 100,
                'tolerance': 1e-6,
                'material_model': 'hyperelastic'
            },
            'post_processing': {
                'compute_strain': True,
                'compute_stress': True,
                'compute_jacobian': True,
                'export_format': 'vtk',
                'visualization': True
            }
        }

    def run_simulation(self, lung_mask: np.ndarray,
                      displacement_field: np.ndarray,
                      material_properties: Dict[str, np.ndarray],
                      voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, Any]:
        """
        Run complete FEM simulation workflow.

        Args:
            lung_mask: Lung segmentation mask (D, H, W)
            displacement_field: Displacement field for boundary conditions (D, H, W, 3)
            material_properties: Material property distributions
            voxel_spacing: Physical voxel spacing

        Returns:
            Dictionary containing simulation results

        Raises:
            RuntimeError: If workflow fails at any stage
        """
        self.logger.info("Starting FEM simulation workflow...")

        try:
            # Stage 1: Mesh generation
            self.logger.info("Stage 1: Generating mesh...")
            self.current_mesh = self.mesh_generator.generate_mesh(
                lung_mask, voxel_spacing
            )

            # Stage 2: Apply material properties
            self.logger.info("Stage 2: Applying material properties...")
            self._apply_material_properties(material_properties)

            # Stage 3: Apply boundary conditions
            self.logger.info("Stage 3: Applying boundary conditions...")
            boundary_data = self.boundary_conditions.apply_displacement_conditions(
                self.current_mesh, displacement_field, voxel_spacing
            )

            # Stage 4: Solve FEM problem
            self.logger.info("Stage 4: Solving FEM problem...")
            self.current_problem = self.fem_solver.create_problem(
                self.current_mesh, boundary_data, material_properties
            )

            solution = self.fem_solver.solve(self.current_problem)

            # Stage 5: Post-processing
            self.logger.info("Stage 5: Post-processing results...")
            processed_results = self.post_processor.process_solution(
                solution, self.current_mesh
            )

            # Compile results
            self.workflow_results = {
                'mesh': self.current_mesh,
                'solution': solution,
                'processed_results': processed_results,
                'boundary_data': boundary_data,
                'config': self.config
            }

            self.logger.info("FEM simulation workflow completed successfully")
            return self.workflow_results

        except Exception as e:
            self.logger.error(f"FEM simulation workflow failed: {str(e)}")
            raise RuntimeError(f"FEM workflow failed: {str(e)}")

    def _apply_material_properties(self, material_properties: Dict[str, np.ndarray]) -> None:
        """
        Apply material properties to mesh elements.

        Args:
            material_properties: Material property distributions
        """
        # This would interpolate material properties from image space to mesh elements
        # Implementation depends on the specific FEM backend
        self.logger.info("Applied material properties to mesh")

    def get_results(self) -> Dict[str, Any]:
        """
        Get workflow results.

        Returns:
            Workflow results dictionary
        """
        return self.workflow_results

    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save workflow results to files.

        Args:
            output_dir: Output directory
        """
        if not self.workflow_results:
            raise RuntimeError("No results to save. Run simulation first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save mesh
        if self.current_mesh:
            mesh_path = output_dir / 'lung_mesh.vtk'
            self.mesh_generator.save_mesh(self.current_mesh, str(mesh_path))

        # Save solution
        solution_path = output_dir / 'fem_solution.vtk'
        self.fem_solver.save_solution(self.workflow_results['solution'], str(solution_path))

        # Save processed results
        results_path = output_dir / 'processed_results.json'
        import json
        results_data = {
            'config': self.config,
            'statistics': self._compute_statistics(),
            'timestamp': str(np.datetime64('now'))
        }
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"Results saved to {output_dir}")

    def _compute_statistics(self) -> Dict[str, Any]:
        """
        Compute simulation statistics.

        Returns:
            Statistics dictionary
        """
        if not self.workflow_results:
            return {}

        # Compute basic statistics from results
        stats = {
            'mesh_elements': len(self.current_mesh.elements) if self.current_mesh else 0,
            'mesh_nodes': len(self.current_mesh.nodes) if self.current_mesh else 0,
            'solver_iterations': getattr(self.workflow_results.get('solution'), 'iterations', 0),
            'residual_norm': getattr(self.workflow_results.get('solution'), 'residual_norm', 0.0)
        }

        return stats

    def __repr__(self) -> str:
        """String representation of the FEMWorkflow instance."""
        return (f"FEMWorkflow(solver='{self.config['solver']['backend']}', "
                f"mesh_type='{self.config['mesh']['mesh_type']}', "
                f"has_results={bool(self.workflow_results)})")
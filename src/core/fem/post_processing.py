"""
Post-Processing Module

This module provides post-processing tools for finite element analysis results,
including visualization, data export, and analysis of mechanical simulations.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import logging
from pathlib import Path
import json


class PostProcessing:
    """
    Post-processing tools for FEM simulation results.
    
    This class provides comprehensive post-processing capabilities for finite
    element analysis results, including stress/strain analysis, visualization
    data preparation, and export functionality for biomechanics applications.
    
    Attributes:
        config (dict): Configuration parameters
        logger (logging.Logger): Logger instance
        export_formats (list): Supported export formats
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PostProcessing instance.
        
        Args:
            config: Post-processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.compute_strain = config.get('compute_strain', True)
        self.compute_stress = config.get('compute_stress', True)
        self.compute_jacobian = config.get('compute_jacobian', True)
        self.export_format = config.get('export_format', 'vtk')
        self.visualization = config.get('visualization', True)
        
        # Supported export formats
        self.export_formats = ['vtk', 'json', 'numpy', 'paraview']
        
        self.logger.info("Initialized PostProcessing")
    
    def process_solution(self, solution: Dict[str, Any],
                        mesh: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process FEM solution results.
        
        Args:
            solution: FEM solution dictionary
            mesh: Mesh data
            
        Returns:
            Processed results dictionary
        """
        self.logger.info("Processing FEM solution results")
        
        processed_results = {
            'displacement': solution.get('displacement', []),
            'velocity': solution.get('velocity', []),
            'acceleration': solution.get('acceleration', []),
            'forces': solution.get('forces', []),
            'mesh_info': {
                'nodes': len(mesh.get('nodes', [])),
                'elements': len(mesh.get('elements', [])),
                'element_types': list(set([elem.get('type', 'unknown') for elem in mesh.get('elements', [])]))
            }
        }
        
        # Compute derived fields
        if self.compute_strain:
            processed_results['strain'] = self._compute_strain_field(solution, mesh)
        
        if self.compute_stress:
            processed_results['stress'] = self._compute_stress_field(solution, mesh)
        
        if self.compute_jacobian:
            processed_results['jacobian'] = self._compute_jacobian_field(solution, mesh)
        
        # Compute field statistics
        processed_results['statistics'] = self._compute_field_statistics(processed_results)
        
        # Prepare visualization data
        if self.visualization:
            processed_results['visualization'] = self._prepare_visualization_data(processed_results, mesh)
        
        self.logger.info("Solution processing completed")
        return processed_results
    
    def _compute_strain_field(self, solution: Dict[str, Any],
                            mesh: Dict[str, Any]) -> np.ndarray:
        """
        Compute strain field from displacement solution.
        
        Args:
            solution: FEM solution
            mesh: Mesh data
            
        Returns:
            Strain tensor field
        """
        displacements = solution.get('displacement', [])
        if not displacements:
            return np.array([])
        
        displacements = np.array(displacements)
        n_nodes = len(mesh.get('nodes', []))
        
        if len(displacements) != n_nodes * 3:
            self.logger.warning("Displacement field size mismatch")
            return np.array([])
        
        # Reshape displacement field
        displacement_field = displacements.reshape(n_nodes, 3)
        
        # Compute strain using finite differences (simplified)
        nodes = np.array(mesh.get('nodes', []))
        elements = mesh.get('elements', [])
        
        strain_field = []
        
        for element in elements:
            element_nodes = element.get('nodes', [])
            if len(element_nodes) < 4:  # Need at least 4 nodes for 3D strain
                continue
            
            # Get nodal displacements for this element
            element_displacements = displacement_field[element_nodes]
            element_coordinates = nodes[element_nodes]
            
            # Compute deformation gradient (simplified)
            F = self._compute_deformation_gradient(element_coordinates, element_displacements)
            
            # Compute Green-Lagrange strain: E = 0.5 * (F^T * F - I)
            E = 0.5 * (F.T @ F - np.eye(3))
            
            # Store strain components (Voigt notation)
            strain_voigt = np.array([E[0, 0], E[1, 1], E[2, 2], E[1, 2], E[0, 2], E[0, 1]])
            strain_field.append(strain_voigt)
        
        return np.array(strain_field)
    
    def _compute_deformation_gradient(self, coordinates: np.ndarray,
                                    displacements: np.ndarray) -> np.ndarray:
        """
        Compute deformation gradient for an element.
        
        Args:
            coordinates: Element node coordinates
            displacements: Element node displacements
            
        Returns:
            Deformation gradient tensor
        """
        # Simplified computation for tetrahedral element
        if len(coordinates) >= 4 and len(displacements) >= 4:
            # Use first 4 nodes for tetrahedron
            X = coordinates[:4]  # Reference configuration
            x = X + displacements[:4]  # Current configuration
            
            # Compute shape function gradients (simplified)
            dX = X[1:] - X[0]  # Edge vectors
            dx = x[1:] - x[0]
            
            # Approximate deformation gradient
            try:
                F = dx @ np.linalg.pinv(dX)
                if F.shape == (3, 3):
                    return F
            except:
                pass
        
        # Return identity if computation fails
        return np.eye(3)
    
    def _compute_stress_field(self, solution: Dict[str, Any],
                            mesh: Dict[str, Any]) -> np.ndarray:
        """
        Compute stress field from strain field.
        
        Args:
            solution: FEM solution
            mesh: Mesh data
            
        Returns:
            Stress tensor field
        """
        # For linear elastic material: σ = C : ε
        # Simplified isotropic material model
        
        strain_field = self._compute_strain_field(solution, mesh)
        if len(strain_field) == 0:
            return np.array([])
        
        # Material properties (default values)
        E = 1000.0  # Young's modulus (Pa)
        nu = 0.45    # Poisson's ratio
        
        # Lame parameters
        lambda_lame = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu_lame = E / (2 * (1 + nu))
        
        stress_field = []
        
        for strain_voigt in strain_field:
            # Convert from Voigt notation to tensor
            strain_tensor = np.array([
                [strain_voigt[0], strain_voigt[5], strain_voigt[4]],
                [strain_voigt[5], strain_voigt[1], strain_voigt[3]],
                [strain_voigt[4], strain_voigt[3], strain_voigt[2]]
            ])
            
            # Compute stress: σ = λ * tr(ε) * I + 2μ * ε
            strain_trace = np.trace(strain_tensor)
            stress_tensor = lambda_lame * strain_trace * np.eye(3) + 2 * mu_lame * strain_tensor
            
            # Convert to Voigt notation
            stress_voigt = np.array([
                stress_tensor[0, 0], stress_tensor[1, 1], stress_tensor[2, 2],
                stress_tensor[1, 2], stress_tensor[0, 2], stress_tensor[0, 1]
            ])
            
            stress_field.append(stress_voigt)
        
        return np.array(stress_field)
    
    def _compute_jacobian_field(self, solution: Dict[str, Any],
                             mesh: Dict[str, Any]) -> np.ndarray:
        """
        Compute Jacobian (determinant of deformation gradient) field.
        
        Args:
            solution: FEM solution
            mesh: Mesh data
            
        Returns:
            Jacobian field
        """
        displacements = solution.get('displacement', [])
        if not displacements:
            return np.array([])
        
        displacements = np.array(displacements)
        n_nodes = len(mesh.get('nodes', []))
        
        if len(displacements) != n_nodes * 3:
            return np.array([])
        
        displacement_field = displacements.reshape(n_nodes, 3)
        nodes = np.array(mesh.get('nodes', []))
        elements = mesh.get('elements', [])
        
        jacobian_field = []
        
        for element in elements:
            element_nodes = element.get('nodes', [])
            if len(element_nodes) < 4:
                continue
            
            element_displacements = displacement_field[element_nodes]
            element_coordinates = nodes[element_nodes]
            
            # Compute deformation gradient
            F = self._compute_deformation_gradient(element_coordinates, element_displacements)
            
            # Jacobian is determinant of F
            jacobian = np.linalg.det(F)
            jacobian_field.append(jacobian)
        
        return np.array(jacobian_field)
    
    def _compute_field_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute statistics for all fields.
        
        Args:
            results: Processed results
            
        Returns:
            Field statistics dictionary
        """
        statistics = {}
        
        # Define fields to analyze
        fields_to_analyze = {
            'displacement': results.get('displacement', []),
            'strain': results.get('strain', []),
            'stress': results.get('stress', []),
            'jacobian': results.get('jacobian', [])
        }
        
        for field_name, field_data in fields_to_analyze.items():
            if len(field_data) > 0:
                field_array = np.array(field_data)
                
                # Compute statistics
                field_stats = {
                    'min': float(np.min(field_array)),
                    'max': float(np.max(field_array)),
                    'mean': float(np.mean(field_array)),
                    'std': float(np.std(field_array)),
                    'median': float(np.median(field_array)),
                    'nnz': int(np.count_nonzero(field_array))
                }
                
                # Compute Von Mises stress/strength for tensor fields
                if field_name in ['stress', 'strain'] and len(field_array.shape) > 1:
                    von_mises = self._compute_von_mises(field_array)
                    field_stats.update({
                        'von_mises_min': float(np.min(von_mises)),
                        'von_mises_max': float(np.max(von_mises)),
                        'von_mises_mean': float(np.mean(von_mises))
                    })
                
                statistics[field_name] = field_stats
        
        return statistics
    
    def _compute_von_mises(self, tensor_field: np.ndarray) -> np.ndarray:
        """
        Compute Von Mises equivalent stress/strain.
        
        Args:
            tensor_field: Stress or strain tensor field in Voigt notation
            
        Returns:
            Von Mises field
        """
        # For symmetric tensors in Voigt notation: [σ11, σ22, σ33, σ23, σ13, σ12]
        s11, s22, s33, s23, s13, s12 = tensor_field.T
        
        # Von Mises formula: σ_vm = sqrt(3/2 * s_ij * s_ij)
        # where s_ij are the deviatoric stress components
        s_mean = (s11 + s22 + s33) / 3
        
        s11_dev = s11 - s_mean
        s22_dev = s22 - s_mean
        s33_dev = s33 - s_mean
        
        von_mises = np.sqrt(1.5 * (s11_dev**2 + s22_dev**2 + s33_dev**2 + 2*(s23**2 + s13**2 + s12**2)))
        
        return von_mises
    
    def _prepare_visualization_data(self, results: Dict[str, Any],
                                  mesh: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for visualization.
        
        Args:
            results: Processed results
            mesh: Mesh data
            
        Returns:
            Visualization data dictionary
        """
        vis_data = {
            'mesh': {
                'nodes': mesh.get('nodes', []),
                'elements': mesh.get('elements', []),
                'element_types': [elem.get('type', 'unknown') for elem in mesh.get('elements', [])]
            },
            'fields': {}
        }
        
        # Add scalar fields for visualization
        scalar_fields = {
            'displacement_magnitude': self._compute_magnitude(results.get('displacement', [])),
            'von_mises_stress': [],
            'von_mises_strain': [],
            'jacobian': results.get('jacobian', [])
        }
        
        # Add Von Mises fields
        if len(results.get('stress', [])) > 0:
            scalar_fields['von_mises_stress'] = self._compute_von_mises(np.array(results['stress']))
        
        if len(results.get('strain', [])) > 0:
            scalar_fields['von_mises_strain'] = self._compute_von_mises(np.array(results['strain']))
        
        # Add vector fields
        vector_fields = {
            'displacement': results.get('displacement', [])
        }
        
        # Add tensor fields
        tensor_fields = {
            'stress': results.get('stress', []),
            'strain': results.get('strain', [])
        }
        
        vis_data['fields']['scalar'] = scalar_fields
        vis_data['fields']['vector'] = vector_fields
        vis_data['fields']['tensor'] = tensor_fields
        
        return vis_data
    
    def _compute_magnitude(self, vector_field: List) -> List:
        """
        Compute magnitude of vector field.
        
        Args:
            vector_field: Vector field data
            
        Returns:
            Magnitude field
        """
        if len(vector_field) == 0:
            return []
        
        vector_array = np.array(vector_field)
        
        # Reshape to (n_points, 3) if needed
        if len(vector_array.shape) == 1:
            n_points = len(vector_array) // 3
            vector_array = vector_array.reshape(n_points, 3)
        elif len(vector_array.shape) > 2:
            # Take first component if it's a tensor field
            vector_array = vector_array[:, :3]
        
        # Compute magnitude
        magnitude = np.linalg.norm(vector_array, axis=1)
        
        return magnitude.tolist()
    
    def export_results(self, results: Dict[str, Any],
                      output_path: Union[str, Path],
                      format_type: Optional[str] = None) -> None:
        """
        Export results to file.
        
        Args:
            results: Processed results
            output_path: Output file path
            format_type: Export format type
        """
        if format_type is None:
            format_type = self.export_format
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'vtk':
            self._export_vtk(results, output_path)
        elif format_type == 'json':
            self._export_json(results, output_path)
        elif format_type == 'numpy':
            self._export_numpy(results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        self.logger.info(f"Results exported to {output_path}")
    
    def _export_vtk(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export results to VTK format."""
        # Simple VTK file format (basic implementation)
        with open(output_path, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Vent4D-Mech FEM Results\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Write points
            vis_data = results.get('visualization', {})
            nodes = vis_data.get('mesh', {}).get('nodes', [])
            elements = vis_data.get('mesh', {}).get('elements', [])
            
            f.write(f"POINTS {len(nodes)} float\n")
            for node in nodes:
                f.write(f"{node[0]} {node[1]} {node[2]}\n")
            
            # Write cells (simplified for tetrahedra)
            total_cell_size = sum([len(elem.get('nodes', [])) + 1 for elem in elements])
            f.write(f"CELLS {len(elements)} {total_cell_size}\n")
            for element in elements:
                element_nodes = element.get('nodes', [])
                if len(element_nodes) >= 4:
                    f.write(f"4 {element_nodes[0]} {element_nodes[1]} {element_nodes[2]} {element_nodes[3]}\n")
            
            f.write(f"CELL_TYPES {len(elements)}\n")
            for _ in elements:
                f.write("10\n")  # VTK_TETRA
    
    def _export_json(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export results to JSON format."""
        # Convert numpy arrays to lists for JSON serialization
        json_data = self._convert_to_json_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _export_numpy(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export results to NumPy format."""
        np.savez_compressed(output_path, **results)
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def get_export_formats(self) -> List[str]:
        """Get supported export formats."""
        return self.export_formats.copy()
    
    def __repr__(self) -> str:
        """String representation of the PostProcessing instance."""
        return f"PostProcessing(export_format='{self.export_format}', visualization={self.visualization})"
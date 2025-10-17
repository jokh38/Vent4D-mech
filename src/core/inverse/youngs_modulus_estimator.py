"""
Young's Modulus Estimator

This module implements inverse problem solving for estimating spatially-varying
Young's modulus distributions from observed deformation patterns in lung tissue.
"""

from typing import Optional, Dict, Any, Tuple, Union, Callable
import numpy as np
import logging
from pathlib import Path

from scipy.optimize import least_squares, minimize
from scipy.interpolate import RegularGridInterpolator

from .inverse_solver import InverseSolver
from .regularization import RegularizationMethods
from .optimization_utils import OptimizationUtils


class YoungsModulusEstimator:
    """
    Estimator for spatially-varying Young's modulus from deformation data.

    This class implements inverse problem solving to estimate Young's modulus
    distributions from observed strain patterns in lung tissue, using regularized
    optimization and physics-based constraints.

    It is designed to be decoupled from the FEM implementation via dependency
    injection. A FEM solver (either a real one like `FEMSolver` or a
    `MockFEMSolver` for testing) must be injected during initialization.

    Attributes:
        config (dict): Configuration parameters
        fem_solver (object): The injected FEM solver instance.
        inverse_solver (InverseSolver): Core inverse problem solver
        regularization (RegularizationMethods): Regularization techniques
        utils (OptimizationUtils): Optimization utilities
        logger (logging.Logger): Logger instance
        material_model (object): Constitutive model for forward problem
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, fem_solver: Optional[Any] = None):
        """
        Initialize YoungsModulusEstimator instance.

        Args:
            config: Configuration parameters.
            fem_solver: An FEM solver instance (real or mock) that has a `run_simulation` method.
        """
        self.config = config or self._get_default_config()

        # Initialize components
        self.inverse_solver = InverseSolver(self.config['solver'])
        self.regularization = RegularizationMethods(self.config['regularization'])
        self.utils = OptimizationUtils()
        self.fem_solver = fem_solver

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Material model (to be set based on configuration)
        self.material_model = None

        # Estimated parameters
        self.estimated_modulus = None
        self.optimization_results = None

        self.logger.info(f"Initialized YoungsModulusEstimator with solver: {type(fem_solver).__name__}")

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration parameters.

        Returns:
            Default configuration dictionary
        """
        return {
            'solver': {
                'method': 'least_squares',  # 'least_squares', 'trust_region', 'gradient_descent'
                'max_iterations': 1000,
                'tolerance': 1e-6,
                'step_size': 1e-3,
                'line_search': True
            },
            'regularization': {
                'method': 'tikhonov',  # 'tikhonov', 'total_variation', 'l1'
                'alpha': 0.01,  # Regularization parameter
                'operator': 'laplacian',  # 'identity', 'gradient', 'laplacian'
                'spatial_weighting': True,
                'edge_preserving': False
            },
            'material': {
                'constitutive_model': 'linear_elastic',
                'poisson_ratio': 0.45,
                'initial_modulus': 5.0,  # kPa
                'bounds': (0.1, 100.0),  # kPa
                'spatial_smoothing': True
            },
            'constraints': {
                'use_microstructure_db': True,
                'min_modulus_ratio': 0.1,
                'max_modulus_ratio': 10.0,
                'smoothness_constraint': True
            },
            'optimization': {
                'parameterization': 'log_space',  # 'linear', 'log_space'
                'gradient_method': 'analytical',  # 'analytical', 'finite_difference'
                'parallel_evaluation': True,
                'batch_size': 100
            }
        }

    def estimate_modulus(self, observed_strain: np.ndarray,
                        deformation_gradient: np.ndarray,
                        dvf: np.ndarray,
                        voxel_spacing: Tuple[float, float, float],
                        initial_guess: Optional[np.ndarray] = None,
                        mask: Optional[np.ndarray] = None,
                        material_model: Optional[object] = None) -> Dict[str, Any]:
        """
        Estimate Young's modulus distribution from observed strain.

        Args:
            observed_strain: Observed strain tensor field (D, H, W, 3, 3)
            deformation_gradient: Deformation gradient field (D, H, W, 3, 3)
            initial_guess: Initial guess for modulus distribution
            mask: Region of interest mask
            material_model: Constitutive model for forward problem

        Returns:
            Dictionary containing estimation results

        Raises:
            ValueError: If input dimensions are invalid
        """
        self.logger.info("Starting Young's modulus estimation...")

        # Validate inputs
        self._validate_inputs(observed_strain, deformation_gradient)

        # Set material model
        if material_model is not None:
            self.material_model = material_model

        # Apply mask if provided
        if mask is not None:
            observed_strain = observed_strain * mask[..., np.newaxis, np.newaxis]
            deformation_gradient = deformation_gradient * mask[..., np.newaxis, np.newaxis]

        # Prepare initial guess
        if initial_guess is None:
            initial_guess = self._prepare_initial_guess(observed_strain.shape[:3])

        # Set up optimization problem
        problem_data = {
            'observed_strain': observed_strain,
            'deformation_gradient': deformation_gradient,
            'mask': mask,
            'material_model': self.material_model,
            'dvf': dvf,
            'voxel_spacing': voxel_spacing
        }

        # Solve inverse problem
        try:
            if self.config['solver']['method'] == 'least_squares':
                results = self._solve_least_squares(initial_guess, problem_data)
            else:
                results = self._solve_general_optimization(initial_guess, problem_data)

            # Process results
            self.estimated_modulus = self._process_solution(results['x'], observed_strain.shape[:3])
            self.optimization_results = results

            # Compute quality metrics
            quality_metrics = self._compute_quality_metrics(
                self.estimated_modulus, observed_strain, problem_data
            )

            # Prepare output
            output = {
                'youngs_modulus': self.estimated_modulus,
                'optimization_results': results,
                'quality_metrics': quality_metrics,
                'residual_strain': self._compute_residual_strain(
                    self.estimated_modulus, problem_data
                ),
                'config': self.config
            }

            self.logger.info("Young's modulus estimation completed successfully")
            return output

        except Exception as e:
            self.logger.error(f"Young's modulus estimation failed: {str(e)}")
            raise

    def _validate_inputs(self, observed_strain: np.ndarray, deformation_gradient: np.ndarray) -> None:
        """
        Validate input dimensions and data.

        Args:
            observed_strain: Observed strain tensor
            deformation_gradient: Deformation gradient tensor

        Raises:
            ValueError: If inputs are invalid
        """
        if observed_strain.ndim != 5 or observed_strain.shape[-2:] != (3, 3):
            raise ValueError(f"Observed strain must be 5D with shape (D, H, W, 3, 3), got {observed_strain.shape}")

        if deformation_gradient.ndim != 5 or deformation_gradient.shape[-2:] != (3, 3):
            raise ValueError(f"Deformation gradient must be 5D with shape (D, H, W, 3, 3), got {deformation_gradient.shape}")

        if observed_strain.shape != deformation_gradient.shape:
            raise ValueError("Observed strain and deformation gradient must have same shape")

        if np.any(np.isnan(observed_strain)) or np.any(np.isnan(deformation_gradient)):
            raise ValueError("Input data contains NaN values")

    def _prepare_initial_guess(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Prepare initial guess for Young's modulus distribution.

        Args:
            shape: Shape of the 3D volume

        Returns:
            Initial guess array
        """
        initial_modulus = self.config['material']['initial_modulus']

        if self.config['optimization']['parameterization'] == 'log_space':
            return np.log(initial_modulus) * np.ones(shape)
        else:
            return initial_modulus * np.ones(shape)

    def _solve_least_squares(self, initial_guess: np.ndarray,
                           problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve inverse problem using least squares optimization.

        Args:
            initial_guess: Initial parameter guess
            problem_data: Problem data dictionary

        Returns:
            Optimization results
        """
        # Flatten initial guess
        x0 = initial_guess.flatten()

        # Define bounds
        bounds = self.config['material']['bounds']
        if self.config['optimization']['parameterization'] == 'log_space':
            bounds = (np.log(bounds[0]), np.log(bounds[1]))

        bounds_lower = bounds[0] * np.ones_like(x0)
        bounds_upper = bounds[1] * np.ones_like(x0)
        bounds = (bounds_lower, bounds_upper)

        # Define residual function
        def residual_function(x):
            return self._compute_residual_vector(x, problem_data)

        # Solve least squares problem
        results = least_squares(
            residual_function,
            x0,
            bounds=bounds,
            method='trf',  # Trust Region Reflective
            ftol=self.config['solver']['tolerance'],
            xtol=self.config['solver']['tolerance'],
            gtol=self.config['solver']['tolerance'],
            max_nfev=self.config['solver']['max_iterations'],
            verbose=2,
            jac=self._compute_jacobian if self.config['optimization']['gradient_method'] == 'analytical' else '2-point'
        )

        return results

    def _solve_general_optimization(self, initial_guess: np.ndarray,
                                  problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve inverse problem using general optimization.

        Args:
            initial_guess: Initial parameter guess
            problem_data: Problem data dictionary

        Returns:
            Optimization results
        """
        # Flatten initial guess
        x0 = initial_guess.flatten()

        # Define objective function
        def objective_function(x):
            residual = self._compute_residual_vector(x, problem_data)
            return np.sum(residual**2)

        # Define bounds
        bounds = self.config['material']['bounds']
        if self.config['optimization']['parameterization'] == 'log_space':
            bounds = (np.log(bounds[0]), np.log(bounds[1]))

        bounds = [bounds] * len(x0)

        # Solve optimization problem
        results = minimize(
            objective_function,
            x0,
            bounds=bounds,
            method='L-BFGS-B',
            tol=self.config['solver']['tolerance'],
            options={'maxiter': self.config['solver']['max_iterations']}
        )

        return results

    def _compute_residual_vector(self, x: np.ndarray, problem_data: Dict[str, Any]) -> np.ndarray:
        """
        Compute residual vector for optimization.

        Args:
            x: Parameter vector (flattened Young's modulus)
            problem_data: Problem data dictionary

        Returns:
            Residual vector
        """
        # Reshape parameter vector
        shape = problem_data['observed_strain'].shape[:3]
        modulus_field = x.reshape(shape)

        # Apply parameterization transformation
        if self.config['optimization']['parameterization'] == 'log_space':
            modulus_field = np.exp(modulus_field)

        # Apply mask if provided
        if problem_data['mask'] is not None:
            modulus_field = modulus_field * problem_data['mask']

        # Compute forward problem
        predicted_strain = self._solve_forward_problem(modulus_field, problem_data)

        # Compute residual
        observed_strain = problem_data['observed_strain']
        residual = predicted_strain - observed_strain

        # Flatten residual
        residual_vector = residual.flatten()

        # Add regularization term
        regularization_term = self.regularization.compute_regularization(
            modulus_field, problem_data
        )
        residual_vector = np.concatenate([residual_vector, regularization_term.flatten()])

        return residual_vector

    def _solve_forward_problem(self, modulus_field: np.ndarray,
                             problem_data: Dict[str, Any]) -> np.ndarray:
        """
        Solve forward problem to compute strain from modulus using the injected FEM solver.
        """
        if not hasattr(self, 'fem_solver') or self.fem_solver is None:
            raise RuntimeError("FEM solver is not injected into YoungsModulusEstimator.")

        self.logger.info("Solving forward problem using injected FEM solver...")

        # Prepare material properties for the FEM solver
        material_properties = {
            'youngs_modulus': modulus_field,
            'poisson_ratio': np.full_like(modulus_field, self.config['material']['poisson_ratio'])
        }

        # The FEM workflow requires the full DVF for boundary conditions,
        # and a lung mask for mesh generation. These should be in problem_data.
        lung_mask = problem_data.get('mask')
        dvf = problem_data.get('dvf')
        voxel_spacing = problem_data.get('voxel_spacing')

        if lung_mask is None or dvf is None or voxel_spacing is None:
            raise ValueError("Forward problem requires lung_mask, dvf, and voxel_spacing in problem_data.")

        # Run the full FEM simulation
        fem_results = self.fem_solver.run_simulation(
            lung_mask=lung_mask,
            displacement_field=dvf,
            material_properties=material_properties,
            voxel_spacing=voxel_spacing
        )

        # The FEM results should contain the computed strain field.
        # This depends on the post-processing implementation.
        predicted_strain = fem_results['processed_results'].get('strain_tensor')

        if predicted_strain is None:
            raise RuntimeError("FEM simulation did not produce a 'strain_tensor' in its results.")

        return predicted_strain

    def _compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for analytical gradient calculation.

        Args:
            x: Parameter vector

        Returns:
            Jacobian matrix
        """
        # This is a placeholder for analytical Jacobian computation
        # In practice, this would involve sensitivity analysis
        return None  # Use numerical differentiation

    def _process_solution(self, x: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Process optimization solution to final modulus distribution.

        Args:
            x: Optimized parameter vector
            shape: Target shape

        Returns:
            Processed Young's modulus distribution
        """
        modulus_field = x.reshape(shape)

        # Apply parameterization transformation
        if self.config['optimization']['parameterization'] == 'log_space':
            modulus_field = np.exp(modulus_field)

        # Apply smoothing if requested
        if self.config['material']['spatial_smoothing']:
            from scipy.ndimage import gaussian_filter
            modulus_field = gaussian_filter(modulus_field, sigma=1.0)

        return modulus_field

    def _compute_quality_metrics(self, estimated_modulus: np.ndarray,
                               observed_strain: np.ndarray,
                               problem_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute quality metrics for the estimation.

        Args:
            estimated_modulus: Estimated Young's modulus
            observed_strain: Observed strain field
            problem_data: Problem data dictionary

        Returns:
            Quality metrics dictionary
        """
        # Compute residual strain
        residual_strain = self._compute_residual_strain(estimated_modulus, problem_data)

        # Compute metrics
        residual_norm = np.linalg.norm(residual_strain)
        residual_mean = np.mean(residual_strain)
        residual_std = np.std(residual_strain)

        # Compute coefficient of determination (R²)
        ss_res = np.sum(residual_strain**2)
        ss_tot = np.sum((observed_strain - np.mean(observed_strain))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Compute modulus statistics
        modulus_mean = np.mean(estimated_modulus)
        modulus_std = np.std(estimated_modulus)
        modulus_range = (np.min(estimated_modulus), np.max(estimated_modulus))

        return {
            'residual_norm': residual_norm,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'r_squared': r_squared,
            'modulus_mean': modulus_mean,
            'modulus_std': modulus_std,
            'modulus_range': modulus_range
        }

    def _compute_residual_strain(self, modulus_field: np.ndarray,
                               problem_data: Dict[str, Any]) -> np.ndarray:
        """
        Compute residual strain field.

        Args:
            modulus_field: Young's modulus field
            problem_data: Problem data dictionary

        Returns:
            Residual strain field
        """
        # Solve forward problem with estimated modulus
        predicted_strain = self._solve_forward_problem(modulus_field, problem_data)
        observed_strain = problem_data['observed_strain']

        return predicted_strain - observed_strain

    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save estimation results to files.

        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.estimated_modulus is not None:
            # Save Young's modulus map
            import nibabel as nib
            modulus_img = nib.Nifti1Image(self.estimated_modulus, np.eye(4))
            nib.save(modulus_img, output_dir / 'youngs_modulus.nii.gz')

        if self.optimization_results is not None:
            # Save optimization results
            import json
            results_data = {
                'optimization_results': {
                    'success': self.optimization_results.get('success', False),
                    'message': self.optimization_results.get('message', ''),
                    'fun': float(self.optimization_results.get('fun', 0.0)) if hasattr(self.optimization_results.get('fun'), '__float__') else self.optimization_results.get('fun'),
                    'nit': self.optimization_results.get('nit', 0),
                    'nfev': self.optimization_results.get('nfev', 0)
                },
                'config': self.config,
                'timestamp': str(np.datetime64('now'))
            }
            with open(output_dir / 'optimization_results.json', 'w') as f:
                json.dump(results_data, f, indent=2)

        self.logger.info(f"Results saved to {output_dir}")

    def get_estimated_modulus(self) -> Optional[np.ndarray]:
        """
        Get estimated Young's modulus distribution.

        Returns:
            Estimated Young's modulus or None if not available
        """
        return self.estimated_modulus

    def get_optimization_results(self) -> Optional[Dict[str, Any]]:
        """
        Get optimization results.

        Returns:
            Optimization results or None if not available
        """
        return self.optimization_results

    def __repr__(self) -> str:
        """String representation of the YoungsModulusEstimator instance."""
        return (f"YoungsModulusEstimator(solver='{self.config['solver']['method']}', "
                f"regularization='{self.config['regularization']['method']}', "
                f"model='{self.config['material']['constitutive_model']}')")
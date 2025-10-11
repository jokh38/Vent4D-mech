"""
Optimization Utilities

This module provides utility functions and classes for optimization problems
in biomechanics, particularly for inverse problem solving and parameter estimation.
"""

from typing import Dict, Any, Optional, Callable, Union, Tuple
import numpy as np
import logging
from scipy.optimize import line_search
from scipy.linalg import svd


class OptimizationUtils:
    """
    Utility functions for optimization problems.
    
    This class provides various utility functions for optimization, including
    parameter transformations, constraint handling, and numerical methods
    for inverse problem solving in biomechanics.
    
    Attributes:
        logger (logging.Logger): Logger instance
    """
    
    def __init__(self):
        """Initialize OptimizationUtils instance."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized OptimizationUtils")
    
    def parameter_transform_log(self, parameters: np.ndarray, 
                               forward: bool = True) -> np.ndarray:
        """
        Apply logarithmic parameter transformation.
        
        Args:
            parameters: Parameter vector
            forward: If True, apply log transform; if False, apply exp transform
            
        Returns:
            Transformed parameters
        """
        if forward:
            # Apply bounds to avoid log(0)
            epsilon = 1e-12
            return np.log(np.maximum(parameters, epsilon))
        else:
            return np.exp(parameters)
    
    def parameter_transform_box(self, parameters: np.ndarray,
                              bounds: Tuple[float, float],
                              forward: bool = True) -> np.ndarray:
        """
        Apply box constraint parameter transformation.
        
        Args:
            parameters: Parameter vector
            bounds: Lower and upper bounds
            forward: If True, transform to unconstrained space
            
        Returns:
            Transformed parameters
        """
        lower, upper = bounds
        
        if forward:
            # Transform from [lower, upper] to (-inf, inf)
            # Using logit transform: log((x - lower) / (upper - x))
            epsilon = 1e-12
            x_clipped = np.clip(parameters, lower + epsilon, upper - epsilon)
            return np.log((x_clipped - lower) / (upper - x_clipped))
        else:
            # Transform from (-inf, inf) to [lower, upper]
            # Using sigmoid transform: lower + (upper - lower) / (1 + exp(-x))
            return lower + (upper - lower) / (1 + np.exp(-parameters))
    
    def compute_jacobian_finite_difference(self, func: Callable,
                                         x: np.ndarray,
                                         epsilon: float = 1e-8) -> np.ndarray:
        """
        Compute Jacobian matrix using finite differences.
        
        Args:
            func: Function returning vector output
            x: Point at which to compute Jacobian
            epsilon: Finite difference step size
            
        Returns:
            Jacobian matrix
        """
        f0 = func(x)
        if isinstance(f0, (int, float)):
            f0 = np.array([f0])
        
        jacobian = np.zeros((len(f0), len(x)))
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            
            f_plus = func(x_plus)
            if isinstance(f_plus, (int, float)):
                f_plus = np.array([f_plus])
            
            jacobian[:, i] = (f_plus - f0) / epsilon
        
        return jacobian
    
    def compute_hessian_finite_difference(self, func: Callable,
                                        x: np.ndarray,
                                        epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute Hessian matrix using finite differences.
        
        Args:
            func: Function returning scalar output
            x: Point at which to compute Hessian
            epsilon: Finite difference step size
            
        Returns:
            Hessian matrix
        """
        n = len(x)
        hessian = np.zeros((n, n))
        
        # Compute diagonal elements
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            f0 = func(x)
            
            hessian[i, i] = (f_plus - 2*f0 + f_minus) / (epsilon**2)
        
        # Compute off-diagonal elements
        for i in range(n):
            for j in range(i+1, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += epsilon
                x_pp[j] += epsilon
                
                x_pm[i] += epsilon
                x_pm[j] -= epsilon
                
                x_mp[i] -= epsilon
                x_mp[j] += epsilon
                
                x_mm[i] -= epsilon
                x_mm[j] -= epsilon
                
                f_pp = func(x_pp)
                f_pm = func(x_pm)
                f_mp = func(x_mp)
                f_mm = func(x_mm)
                
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
                hessian[j, i] = hessian[i, j]
        
        return hessian
    
    def regularization_matrix_laplacian(self, shape: Tuple[int, ...],
                                      stencil: str = '7_point') -> np.ndarray:
        """
        Generate Laplacian regularization matrix.
        
        Args:
            shape: Shape of the parameter field
            stencil: Type of stencil ('7_point', '27_point')
            
        Returns:
            Regularization matrix
        """
        n_pixels = np.prod(shape)
        
        if stencil == '7_point' and len(shape) == 3:
            # 7-point stencil for 3D
            matrix = np.zeros((n_pixels, n_pixels))
            
            # Helper function to convert 3D index to 1D
            def idx_3d_to_1d(i, j, k):
                return i * shape[1] * shape[2] + j * shape[2] + k
            
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        idx = idx_3d_to_1d(i, j, k)
                        
                        # Center point
                        matrix[idx, idx] = -6
                        
                        # 6 neighbors
                        neighbors = [
                            (i-1, j, k), (i+1, j, k),
                            (i, j-1, k), (i, j+1, k),
                            (i, j, k-1), (i, j, k+1)
                        ]
                        
                        for ni, nj, nk in neighbors:
                            if (0 <= ni < shape[0] and 
                                0 <= nj < shape[1] and 
                                0 <= nk < shape[2]):
                                neighbor_idx = idx_3d_to_1d(ni, nj, nk)
                                matrix[idx, neighbor_idx] = 1
        else:
            # Simplified identity-based regularization for other cases
            matrix = np.eye(n_pixels)
        
        return matrix
    
    def regularization_matrix_total_variation(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Generate total variation regularization operator.
        
        Args:
            shape: Shape of the parameter field
            
        Returns:
            TV regularization operator
        """
        if len(shape) == 3:
            # 3D total variation
            n_pixels = np.prod(shape)
            
            # Gradient operators
            grad_x = np.zeros((n_pixels, n_pixels))
            grad_y = np.zeros((n_pixels, n_pixels))
            grad_z = np.zeros((n_pixels, n_pixels))
            
            def idx_3d_to_1d(i, j, k):
                return i * shape[1] * shape[2] + j * shape[2] + k
            
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        idx = idx_3d_to_1d(i, j, k)
                        
                        # X-direction gradient
                        if i < shape[0] - 1:
                            idx_next = idx_3d_to_1d(i+1, j, k)
                            grad_x[idx, idx] = -1
                            grad_x[idx, idx_next] = 1
                        
                        # Y-direction gradient
                        if j < shape[1] - 1:
                            idx_next = idx_3d_to_1d(i, j+1, k)
                            grad_y[idx, idx] = -1
                            grad_y[idx, idx_next] = 1
                        
                        # Z-direction gradient
                        if k < shape[2] - 1:
                            idx_next = idx_3d_to_1d(i, j, k+1)
                            grad_z[idx, idx] = -1
                            grad_z[idx, idx_next] = 1
            
            # Stack gradient operators
            tv_operator = np.vstack([grad_x, grad_y, grad_z])
            
        else:
            # Simplified case
            n_pixels = np.prod(shape)
            tv_operator = np.eye(n_pixels)
        
        return tv_operator
    
    def line_search_armijo(self, func: Callable,
                          x: np.ndarray,
                          direction: np.ndarray,
                          gradient: np.ndarray,
                          initial_step: float = 1.0,
                          alpha: float = 0.5,
                          beta: float = 0.8) -> float:
        """
        Perform Armijo line search.
        
        Args:
            func: Objective function
            x: Current point
            direction: Search direction
            gradient: Current gradient
            initial_step: Initial step size
            alpha: Backtracking factor
            beta: Sufficient decrease parameter
            
        Returns:
            Step size
        """
        step = initial_step
        f_current = func(x)
        
        # Ensure direction is descent direction
        if np.dot(gradient, direction) >= 0:
            direction = -gradient
        
        while step > 1e-10:
            x_new = x + step * direction
            f_new = func(x_new)
            
            # Armijo condition
            expected_decrease = beta * step * np.dot(gradient, direction)
            
            if f_current - f_new >= expected_decrease:
                break
            
            step *= alpha
        
        return step
    
    def compute_condition_number(self, matrix: np.ndarray) -> float:
        """
        Compute condition number of a matrix.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Condition number
        """
        try:
            singular_values = svd(matrix, compute_uv=False)
            return np.max(singular_values) / np.max([np.min(singular_values), 1e-12])
        except:
            return np.inf
    
    def adaptive_regularization_parameter(self, residual_norm: float,
                                        regularization_norm: float,
                                        target_ratio: float = 1.0) -> float:
        """
        Compute adaptive regularization parameter using discrepancy principle.
        
        Args:
            residual_norm: Norm of data misfit
            regularization_norm: Norm of regularization term
            target_ratio: Target ratio between terms
            
        Returns:
            Regularization parameter
        """
        if regularization_norm < 1e-12:
            return 0.01
        
        # Adjust regularization to achieve target ratio
        alpha = (residual_norm / regularization_norm) * target_ratio
        
        # Bound the regularization parameter
        alpha = np.clip(alpha, 1e-6, 1e6)
        
        return alpha
    
    def multi_objective_weighting(self, objectives: list,
                                weights: Optional[list] = None) -> Callable:
        """
        Create multi-objective function with weights.
        
        Args:
            objectives: List of objective functions
            weights: List of weights for each objective
            
        Returns:
            Weighted multi-objective function
        """
        if weights is None:
            weights = [1.0] * len(objectives)
        
        if len(objectives) != len(weights):
            raise ValueError("Number of objectives must match number of weights")
        
        def weighted_objective(x):
            total = 0.0
            for obj_func, weight in zip(objectives, weights):
                value = obj_func(x)
                if isinstance(value, np.ndarray):
                    value = np.sum(value**2)
                total += weight * value
            return total
        
        return weighted_objective
    
    def convergence_criterion(self, x_current: np.ndarray,
                            x_previous: np.ndarray,
                            f_current: float,
                            f_previous: float,
                            tol_x: float = 1e-6,
                            tol_f: float = 1e-8) -> bool:
        """
        Check convergence criteria.
        
        Args:
            x_current: Current parameter vector
            x_previous: Previous parameter vector
            f_current: Current objective value
            f_previous: Previous objective value
            tol_x: Parameter tolerance
            tol_f: Function tolerance
            
        Returns:
            True if converged
        """
        # Parameter convergence
        param_change = np.linalg.norm(x_current - x_previous)
        param_norm = np.linalg.norm(x_current)
        param_converged = param_change < tol_x * (1 + param_norm)
        
        # Function convergence
        func_change = abs(f_current - f_previous)
        func_norm = abs(f_current)
        func_converged = func_change < tol_f * (1 + func_norm)
        
        return param_converged or func_converged
    
    def __repr__(self) -> str:
        """String representation of the OptimizationUtils instance."""
        return "OptimizationUtils()"
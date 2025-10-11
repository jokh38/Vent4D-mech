"""
Inverse Problem Solver

This module provides core inverse problem solving capabilities for estimating
material parameters from observed deformation data in biomechanics applications.
"""

from typing import Dict, Any, Optional, Callable, Union
import numpy as np
import logging
from scipy.optimize import least_squares, minimize


class InverseSolver:
    """
    Core inverse problem solver for parameter estimation.
    
    This class provides a unified interface for various inverse problem solving
    methods including least squares optimization, trust region methods, and
    gradient descent algorithms for estimating material parameters.
    
    Attributes:
        config (dict): Configuration parameters
        logger (logging.Logger): Logger instance
        solver_method (str): Current solver method
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize InverseSolver instance.
        
        Args:
            config: Solver configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.solver_method = config.get('method', 'least_squares')
        
        # Initialize solver state
        self.current_iteration = 0
        self.best_solution = None
        self.best_residual = np.inf
        
        self.logger.info(f"Initialized InverseSolver with method: {self.solver_method}")
    
    def solve(self, objective_func: Callable,
              initial_guess: np.ndarray,
              bounds: Optional[tuple] = None,
              constraints: Optional[list] = None,
              callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Solve inverse problem using configured method.
        
        Args:
            objective_func: Objective function to minimize
            initial_guess: Initial parameter guess
            bounds: Parameter bounds (lower, upper)
            constraints: Optimization constraints
            callback: Callback function for progress monitoring
            
        Returns:
            Optimization results dictionary
        """
        self.logger.info(f"Starting inverse problem solving with {self.solver_method}")
        
        if self.solver_method == 'least_squares':
            return self._solve_least_squares(objective_func, initial_guess, bounds, callback)
        elif self.solver_method == 'trust_region':
            return self._solve_trust_region(objective_func, initial_guess, bounds, callback)
        elif self.solver_method == 'gradient_descent':
            return self._solve_gradient_descent(objective_func, initial_guess, bounds, callback)
        else:
            raise ValueError(f"Unsupported solver method: {self.solver_method}")
    
    def _solve_least_squares(self, objective_func: Callable,
                           initial_guess: np.ndarray,
                           bounds: Optional[tuple] = None,
                           callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Solve using least squares optimization.
        
        Args:
            objective_func: Residual function
            initial_guess: Initial parameter guess
            bounds: Parameter bounds
            callback: Progress callback
            
        Returns:
            Optimization results
        """
        # Enhanced callback for iteration tracking
        def enhanced_callback(xk):
            self.current_iteration += 1
            if callback is not None:
                callback(xk)
            
            # Track best solution
            residual = np.linalg.norm(objective_func(xk))
            if residual < self.best_residual:
                self.best_residual = residual
                self.best_solution = xk.copy()
        
        results = least_squares(
            objective_func,
            initial_guess,
            bounds=bounds,
            method='trf',
            ftol=self.config.get('tolerance', 1e-6),
            xtol=self.config.get('tolerance', 1e-6),
            gtol=self.config.get('tolerance', 1e-6),
            max_nfev=self.config.get('max_iterations', 1000),
            verbose=0,
            callback=enhanced_callback
        )
        
        return self._format_results(results)
    
    def _solve_trust_region(self, objective_func: Callable,
                          initial_guess: np.ndarray,
                          bounds: Optional[tuple] = None,
                          callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Solve using trust region methods.
        
        Args:
            objective_func: Objective function
            initial_guess: Initial parameter guess
            bounds: Parameter bounds
            callback: Progress callback
            
        Returns:
            Optimization results
        """
        # Define wrapper for minimization
        def objective_wrapper(x):
            residuals = objective_func(x)
            return np.sum(residuals**2)
        
        results = minimize(
            objective_wrapper,
            initial_guess,
            method='trust-constr',
            bounds=bounds,
            tol=self.config.get('tolerance', 1e-6),
            options={'maxiter': self.config.get('max_iterations', 1000)},
            callback=callback
        )
        
        return self._format_results(results)
    
    def _solve_gradient_descent(self, objective_func: Callable,
                              initial_guess: np.ndarray,
                              bounds: Optional[tuple] = None,
                              callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Solve using gradient descent optimization.
        
        Args:
            objective_func: Objective function
            initial_guess: Initial parameter guess
            bounds: Parameter bounds
            callback: Progress callback
            
        Returns:
            Optimization results
        """
        # Simple gradient descent implementation
        x = initial_guess.copy()
        step_size = self.config.get('step_size', 1e-3)
        max_iterations = self.config.get('max_iterations', 1000)
        tolerance = self.config.get('tolerance', 1e-6)
        
        # Use line search if configured
        line_search = self.config.get('line_search', True)
        
        for iteration in range(max_iterations):
            self.current_iteration = iteration
            
            # Compute objective and gradient
            current_objective = objective_func(x)
            if isinstance(current_objective, np.ndarray):
                current_objective = np.sum(current_objective**2)
            
            # Numerical gradient
            gradient = self._compute_numerical_gradient(objective_func, x)
            
            # Line search for step size
            if line_search:
                step_size = self._line_search(objective_func, x, gradient, step_size)
            
            # Update parameters
            x_new = x - step_size * gradient
            
            # Apply bounds if provided
            if bounds is not None:
                x_new = np.clip(x_new, bounds[0], bounds[1])
            
            # Check convergence
            if np.linalg.norm(x_new - x) < tolerance:
                break
            
            x = x_new
            
            if callback is not None:
                callback(x)
        
        # Return results in scipy format
        results = {
            'x': x,
            'success': True,
            'fun': current_objective,
            'nit': iteration + 1,
            'message': 'Gradient descent converged'
        }
        
        return self._format_results(results)
    
    def _compute_numerical_gradient(self, objective_func: Callable,
                                  x: np.ndarray,
                                  epsilon: float = 1e-8) -> np.ndarray:
        """
        Compute numerical gradient using finite differences.
        
        Args:
            objective_func: Objective function
            x: Current parameter vector
            epsilon: Finite difference step size
            
        Returns:
            Gradient vector
        """
        gradient = np.zeros_like(x)
        base_value = objective_func(x)
        
        if isinstance(base_value, np.ndarray):
            base_value = np.sum(base_value**2)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            
            value_plus = objective_func(x_plus)
            if isinstance(value_plus, np.ndarray):
                value_plus = np.sum(value_plus**2)
            
            gradient[i] = (value_plus - base_value) / epsilon
        
        return gradient
    
    def _line_search(self, objective_func: Callable,
                    x: np.ndarray,
                    gradient: np.ndarray,
                    initial_step_size: float,
                    max_iterations: int = 20) -> float:
        """
        Perform backtracking line search.
        
        Args:
            objective_func: Objective function
            x: Current parameter vector
            gradient: Search direction
            initial_step_size: Initial step size
            max_iterations: Maximum line search iterations
            
        Returns:
            Optimal step size
        """
        step_size = initial_step_size
        alpha = 0.5  # Backtracking factor
        beta = 0.8   # Sufficient decrease parameter
        
        current_value = objective_func(x)
        if isinstance(current_value, np.ndarray):
            current_value = np.sum(current_value**2)
        
        for _ in range(max_iterations):
            x_new = x - step_size * gradient
            
            new_value = objective_func(x_new)
            if isinstance(new_value, np.ndarray):
                new_value = np.sum(new_value**2)
            
            # Armijo condition
            expected_decrease = beta * step_size * np.dot(gradient, gradient)
            
            if current_value - new_value >= expected_decrease:
                break
            
            step_size *= alpha
        
        return step_size
    
    def _format_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format optimization results consistently.
        
        Args:
            results: Raw optimization results
            
        Returns:
            Formatted results dictionary
        """
        formatted = {
            'x': results.get('x'),
            'success': results.get('success', False),
            'message': results.get('message', ''),
            'fun': results.get('fun', 0.0),
            'nit': results.get('nit', 0),
            'nfev': results.get('nfev', 0),
            'method': self.solver_method,
            'config': self.config
        }
        
        # Add method-specific results
        if hasattr(results, 'cost'):
            formatted['cost'] = results.cost
        if hasattr(results, 'jac'):
            formatted['jac'] = results.jac
        if hasattr(results, 'optimality'):
            formatted['optimality'] = results.optimality
        
        return formatted
    
    def get_solver_info(self) -> Dict[str, Any]:
        """
        Get information about the current solver configuration.
        
        Returns:
            Solver information dictionary
        """
        return {
            'method': self.solver_method,
            'config': self.config,
            'current_iteration': self.current_iteration,
            'best_residual': self.best_residual,
            'has_solution': self.best_solution is not None
        }
    
    def reset(self) -> None:
        """Reset solver state."""
        self.current_iteration = 0
        self.best_solution = None
        self.best_residual = np.inf
        self.logger.info("Reset inverse solver state")
    
    def __repr__(self) -> str:
        """String representation of the InverseSolver instance."""
        return f"InverseSolver(method='{self.solver_method}', tolerance={self.config.get('tolerance', 1e-6)})"
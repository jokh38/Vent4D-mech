"""
Material Fitting

This module provides material parameter estimation capabilities for fitting
constitutive models to experimental data using various optimization methods.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import warnings
import logging

# Try to import numpy and scipy, but make them optional
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Some functionality will be limited.")

try:
    from scipy.optimize import minimize, curve_fit, differential_evolution
    from scipy.stats import linregress
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced fitting methods will be limited.")

from .constitutive_models import ConstitutiveModel, create_model


class MaterialFitter:
    """
    Material parameter fitter for constitutive models.

    This class provides comprehensive parameter estimation capabilities for
    fitting constitutive models to experimental stress-strain data using
    various optimization methods.
    """

    def __init__(self, model_type: str, logger: Optional[logging.Logger] = None):
        """
        Initialize MaterialFitter.

        Args:
            model_type: Type of constitutive model to fit
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_type = model_type
        self.fitting_history = []
        self.current_fit = None

        # Validate model type
        try:
            self.test_model = create_model(model_type)
        except ValueError as e:
            raise ValueError(f"Invalid model type: {model_type}. {e}")

    def fit_uniaxial_data(self, strain_data: List[float], stress_data: List[float],
                         method: str = 'least_squares', bounds: Optional[Dict[str, tuple]] = None,
                         initial_guess: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Fit material parameters to uniaxial stress-strain data.

        Args:
            strain_data: Engineering strain values
            stress_data: Engineering stress values (kPa)
            method: Fitting method ('least_squares', 'differential_evolution', 'manual')
            bounds: Parameter bounds for optimization
            initial_guess: Initial parameter guesses

        Returns:
            Fitting results dictionary

        Raises:
            ImportError: If required libraries are not available
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for material fitting")

        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for material fitting")

        self.logger.info(f"Fitting {self.model_type} model to uniaxial data using {method}")

        # Convert to numpy arrays
        strain = np.array(strain_data)
        stress = np.array(stress_data)

        # Validate data
        if len(strain) != len(stress):
            raise ValueError("Strain and stress data must have same length")

        if len(strain) < 3:
            raise ValueError("Need at least 3 data points for fitting")

        # Define objective function
        def objective_function(params):
            """Objective function for optimization."""
            # Create model with current parameters
            param_dict = self._array_to_dict(params)
            model = create_model(self.model_type, **param_dict)

            # Convert to 3D strain tensors (uniaxial condition)
            strain_tensors = self._uniaxial_strain_to_tensor(strain)

            # Compute stress
            try:
                predicted_stress = []
                for strain_tensor in strain_tensors:
                    stress_tensor = model.compute_stress(strain_tensor)
                    # Extract axial stress component
                    predicted_stress.append(stress_tensor[0, 0])

                predicted_stress = np.array(predicted_stress)

                # Compute residual
                residual = predicted_stress - stress
                return np.sum(residual**2)

            except Exception as e:
                self.logger.warning(f"Error computing stress: {e}")
                return 1e10  # Large penalty for invalid parameters

        # Get parameter bounds and initial guess
        param_names, param_bounds, param_initial = self._get_fitting_parameters(bounds, initial_guess)

        # Perform fitting
        if method == 'least_squares':
            result = minimize(
                objective_function,
                x0=param_initial,
                bounds=param_bounds,
                method='L-BFGS-B'
            )
        elif method == 'differential_evolution':
            result = differential_evolution(
                objective_function,
                bounds=param_bounds,
                maxiter=1000,
                popsize=15
            )
        else:
            raise ValueError(f"Unknown fitting method: {method}")

        # Process results
        fitted_params = self._array_to_dict(result.x)
        model = create_model(self.model_type, **fitted_params)

        # Calculate goodness of fit
        strain_tensors = self._uniaxial_strain_to_tensor(strain)
        predicted_stress = []
        for strain_tensor in strain_tensors:
            stress_tensor = model.compute_stress(strain_tensor)
            predicted_stress.append(stress_tensor[0, 0])
        predicted_stress = np.array(predicted_stress)

        r_squared = self._calculate_r_squared(stress, predicted_stress)
        rmse = np.sqrt(np.mean((stress - predicted_stress)**2))
        mae = np.mean(np.abs(stress - predicted_stress))

        fit_results = {
            'model_type': self.model_type,
            'fitted_parameters': fitted_params,
            'optimization_result': result,
            'goodness_of_fit': {
                'r_squared': r_squared,
                'rmse': rmse,
                'mae': mae
            },
            'data': {
                'strain': strain_data,
                'stress': stress_data,
                'predicted_stress': predicted_stress.tolist()
            },
            'method': method
        }

        self.current_fit = fit_results
        self.fitting_history.append(fit_results)

        self.logger.info(f"Fitting completed. R² = {r_squared:.4f}, RMSE = {rmse:.4f} kPa")

        return fit_results

    def fit_biaxial_data(self, strain_x: List[float], strain_y: List[float],
                        stress_x: List[float], stress_y: List[float],
                        method: str = 'least_squares') -> Dict[str, Any]:
        """
        Fit material parameters to biaxial stress-strain data.

        Args:
            strain_x: Engineering strain in x-direction
            strain_y: Engineering strain in y-direction
            stress_x: Engineering stress in x-direction (kPa)
            stress_y: Engineering stress in y-direction (kPa)
            method: Fitting method

        Returns:
            Fitting results dictionary
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for material fitting")

        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for material fitting")

        self.logger.info(f"Fitting {self.model_type} model to biaxial data using {method}")

        # Convert to numpy arrays
        strain_x_arr = np.array(strain_x)
        strain_y_arr = np.array(strain_y)
        stress_x_arr = np.array(stress_x)
        stress_y_arr = np.array(stress_y)

        # Validate data
        if not (len(strain_x) == len(strain_y) == len(stress_x) == len(stress_y)):
            raise ValueError("All data arrays must have same length")

        # Define objective function
        def objective_function(params):
            """Objective function for biaxial fitting."""
            param_dict = self._array_to_dict(params)
            model = create_model(self.model_type, **param_dict)

            # Convert to 3D strain tensors (biaxial condition)
            strain_tensors = self._biaxial_strain_to_tensor(strain_x_arr, strain_y_arr)

            try:
                predicted_stress_x = []
                predicted_stress_y = []

                for strain_tensor in strain_tensors:
                    stress_tensor = model.compute_stress(strain_tensor)
                    predicted_stress_x.append(stress_tensor[0, 0])  # σ_xx
                    predicted_stress_y.append(stress_tensor[1, 1])  # σ_yy

                predicted_stress_x = np.array(predicted_stress_x)
                predicted_stress_y = np.array(predicted_stress_y)

                # Compute combined residual
                residual_x = predicted_stress_x - stress_x_arr
                residual_y = predicted_stress_y - stress_y_arr
                return np.sum(residual_x**2) + np.sum(residual_y**2)

            except Exception as e:
                return 1e10

        # Get parameter bounds and initial guess
        param_names, param_bounds, param_initial = self._get_fitting_parameters()

        # Perform fitting
        if method == 'least_squares':
            result = minimize(
                objective_function,
                x0=param_initial,
                bounds=param_bounds,
                method='L-BFGS-B'
            )
        else:
            raise ValueError(f"Unknown fitting method: {method}")

        # Process results
        fitted_params = self._array_to_dict(result.x)
        model = create_model(self.model_type, **fitted_params)

        # Calculate predicted stresses
        strain_tensors = self._biaxial_strain_to_tensor(strain_x_arr, strain_y_arr)
        predicted_stress_x = []
        predicted_stress_y = []

        for strain_tensor in strain_tensors:
            stress_tensor = model.compute_stress(strain_tensor)
            predicted_stress_x.append(stress_tensor[0, 0])
            predicted_stress_y.append(stress_tensor[1, 1])

        # Calculate goodness of fit
        predicted_stress_x = np.array(predicted_stress_x)
        predicted_stress_y = np.array(predicted_stress_y)

        combined_stress = np.concatenate([stress_x_arr, stress_y_arr])
        combined_predicted = np.concatenate([predicted_stress_x, predicted_stress_y])
        r_squared = self._calculate_r_squared(combined_stress, combined_predicted)
        rmse = np.sqrt(np.mean((combined_stress - combined_predicted)**2))

        fit_results = {
            'model_type': self.model_type,
            'fitted_parameters': fitted_params,
            'optimization_result': result,
            'goodness_of_fit': {
                'r_squared': r_squared,
                'rmse': rmse
            },
            'data': {
                'strain_x': strain_x,
                'strain_y': strain_y,
                'stress_x': stress_x,
                'stress_y': stress_y,
                'predicted_stress_x': predicted_stress_x.tolist(),
                'predicted_stress_y': predicted_stress_y.tolist()
            },
            'method': method
        }

        self.current_fit = fit_results
        self.fitting_history.append(fit_results)

        self.logger.info(f"Biaxial fitting completed. R² = {r_squared:.4f}, RMSE = {rmse:.4f} kPa")

        return fit_results

    def _get_fitting_parameters(self, bounds: Optional[Dict[str, tuple]] = None,
                              initial_guess: Optional[Dict[str, float]] = None) -> tuple:
        """Get parameter names, bounds, and initial values."""
        if self.model_type == 'neo_hookean':
            param_names = ['C10']
            default_bounds = [(0.001, 100.0)]
            default_initial = [0.1]
        elif self.model_type == 'mooney_rivlin':
            param_names = ['C10', 'C01']
            default_bounds = [(0.001, 100.0), (0.0, 50.0)]
            default_initial = [0.1, 0.02]
        elif self.model_type == 'yeoh':
            param_names = ['C10', 'C20', 'C30']
            default_bounds = [(0.001, 100.0), (0.0, 10.0), (0.0, 1.0)]
            default_initial = [0.1, 0.01, 0.001]
        elif self.model_type == 'ogden':
            param_names = ['mu1', 'alpha1', 'mu2', 'alpha2']
            default_bounds = [(0.001, 100.0), (0.1, 10.0), (0.001, 100.0), (-10.0, -0.1)]
            default_initial = [0.5, 2.0, 0.1, -2.0]
        elif self.model_type == 'linear_elastic':
            param_names = ['youngs_modulus']
            default_bounds = [(0.1, 1000.0)]
            default_initial = [5.0]
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Apply user-specified bounds
        if bounds:
            for i, param in enumerate(param_names):
                if param in bounds:
                    default_bounds[i] = bounds[param]

        # Apply user-specified initial guess
        if initial_guess:
            for i, param in enumerate(param_names):
                if param in initial_guess:
                    default_initial[i] = initial_guess[param]

        return param_names, default_bounds, default_initial

    def _array_to_dict(self, param_array: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        param_names, _, _ = self._get_fitting_parameters()
        return {name: value for name, value in zip(param_names, param_array)}

    def _uniaxial_strain_to_tensor(self, strain: np.ndarray) -> List[np.ndarray]:
        """Convert uniaxial strain to 3D strain tensor."""
        strain_tensors = []
        for eps in strain:
            # Green-Lagrange strain tensor for uniaxial tension
            E = np.array([
                [eps, 0, 0],
                [0, -0.5*eps**2, 0],  # Assuming incompressibility
                [0, 0, -0.5*eps**2]
            ])
            strain_tensors.append(E)
        return strain_tensors

    def _biaxial_strain_to_tensor(self, strain_x: np.ndarray, strain_y: np.ndarray) -> List[np.ndarray]:
        """Convert biaxial strain to 3D strain tensor."""
        strain_tensors = []
        for eps_x, eps_y in zip(strain_x, strain_y):
            # Green-Lagrange strain tensor for biaxial tension
            E = np.array([
                [eps_x, 0, 0],
                [0, eps_y, 0],
                [0, 0, -(eps_x + eps_y + eps_x*eps_y + eps_y*eps_x)]  # Incompressibility
            ])
            strain_tensors.append(E)
        return strain_tensors

    def _calculate_r_squared(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate coefficient of determination (R²)."""
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def validate_fitting_data(self, strain_data: List[float], stress_data: List[float],
                             data_type: str = 'uniaxial') -> bool:
        """
        Validate fitting data.

        Args:
            strain_data: Strain data
            stress_data: Stress data
            data_type: Type of data ('uniaxial', 'biaxial')

        Returns:
            True if data is valid
        """
        if not NUMPY_AVAILABLE:
            return False

        # Convert to numpy arrays
        strain = np.array(strain_data)
        stress = np.array(stress_data)

        # Basic checks
        if len(strain) != len(stress):
            self.logger.error("Strain and stress data must have same length")
            return False

        if len(strain) < 3:
            self.logger.error("Need at least 3 data points for fitting")
            return False

        # Check for invalid values
        if np.any(np.isnan(strain)) or np.any(np.isnan(stress)):
            self.logger.error("Data contains NaN values")
            return False

        if np.any(np.isinf(strain)) or np.any(np.isinf(stress)):
            self.logger.error("Data contains infinite values")
            return False

        # Check for reasonable ranges
        if np.any(strain < -0.5) or np.any(strain > 2.0):
            self.logger.warning("Strain values outside typical range (-0.5 to 2.0)")

        if np.any(stress < -1000) or np.any(stress > 1000):
            self.logger.warning("Stress values outside typical range (-1000 to 1000 kPa)")

        # Check monotonicity for uniaxial data
        if data_type == 'uniaxial':
            if not np.all(np.diff(strain) > 0):
                self.logger.warning("Strain data is not monotonically increasing")

        return True

    def generate_synthetic_data(self, true_params: Dict[str, float],
                             strain_range: tuple = (0.0, 0.5), num_points: int = 20,
                             noise_level: float = 0.02, data_type: str = 'uniaxial') -> Dict[str, List[float]]:
        """
        Generate synthetic stress-strain data for testing.

        Args:
            true_params: True material parameters
            strain_range: Range of strain values
            num_points: Number of data points
            noise_level: Level of noise to add (fraction of stress)
            data_type: Type of data to generate

        Returns:
            Dictionary containing synthetic data
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for synthetic data generation")

        self.logger.info(f"Generating synthetic {data_type} data")

        # Create model with true parameters
        model = create_model(self.model_type, **true_params)

        # Generate strain values
        strain_values = np.linspace(strain_range[0], strain_range[1], num_points)

        # Compute stress values
        if data_type == 'uniaxial':
            strain_tensors = self._uniaxial_strain_to_tensor(strain_values)
            stress_values = []
            for strain_tensor in strain_tensors:
                stress_tensor = model.compute_stress(strain_tensor)
                stress_values.append(stress_tensor[0, 0])  # Axial stress
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        stress_values = np.array(stress_values)

        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.mean(np.abs(stress_values)), len(stress_values))
            stress_values += noise

        return {
            'strain': strain_values.tolist(),
            'stress': stress_values.tolist(),
            'true_parameters': true_params,
            'noise_level': noise_level
        }

    def get_fitting_summary(self) -> Dict[str, Any]:
        """
        Get summary of fitting operations.

        Returns:
            Fitting summary dictionary
        """
        if not self.fitting_history:
            return {'message': 'No fitting operations performed'}

        return {
            'total_fits': len(self.fitting_history),
            'model_type': self.model_type,
            'current_fit': self.current_fit,
            'fit_history': self.fitting_history
        }

    def clear_history(self) -> None:
        """Clear fitting history."""
        self.fitting_history.clear()
        self.current_fit = None
        self.logger.debug("Fitting history cleared")

    def __repr__(self) -> str:
        """String representation of the MaterialFitter instance."""
        return f"MaterialFitter(model_type={self.model_type}, fits={len(self.fitting_history)})"


# Convenience function for creating a fitter
def create_material_fitter(model_type: str, logger: Optional[logging.Logger] = None) -> MaterialFitter:
    """
    Create a material fitter instance.

    Args:
        model_type: Type of constitutive model
        logger: Optional logger instance

    Returns:
        MaterialFitter instance
    """
    return MaterialFitter(model_type=model_type, logger=logger)
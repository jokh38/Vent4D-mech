"""
Base Component Module for Vent4D-Mech

This module provides the base component class that standardizes interfaces
across all Vent4D-Mech components, ensuring consistent behavior,
configuration handling, and error management throughout the framework.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, Union
import numpy as np


class BaseComponent(ABC):
    """
    Base class for all Vent4D-Mech components.

    This class provides a standardized interface for all components in the
    Vent4D-Mech framework, ensuring consistent initialization, configuration
    handling, processing patterns, and error management.

    Attributes:
        config (Dict[str, Any]): Configuration parameters for the component
        gpu (bool): Whether GPU acceleration is enabled
        logger (logging.Logger): Logger instance for the component
        component_name (str): Name of the component class
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, gpu: bool = True):
        """
        Initialize the base component.

        Args:
            config: Optional configuration parameters. If None, default config is used.
            gpu: Whether to enable GPU acceleration for supported operations

        Raises:
            ConfigurationError: If configuration validation fails
        """
        self.config = config or self._get_default_config()
        self.gpu = gpu
        self.component_name = self.__class__.__name__

        # Initialize logger
        self.logger = logging.getLogger(f"vent4d_mech.{self.component_name}")

        # Validate configuration
        self._validate_config()

        # Log initialization
        self.logger.info(f"Initialized {self.component_name} with GPU={self.gpu}")

    @abstractmethod
    def process(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main processing method for the component.

        This method must be implemented by all subclasses and should contain
        the primary computation logic for the component.

        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments

        Returns:
            Dictionary containing processing results

        Raises:
            ComputationError: If processing fails
            ValidationError: If input validation fails
        """
        pass

    def _validate_config(self) -> None:
        """
        Validate component configuration.

        This method should be overridden by subclasses to perform
        component-specific configuration validation.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Basic validation - ensure config is a dictionary
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary")

    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the component.

        This method must be implemented by subclasses to provide
        sensible default configuration parameters.

        Returns:
            Default configuration dictionary
        """
        pass

    def get_component_info(self) -> Dict[str, Any]:
        """
        Get component information.

        Returns:
            Dictionary containing component information including:
            - name: Component class name
            - config: Current configuration
            - gpu_enabled: GPU acceleration status
            - logger_name: Logger name
        """
        return {
            'name': self.component_name,
            'config': self.config,
            'gpu_enabled': self.gpu,
            'logger_name': self.logger.name
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update component configuration.

        Args:
            new_config: New configuration parameters to merge

        Raises:
            ConfigurationError: If new configuration is invalid
        """
        # Create merged configuration
        updated_config = self.config.copy()
        updated_config.update(new_config)

        # Temporarily update config for validation
        old_config = self.config
        self.config = updated_config

        try:
            # Validate new configuration
            self._validate_config()

            # If validation passes, log the update
            self.logger.info(f"Updated configuration with keys: {list(new_config.keys())}")

        except Exception as e:
            # If validation fails, restore old config
            self.config = old_config
            self.logger.error(f"Configuration update failed: {str(e)}")
            raise

    def _validate_input_array(self, array: np.ndarray, name: str,
                             expected_dims: Optional[int] = None,
                             expected_shape: Optional[tuple] = None) -> None:
        """
        Validate input numpy arrays.

        Args:
            array: Input array to validate
            name: Name of the array (for error messages)
            expected_dims: Expected number of dimensions
            expected_shape: Expected shape tuple

        Raises:
            ValidationError: If array validation fails
        """
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")

        # Check dimensions
        if expected_dims is not None and array.ndim != expected_dims:
            raise ValueError(f"{name} must have {expected_dims} dimensions, got {array.ndim}")

        # Check shape
        if expected_shape is not None:
            for i, expected_size in enumerate(expected_shape):
                if expected_size is not None and array.shape[i] != expected_size:
                    raise ValueError(f"{name} dimension {i} must have size {expected_size}, got {array.shape[i]}")

        # Check for invalid values
        if np.any(np.isnan(array)):
            raise ValueError(f"{name} contains NaN values")

        if np.any(np.isinf(array)):
            raise ValueError(f"{name} contains infinite values")

    def _package_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Package processing results with metadata.

        Args:
            results: Raw processing results

        Returns:
            Packaged results with metadata
        """
        packaged_results = results.copy()
        packaged_results['_metadata'] = {
            'component': self.component_name,
            'gpu_enabled': self.gpu,
            'config': self.config
        }

        return packaged_results

    def _log_processing_start(self, operation_name: str, **kwargs) -> None:
        """
        Log the start of a processing operation.

        Args:
            operation_name: Name of the operation
            **kwargs: Additional operation parameters to log
        """
        params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"Starting {operation_name}: {params_str}")

    def _log_processing_end(self, operation_name: str, success: bool = True) -> None:
        """
        Log the end of a processing operation.

        Args:
            operation_name: Name of the operation
            success: Whether the operation was successful
        """
        status = "completed successfully" if success else "failed"
        self.logger.info(f"{operation_name} {status}")

    def __repr__(self) -> str:
        """String representation of the component."""
        return f"{self.component_name}(gpu={self.gpu}, config_keys={list(self.config.keys())})"
"""
Configuration Module

This module provides configuration management for the Vent4D-Mech framework,
including default parameters, validation, and utilities for loading and saving
configuration files.

Features both legacy validation and enhanced Pydantic-based validation for
improved type safety and developer experience.
"""

from .config_manager import ConfigManager
from .default_config import DefaultConfig
from .config_validation import ConfigValidation

# Pydantic schemas (optional import)
try:
    from .schemas import (
        Vent4DMechConfig,
        RegistrationConfig,
        MechanicalConfig,
        PerformanceConfig,
        LoggingConfig,
        ValidationConfig,
        create_default_config,
        create_minimal_config,
        validate_config_dict,
        config_to_dict,
        config_to_json,
        config_from_json
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create dummy objects to avoid import errors
    Vent4DMechConfig = None
    RegistrationConfig = None
    MechanicalConfig = None
    PerformanceConfig = None
    LoggingConfig = None
    ValidationConfig = None
    create_default_config = None
    create_minimal_config = None
    validate_config_dict = None
    config_to_dict = None
    config_to_json = None
    config_from_json = None

__all__ = [
    "ConfigManager",
    "DefaultConfig",
    "ConfigValidation",
    # Pydantic schemas (only available if Pydantic is installed)
    "Vent4DMechConfig",
    "RegistrationConfig",
    "MechanicalConfig",
    "PerformanceConfig",
    "LoggingConfig",
    "ValidationConfig",
    "create_default_config",
    "create_minimal_config",
    "validate_config_dict",
    "config_to_dict",
    "config_to_json",
    "config_from_json",
    "PYDANTIC_AVAILABLE"
]
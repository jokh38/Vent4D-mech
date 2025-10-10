"""
Configuration Module

This module provides configuration management for the Vent4D-Mech framework,
including default parameters, validation, and utilities for loading and saving
configuration files.
"""

from .config_manager import ConfigManager
from .default_config import DefaultConfig
from .config_validation import ConfigValidation

__all__ = [
    "ConfigManager",
    "DefaultConfig",
    "ConfigValidation"
]
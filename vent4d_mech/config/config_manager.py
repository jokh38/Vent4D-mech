"""
Configuration Manager

This module provides centralized configuration management for the Vent4D-Mech
framework, including loading, saving, validation, and merging of configuration
parameters.
"""

from typing import Optional, Dict, Any, Union
import logging
from pathlib import Path
import yaml
import json

from .default_config import DefaultConfig
from .config_validation import ConfigValidation


class ConfigManager:
    """
    Centralized configuration manager for Vent4D-Mech.

    This class provides comprehensive configuration management capabilities,
    including loading from files, validation, merging, and environment-specific
    configurations.

    Attributes:
        config (dict): Current configuration
        validator (ConfigValidation): Configuration validator
        logger (logging.Logger): Logger instance
        config_sources (list): List of configuration file sources
    """

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigManager instance.

        Args:
            config_file: Optional configuration file path
        """
        self.logger = logging.getLogger(__name__)
        self.validator = ConfigValidation()
        self.config_sources = []

        # Load default configuration
        self.config = DefaultConfig.get_default_config()

        # Load additional configuration if provided
        if config_file:
            self.load_config(config_file)

        self.logger.info("Initialized ConfigManager")

    def load_config(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from file.

        Args:
            config_file: Configuration file path

        Raises:
            FileNotFoundError: If configuration file is not found
            ValueError: If configuration is invalid
        """
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            # Load configuration based on file extension
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

            # Validate configuration
            self.validator.validate_config(user_config)

            # Merge with existing configuration
            self.config = self._merge_configs(self.config, user_config)
            self.config_sources.append(str(config_path))

            self.logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise

    def save_config(self, config_file: Union[str, Path], format: str = 'yaml') -> None:
        """
        Save current configuration to file.

        Args:
            config_file: Output configuration file path
            format: File format ('yaml' or 'json')
        """
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format.lower() == 'yaml':
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Saved configuration to {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key is not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

        self.logger.info(f"Set configuration: {key} = {value}")

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with multiple values.

        Args:
            updates: Dictionary of updates
        """
        self.config = self._merge_configs(self.config, updates)
        self.logger.info("Updated configuration with multiple values")

    def _merge_configs(self, base_config: Dict[str, Any],
                      update_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.

        Args:
            base_config: Base configuration
            update_config: Update configuration

        Returns:
            Merged configuration
        """
        merged = base_config.copy()

        for key, value in update_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a configuration section.

        Args:
            section: Section name

        Returns:
            Configuration section
        """
        return self.config.get(section, {})

    def validate_config(self) -> bool:
        """
        Validate current configuration.

        Returns:
            True if configuration is valid
        """
        try:
            self.validator.validate_config(self.config)
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.

        Returns:
            Configuration summary
        """
        return {
            'config_sources': self.config_sources,
            'sections': list(self.config.keys()),
            'total_keys': self._count_keys(self.config)
        }

    def _count_keys(self, config: Dict[str, Any]) -> int:
        """
        Count total number of keys in configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Total key count
        """
        count = 0
        for value in config.values():
            if isinstance(value, dict):
                count += self._count_keys(value)
            else:
                count += 1
        return count

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using bracket notation."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return self.get(key) is not None

    def __repr__(self) -> str:
        """String representation of the ConfigManager instance."""
        return f"ConfigManager(sources={len(self.config_sources)}, sections={list(self.config.keys())})"
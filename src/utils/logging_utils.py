"""
Logging Utilities

This module provides structured logging utilities for the Vent4D-Mech framework,
ensuring consistent logging across all components with configurable levels
and output formats.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


class LoggingUtils:
    """
    Structured logging utilities for Vent4D-Mech framework.

    This class provides comprehensive logging capabilities with support for
    multiple output formats, structured logging, and configurable log levels.
    It ensures consistent logging across all framework components.

    Attributes:
        logger (logging.Logger): Logger instance
        name (str): Logger name
        handlers (list): List of configured handlers
    """

    def __init__(self, name: Optional[str] = None, level: str = "INFO",
                 log_file: Optional[str] = None, structured: bool = False):
        """
        Initialize LoggingUtils.

        Args:
            name: Logger name (defaults to calling module name)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            structured: Whether to use structured logging format
        """
        self.name = name or self._get_calling_module_name()
        self.logger = logging.getLogger(self.name)
        self.structured = structured
        self.handlers = []

        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Set logging level
        self.set_level(level)

        # Configure console handler
        self._add_console_handler(structured)

        # Configure file handler if specified
        if log_file:
            self._add_file_handler(log_file, structured)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def _get_calling_module_name(self) -> str:
        """Get the name of the calling module."""
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            return frame.f_back.f_globals.get('__name__', 'vent4d_mech')
        return 'vent4d_mech'

    def set_level(self, level: str) -> None:
        """
        Set logging level.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Raises:
            ValueError: If invalid logging level provided
        """
        level = level.upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        if level not in valid_levels:
            raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")

        self.logger.setLevel(getattr(logging, level))

    def _add_console_handler(self, structured: bool = False) -> None:
        """Add console handler with appropriate formatter."""
        handler = logging.StreamHandler(sys.stdout)

        if structured:
            formatter = self._get_structured_formatter()
        else:
            formatter = self._get_simple_formatter()

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.handlers.append(handler)

    def _add_file_handler(self, log_file: str, structured: bool = False) -> None:
        """Add file handler with appropriate formatter."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(log_path)

        if structured:
            formatter = self._get_structured_formatter(include_timestamp=True)
        else:
            formatter = self._get_detailed_formatter()

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.handlers.append(handler)

    def _get_simple_formatter(self) -> logging.Formatter:
        """Get simple log formatter."""
        return logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )

    def _get_detailed_formatter(self) -> logging.Formatter:
        """Get detailed log formatter with timestamp."""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _get_structured_formatter(self, include_timestamp: bool = False) -> logging.Formatter:
        """Get structured JSON log formatter."""
        class StructuredFormatter(logging.Formatter):
            def __init__(self, include_timestamp: bool = False):
                super().__init__()
                self.include_timestamp = include_timestamp

            def format(self, record):
                log_entry = {
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }

                if self.include_timestamp:
                    log_entry['timestamp'] = datetime.fromtimestamp(record.created).isoformat()

                if record.exc_info:
                    log_entry['exception'] = self.formatException(record.exc_info)

                return json.dumps(log_entry)

        return StructuredFormatter(include_timestamp)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def _log_with_context(self, level: int, message: str, **kwargs) -> None:
        """
        Log message with additional context.

        Args:
            level: Logging level
            message: Log message
            **kwargs: Additional context key-value pairs
        """
        if kwargs:
            context_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            full_message = f"{message} | {context_str}"
        else:
            full_message = message

        self.logger.log(level, full_message)

    def log_function_call(self, func_name: str, args: tuple = (), kwargs: dict = None) -> None:
        """
        Log function call with parameters.

        Args:
            func_name: Function name
            args: Function arguments
            kwargs: Function keyword arguments
        """
        kwargs = kwargs or {}

        # Convert args to string representation
        args_str = ", ".join([str(arg) for arg in args])
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])

        params = []
        if args_str:
            params.append(args_str)
        if kwargs_str:
            params.append(kwargs_str)

        params_str = ", ".join(params)

        self.debug(f"Calling {func_name}({params_str})")

    def log_performance(self, operation: str, duration: float, **metrics) -> None:
        """
        Log performance metrics.

        Args:
            operation: Operation name
            duration: Duration in seconds
            **metrics: Additional performance metrics
        """
        self.info(
            f"Performance: {operation}",
            duration_s=duration,
            **metrics
        )

    def log_data_info(self, data_name: str, shape: tuple = None, dtype: str = None, **info) -> None:
        """
        Log data information.

        Args:
            data_name: Name of the data
            shape: Data shape
            dtype: Data type
            **info: Additional data information
        """
        context = {'data': data_name}
        if shape is not None:
            context['shape'] = shape
        if dtype is not None:
            context['dtype'] = dtype
        context.update(info)

        self.debug(f"Data info: {data_name}", **context)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration information.

        Args:
            config: Configuration dictionary
        """
        config_str = json.dumps(config, indent=2, default=str)
        self.debug(f"Configuration: {config_str}")

    def log_error_with_traceback(self, message: str, exception: Exception) -> None:
        """
        Log error with full traceback.

        Args:
            message: Error message
            exception: Exception instance
        """
        self.logger.error(f"{message}: {str(exception)}", exc_info=True)

    def create_child_logger(self, name: str) -> 'LoggingUtils':
        """
        Create a child logger with inherited configuration.

        Args:
            name: Child logger name

        Returns:
            New LoggingUtils instance with child logger
        """
        child_name = f"{self.name}.{name}"
        child_logger = LoggingUtils(
            name=child_name,
            level=logging.getLevelName(self.logger.level),
            structured=self.structured
        )
        return child_logger

    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger

    def __repr__(self) -> str:
        """String representation of the LoggingUtils instance."""
        return f"LoggingUtils(name='{self.name}', level='{logging.getLevelName(self.logger.level)}')"


# Convenience function for getting a configured logger
def get_logger(name: Optional[str] = None, level: str = "INFO",
               log_file: Optional[str] = None, structured: bool = False) -> LoggingUtils:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        structured: Whether to use structured logging

    Returns:
        Configured LoggingUtils instance
    """
    return LoggingUtils(name=name, level=level, log_file=log_file, structured=structured)


# Module-level logger instance
default_logger = get_logger('vent4d_mech')
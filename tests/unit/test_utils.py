"""
Utilities Unit Tests

This module contains unit tests for utility functions and classes.
"""

import pytest
import sys
import os
import tempfile
import logging

from src.utils import logging_utils


class TestLoggingUtils:
    """Test cases for LoggingUtils class."""

    def test_logging_utils_initialization(self):
        """Test LoggingUtils initialization."""
        logger = logging_utils.LoggingUtils()

        assert logger is not None
        assert logger.logger is not None
        assert hasattr(logger, 'set_level')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')

    def test_logging_utils_with_custom_name(self):
        """Test LoggingUtils with custom name."""
        custom_name = "test_logger"
        logger = logging_utils.LoggingUtils(name=custom_name)

        assert logger.name == custom_name

    def test_set_logging_level(self):
        """Test setting logging levels."""
        logger = logging_utils.LoggingUtils()

        # Test valid levels
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            logger.set_level(level)
            # Should not raise any exceptions

        # Test invalid level
        with pytest.raises(ValueError):
            logger.set_level('INVALID_LEVEL')

    def test_logging_methods(self):
        """Test different logging methods."""
        logger = logging_utils.LoggingUtils(level='DEBUG')

        # Test all logging methods
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # Should not raise any exceptions

    def test_logging_with_context(self):
        """Test logging with additional context."""
        logger = logging_utils.LoggingUtils()

        logger.info("Test message", param1="value1", param2=42)

        # Should not raise any exceptions

    def test_performance_logging(self):
        """Test performance logging."""
        logger = logging_utils.LoggingUtils()

        logger.log_performance("test_operation", 0.123, iterations=100)

        # Should not raise any exceptions

    def test_data_info_logging(self):
        """Test data information logging."""
        logger = logging_utils.LoggingUtils()

        logger.log_data_info("test_data", shape=(64, 64, 64), dtype="float32")

        # Should not raise any exceptions

    def test_config_logging(self):
        """Test configuration logging."""
        logger = logging_utils.LoggingUtils()

        test_config = {
            'param1': 'value1',
            'param2': 42,
            'nested': {'sub_param': 'sub_value'}
        }

        logger.log_config(test_config)

        # Should not raise any exceptions

    def test_child_logger_creation(self):
        """Test creating child loggers."""
        parent_logger = logging_utils.LoggingUtils(name="parent")
        child_logger = parent_logger.create_child_logger("child")

        assert child_logger is not None
        assert "parent.child" in child_logger.name

    def test_get_logger_method(self):
        """Test getting the underlying logger instance."""
        logger = logging_utils.LoggingUtils()
        underlying_logger = logger.get_logger()

        assert underlying_logger is not None
        assert hasattr(underlying_logger, 'info')

    def test_logging_utils_repr(self):
        """Test string representation of LoggingUtils."""
        logger = logging_utils.LoggingUtils(name="test_logger", level="INFO")
        repr_str = repr(logger)

        assert "LoggingUtils" in repr_str
        assert "test_logger" in repr_str
        assert "INFO" in repr_str


class TestPerformanceLogging:
    """Test cases for performance logging functionality."""

    def test_log_function_call(self):
        """Test function call logging."""
        logger = logging_utils.LoggingUtils()

        logger.log_function_call("test_function", (1, 2, 3), {'param': 'value'})

        # Should not raise any exceptions

    def test_log_error_with_traceback(self):
        """Test error logging with traceback."""
        logger = logging_utils.LoggingUtils()

        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.log_error_with_traceback("Test error message", e)

        # Should not raise any exceptions


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_get_logger_function(self):
        """Test get_logger convenience function."""
        logger = logging_utils.get_logger(name="convenience_test", level="INFO")

        assert logger is not None
        assert isinstance(logger, logging_utils.LoggingUtils)

    def test_default_logger_instance(self):
        """Test default logger instance."""
        default_logger = logging_utils.default_logger

        assert default_logger is not None
        assert isinstance(default_logger, logging_utils.LoggingUtils)


class TestLoggingFileOperations:
    """Test cases for logging file operations."""

    def test_logging_with_file_output(self, tmp_path):
        """Test logging to file."""
        log_file = tmp_path / "test_log.txt"

        logger = logging_utils.LoggingUtils(
            name="file_test",
            log_file=str(log_file),
            level="INFO"
        )

        logger.info("Test message to file")

        # Check if log file was created
        assert log_file.exists()

    def test_logging_structured_format(self, tmp_path):
        """Test structured logging format."""
        log_file = tmp_path / "structured_log.txt"

        logger = logging_utils.LoggingUtils(
            name="structured_test",
            log_file=str(log_file),
            level="INFO",
            structured=True
        )

        logger.info("Test structured message", param1="value1")

        # Should not raise any exceptions
        assert log_file.exists()


class TestLoggingConfiguration:
    """Test cases for logging configuration."""

    def test_multiple_logger_instances(self):
        """Test creating multiple logger instances."""
        logger1 = logging_utils.LoggingUtils(name="logger1")
        logger2 = logging_utils.LoggingUtils(name="logger2")

        assert logger1.name != logger2.name
        assert logger1.logger is not logger2.logger

    def test_logger_level_inheritance(self):
        """Test logger level inheritance."""
        parent_logger = logging_utils.LoggingUtils(name="parent", level="WARNING")
        child_logger = parent_logger.create_child_logger("child")

        # Child should inherit or be able to set its own level
        child_logger.set_level("DEBUG")

        # Should not raise any exceptions

    def test_logging_error_handling(self):
        """Test logging error handling."""
        logger = logging_utils.LoggingUtils()

        # Test logging with problematic data
        logger.info("Test message", complex_data=object())

        # Should not raise any exceptions

    def test_logging_performance_overhead(self):
        """Test logging performance overhead."""
        logger = logging_utils.LoggingUtils(level="INFO")

        import time

        # Time logging operations
        start_time = time.time()
        for i in range(100):
            logger.info(f"Performance test message {i}")
        end_time = time.time()

        duration = end_time - start_time

        # Should complete reasonably quickly (less than 1 second for 100 messages)
        assert duration < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
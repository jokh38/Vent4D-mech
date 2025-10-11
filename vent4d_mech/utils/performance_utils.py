"""
Performance Utilities

This module provides performance monitoring and profiling utilities for the Vent4D-Mech framework,
including timing, memory usage tracking, and benchmarking capabilities.
"""

import time
import psutil
import threading
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager
from functools import wraps
import numpy as np
from .logging_utils import LoggingUtils


class PerformanceUtils:
    """
    Performance monitoring and profiling utilities.

    This class provides comprehensive performance monitoring capabilities including
    timing, memory usage tracking, and benchmarking for scientific computing workflows.

    Attributes:
        logger (LoggingUtils): Logger instance
        timers (dict): Active timers
        memory_snapshots (list): Memory usage snapshots
    """

    def __init__(self, logger: Optional[LoggingUtils] = None):
        """
        Initialize PerformanceUtils.

        Args:
            logger: Optional logger instance (creates default if None)
        """
        self.logger = logger or LoggingUtils('performance_utils')
        self.timers = {}
        self.memory_snapshots = []
        self.process = psutil.Process()

    @contextmanager
    def timer(self, name: str, log_result: bool = True, **context):
        """
        Context manager for timing operations.

        Args:
            name: Timer name
            log_result: Whether to log the timing result
            **context: Additional context for logging

        Yields:
            None

        Example:
            with perf_utils.timer("image_processing"):
                process_image()
        """
        start_time = time.time()
        start_memory = self.get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.get_memory_usage()
            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            if log_result:
                self.logger.log_performance(
                    operation=name,
                    duration=duration,
                    memory_mb=end_memory,
                    memory_delta_mb=memory_delta,
                    **context
                )

    def time_function(self, func_name: Optional[str] = None, log_result: bool = True):
        """
        Decorator for timing function execution.

        Args:
            func_name: Custom function name (uses function name if None)
            log_result: Whether to log the timing result

        Returns:
            Decorated function

        Example:
            @perf_utils.time_function()
            def process_data(data):
                return data * 2
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or f"{func.__module__}.{func.__name__}"

                start_time = time.time()
                start_memory = self.get_memory_usage()

                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    result = None
                    raise
                finally:
                    end_time = time.time()
                    end_memory = self.get_memory_usage()
                    duration = end_time - start_time
                    memory_delta = end_memory - start_memory

                    if log_result:
                        self.logger.log_performance(
                            operation=name,
                            duration=duration,
                            memory_mb=end_memory,
                            memory_delta_mb=memory_delta,
                            success=success
                        )

                return result

            return wrapper
        return decorator

    def start_timer(self, name: str) -> None:
        """
        Start a named timer.

        Args:
            name: Timer name

        Example:
            perf_utils.start_timer("computation")
            # ... do work ...
            duration = perf_utils.end_timer("computation")
        """
        if name in self.timers:
            self.logger.warning(f"Timer '{name}' already exists, overwriting")

        self.timers[name] = {
            'start_time': time.time(),
            'start_memory': self.get_memory_usage(),
            'thread_id': threading.current_thread().ident
        }

    def end_timer(self, name: str, **context) -> float:
        """
        End a named timer and return duration.

        Args:
            name: Timer name
            **context: Additional context for logging

        Returns:
            Duration in seconds

        Raises:
            KeyError: If timer not found
        """
        if name not in self.timers:
            raise KeyError(f"Timer '{name}' not found. Use start_timer() first.")

        timer_data = self.timers.pop(name)
        end_time = time.time()
        end_memory = self.get_memory_usage()

        # Verify thread consistency
        current_thread = threading.current_thread().ident
        if timer_data['thread_id'] != current_thread:
            self.logger.warning(
                f"Timer '{name}' started on thread {timer_data['thread_id']} "
                f"but ended on thread {current_thread}"
            )

        duration = end_time - timer_data['start_time']
        memory_delta = end_memory - timer_data['start_memory']

        self.logger.log_performance(
            operation=name,
            duration=duration,
            memory_mb=end_memory,
            memory_delta_mb=memory_delta,
            **context
        )

        return duration

    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.

        Returns:
            CPU usage percentage
        """
        return psutil.cpu_percent(interval=None)

    def take_memory_snapshot(self, name: str) -> None:
        """
        Take a memory usage snapshot.

        Args:
            name: Snapshot name
        """
        snapshot = {
            'name': name,
            'timestamp': time.time(),
            'memory_mb': self.get_memory_usage(),
            'cpu_percent': self.get_cpu_usage()
        }
        self.memory_snapshots.append(snapshot)
        self.logger.debug(f"Memory snapshot '{name}': {snapshot['memory_mb']:.2f} MB")

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory snapshots.

        Returns:
            Memory usage summary
        """
        if not self.memory_snapshots:
            return {'message': 'No memory snapshots available'}

        memory_values = [s['memory_mb'] for s in self.memory_snapshots]

        return {
            'total_snapshots': len(self.memory_snapshots),
            'min_memory_mb': min(memory_values),
            'max_memory_mb': max(memory_values),
            'avg_memory_mb': np.mean(memory_values),
            'std_memory_mb': np.std(memory_values),
            'snapshots': self.memory_snapshots
        }

    def benchmark_function(self, func: Callable, *args, num_runs: int = 5,
                          warmup_runs: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a function with multiple runs.

        Args:
            func: Function to benchmark
            *args: Function arguments
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs (not timed)
            **kwargs: Function keyword arguments

        Returns:
            Benchmark results

        Example:
            results = perf_utils.benchmark_function(
                np.linalg.svd, random_matrix, num_runs=10
            )
        """
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Warmup run failed: {e}")
                return {'error': str(e)}

        # Benchmark runs
        durations = []
        memory_usages = []
        results = []

        for i in range(num_runs):
            start_time = time.time()
            start_memory = self.get_memory_usage()

            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                result = None

            end_time = time.time()
            end_memory = self.get_memory_usage()

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            durations.append(duration)
            memory_usages.append(end_memory)
            results.append(result)

            self.logger.debug(
                f"Benchmark run {i+1}/{num_runs}: {duration:.4f}s, "
                f"memory: {end_memory:.2f}MB"
            )

        # Calculate statistics
        successful_runs = [d for i, d in enumerate(durations) if results[i] is not None or success]

        if not successful_runs:
            return {'error': 'All benchmark runs failed'}

        return {
            'function_name': f"{func.__module__}.{func.__name__}",
            'num_runs': num_runs,
            'successful_runs': len(successful_runs),
            'warmup_runs': warmup_runs,
            'duration_stats': {
                'min': np.min(successful_runs),
                'max': np.max(successful_runs),
                'mean': np.mean(successful_runs),
                'std': np.std(successful_runs),
                'median': np.median(successful_runs)
            },
            'memory_stats': {
                'final_mb': memory_usages[-1] if memory_usages else None,
                'peak_mb': np.max(memory_usages) if memory_usages else None
            },
            'durations': durations,
            'memory_usages': memory_usages
        }

    def monitor_system_resources(self, interval: float = 1.0,
                                duration: float = 60.0) -> Dict[str, List]:
        """
        Monitor system resources over time.

        Args:
            interval: Sampling interval in seconds
            duration: Total monitoring duration in seconds

        Returns:
            Resource monitoring data
        """
        import time as time_module

        timestamps = []
        cpu_percentages = []
        memory_usages = []

        end_time = time_module.time() + duration

        while time_module.time() < end_time:
            timestamps.append(time_module.time())
            cpu_percentages.append(self.get_cpu_usage())
            memory_usages.append(self.get_memory_usage())

            time_module.sleep(interval)

        return {
            'timestamps': timestamps,
            'cpu_percentages': cpu_percentages,
            'memory_usages': memory_usages,
            'sampling_interval': interval,
            'total_duration': duration
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.

        Returns:
            Performance report
        """
        return {
            'current_memory_mb': self.get_memory_usage(),
            'current_cpu_percent': self.get_cpu_usage(),
            'memory_summary': self.get_memory_summary(),
            'active_timers': list(self.timers.keys()),
            'total_snapshots': len(self.memory_snapshots)
        }

    def clear_snapshots(self) -> None:
        """Clear all memory snapshots."""
        self.memory_snapshots.clear()
        self.logger.debug("Memory snapshots cleared")

    def clear_timers(self) -> None:
        """Clear all active timers with warning."""
        if self.timers:
            self.logger.warning(f"Clearing {len(self.timers)} active timers")
            self.timers.clear()

    def __repr__(self) -> str:
        """String representation of the PerformanceUtils instance."""
        return f"PerformanceUtils(active_timers={len(self.timers)}, snapshots={len(self.memory_snapshots)})"


# Convenience function for getting a performance monitor
def get_performance_monitor(logger: Optional[LoggingUtils] = None) -> PerformanceUtils:
    """
    Get a configured performance monitor instance.

    Args:
        logger: Optional logger instance

    Returns:
        Configured PerformanceUtils instance
    """
    return PerformanceUtils(logger=logger)


# Module-level performance monitor instance
default_perf_monitor = get_performance_monitor()
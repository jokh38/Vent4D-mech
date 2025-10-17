"""
Performance Optimization - Parallel Processing Utilities

This module provides parallel processing utilities for the Vent4D-Mech framework
to improve performance through multi-threading and multi-processing for
computationally intensive operations.

Key Features:
- Multi-process data processing for large datasets
- Thread-safe operations for I/O bound tasks
- Memory-efficient chunking strategies
- Progress monitoring and error handling
"""

import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator
import numpy as np
from queue import Queue
import traceback

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ParallelProcessor:
    """
    Advanced parallel processing manager for Vent4D-Mech.

    Provides intelligent parallel processing with automatic resource detection,
    memory management, and progress monitoring.
    """

    def __init__(
        self,
        n_processes: Optional[int] = None,
        use_processes: bool = True,
        chunk_size: Optional[int] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize parallel processor.

        Args:
            n_processes: Number of processes to use. If None, auto-detects.
            use_processes: Whether to use processes (True) or threads (False)
            chunk_size: Size of data chunks for each worker
            enable_monitoring: Whether to enable performance monitoring
        """
        self.logger = logging.getLogger(__name__)
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self.enable_monitoring = enable_monitoring

        # Determine optimal number of processes
        if n_processes is None:
            self.n_processes = self._detect_optimal_processes()
        else:
            self.n_processes = min(n_processes, mp.cpu_count())

        # Performance monitoring
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._start_time = None
        self._end_time = None

        self.logger.info(
            f"Initialized ParallelProcessor: n_processes={self.n_processes}, "
            f"use_processes={use_processes}, chunk_size={chunk_size}"
        )

    def _detect_optimal_processes(self) -> int:
        """
        Detect optimal number of processes based on system resources.

        Returns:
            Optimal number of processes
        """
        cpu_count = mp.cpu_count()

        if PSUTIL_AVAILABLE:
            # Consider memory constraints
            memory_gb = psutil.virtual_memory().total / (1024**3)
            # Heuristic: use 1 process per 2GB of RAM, up to CPU count
            memory_limited_processes = min(int(memory_gb / 2), cpu_count)
            return max(1, memory_limited_processes)
        else:
            # Fallback to CPU count with a conservative limit
            return max(1, min(cpu_count, 4))

    def _chunk_data(self, data: np.ndarray, n_chunks: int) -> List[np.ndarray]:
        """
        Split data into chunks for parallel processing.

        Args:
            data: Input data array
            n_chunks: Number of chunks to create

        Returns:
            List of data chunks
        """
        if self.chunk_size is not None:
            # Use fixed chunk size
            chunk_size = min(self.chunk_size, data.shape[0])
            n_chunks = max(1, data.shape[0] // chunk_size)

        # Calculate chunk boundaries
        chunk_size = data.shape[0] // n_chunks
        chunks = []

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_chunks - 1 else data.shape[0]
            chunk = data[start_idx:end_idx]
            chunks.append(chunk)

        return chunks

    def process_array(
        self,
        func: Callable,
        data: np.ndarray,
        axis: int = 0,
        combine_results: bool = True,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Process array data in parallel.

        Args:
            func: Function to apply to each chunk
            data: Input array data
            axis: Axis along which to split the data
            combine_results: Whether to combine results into single array
            **kwargs: Additional arguments to pass to function

        Returns:
            Processed data (combined or as list of chunks)
        """
        if data.size == 0:
            return data if combine_results else []

        self._start_time = time.time()
        self._total_tasks = self.n_processes

        # Ensure we're splitting along the first axis
        if axis != 0:
            data = np.moveaxis(data, axis, 0)

        # Split data into chunks
        chunks = self._chunk_data(data, self.n_processes)
        self.logger.info(f"Processing {data.shape} array in {len(chunks)} chunks")

        # Prepare tasks
        tasks = [(func, chunk, kwargs) for chunk in chunks]

        # Process chunks
        results = self._execute_tasks(tasks)

        # Combine results if requested
        if combine_results and results:
            try:
                combined_result = np.concatenate(results, axis=0)
                # Move axis back to original position if needed
                if axis != 0:
                    combined_result = np.moveaxis(combined_result, 0, axis)
                return combined_result
            except ValueError as e:
                self.logger.warning(f"Could not combine results: {e}. Returning list of results.")
                return results
        else:
            return results

    def process_multiple(
        self,
        func: Callable,
        data_list: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        Process multiple data items in parallel.

        Args:
            func: Function to apply to each data item
            data_list: List of data items to process
            **kwargs: Additional arguments to pass to function

        Returns:
            List of processed results
        """
        if not data_list:
            return []

        self._start_time = time.time()
        self._total_tasks = len(data_list)

        # Prepare tasks
        tasks = [(func, data_item, kwargs) for data_item in data_list]

        # Execute tasks
        return self._execute_tasks(tasks)

    def _execute_tasks(self, tasks: List[Tuple[Callable, Any, Dict]]) -> List[Any]:
        """
        Execute a list of tasks in parallel.

        Args:
            tasks: List of (function, data, kwargs) tuples

        Returns:
            List of results
        """
        results = []
        errors = []

        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with executor_class(max_workers=self.n_processes) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(_execute_single_task, task): task
                for task in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._completed_tasks += 1
                except Exception as e:
                    error_info = {
                        'task': task,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                    errors.append(error_info)
                    self._failed_tasks += 1
                    self.logger.error(f"Task failed: {e}")

        self._end_time = time.time()
        self._log_performance_summary(len(tasks), len(errors))

        if errors and self.enable_monitoring:
            self.logger.warning(f"{len(errors)} tasks failed during processing")

        return results

    def _log_performance_summary(self, total_tasks: int, failed_tasks: int) -> None:
        """Log performance summary."""
        if self._start_time and self._end_time:
            duration = self._end_time - self._start_time
            success_rate = (total_tasks - failed_tasks) / total_tasks if total_tasks > 0 else 0
            throughput = total_tasks / duration if duration > 0 else 0

            self.logger.info(
                f"Parallel processing completed: {total_tasks} tasks, "
                f"{failed_tasks} failed, {duration:.2f}s, "
                f"{throughput:.1f} tasks/s, {success_rate:.1%} success rate"
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary containing performance metrics
        """
        stats = {
            'n_processes': self.n_processes,
            'use_processes': self.use_processes,
            'chunk_size': self.chunk_size,
            'total_tasks': self._total_tasks,
            'completed_tasks': self._completed_tasks,
            'failed_tasks': self._failed_tasks,
            'success_rate': self._completed_tasks / self._total_tasks if self._total_tasks > 0 else 0
        }

        if self._start_time and self._end_time:
            stats.update({
                'duration_seconds': self._end_time - self._start_time,
                'throughput_tasks_per_second': self._total_tasks / (self._end_time - self._start_time)
            })

        if PSUTIL_AVAILABLE:
            stats['system_info'] = {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            }

        return stats


def _execute_single_task(task: Tuple[Callable, Any, Dict]) -> Any:
    """
    Execute a single task in a worker process/thread.

    Args:
        task: Tuple of (function, data, kwargs)

    Returns:
        Task result

    Raises:
        Exception: If task execution fails
    """
    func, data, kwargs = task
    return func(data, **kwargs)


def parallel_map(
    func: Callable,
    data: Union[List[Any], np.ndarray],
    n_processes: Optional[int] = None,
    use_processes: bool = True,
    chunk_size: Optional[int] = None,
    **kwargs
) -> List[Any]:
    """
    Simple parallel map function.

    Args:
        func: Function to apply to each data item
        data: List or array of data to process
        n_processes: Number of processes to use
        use_processes: Whether to use processes (True) or threads (False)
        chunk_size: Size of chunks for array processing
        **kwargs: Additional arguments to pass to function

    Returns:
        List of results
    """
    processor = ParallelProcessor(
        n_processes=n_processes,
        use_processes=use_processes,
        chunk_size=chunk_size
    )

    if isinstance(data, np.ndarray):
        results = processor.process_array(func, data, combine_results=False, **kwargs)
        # Flatten results if they're arrays
        if results and isinstance(results[0], np.ndarray):
            return list(results)
        return results
    else:
        return processor.process_multiple(func, data, **kwargs)


def parallel_for(
    func: Callable,
    iterable: List[Any],
    n_processes: Optional[int] = None,
    use_processes: bool = True,
    show_progress: bool = False,
    **kwargs
) -> None:
    """
    Parallel for loop that executes a function for each item in an iterable.

    Args:
        func: Function to execute (receives item and kwargs)
        iterable: List of items to process
        n_processes: Number of processes to use
        use_processes: Whether to use processes (True) or threads (False)
        show_progress: Whether to show progress
        **kwargs: Additional arguments to pass to function
    """
    processor = ParallelProcessor(
        n_processes=n_processes,
        use_processes=use_processes,
        enable_monitoring=show_progress
    )

    def wrapper_func(args):
        item, local_kwargs = args
        return func(item, **local_kwargs)

    tasks = [(item, kwargs) for item in iterable]
    processor._execute_tasks([(wrapper_func, task, {}) for task in tasks])


class ThreadSafeCounter:
    """Thread-safe counter for progress monitoring."""

    def __init__(self, initial_value: int = 0):
        """Initialize counter."""
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, increment: int = 1) -> int:
        """
        Increment counter.

        Args:
            increment: Amount to increment by

        Returns:
            New counter value
        """
        with self._lock:
            self._value += increment
            return self._value

    def get_value(self) -> int:
        """
        Get current counter value.

        Returns:
            Current counter value
        """
        with self._lock:
            return self._value

    def reset(self, value: int = 0) -> None:
        """
        Reset counter to specified value.

        Args:
            value: New counter value
        """
        with self._lock:
            self._value = value


class ProgressMonitor:
    """Progress monitoring for parallel tasks."""

    def __init__(self, total_tasks: int, update_interval: float = 1.0):
        """
        Initialize progress monitor.

        Args:
            total_tasks: Total number of tasks to complete
            update_interval: Progress update interval in seconds
        """
        self.total_tasks = total_tasks
        self.update_interval = update_interval
        self.completed_tasks = ThreadSafeCounter(0)
        self.failed_tasks = ThreadSafeCounter(0)
        self.logger = logging.getLogger(__name__)

    def task_completed(self) -> None:
        """Mark a task as completed."""
        self.completed_tasks.increment()

    def task_failed(self) -> None:
        """Mark a task as failed."""
        self.failed_tasks.increment()

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress.

        Returns:
            Dictionary containing progress information
        """
        completed = self.completed_tasks.get_value()
        failed = self.failed_tasks.get_value()
        total_processed = completed + failed
        progress_percent = (total_processed / self.total_tasks) * 100 if self.total_tasks > 0 else 0

        return {
            'completed': completed,
            'failed': failed,
            'total_processed': total_processed,
            'total_tasks': self.total_tasks,
            'progress_percent': progress_percent
        }

    def log_progress(self) -> None:
        """Log current progress."""
        progress = self.get_progress()
        self.logger.info(
            f"Progress: {progress['completed']} completed, "
            f"{progress['failed']} failed, "
            f"{progress['progress_percent']:.1f}% complete"
        )


def process_in_chunks(
    func: Callable,
    data: np.ndarray,
    chunk_size: int,
    overlap: int = 0,
    **kwargs
) -> np.ndarray:
    """
    Process large arrays in chunks to manage memory usage.

    Args:
        func: Function to apply to each chunk
        data: Input array
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        **kwargs: Additional arguments to pass to function

    Returns:
        Processed array
    """
    if data.size == 0:
        return data

    n_samples = data.shape[0]
    results = []

    # Process chunks with overlap
    for i in range(0, n_samples, chunk_size - overlap):
        start_idx = i
        end_idx = min(i + chunk_size, n_samples)
        chunk = data[start_idx:end_idx]

        # Process chunk
        result_chunk = func(chunk, **kwargs)

        # Handle overlap in results
        if overlap > 0 and i > 0:
            # Remove overlapping portion from beginning of result
            result_chunk = result_chunk[overlap:]

        # Handle last chunk
        if end_idx == n_samples and overlap > 0:
            # Include full last chunk
            pass

        results.append(result_chunk)

    # Combine results
    if results:
        return np.concatenate(results, axis=0)
    else:
        return np.array([])
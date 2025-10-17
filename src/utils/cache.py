"""
Performance Optimization - Caching Utilities

This module provides caching utilities for the Vent4D-Mech framework to improve
performance by avoiding repeated expensive computations, particularly for
material model calculations and image processing operations.

Key Features:
- LRU caching for expensive computations
- Hash-based caching for numpy arrays
- Memory-aware cache management
- Cache statistics and monitoring
"""

import hashlib
import logging
import time
from functools import lru_cache, wraps
from typing import Any, Dict, Optional, Callable, Tuple, Union
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class CacheManager:
    """
    Advanced cache manager for Vent4D-Mech computations.

    Provides intelligent caching with memory management, statistics,
    and monitoring capabilities.
    """

    def __init__(self, max_cache_size_mb: int = 1024, enable_monitoring: bool = True):
        """
        Initialize cache manager.

        Args:
            max_cache_size_mb: Maximum cache size in megabytes
            enable_monitoring: Whether to enable cache monitoring
        """
        self.max_cache_size_mb = max_cache_size_mb
        self.enable_monitoring = enable_monitoring
        self.logger = logging.getLogger(__name__)

        # Cache storage
        self._cache: Dict[str, Tuple[Any, float, int]] = {}  # key -> (value, timestamp, size)
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

        # Memory tracking
        self._current_cache_size = 0
        self._peak_cache_size = 0

        if PSUTIL_AVAILABLE and enable_monitoring:
            self._memory_monitor = MemoryMonitor()
        else:
            self._memory_monitor = None

    def _estimate_object_size(self, obj: Any) -> int:
        """
        Estimate memory usage of an object in bytes.

        Args:
            obj: Object to size

        Returns:
            Estimated size in bytes
        """
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_object_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._estimate_object_size(k) + self._estimate_object_size(v)
                      for k, v in obj.items())
        else:
            # Rough estimate for other objects
            return len(str(obj).encode('utf-8'))

    def _make_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        Create a cache key from function arguments.

        Args:
            func_name: Name of the function
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            Cache key string
        """
        # Convert numpy arrays to hashable representation
        def process_arg(arg):
            if isinstance(arg, np.ndarray):
                return f"array_{hash_array(arg)}"
            elif isinstance(arg, (list, tuple)):
                return tuple(process_arg(a) for a in arg)
            elif isinstance(arg, dict):
                return tuple(sorted((k, process_arg(v)) for k, v in arg.items()))
            else:
                return arg

        processed_args = tuple(process_arg(arg) for arg in args)
        processed_kwargs = tuple(sorted((k, process_arg(v)) for k, v in kwargs.items()))

        key_data = f"{func_name}_{processed_args}_{processed_kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _evict_if_needed(self, new_item_size: int) -> None:
        """
        Evict cache entries if needed to make room for new item.

        Args:
            new_item_size: Size of the new item to be cached
        """
        target_size = self.max_cache_size_mb * 1024 * 1024  # Convert to bytes

        while (self._current_cache_size + new_item_size) > target_size and self._cache:
            # Find least recently used item (based on timestamp)
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            oldest_size = self._estimate_object_size(self._cache[oldest_key][0])

            # Remove oldest item
            del self._cache[oldest_key]
            self._current_cache_size -= oldest_size
            self._cache_evictions += 1

            self.logger.debug(f"Evicted cache entry {oldest_key} (size: {oldest_size} bytes)")

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found
        """
        if key in self._cache:
            value, timestamp, _ = self._cache[key]
            # Update timestamp (LRU behavior)
            self._cache[key] = (value, time.time(), self._estimate_object_size(value))
            self._cache_hits += 1
            return value
        else:
            self._cache_misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Item to cache
        """
        # Check if item already exists
        if key in self._cache:
            old_size = self._estimate_object_size(self._cache[key][0])
            self._current_cache_size -= old_size

        # Estimate size of new item
        item_size = self._estimate_object_size(value)

        # Evict if needed
        self._evict_if_needed(item_size)

        # Add item to cache
        self._cache[key] = (value, time.time(), item_size)
        self._current_cache_size += item_size
        self._peak_cache_size = max(self._peak_cache_size, self._current_cache_size)

        self.logger.debug(f"Cached item {key} (size: {item_size} bytes)")

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._current_cache_size = 0
        self.logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0

        stats = {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size_items': len(self._cache),
            'cache_size_bytes': self._current_cache_size,
            'cache_size_mb': self._current_cache_size / (1024 * 1024),
            'peak_cache_size_mb': self._peak_cache_size / (1024 * 1024),
            'max_cache_size_mb': self.max_cache_size_mb,
            'cache_evictions': self._cache_evictions
        }

        if self._memory_monitor:
            stats.update(self._memory_monitor.get_memory_stats())

        return stats

    def __repr__(self) -> str:
        """String representation of cache manager."""
        return f"CacheManager(size={len(self._cache)}, memory_mb={self._current_cache_size/(1024*1024):.2f})"


class MemoryMonitor:
    """Memory usage monitoring for cache management."""

    def __init__(self):
        """Initialize memory monitor."""
        self.process = psutil.Process()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.

        Returns:
            Dictionary containing memory statistics
        """
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        return {
            'process_memory_mb': memory_info.rss / (1024 * 1024),
            'process_memory_percent': memory_percent,
            'system_memory_available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }


def hash_array(arr: np.ndarray) -> int:
    """
    Create a hash of a numpy array for caching purposes.

    Args:
        arr: Numpy array to hash

    Returns:
        Integer hash value
    """
    # Create a deterministic hash from array data, shape, and dtype
    data = arr.tobytes()
    shape = str(arr.shape).encode('utf-8')
    dtype = str(arr.dtype).encode('utf-8')

    combined = data + b'|' + shape + b'|' + dtype
    return int(hashlib.md5(combined).hexdigest(), 16)


def cached_computation(maxsize: int = 128, include_args: bool = True):
    """
    Decorator for caching expensive computations with numpy arrays.

    Args:
        maxsize: Maximum number of cached results
        include_args: Whether to include function arguments in cache key

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_stats = {'hits': 0, 'misses': 0}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            if include_args:
                key_parts = [func.__name__]

                # Process arguments
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        key_parts.append(f"arr_{hash_array(arg)}")
                    else:
                        key_parts.append(str(arg))

                # Process keyword arguments
                for k, v in sorted(kwargs.items()):
                    if isinstance(v, np.ndarray):
                        key_parts.append(f"{k}_arr_{hash_array(v)}")
                    else:
                        key_parts.append(f"{k}_{v}")

                cache_key = "_".join(key_parts)
            else:
                cache_key = func.__name__

            # Check cache
            if cache_key in cache:
                cache_stats['hits'] += 1
                return cache[cache_key]

            # Compute result
            result = func(*args, **kwargs)
            cache_stats['misses'] += 1

            # Manage cache size
            if len(cache) >= maxsize:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(cache))
                del cache[oldest_key]

            # Cache result
            cache[cache_key] = result
            return result

        # Add cache statistics method
        def get_cache_stats():
            total = cache_stats['hits'] + cache_stats['misses']
            hit_rate = cache_stats['hits'] / total if total > 0 else 0
            return {
                'hits': cache_stats['hits'],
                'misses': cache_stats['misses'],
                'hit_rate': hit_rate,
                'cache_size': len(cache)
            }

        wrapper.get_cache_stats = get_cache_stats
        wrapper.cache_clear = lambda: cache.clear()
        return wrapper

    return decorator


def material_model_cache(func: Callable) -> Callable:
    """
    Specialized cache decorator for material model computations.

    Material model computations are expensive and benefit greatly from caching
    since the same strain states are often computed multiple times.

    Args:
        func: Material model function to cache

    Returns:
        Cached function
    """
    return cached_computation(maxsize=256, include_args=True)(func)


def image_processing_cache(func: Callable) -> Callable:
    """
    Specialized cache decorator for image processing operations.

    Image processing operations can be expensive, especially for large volumes.
    This cache is optimized for numpy array inputs.

    Args:
        func: Image processing function to cache

    Returns:
        Cached function
    """
    return cached_computation(maxsize=64, include_args=True)(func)


# Global cache manager instance
_global_cache_manager = None


def get_global_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.

    Returns:
        Global cache manager
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def configure_global_cache(max_cache_size_mb: int = 1024, enable_monitoring: bool = True) -> None:
    """
    Configure the global cache manager.

    Args:
        max_cache_size_mb: Maximum cache size in megabytes
        enable_monitoring: Whether to enable cache monitoring
    """
    global _global_cache_manager
    _global_cache_manager = CacheManager(max_cache_size_mb, enable_monitoring)


def cache_performance_report() -> Dict[str, Any]:
    """
    Generate a performance report for all caches.

    Returns:
        Dictionary containing cache performance metrics
    """
    report = {
        'timestamp': time.time(),
        'global_cache': get_global_cache_manager().get_stats() if _global_cache_manager else None
    }

    # Add system memory info if available
    if PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        report['system_memory'] = {
            'total_gb': vm.total / (1024**3),
            'available_gb': vm.available / (1024**3),
            'used_percent': vm.percent
        }

    return report


class CacheContext:
    """
    Context manager for temporary cache configuration.

    Useful for performance testing and optimization.
    """

    def __init__(self, max_cache_size_mb: int = 2048, enable_monitoring: bool = True):
        """
        Initialize cache context.

        Args:
            max_cache_size_mb: Maximum cache size in megabytes
            enable_monitoring: Whether to enable cache monitoring
        """
        self.max_cache_size_mb = max_cache_size_mb
        self.enable_monitoring = enable_monitoring
        self.original_manager = None

    def __enter__(self) -> CacheManager:
        """Enter cache context."""
        global _global_cache_manager
        self.original_manager = _global_cache_manager
        _global_cache_manager = CacheManager(self.max_cache_size_mb, self.enable_monitoring)
        return _global_cache_manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit cache context."""
        global _global_cache_manager
        _global_cache_manager = self.original_manager
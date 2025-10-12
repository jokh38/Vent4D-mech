"""
Utilities Module

This module provides utility functions and classes for the Vent4D-Mech framework,
including data loading/saving, image processing, visualization, validation,
and performance optimization.

Key features:
- Medical image I/O utilities
- Data preprocessing and postprocessing
- Visualization tools
- Validation and quality metrics
- Performance monitoring and optimization
- Caching and parallel processing utilities
- Logging utilities
"""

from .io_utils import IOUtils
from .image_utils import ImageUtils
from .visualization import Visualization
from .validation_utils import ValidationUtils
from .performance_utils import PerformanceUtils
from .logging_utils import LoggingUtils

# Performance optimization utilities
from .cache import (
    CacheManager,
    cached_computation,
    material_model_cache,
    image_processing_cache,
    hash_array,
    get_global_cache_manager,
    configure_global_cache,
    cache_performance_report,
    CacheContext
)
from .parallel import (
    ParallelProcessor,
    parallel_map,
    parallel_for,
    ThreadSafeCounter,
    ProgressMonitor,
    process_in_chunks
)

__all__ = [
    # Core utilities
    "IOUtils",
    "ImageUtils",
    "Visualization",
    "ValidationUtils",
    "PerformanceUtils",
    "LoggingUtils",

    # Caching utilities
    "CacheManager",
    "cached_computation",
    "material_model_cache",
    "image_processing_cache",
    "hash_array",
    "get_global_cache_manager",
    "configure_global_cache",
    "cache_performance_report",
    "CacheContext",

    # Parallel processing utilities
    "ParallelProcessor",
    "parallel_map",
    "parallel_for",
    "ThreadSafeCounter",
    "ProgressMonitor",
    "process_in_chunks"
]
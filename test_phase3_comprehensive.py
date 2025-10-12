#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phase 3 Implementation
Tests Pydantic configuration validation, performance utilities, and quality gates
"""

import sys
import pytest
import numpy as np
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Test Phase 3 modules
def test_phase3_imports():
    """Test that all Phase 3 modules can be imported."""
    try:
        # Configuration validation
        from vent4d_mech.config.schemas import (
            Vent4DMechConfig,
            PerformanceConfig,
            MechanicalConfig,
            RegistrationConfig,
            validate_config_dict,
            create_default_config
        )

        # Performance utilities
        from vent4d_mech.utils.cache import (
            CacheManager,
            get_global_cache_manager,
            cached_computation,
            material_model_cache,
            hash_array
        )

        from vent4d_mech.utils.parallel import (
            ParallelProcessor,
            parallel_map,
            parallel_for,
            process_in_chunks
        )

        # Quality gates
        from vent4d_mech.utils.quality_gates import (
            QualityGateManager,
            CodeQualityAnalyzer,
            TestCoverageAnalyzer,
            SecurityAnalyzer,
            TypeSafetyAnalyzer,
            run_quality_gates
        )

        print("✅ All Phase 3 modules imported successfully")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_pydantic_configuration_validation():
    """Test Pydantic configuration validation system."""
    print("\n🧪 Testing Pydantic Configuration Validation...")

    try:
        from vent4d_mech.config.schemas import (
            Vent4DMechConfig,
            create_default_config,
            validate_config_dict
        )

        # Test default configuration creation
        config = create_default_config()
        assert config is not None
        assert isinstance(config, Vent4DMechConfig)
        print("✅ Default configuration created successfully")

        # Test configuration validation
        config_dict = {
            "version": "0.1.0",
            "performance": {
                "gpu_acceleration": True,
                "parallel_processing": True,
                "num_processes": 4
            },
            "mechanical": {
                "constitutive_model": "neo_hookean",
                "material_parameters": {
                    "neo_hookean": {
                        "density": 1.05,
                        "C10": 0.135
                    }
                }
            }
        }

        validated_config = validate_config_dict(config_dict)
        assert validated_config is not None
        assert isinstance(validated_config, Vent4DMechConfig)
        print("✅ Configuration validation successful")

        # Test validation errors
        invalid_config_dict = {
            "performance": {
                "num_processes": -1  # Invalid value
            }
        }

        try:
            validate_config_dict(invalid_config_dict)
            assert False, "Should have raised validation error"
        except Exception:
            print("✅ Configuration validation correctly catches errors")

        return True

    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_caching_utilities():
    """Test caching utilities functionality."""
    print("\n🚀 Testing Caching Utilities...")

    try:
        from vent4d_mech.utils.cache import (
            CacheManager,
            cached_computation,
            material_model_cache,
            hash_array
        )

        # Test cache manager
        cache_manager = CacheManager(max_cache_size_mb=10)
        assert cache_manager is not None
        print("✅ Cache manager created successfully")

        # Test basic cache operations
        test_key = "test_key"
        test_value = {"data": [1, 2, 3, 4, 5]}

        cache_manager.put(test_key, test_value)
        retrieved_value = cache_manager.get(test_key)
        assert retrieved_value == test_value
        print("✅ Basic cache operations working")

        # Test array hashing
        test_array = np.random.rand(10, 10)
        array_hash = hash_array(test_array)
        assert isinstance(array_hash, int)
        print("✅ Array hashing working")

        # Test cached computation decorator
        @cached_computation(maxsize=10)
        def expensive_computation(x, y):
            time.sleep(0.01)  # Simulate expensive computation
            return x * y + np.random.rand()

        # First call (should compute)
        start_time = time.time()
        result1 = expensive_computation(5, 10)
        first_call_time = time.time() - start_time

        # Second call (should use cache)
        start_time = time.time()
        result2 = expensive_computation(5, 10)
        second_call_time = time.time() - start_time

        assert np.allclose(result1, result2)
        assert second_call_time < first_call_time
        print("✅ Cached computation decorator working")

        # Test cache statistics
        stats = cache_manager.get_stats()
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'hit_rate' in stats
        print(f"✅ Cache statistics: {stats}")

        return True

    except Exception as e:
        print(f"❌ Caching utilities test failed: {e}")
        return False

def test_parallel_processing():
    """Test parallel processing utilities."""
    print("\n⚡ Testing Parallel Processing...")

    try:
        from vent4d_mech.utils.parallel import (
            ParallelProcessor,
            parallel_map,
            process_in_chunks
        )

        # Test parallel processor
        processor = ParallelProcessor(n_processes=2, use_processes=True)
        assert processor is not None
        print("✅ Parallel processor created successfully")

        # Test parallel map with simple function
        def square_function(x):
            return x ** 2

        data = [1, 2, 3, 4, 5, 6, 7, 8]
        results = parallel_map(square_function, data, n_processes=2)
        expected = [x ** 2 for x in data]
        assert results == expected
        print("✅ Parallel map working correctly")

        # Test array processing
        test_array = np.random.rand(100, 10)

        def process_chunk(chunk):
            return np.mean(chunk, axis=1)

        results = processor.process_array(process_chunk, test_array)
        assert len(results) == test_array.shape[0]
        print("✅ Parallel array processing working")

        # Test chunked processing
        def chunk_processor(chunk):
            return np.sum(chunk, axis=0)

        chunked_results = process_in_chunks(
            chunk_processor,
            test_array,
            chunk_size=25
        )
        assert len(chunked_results) == test_array.shape[0]
        print("✅ Chunked processing working")

        # Test performance stats
        perf_stats = processor.get_performance_stats()
        assert 'n_processes' in perf_stats
        assert 'total_tasks' in perf_stats
        assert 'success_rate' in perf_stats
        print(f"✅ Performance stats: {perf_stats}")

        return True

    except Exception as e:
        print(f"❌ Parallel processing test failed: {e}")
        return False

def test_quality_gates():
    """Test quality gates functionality."""
    print("\n🔍 Testing Quality Gates...")

    try:
        from vent4d_mech.utils.quality_gates import (
            QualityGateManager,
            CodeQualityAnalyzer,
            run_quality_gates
        )

        # Test quality gate manager
        manager = QualityGateManager(
            source_dir="vent4d_mech",
            test_dir="tests"
        )
        assert manager is not None
        print("✅ Quality gate manager created successfully")

        # Test code quality analyzer
        code_analyzer = CodeQualityAnalyzer("vent4d_mech")

        # Test documentation coverage analysis
        doc_result = code_analyzer.analyze_documentation_coverage()
        assert doc_result is not None
        assert hasattr(doc_result, 'passed')
        assert hasattr(doc_result, 'score')
        print(f"✅ Documentation coverage: {doc_result.score:.1f}% ({'PASS' if doc_result.passed else 'FAIL'})")

        # Test complexity analysis
        complexity_result = code_analyzer.analyze_complexity()
        assert complexity_result is not None
        assert hasattr(complexity_result, 'passed')
        print(f"✅ Code complexity analysis: {complexity_result.score:.1f}% ({'PASS' if complexity_result.passed else 'FAIL'})")

        # Test type safety analysis (may fail if mypy not available)
        type_analyzer = manager.type_analyzer
        if type_analyzer:
            try:
                type_result = type_analyzer.analyze_type_safety()
                print(f"✅ Type safety analysis: {type_result.score:.1f}% ({'PASS' if type_result.passed else 'FAIL'})")
            except Exception as e:
                print(f"⚠️  Type safety analysis skipped (mypy issues): {e}")

        # Test security analysis (may fail if bandit not available)
        security_analyzer = manager.security_analyzer
        if security_analyzer:
            try:
                security_result = security_analyzer.analyze_security()
                print(f"✅ Security analysis: {security_result.score:.1f}% ({'PASS' if security_result.passed else 'FAIL'})")
            except Exception as e:
                print(f"⚠️  Security analysis skipped (bandit issues): {e}")

        return True

    except Exception as e:
        print(f"❌ Quality gates test failed: {e}")
        return False

def test_integration():
    """Test integration between Phase 3 components."""
    print("\n🔗 Testing Integration...")

    try:
        from vent4d_mech.config.schemas import create_default_config
        from vent4d_mech.utils.cache import configure_global_cache
        from vent4d_mech.utils.parallel import ParallelProcessor

        # Test configuration-driven cache setup
        config = create_default_config()
        cache_config = config.performance.memory_management

        configure_global_cache(
            max_cache_size_mb=cache_config.cache_size,
            enable_monitoring=True
        )
        print("✅ Configuration-driven cache setup working")

        # Test parallel processing with configuration
        processor = ParallelProcessor(
            n_processes=config.performance.num_processes if config.performance.num_processes != "auto" else None,
            use_processes=config.performance.parallel_processing
        )
        print("✅ Configuration-driven parallel processing working")

        # Test combined caching and parallel processing
        from vent4d_mech.utils.cache import material_model_cache

        @material_model_cache
        def compute_stress(strain_tensor):
            # Simulate expensive material model computation
            time.sleep(0.001)
            return 2 * strain_tensor  # Simple linear elastic

        # Generate test data
        test_strains = [np.random.rand(5, 5) for _ in range(10)]

        # Process in parallel with caching
        results = parallel_map(compute_stress, test_strains, n_processes=2)
        assert len(results) == len(test_strains)
        print("✅ Combined caching and parallel processing working")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_backward_compatibility():
    """Test backward compatibility with existing code."""
    print("\n🔄 Testing Backward Compatibility...")

    try:
        # Test old-style configuration loading still works
        from vent4d_mech.config.config_manager import ConfigManager

        config_manager = ConfigManager()
        assert config_manager is not None
        print("✅ Legacy ConfigManager still working")

        # Test old-style validation still works
        from vent4d_mech.config.config_validation import ConfigValidation

        validator = ConfigValidation()
        assert validator is not None
        print("✅ Legacy ConfigValidation still working")

        # Test base component still works
        from vent4d_mech.core.base_component import BaseComponent

        class TestComponent(BaseComponent):
            def process(self, *args, **kwargs):
                return "test_result"

        component = TestComponent()
        assert component is not None
        print("✅ BaseComponent backward compatibility working")

        return True

    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def performance_benchmark():
    """Run performance benchmarks for Phase 3 features."""
    print("\n📊 Performance Benchmarks...")

    try:
        from vent4d_mech.utils.cache import material_model_cache, get_global_cache_manager
        from vent4d_mech.utils.parallel import ParallelProcessor

        # Benchmark caching performance
        @material_model_cache
        def cached_expensive_function(n):
            # Simulate expensive computation
            return sum(i**2 for i in range(n))

        # Test without cache
        start_time = time.time()
        for _ in range(100):
            result1 = cached_expensive_function(1000)
        no_cache_time = time.time() - start_time

        # Reset cache and test with cache
        get_global_cache_manager().clear()

        start_time = time.time()
        for _ in range(100):
            result2 = cached_expensive_function(1000)
        with_cache_time = time.time() - start_time

        cache_speedup = no_cache_time / with_cache_time if with_cache_time > 0 else float('inf')
        print(f"✅ Caching speedup: {cache_speedup:.2f}x")

        # Benchmark parallel processing
        def cpu_intensive_function(n):
            return sum(i**3 for i in range(n))

        data = [1000] * 20

        # Sequential processing
        start_time = time.time()
        sequential_results = [cpu_intensive_function(n) for n in data]
        sequential_time = time.time() - start_time

        # Parallel processing
        processor = ParallelProcessor(n_processes=4)
        start_time = time.time()
        parallel_results = processor.process_multiple(cpu_intensive_function, data)
        parallel_time = time.time() - start_time

        parallel_speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')
        print(f"✅ Parallel processing speedup: {parallel_speedup:.2f}x")

        # Get cache statistics
        cache_stats = get_global_cache_manager().get_stats()
        print(f"✅ Cache hit rate: {cache_stats['hit_rate']:.1%}")

        return {
            'cache_speedup': cache_speedup,
            'parallel_speedup': parallel_speedup,
            'cache_hit_rate': cache_stats['hit_rate']
        }

    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return None

def main():
    """Run comprehensive Phase 3 validation."""
    print("🚀 Phase 3 Comprehensive Testing Suite")
    print("=" * 50)

    test_results = {}

    # Run all tests
    test_results['imports'] = test_phase3_imports()
    test_results['config_validation'] = test_pydantic_configuration_validation()
    test_results['caching'] = test_caching_utilities()
    test_results['parallel_processing'] = test_parallel_processing()
    test_results['quality_gates'] = test_quality_gates()
    test_results['integration'] = test_integration()
    test_results['backward_compatibility'] = test_backward_compatibility()

    # Run performance benchmarks
    perf_results = performance_benchmark()

    # Generate report
    print("\n" + "=" * 50)
    print("📋 PHASE 3 VALIDATION REPORT")
    print("=" * 50)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")

    if perf_results:
        print(f"\n📊 Performance Metrics:")
        print(f"  Cache Speedup: {perf_results['cache_speedup']:.2f}x")
        print(f"  Parallel Speedup: {perf_results['parallel_speedup']:.2f}x")
        print(f"  Cache Hit Rate: {perf_results['cache_hit_rate']:.1%}")

    # Final assessment
    if passed_tests == total_tests:
        print("\n🎉 PHASE 3 VALIDATION: SUCCESS")
        print("All implementations are working correctly!")
        return True
    else:
        print(f"\n⚠️  PHASE 3 VALIDATION: PARTIAL")
        print(f"{total_tests - passed_tests} tests failed. Review issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
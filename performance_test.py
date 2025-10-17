#!/usr/bin/env python3
"""
Performance Benchmarking Script for Vent4D-Mech Phase 2 Components

This script tests the performance characteristics of the new Phase 2 components
including BaseComponent, exception handling, type validation, and mechanical modeling.
"""

import time
import numpy as np
import sys
import tracemalloc
from pathlib import Path
from typing import Dict, Any

# Add src to path to find the module
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.base_component import BaseComponent
from core.exceptions import (
    ConfigurationError, ValidationError, ComputationError
)
from core.mechanical.mechanical_modeler import MechanicalModeler


class PerformanceBenchmark:
    """Performance benchmarking utility for Phase 2 components."""

    def __init__(self):
        self.results = {}

    def benchmark_base_component(self) -> Dict[str, Any]:
        """Benchmark BaseComponent performance."""
        print("Benchmarking BaseComponent...")

        class TestComponent(BaseComponent):
            def _get_default_config(self) -> dict:
                return {
                    'param1': 'value1',
                    'param2': 42,
                    'nested': {'inner': 'value'}
                }

            def process(self, data: np.ndarray) -> dict:
                # Mock processing
                result = np.mean(data)
                return self._package_results({'result': result})

        # Benchmark initialization
        init_times = []
        for _ in range(100):
            start = time.perf_counter()
            component = TestComponent()
            end = time.perf_counter()
            init_times.append(end - start)

        # Benchmark processing
        component = TestComponent()
        test_data = np.random.rand(64, 64, 64).astype(np.float32)

        process_times = []
        for _ in range(50):
            start = time.perf_counter()
            result = component.process(test_data)
            end = time.perf_counter()
            process_times.append(end - start)

        # Benchmark memory usage
        tracemalloc.start()
        components = [TestComponent() for _ in range(10)]
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            'initialization': {
                'mean_time': np.mean(init_times) * 1000,  # ms
                'std_time': np.std(init_times) * 1000,
                'min_time': np.min(init_times) * 1000,
                'max_time': np.max(init_times) * 1000
            },
            'processing': {
                'mean_time': np.mean(process_times) * 1000,
                'std_time': np.std(process_times) * 1000,
                'min_time': np.min(process_times) * 1000,
                'max_time': np.max(process_times) * 1000
            },
            'memory': {
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024
            }
        }

    def benchmark_mechanical_modeler(self) -> Dict[str, Any]:
        """Benchmark MechanicalModeler performance."""
        print("Benchmarking MechanicalModeler...")

        try:
            # Test different data sizes
            sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
            results = {}

            for size in sizes:
                print(f"  Testing size {size}...")

                modeler = MechanicalModeler(gpu=False)  # Use CPU for consistency

                # Generate test data
                strain_tensor = np.random.rand(*size, 3, 3).astype(np.float64)
                deformation_gradient = np.random.rand(*size, 3, 3).astype(np.float64)
                deformation_gradient += np.eye(3).reshape((1,) * len(size) + (3, 3))

                # Benchmark stress computation
                stress_times = []
                for _ in range(10):
                    start = time.perf_counter()
                    try:
                        result = modeler.process(strain_tensor, deformation_gradient, 'stress')
                        end = time.perf_counter()
                        stress_times.append(end - start)
                    except Exception as e:
                        print(f"    Error in stress computation: {e}")
                        break

                if stress_times:
                    results[f'size_{size[0]}'] = {
                        'stress_mean_time': np.mean(stress_times) * 1000,
                        'stress_std_time': np.std(stress_times) * 1000
                    }

            return results

        except Exception as e:
            print(f"MechanicalModeler benchmark failed: {e}")
            return {'error': str(e)}

    def benchmark_exceptions(self) -> Dict[str, Any]:
        """Benchmark exception handling performance."""
        print("Benchmarking exception handling...")

        # Test exception creation
        creation_times = []
        for _ in range(1000):
            start = time.perf_counter()
            error = ConfigurationError(
                "Test error message",
                component="TestComponent",
                config_key="test_key"
            )
            end = time.perf_counter()
            creation_times.append(end - start)

        # Test exception serialization
        error = ConfigurationError(
            "Test error message",
            component="TestComponent",
            config_key="test_key"
        )

        serialization_times = []
        for _ in range(1000):
            start = time.perf_counter()
            error_dict = error.to_dict()
            end = time.perf_counter()
            serialization_times.append(end - start)

        return {
            'creation': {
                'mean_time': np.mean(creation_times) * 1000,
                'std_time': np.std(creation_times) * 1000
            },
            'serialization': {
                'mean_time': np.mean(serialization_times) * 1000,
                'std_time': np.std(serialization_times) * 1000
            }
        }

    def benchmark_type_validation(self) -> Dict[str, Any]:
        """Benchmark type validation functions."""
        print("Benchmarking type validation...")

        from core.types import (
            validate_tensor_array, is_strain_tensor, is_stress_tensor
        )

        # Test different array sizes
        sizes = [(32, 32, 32, 3, 3), (64, 64, 64, 3, 3)]
        results = {}

        for size in sizes:
            test_array = np.random.rand(*size).astype(np.float64)

            # Benchmark validate_tensor_array
            validation_times = []
            for _ in range(100):
                start = time.perf_counter()
                is_valid = validate_tensor_array(test_array, expected_dims=5)
                end = time.perf_counter()
                validation_times.append(end - start)

            # Benchmark type-specific validators
            strain_times = []
            for _ in range(100):
                start = time.perf_counter()
                is_strain = is_strain_tensor(test_array)
                end = time.perf_counter()
                strain_times.append(end - start)

            stress_times = []
            for _ in range(100):
                start = time.perf_counter()
                is_stress = is_stress_tensor(test_array)
                end = time.perf_counter()
                stress_times.append(end - start)

            results[f'size_{size[0]}'] = {
                'validation_mean_time': np.mean(validation_times) * 1000,
                'strain_check_mean_time': np.mean(strain_times) * 1000,
                'stress_check_mean_time': np.mean(stress_times) * 1000
            }

        return results

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("Starting Vent4D-Mech Phase 2 Performance Benchmarks...")
        print("=" * 60)

        try:
            self.results['base_component'] = self.benchmark_base_component()
            print("✓ BaseComponent benchmark completed")
        except Exception as e:
            print(f"✗ BaseComponent benchmark failed: {e}")
            self.results['base_component'] = {'error': str(e)}

        try:
            self.results['mechanical_modeler'] = self.benchmark_mechanical_modeler()
            print("✓ MechanicalModeler benchmark completed")
        except Exception as e:
            print(f"✗ MechanicalModeler benchmark failed: {e}")
            self.results['mechanical_modeler'] = {'error': str(e)}

        try:
            self.results['exceptions'] = self.benchmark_exceptions()
            print("✓ Exception handling benchmark completed")
        except Exception as e:
            print(f"✗ Exception handling benchmark failed: {e}")
            self.results['exceptions'] = {'error': str(e)}

        try:
            self.results['type_validation'] = self.benchmark_type_validation()
            print("✓ Type validation benchmark completed")
        except Exception as e:
            print(f"✗ Type validation benchmark failed: {e}")
            self.results['type_validation'] = {'error': str(e)}

        print("=" * 60)
        print("All benchmarks completed!")

        return self.results

    def print_summary(self):
        """Print performance summary."""
        print("\n" + "=" * 60)
        print("VENT4D-MECH PHASE 2 PERFORMANCE SUMMARY")
        print("=" * 60)

        # BaseComponent summary
        if 'base_component' in self.results and 'error' not in self.results['base_component']:
            bc = self.results['base_component']
            print(f"\nBaseComponent Performance:")
            print(f"  Initialization: {bc['initialization']['mean_time']:.3f} ± {bc['initialization']['std_time']:.3f} ms")
            print(f"  Processing:     {bc['processing']['mean_time']:.3f} ± {bc['processing']['std_time']:.3f} ms")
            print(f"  Memory (10 instances): {bc['memory']['peak_mb']:.2f} MB peak")

        # MechanicalModeler summary
        if 'mechanical_modeler' in self.results and 'error' not in self.results['mechanical_modeler']:
            mm = self.results['mechanical_modeler']
            print(f"\nMechanicalModeler Performance:")
            for size_key, data in mm.items():
                if 'stress_mean_time' in data:
                    print(f"  {size_key}: {data['stress_mean_time']:.3f} ± {data['stress_std_time']:.3f} ms")

        # Exception handling summary
        if 'exceptions' in self.results and 'error' not in self.results['exceptions']:
            ex = self.results['exceptions']
            print(f"\nException Handling Performance:")
            print(f"  Creation: {ex['creation']['mean_time']:.3f} ± {ex['creation']['std_time']:.3f} ms")
            print(f"  Serialization: {ex['serialization']['mean_time']:.3f} ± {ex['serialization']['std_time']:.3f} ms")

        # Type validation summary
        if 'type_validation' in self.results and 'error' not in self.results['type_validation']:
            tv = self.results['type_validation']
            print(f"\nType Validation Performance:")
            for size_key, data in tv.items():
                if 'validation_mean_time' in data:
                    print(f"  {size_key}:")
                    print(f"    General validation: {data['validation_mean_time']:.3f} ms")
                    print(f"    Strain check:       {data['strain_check_mean_time']:.3f} ms")
                    print(f"    Stress check:       {data['stress_check_mean_time']:.3f} ms")


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    benchmark.print_summary()

    # Save results
    import json
    with open('performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to performance_results.json")
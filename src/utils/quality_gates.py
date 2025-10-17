"""
Quality Gates and Metrics

This module provides comprehensive quality gates and metrics collection for the
Vent4D-Mech framework, ensuring code quality, performance standards, and
security compliance are maintained throughout development.

Key Features:
- Code quality metrics collection
- Performance benchmarks and monitoring
- Security vulnerability scanning
- Test coverage validation
- Documentation completeness checks
- CI/CD integration utilities
"""

import logging
import subprocess
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
import ast
import sys

try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

try:
    import bandit
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False

try:
    import mypy
    MYPY_AVAILABLE = True
except ImportError:
    MYPY_AVAILABLE = False


class QualityGateResult:
    """Result of a quality gate check."""

    def __init__(
        self,
        name: str,
        passed: bool,
        score: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ):
        """
        Initialize quality gate result.

        Args:
            name: Name of the quality gate
            passed: Whether the gate passed
            score: Optional score (0-100)
            details: Additional details about the result
            errors: List of errors encountered
            warnings: List of warnings encountered
        """
        self.name = name
        self.passed = passed
        self.score = score
        self.details = details or {}
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'passed': self.passed,
            'score': self.score,
            'details': self.details,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': self.timestamp
        }


class CodeQualityAnalyzer:
    """Analyzes code quality metrics."""

    def __init__(self, source_dir: str = "vent4d_mech"):
        """
        Initialize code quality analyzer.

        Args:
            source_dir: Source code directory to analyze
        """
        self.source_dir = Path(source_dir)
        self.logger = logging.getLogger(__name__)

    def analyze_complexity(self) -> QualityGateResult:
        """
        Analyze cyclomatic complexity of source code.

        Returns:
            Quality gate result for complexity analysis
        """
        self.logger.info("Analyzing code complexity...")

        max_complexity = 10
        complexity_issues = []
        total_functions = 0
        complex_functions = 0

        try:
            # Use radon for complexity analysis if available
            try:
                import radon.complexity as radon_cc
                from radon.visitors import ComplexityVisitor

                for py_file in self.source_dir.rglob("*.py"):
                    if "__pycache__" in str(py_file):
                        continue

                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            source_code = f.read()

                        visitor = ComplexityVisitor.from_code(source_code)
                        for item in visitor.functions:
                            total_functions += 1
                            if item.complexity > max_complexity:
                                complexity_issues.append(
                                    f"{py_file}:{item.lineno}: {item.name} (complexity: {item.complexity})"
                                )
                                complex_functions += 1

                    except Exception as e:
                        self.logger.warning(f"Could not analyze {py_file}: {e}")

            except ImportError:
                # Fallback: basic AST analysis
                for py_file in self.source_dir.rglob("*.py"):
                    if "__pycache__" in str(py_file):
                        continue

                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            source_code = f.read()

                        tree = ast.parse(source_code)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                total_functions += 1
                                # Simple complexity heuristic based on control structures
                                complexity = self._calculate_simple_complexity(node)
                                if complexity > max_complexity:
                                    complexity_issues.append(
                                        f"{py_file}:{node.lineno}: {node.name} (complexity: {complexity})"
                                    )
                                    complex_functions += 1

                    except Exception as e:
                        self.logger.warning(f"Could not analyze {py_file}: {e}")

            # Calculate score
            if total_functions > 0:
                complexity_ratio = complex_functions / total_functions
                score = max(0, 100 - (complexity_ratio * 100))
            else:
                score = 100

            passed = len(complexity_issues) == 0

            return QualityGateResult(
                name="Code Complexity",
                passed=passed,
                score=score,
                details={
                    'total_functions': total_functions,
                    'complex_functions': complex_functions,
                    'complexity_issues': complexity_issues,
                    'max_complexity_threshold': max_complexity
                },
                warnings=complexity_issues if passed else []
            )

        except Exception as e:
            return QualityGateResult(
                name="Code Complexity",
                passed=False,
                errors=[f"Complexity analysis failed: {e}"]
            )

    def _calculate_simple_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate simple complexity metric for a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def analyze_documentation_coverage(self) -> QualityGateResult:
        """
        Analyze documentation coverage.

        Returns:
            Quality gate result for documentation coverage
        """
        self.logger.info("Analyzing documentation coverage...")

        min_coverage = 80
        documented_items = 0
        total_items = 0
        undocumented_items = []

        try:
            for py_file in self.source_dir.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()

                    tree = ast.parse(source_code)

                    # Check modules
                    total_items += 1
                    if ast.get_docstring(tree):
                        documented_items += 1
                    else:
                        undocumented_items.append(f"Module: {py_file}")

                    # Check classes and functions
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            total_items += 1
                            if ast.get_docstring(node):
                                documented_items += 1
                            else:
                                undocumented_items.append(f"{py_file}:{node.lineno}: {node.name}")

                except Exception as e:
                    self.logger.warning(f"Could not analyze {py_file}: {e}")

            # Calculate coverage
            coverage = (documented_items / total_items * 100) if total_items > 0 else 0
            passed = coverage >= min_coverage

            return QualityGateResult(
                name="Documentation Coverage",
                passed=passed,
                score=coverage,
                details={
                    'documented_items': documented_items,
                    'total_items': total_items,
                    'coverage_percentage': coverage,
                    'min_coverage_threshold': min_coverage,
                    'undocumented_items': undocumented_items[:20]  # Limit to first 20
                },
                warnings=undocumented_items if passed else []
            )

        except Exception as e:
            return QualityGateResult(
                name="Documentation Coverage",
                passed=False,
                errors=[f"Documentation analysis failed: {e}"]
            )


class TestCoverageAnalyzer:
    """Analyzes test coverage metrics."""

    def __init__(self, source_dir: str = "vent4d_mech", test_dir: str = "tests"):
        """
        Initialize test coverage analyzer.

        Args:
            source_dir: Source code directory
            test_dir: Test directory
        """
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.logger = logging.getLogger(__name__)

    def analyze_coverage(self) -> QualityGateResult:
        """
        Analyze test coverage using coverage.py.

        Returns:
            Quality gate result for test coverage
        """
        if not COVERAGE_AVAILABLE:
            return QualityGateResult(
                name="Test Coverage",
                passed=False,
                errors=["coverage.py not available. Install with: pip install coverage"]
            )

        self.logger.info("Analyzing test coverage...")

        min_coverage = 80
        try:
            # Run coverage analysis
            cov = coverage.Coverage(source=[str(self.source_dir)])
            cov.start()

            # Discover and run tests
            try:
                import pytest
                exit_code = pytest.main([str(self.test_dir), "-q"])
            except ImportError:
                # Fallback: simple test discovery
                exit_code = self._run_simple_tests()

            cov.stop()
            cov.save()

            # Generate coverage report
            total_coverage = cov.report(file=open("/dev/null", "w"))

            # Get detailed coverage data
            coverage_data = {}
            for filename in cov.get_data().measured_files():
                if self.source_dir in Path(filename).parents:
                    analysis = cov.analysis2(filename)
                    coverage_data[filename] = {
                        'statements': analysis[1],
                        'missing': analysis[2],
                        'covered': analysis[3],
                        'coverage_percent': analysis[4]
                    }

            passed = total_coverage >= min_coverage

            return QualityGateResult(
                name="Test Coverage",
                passed=passed,
                score=total_coverage,
                details={
                    'total_coverage_percentage': total_coverage,
                    'min_coverage_threshold': min_coverage,
                    'coverage_data': coverage_data,
                    'pytest_exit_code': exit_code if 'exit_code' in locals() else None
                }
            )

        except Exception as e:
            return QualityGateResult(
                name="Test Coverage",
                passed=False,
                errors=[f"Coverage analysis failed: {e}"]
            )

    def _run_simple_tests(self) -> int:
        """Simple test runner fallback."""
        exit_code = 0
        for test_file in self.test_dir.rglob("test_*.py"):
            try:
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                if result.returncode != 0:
                    exit_code = 1
                    self.logger.error(f"Test {test_file} failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                exit_code = 1
                self.logger.error(f"Test {test_file} timed out")
            except Exception as e:
                exit_code = 1
                self.logger.error(f"Could not run test {test_file}: {e}")

        return exit_code


class SecurityAnalyzer:
    """Analyzes security vulnerabilities."""

    def __init__(self, source_dir: str = "vent4d_mech"):
        """
        Initialize security analyzer.

        Args:
            source_dir: Source code directory to analyze
        """
        self.source_dir = Path(source_dir)
        self.logger = logging.getLogger(__name__)

    def analyze_security(self) -> QualityGateResult:
        """
        Analyze security vulnerabilities using bandit.

        Returns:
            Quality gate result for security analysis
        """
        if not BANDIT_AVAILABLE:
            return QualityGateResult(
                name="Security Analysis",
                passed=False,
                errors=["bandit not available. Install with: pip install bandit"]
            )

        self.logger.info("Running security analysis...")

        max_high_issues = 0
        max_medium_issues = 2
        max_low_issues = 10

        try:
            # Run bandit security analysis
            result = subprocess.run(
                [
                    "bandit",
                    "-r", str(self.source_dir),
                    "-f", "json",
                    "-q",  # Quiet mode
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                # Parse bandit output
                try:
                    bandit_data = json.loads(result.stdout)
                    issues = bandit_data.get('results', [])

                    high_issues = [i for i in issues if i.get('issue_severity') == 'HIGH']
                    medium_issues = [i for i in issues if i.get('issue_severity') == 'MEDIUM']
                    low_issues = [i for i in issues if i.get('issue_severity') == 'LOW']

                    passed = (
                        len(high_issues) <= max_high_issues and
                        len(medium_issues) <= max_medium_issues and
                        len(low_issues) <= max_low_issues
                    )

                    # Calculate security score
                    total_issues = len(issues)
                    if total_issues == 0:
                        score = 100
                    else:
                        # Weight issues by severity
                        weighted_score = len(high_issues) * 10 + len(medium_issues) * 5 + len(low_issues) * 1
                        score = max(0, 100 - weighted_score)

                    return QualityGateResult(
                        name="Security Analysis",
                        passed=passed,
                        score=score,
                        details={
                            'total_issues': total_issues,
                            'high_issues': len(high_issues),
                            'medium_issues': len(medium_issues),
                            'low_issues': len(low_issues),
                            'thresholds': {
                                'max_high': max_high_issues,
                                'max_medium': max_medium_issues,
                                'max_low': max_low_issues
                            },
                            'issues': [
                                {
                                    'file': issue.get('filename'),
                                    'line': issue.get('line_number'),
                                    'severity': issue.get('issue_severity'),
                                    'confidence': issue.get('issue_cseverity'),
                                    'text': issue.get('issue_text'),
                                    'test_id': issue.get('test_id'),
                                    'test_name': issue.get('test_name')
                                }
                                for issue in issues[:20]  # Limit to first 20
                            ]
                        },
                        warnings=[f"Security issue: {i.get('issue_text')}" for i in issues] if passed else []
                    )

                except json.JSONDecodeError:
                    return QualityGateResult(
                        name="Security Analysis",
                        passed=False,
                        errors=["Could not parse bandit output"]
                    )
            else:
                return QualityGateResult(
                    name="Security Analysis",
                    passed=False,
                    errors=[f"Bandit execution failed: {result.stderr}"]
                )

        except subprocess.TimeoutExpired:
            return QualityGateResult(
                name="Security Analysis",
                passed=False,
                errors=["Security analysis timed out"]
            )
        except Exception as e:
            return QualityGateResult(
                name="Security Analysis",
                passed=False,
                errors=[f"Security analysis failed: {e}"]
            )


class TypeSafetyAnalyzer:
    """Analyzes type safety using mypy."""

    def __init__(self, source_dir: str = "vent4d_mech"):
        """
        Initialize type safety analyzer.

        Args:
            source_dir: Source code directory to analyze
        """
        self.source_dir = Path(source_dir)
        self.logger = logging.getLogger(__name__)

    def analyze_type_safety(self) -> QualityGateResult:
        """
        Analyze type safety using mypy.

        Returns:
            Quality gate result for type safety analysis
        """
        if not MYPY_AVAILABLE:
            return QualityGateResult(
                name="Type Safety",
                passed=False,
                errors=["mypy not available. Install with: pip install mypy"]
            )

        self.logger.info("Running type safety analysis...")

        try:
            # Run mypy type checking
            result = subprocess.run(
                [
                    "mypy",
                    str(self.source_dir),
                    "--show-error-codes",
                    "--no-error-summary",
                    "--json-report", "/tmp/mypy_report"
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Parse mypy results
            mypy_errors = []
            mypy_warnings = []

            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        if 'error:' in line:
                            mypy_errors.append(line)
                        elif 'warning:' in line:
                            mypy_warnings.append(line)

            # Try to read JSON report if available
            json_data = {}
            try:
                with open("/tmp/mypy_report/index.json", 'r') as f:
                    json_data = json.load(f)
            except:
                pass

            passed = result.returncode == 0
            total_issues = len(mypy_errors) + len(mypy_warnings)

            # Calculate type safety score
            if total_issues == 0:
                score = 100
            else:
                score = max(0, 100 - (len(mypy_errors) * 10 + len(mypy_warnings) * 5))

            return QualityGateResult(
                name="Type Safety",
                passed=passed,
                score=score,
                details={
                    'mypy_return_code': result.returncode,
                    'total_issues': total_issues,
                    'errors': len(mypy_errors),
                    'warnings': len(mypy_warnings),
                    'error_messages': mypy_errors[:10],  # Limit to first 10
                    'warning_messages': mypy_warnings[:10],  # Limit to first 10
                    'json_report': json_data
                },
                errors=mypy_errors,
                warnings=mypy_warnings
            )

        except subprocess.TimeoutExpired:
            return QualityGateResult(
                name="Type Safety",
                passed=False,
                errors=["Type safety analysis timed out"]
            )
        except Exception as e:
            return QualityGateResult(
                name="Type Safety",
                passed=False,
                errors=[f"Type safety analysis failed: {e}"]
            )


class QualityGateManager:
    """Manages and runs all quality gates."""

    def __init__(self, source_dir: str = "vent4d_mech", test_dir: str = "tests"):
        """
        Initialize quality gate manager.

        Args:
            source_dir: Source code directory
            test_dir: Test directory
        """
        self.source_dir = source_dir
        self.test_dir = test_dir
        self.logger = logging.getLogger(__name__)

        # Initialize analyzers
        self.code_analyzer = CodeQualityAnalyzer(source_dir)
        self.coverage_analyzer = TestCoverageAnalyzer(source_dir, test_dir)
        self.security_analyzer = SecurityAnalyzer(source_dir)
        self.type_analyzer = TypeSafetyAnalyzer(source_dir)

    def run_all_gates(self) -> Dict[str, QualityGateResult]:
        """
        Run all quality gates.

        Returns:
            Dictionary of quality gate results
        """
        self.logger.info("Running all quality gates...")

        results = {}

        # Code quality gates
        results['complexity'] = self.code_analyzer.analyze_complexity()
        results['documentation'] = self.code_analyzer.analyze_documentation_coverage()

        # Test coverage
        results['test_coverage'] = self.coverage_analyzer.analyze_coverage()

        # Security
        results['security'] = self.security_analyzer.analyze_security()

        # Type safety
        results['type_safety'] = self.type_analyzer.analyze_type_safety()

        return results

    def generate_report(self, results: Dict[str, QualityGateResult]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.

        Args:
            results: Quality gate results

        Returns:
            Comprehensive quality report
        """
        total_gates = len(results)
        passed_gates = sum(1 for r in results.values() if r.passed)
        overall_passed = passed_gates == total_gates

        # Calculate overall score
        scores = [r.score for r in results.values() if r.score is not None]
        overall_score = sum(scores) / len(scores) if scores else 0

        report = {
            'summary': {
                'total_gates': total_gates,
                'passed_gates': passed_gates,
                'failed_gates': total_gates - passed_gates,
                'overall_passed': overall_passed,
                'overall_score': overall_score,
                'timestamp': time.time()
            },
            'gates': {name: result.to_dict() for name, result in results.items()},
            'recommendations': self._generate_recommendations(results)
        }

        return report

    def _generate_recommendations(self, results: Dict[str, QualityGateResult]) -> List[str]:
        """Generate improvement recommendations based on results."""
        recommendations = []

        for name, result in results.items():
            if not result.passed:
                if name == "Code Complexity":
                    recommendations.append(
                        "Consider refactoring complex functions to reduce cyclomatic complexity. "
                        "Break down large functions into smaller, more focused units."
                    )
                elif name == "Documentation Coverage":
                    recommendations.append(
                        "Add docstrings to all public functions, classes, and modules. "
                        "Include parameter descriptions and return value specifications."
                    )
                elif name == "Test Coverage":
                    recommendations.append(
                        "Increase test coverage by adding unit tests for uncovered code paths. "
                        "Focus on critical functionality and edge cases."
                    )
                elif name == "Security Analysis":
                    recommendations.append(
                        "Address security vulnerabilities identified by bandit. "
                        "Review and fix high and medium severity issues first."
                    )
                elif name == "Type Safety":
                    recommendations.append(
                        "Add type hints to function signatures and variables. "
                        "Fix mypy type checking errors to improve code safety."
                    )

        return recommendations

    def save_report(self, report: Dict[str, Any], output_file: str = "quality_report.json") -> None:
        """
        Save quality report to file.

        Args:
            report: Quality report
            output_file: Output file path
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Quality report saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Could not save quality report: {e}")


def run_quality_gates(source_dir: str = "vent4d_mech", test_dir: str = "tests") -> Dict[str, Any]:
    """
    Run all quality gates and return results.

    Args:
        source_dir: Source code directory
        test_dir: Test directory

    Returns:
        Quality report
    """
    manager = QualityGateManager(source_dir, test_dir)
    results = manager.run_all_gates()
    report = manager.generate_report(results)
    manager.save_report(report)
    return report
# Vent4D-Mech Development Guidelines

## Current Project Status

**Project State**: Alpha v0.1.0 - Core architecture complete, ready for feature implementation and testing

## Project Structure and Patterns

### Modular Architecture (Fully Implemented)
- Each core module (registration, deformation, mechanical, etc.) is self-contained and fully present
- All modules follow consistent patterns with proper initialization files
- Use dependency injection for configuration and optional dependencies
- Implement fallback mechanisms when GPU/deep learning dependencies are unavailable
- Follow the pipeline pattern where each stage can run independently or as part of the complete workflow

### Configuration-Driven Design
- All modules should accept configuration dictionaries
- Use sensible defaults that work for most use cases
- Support both YAML files and programmatic configuration
- Validate configuration parameters and provide clear error messages
- **Note**: Verify config/default.yaml exists and is comprehensive

### GPU/CPU Duality
- Always provide CPU fallbacks for GPU operations
- Use CuPy for NumPy-compatible GPU acceleration
- Check GPU availability with try/except blocks, not version checks
- Implement memory management for large 3D volumes

## Current Implementation Priorities

### High Priority - Missing Components
1. **CLI Interface**: Implement `vent4d_mech/cli.py` for the `vent4d-mech` command referenced in setup.py
2. **Configuration Files**: Ensure `config/default.yaml` exists and covers all modules
3. **Testing Suite**: Implement comprehensive tests in `tests/` directory
4. **Examples**: Populate `examples/` directory with working demonstrations

### Medium Priority - Enhancement
1. **Documentation**: Verify all docstrings are complete and follow consistent style
2. **Performance**: Optimize GPU memory management and parallel processing
3. **Validation**: Implement validation tools for SPECT/CT comparison
4. **Integration**: Ensure seamless module integration and error handling

## Code Implementation Guidelines

### Adding New Constitutive Models
1. Create model class in `vent4d_mech/core/mechanical/` inheriting from base `ConstitutiveModel`
2. Implement required methods: `compute_stress()`, `compute_tangent_modulus()`
3. Add configuration parameters to config system
4. Update unit tests in `tests/test_mechanical.py`
5. Add entry to model registry in `mechanical_modeler.py`

### Extending Registration Methods
1. Create new registration class in `vent4d_mech/core/registration/`
2. Inherit from base `ImageRegistration` class
3. Implement `register()` method with proper error handling
4. Add method-specific configuration options
5. Register method in `__init__.py` and configuration schema
6. Add unit tests with synthetic data

### Integration with External Data Sources
1. Extend `vent4d_mech/core/microstructure/microstructure_db.py`
2. Add configuration options for API endpoints and authentication
3. Implement data caching and validation mechanisms
4. Add error handling for network issues and data format problems
5. Create synthetic data fallbacks for testing

### CLI Implementation (Priority)
1. Create `vent4d_mech/cli.py` with Click-based command interface
2. Implement commands for: run, configure, validate, test
3. Add proper argument parsing and help text
4. Include configuration file validation
5. Add progress reporting and logging

## Performance Considerations

### Memory Management
- Use batch processing for large 3D volumes
- Implement memory-efficient array operations
- Use generators for data streaming when possible
- Monitor memory usage in long-running computations

### GPU Optimization
- Minimize CPU-GPU data transfers
- Use in-place operations where possible
- Implement custom CUDA kernels for performance-critical operations
- Use memory pooling for repeated allocations

### Parallel Processing
- Use joblib for patient-level parallelization
- Implement multiprocessing for CPU-bound tasks
- Consider Dask for out-of-core computation on large datasets
- Use threading for I/O-bound operations

## Testing Strategy

### Required Test Implementation
- **Unit Tests**: Individual component functionality with synthetic data (missing)
- **Integration Tests**: Pipeline workflow validation (missing)
- **Performance Tests**: GPU acceleration and memory usage validation (missing)
- **Validation Tests**: Comparison against clinical SPECT/CT data (missing)

### Test Data Management
- Use synthetic data generation for reproducible testing
- Include phantom studies for algorithm validation
- Store test data in `tests/data/` with clear documentation
- Implement data versioning for clinical validation datasets

### Continuous Integration Setup
- Test on Python 3.9, 3.10, 3.11
- Test both CPU and GPU environments
- Include code quality checks (black, flake8, mypy)
- Add performance regression tests

## Documentation Requirements

### Current State
- ✅ Comprehensive README.md with installation and usage instructions
- ✅ Module documentation structure in place
- ⚠️ API documentation may need verification
- ⚠️ Tutorial examples need implementation

### Code Documentation Standards
- All public APIs must have comprehensive docstrings
- Include usage examples in docstrings
- Document mathematical formulations with proper notation
- Provide references to relevant literature
- Follow consistent Google/NumPy style docstrings

### User Documentation
- ✅ Clear installation instructions in README
- ✅ Configuration reference with examples
- ⏳ Step-by-step tutorials for common use cases (in examples/)
- ⏳ Performance tuning guidelines
- ⏳ Troubleshooting guide for common issues

## Release and Distribution

### Current Status
- ✅ Version management set to v0.1.0
- ✅ Setup.py complete with proper configuration
- ✅ Package distribution ready
- ⏳ CHANGELOG needs implementation
- ⏳ Release testing procedures need establishment

### Quality Assurance
- Perform integration testing before releases
- Validate performance on benchmark datasets
- Review documentation completeness
- Check backward compatibility
- Ensure all tests pass in CI environment

## Development Workflow

### Before Adding Features
1. Use `architecture-guardian` to analyze scope and impact
2. Create task-specific checklist
3. Identify potential architectural violations
4. Plan testing strategy
5. Update relevant memory files

### After Implementation
1. Use `static-code-analyzer` for code quality checks
2. Use `test-automation-validator` for comprehensive testing
3. Review and address all identified issues
4. Verify documentation completeness
5. Update Serena project knowledge base

## Quality Gates
- No code changes should be committed without passing static analysis
- All tests must pass before merging changes
- Documentation must be updated for any API changes
- Architecture rules must never be violated
- CLI interface must be tested for all new features
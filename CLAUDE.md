# CLAUDE.md

This file provides development workflow guidelines for using subagents to ensure code quality and architectural integrity.

## Development Workflow with Subagents

This project implements a structured development workflow using specialized subagents to ensure code quality and architectural integrity.

### Before File Editing: Prediction and Prevention

**1. Change Scope Prediction**
- Use `architecture-guardian` subagent to predict all files that will be affected by the changes
- Identify dependencies and potential ripple effects across the codebase
- Present a comprehensive list of files that may need modification
- Analyze import relationships and module interdependencies

**2. Procedure Guidance**
- Provide a task-specific checklist for the current operation (e.g., renaming, refactoring, adding features)
- Include step-by-step instructions that follow best practices
- Ensure all necessary preconditions and prerequisites are identified
- Define success criteria and validation steps

**3. Architecture Rule Warning**
- Use `architecture-guardian` subagent to identify potential architecture rule violations
- Warn about patterns that could break modularity, circular dependencies, or other architectural invariants
- Suggest alternative approaches that maintain architectural integrity
- Identify potential breaking changes and migration requirements

### After File Editing: Verification and Synchronization

**1. Automated Inspection**
- Static analysis is automatically configured via pre-commit hooks
- Use `static-code-analyzer` subagent to run comprehensive analysis
- Check for architecture rule violations, code quality issues, and potential bugs
- Review hook results and address any identified issues immediately
- Verify compliance with coding standards and style guides

**2. Code Testing**
- Use `test-automation-validator` subagent to run the full test suite
- Validate that all tests pass and coverage requirements are met
- If tests fail, diagnose issues and implement fixes
- Ensure new functionality is properly tested with appropriate test cases
- Run integration tests to verify system-wide functionality

**3. Documentation**
- Update docstrings to follow the project's documentation style
- Include clear parameter descriptions, return value specifications, and usage examples
- Document any new APIs or modified interfaces
- Ensure all public methods and classes have comprehensive documentation
- Update README files and API documentation as needed

**4. Knowledge Synchronization**
- Update Serena project documentation when code structure or APIs change
- Sync memory stores with new architectural decisions or patterns
- Update configuration files if new parameters or options are added
- Ensure all documentation reflects the current state of the codebase
- Inform team members of significant architectural changes

## Subagent Usage Guidelines

### Architecture Guardian
- **When to use**: Before making structural changes, during refactoring, when adding new modules
- **Purpose**: Maintain architectural integrity and prevent technical debt
- **Key functions**:
  - Dependency analysis and impact assessment
  - Pattern validation and architectural rule enforcement
  - Circular dependency detection
  - Module boundary verification
  - Interface compatibility checking

### Static Code Analyzer
- **When to use**: After code changes, before commits, during code reviews
- **Purpose**: Ensure code quality and catch potential issues early
- **Key functions**:
  - Syntax checking and parsing validation
  - Linting and style guide compliance
  - Type checking and inference
  - Security scanning and vulnerability detection
  - Performance anti-pattern identification

### Test Automation Validator
- **When to use**: After implementation, before merging, when validating fixes
- **Purpose**: Ensure functional correctness and test coverage
- **Key functions**:
  - Test execution and result validation
  - Coverage analysis and reporting
  - Performance regression testing
  - Integration test validation
  - Requirements compliance verification

## General Development Patterns

### For Any Code Modification
1. **Preparation Phase**
   - Use `architecture-guardian` to analyze scope and impact
   - Create a task-specific checklist
   - Identify potential architectural violations
   - Plan testing strategy

2. **Implementation Phase**
   - Make code changes following the identified patterns
   - Ensure code follows established style guidelines
   - Write comprehensive tests for new functionality
   - Update documentation as you code

3. **Validation Phase**
   - Use `static-code-analyzer` for code quality checks
   - Use `test-automation-validator` for comprehensive testing
   - Review and address all identified issues
   - Verify documentation completeness

4. **Integration Phase**
   - Update Serena project knowledge base
   - Sync relevant documentation and configurations
   - Communicate changes to team members
   - Ensure all systems reflect current state

### Quality Gates
- No code changes should be committed without passing static analysis
- All tests must pass before merging changes
- Documentation must be updated for any API changes
- Architecture rules must never be violated

### Continuous Improvement
- Regularly review and update architectural guidelines
- Enhance test coverage based on identified gaps
- Refactor code when quality metrics indicate issues
- Maintain up-to-date documentation and knowledge bases

## Hook Configuration

Pre-commit hooks are automatically configured to run:
- Code formatting
- Linting and style checks
- Static analysis validation
- Basic syntax verification

These hooks ensure code quality standards are maintained before any commit is made, providing immediate feedback on potential issues.

## Error Handling and Recovery

When subagents identify issues:
1. **Architecture Violations**: Refactor to maintain architectural integrity
2. **Static Analysis Issues**: Fix code quality problems immediately
3. **Test Failures**: Debug and resolve all failing tests
4. **Documentation Gaps**: Complete missing documentation before proceeding

This systematic approach ensures consistent code quality, architectural integrity, and comprehensive documentation throughout the development lifecycle.
# CLAUDE.md

**MANDATORY DEVELOPMENT WORKFLOW GUIDELINES**
*For AI-Assisted Development with Subagents*

This document provides **MANDATORY** development workflow guidelines for using subagents to ensure code quality, architectural integrity, and reliable outcomes. **These guidelines must be followed exactly - no exceptions, no shortcuts.**

## 🚨 CRITICAL: MANDATORY COMPLIANCE REQUIRED

### ABSOLUTE REQUIREMENTS
- **100% compliance** with all workflow steps is required
- **No exceptions** to guidelines without explicit approval
- **Quality gates are blocking** - work stops until issues resolved
- **Subagents are integrated tools**, not post-validation checkboxes
- **Test-Driven Development (TDD)** is mandatory for all new features

### VIOLATIONS CONSEQUENCES
- **Process violations** = Immediate stop and reassessment
- **Quality gate failures** = Fix before proceeding
- **Guideline non-compliance** = Root cause analysis required
- **Defect acceptance** = Unacceptable - fix all issues before considering work complete

---

## 📋 MANDATORY DEVELOPMENT WORKFLOW

### Phase 1: PRE-IMPLEMENTATION ANALYSIS (MANDATORY)

#### 1.1 Change Scope Prediction (MANDATORY)
**When**: BEFORE any code changes are made
**Tool**: `architecture-guardian` subagent

**Required Analysis**:
- [ ] Predict ALL files that will be affected by the changes
- [ ] Identify dependencies and potential ripple effects across the codebase
- [ ] Present comprehensive list of files that may need modification
- [ ] Analyze import relationships and module interdependencies
- [ ] Assess impact on existing functionality
- [ ] Identify potential breaking changes

**Success Criteria**:
- Complete impact analysis documented
- All affected files identified
- No critical architectural violations predicted
- Dependency risks assessed and documented

#### 1.2 Procedure Guidance (MANDATORY)
**When**: BEFORE starting implementation
**Action**: Create task-specific checklist

**Required Elements**:
- [ ] Step-by-step instructions following best practices
- [ ] Success criteria clearly defined and measurable
- [ ] All necessary preconditions identified
- [ ] Validation checkpoints established
- [ ] Testing strategy planned
- [ ] Risk assessment completed
- [ ] Rollback plan defined

#### 1.3 Architecture Rule Warning (MANDATORY)
**When**: BEFORE implementation begins
**Tool**: `architecture-guardian` subagent

**Required Validation**:
- [ ] Identify ALL potential architecture rule violations
- [ ] Assess impact on modularity and maintainability
- [ ] Check for circular dependency risks
- [ ] Validate module boundary integrity
- [ ] Analyze interface compatibility
- [ ] Suggest alternative approaches if violations found
- [ ] Document migration requirements for breaking changes

**Blocking Conditions**:
- ANY critical architectural violation = STOP until resolved
- Unclear impact assessment = STOP until clarified
- Missing alternative approaches = STOP until provided

### Phase 2: IMPLEMENTATION (GUIDED)

#### 2.1 Test-Driven Development (MANDATORY)
**Rule**: Tests MUST be written BEFORE implementation

**TDD Workflow**:
1. Write failing test for new functionality
2. Implement minimal code to make test pass
3. Refactor while maintaining test coverage
4. Add additional tests for edge cases
5. Ensure 100% test pass rate AT ALL TIMES

**Quality Requirements**:
- [ ] Tests written before implementation
- [ ] 100% test pass rate maintained continuously
- [ ] Test coverage 80%+ for implemented features
- [ ] No tests skipped or ignored
- [ ] All edge cases covered

#### 2.2 Incremental Development (MANDATORY)
**Rule**: Small, focused changes with continuous validation

**Implementation Requirements**:
- [ ] Changes are small and focused (single feature or bug fix)
- [ ] Tests pass after each small change
- [ ] Documentation updated with each change
- [ ] No quality degradation introduced
- [ ] Continuous integration testing maintained

#### 2.3 Quality Gates (MANDATORY)
**Rule**: ALL quality gates must be passed before proceeding

**Blocking Quality Gates**:
- [ ] **Test Pass Rate**: 100% (NO exceptions)
- [ ] **Test Coverage**: 80%+ for implemented features
- [ ] **Static Analysis**: Zero critical issues
- [ ] **Type Checking**: Passes without errors
- [ ] **Documentation**: Complete and accurate
- [ ] **Architecture**: No violations

**Consequences**:
- ANY quality gate failure = STOP development until fixed
- NO workarounds or exceptions permitted
- Root cause analysis required for all failures

### Phase 3: POST-IMPLEMENTATION VERIFICATION (MANDATORY)

#### 3.1 Automated Inspection (MANDATORY)
**When**: AFTER code changes, BEFORE commits
**Tool**: `static-code-analyzer` subagent

**Required Analysis**:
- [ ] Syntax checking and parsing validation
- [ ] Linting and style guide compliance
- [ ] Type checking and inference
- [ ] Security scanning and vulnerability detection
- [ ] Performance anti-pattern identification
- [ ] Code quality metrics assessment

**Success Criteria**:
- No syntax errors
- No critical linting issues
- Type checking passes
- No security vulnerabilities
- Performance acceptable

#### 3.2 Code Testing (MANDATORY)
**When**: AFTER implementation, BEFORE considering work complete
**Tool**: `test-automation-validator` subagent

**Required Testing**:
- [ ] Full test suite execution
- [ ] Test coverage requirements validation (80%+)
- [ ] Integration test execution and validation
- [ ] Performance benchmark execution
- [ ] Requirements compliance verification
- [ ] Error handling validation

**Success Criteria**:
- All tests pass (100%)
- Coverage meets requirements
- Integration tests pass
- Performance benchmarks met
- Documentation validated

#### 3.3 Documentation and Knowledge Sync (MANDATORY)
**When**: AFTER testing validation, BEFORE work considered complete

**Required Documentation**:
- [ ] Update docstrings following project style guide
- [ ] Include clear parameter descriptions and return value specifications
- [ ] Add usage examples for all new functionality
- [ ] Document all new APIs or modified interfaces
- [ ] Update README files and API documentation
- [ ] Update architectural decision records
- [ ] Sync knowledge bases with new decisions

---

## 🛠️ SUBAGENT USAGE GUIDELINES

### Architecture Guardian
- **When**: BEFORE making structural changes, during refactoring, when adding new modules
- **Purpose**: Maintain architectural integrity and prevent technical debt
- **MANDATORY Integration**: Must be used BEFORE any code changes
- **Key Functions**:
  - Dependency analysis and impact assessment
  - Pattern validation and architectural rule enforcement
  - Circular dependency detection
  - Module boundary verification
  - Interface compatibility checking

### Static Code Analyzer
- **When**: AFTER code changes, BEFORE commits, during code reviews
- **Purpose**: Ensure code quality and catch potential issues early
- **MANDATORY Integration**: Must be used AFTER implementation
- **Key Functions**:
  - Syntax checking and parsing validation
  - Linting and style guide compliance
  - Type checking and inference
  - Security scanning and vulnerability detection
  - Performance anti-pattern identification

### Test Automation Validator
- **When**: AFTER implementation, BEFORE merging, when validating fixes
- **Purpose**: Ensure functional correctness and test coverage
- **MANDATORY Integration**: Must be used AFTER testing
- **Key Functions**:
  - Test execution and result validation
  - Coverage analysis and reporting
  - Performance regression testing
  - Integration test validation
  - Requirements compliance verification

---

## 📊 QUALITY STANDARDS AND METRICS

### Test Coverage Requirements
- **Minimum**: 80% coverage for implemented features
- **Target**: 90%+ coverage for core modules
- **Ideal**: 95%+ coverage for critical components
- **Acceptable**: 100% pass rate (no failing tests)

### Code Quality Requirements
- **Test Pass Rate**: 100% (no exceptions allowed)
- **Static Analysis**: Zero critical issues
- **Type Safety**: Strong typing throughout codebase
- **Documentation**: Complete, accurate, and up-to-date
- **Security**: No vulnerabilities in committed code

### Process Requirements
- **Guideline Compliance**: 100% adherence to all workflow steps
- **Incremental Development**: All changes made in small, testable increments
- **Test-Driven Development**: All new features follow TDD methodology
- **Quality Gates**: All passed without exception
- **Documentation Sync**: All changes documented immediately

---

## 🚫 ABSOLUTE PROHIBITIONS

### NEVER Do These Things
1. **Skip pre-implementation analysis** - ALWAYS use architecture-guardian first
2. **Commit code with failing tests** - Fix ALL failures before committing
3. **Ignore static analysis findings** - Address ALL issues before proceeding
4. **Use subagents as post-validation checkboxes** - Integrate throughout development
5. **Make big-bang changes** - Implement incrementally with continuous validation
6. **Accept known defects** - Fix ALL issues before considering work complete
7. **Skip documentation** - Update docs with EVERY change
8. **Violate established guidelines** - NO exceptions to process requirements
9. **Proceed with unresolved quality gate failures** - STOP until issues resolved
10. **Make assumptions without validation** - Always verify through analysis

### Quality Compromise Is Not Acceptable
- Speed over quality is NEVER acceptable
- Technical debt accumulation is NEVER acceptable
- Known defect acceptance is NEVER acceptable
- Process shortcut taking is NEVER acceptable
- Guideline violation is NEVER acceptable

---

## 🔍 VALIDATION CHECKPOINTS

### Before Starting Any Work
- [ ] Architecture analysis completed and documented
- [ ] Change scope predicted and verified
- [ ] Task checklist created with success criteria
- [ ] Risk assessment completed with mitigation strategies
- [ ] Rollback plan defined and documented

### During Implementation
- [ ] TDD approach strictly followed
- [ ] Small incremental changes implemented
- [ ] Continuous testing maintained (100% pass rate)
- [ ] Documentation updated with each change
- [ ] No quality degradation introduced
- [ ] All quality gates passed

### Before Committing Changes
- [ ] All tests pass (100% pass rate)
- [ ] Test coverage meets requirements (80%+)
- [ ] Static analysis clean (no critical issues)
- [ ] Type checking passes without errors
- [ ] Security scan passes without vulnerabilities
- [ ] Documentation complete and accurate
- [ ] Architecture compliance verified

### After Implementation
- [ ] Integration tests pass
- [ ] Performance benchmarks met
- [ ] System-wide functionality validated
- [ ] Knowledge bases updated
- [ ] Team notifications sent
- [ ] Lessons learned documented

---

## 🔄 CONTINUOUS IMPROVEMENT

### After Each Development Session
1. **Document lessons learned** in session summary
2. **Update guidelines** if process improvements identified
3. **Identify process improvements** for future sessions
4. **Share insights** with team members
5. **Assess quality metrics** and trends

### Regular Quality Reviews
- **Weekly**: Review test coverage and quality metrics
- **Monthly**: Assess guideline compliance and process effectiveness
- **Quarterly**: Evaluate tool effectiveness and adoption
- **Semi-annually**: Comprehensive process review and updates

### Knowledge Management
- **Update architectural decisions** in knowledge base
- **Document common patterns** and anti-patterns
- **Share best practices** with team
- **Maintain up-to-date documentation**

---

## 📞 GETTING HELP AND ESCALATION

### When Guidelines Are Unclear
1. **Ask for clarification** BEFORE starting work
2. **Err on the side of caution** - choose quality over speed
3. **Document uncertainty** for future reference and team learning
4. **Escalate to team lead** if immediate guidance needed
5. **Stop work** until clarity achieved

### When Tools Fail or Issues Arise
1. **Document the issue immediately** with full context
2. **Find alternative approaches** if possible while maintaining quality
3. **Notify team** of tool problems and impacts
4. **Update guidelines** with workarounds and solutions
5. **Assess impact** on quality gates and adjust accordingly

### When Quality Standards Cannot Be Met
1. **STOP IMMEDIATELY** - do not proceed
2. **ASSESS ROOT CAUSE** of quality issues
3. **DOCUMENT THOROUGHLY** the specific problems
4. **ESCALATE** with detailed analysis and recommendations
5. **DO NOT COMPROMISE** on quality standards under any circumstances

---

## 🎖️ FINAL PRINCIPLES AND SUCCESS CRITERIA

### Quality First Principles
- **Never compromise quality for speed or convenience**
- **Fix issues immediately** when they are discovered
- **Maintain high standards consistently** across all work
- **Take pride in professional work** and craftsmanship
- **Learn from mistakes** and improve continuously

### Process Discipline Principles
- **Follow guidelines exactly** as written - no exceptions
- **Don't make assumptions** without validation
- **Document all exceptions** and reasoning for decisions
- **Maintain transparency** in all development activities
- **Seek feedback regularly** and incorporate improvements

### Team Collaboration Principles
- **Share knowledge openly** and document decisions
- **Review each other's work** constructively
- **Maintain consistent standards** across all team members
- **Support continuous improvement** initiatives
- **Hold each other accountable** for quality and process adherence

---

**REMEMBER**: AI-assisted development amplifies both good and bad practices. Strict adherence to these guidelines is essential for maintaining quality, reliability, and professional standards in AI-assisted development workflows.

**VIOLATIONS of these guidelines will result in immediate work stoppage and root cause analysis to identify systemic issues and prevent recurrence.**

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
USER INTENTION - Code Review Workflow Setup:

The user wants to implement a code review system using specialized subagents before making code revisions. The goal is to:

1. Use a "code reviewer" subagent to analyze the impact of changes before implementation
2. Prevent unintended changes to the codebase 
3. Ensure comprehensive review of how changes affect the entire system

KEY REQUIREMENTS:
- Code reviewer should analyze potential impacts before changes are made
- Should identify dependencies, breaking changes, and side effects
- Should help maintain codebase integrity and architectural consistency
- Should be integrated into the development workflow as a pre-revision step

IMPLEMENTATION APPROACH:
- Use architecture-guardian agent for architectural impact analysis
- Use static-code-analyzer agent for code quality and dependency analysis
- Create a workflow that triggers these agents before code modifications
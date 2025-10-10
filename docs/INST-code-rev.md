# AI Code Assistant Guidelines for Knowledge Repository Project

## Overview

This document provides guidelines for AI code assistants working on the Knowledge Repository project. It contains insights from analyzing git history where previous AI assistance required multiple iterations to resolve issues.

## Project Context

**Project Type**: Personal knowledge management system
**Primary Usage**: Local development environment for single user
**Key Components**: FastAPI backend, Obsidian integration, LLM-powered content processing
**Deployment**: Single machine, development-focused

## Common AI Challenge Patterns

### 1. Authentication & Security Issues

**Problem**: AI assistants consistently over-engineered authentication for a local personal tool.

**Historical Issues**:
- Multiple authentication system rewrites (commits a19ac54, 0e53e0c)
- CORS configuration iterations (commit 0ce7d40)
- JWT implementation for single-user local application

**Guidelines**:
- **Default to no-auth** for local development
- Use simple localhost CORS origins: `["http://localhost:7860", "http://localhost:8000", "http://127.0.0.1:7860", "http://127.0.0.1:8000", "*"]`
- Only add authentication if explicitly requested for production deployment
- Consider environment variables: `ENVIRONMENT=development|production`

### 2. Logging Configuration Problems

**Problem**: AI struggled with appropriate logging levels, creating either too much noise or insufficient visibility.

**Historical Issues**:
- Multiple logging system overhauls (commits 34ceec7, 63b894c, 8559cfd)
- Console output capture duplication
- DEBUG vs INFO level confusion

**Guidelines**:
- **DEBUG level** for development console output
- **INFO level** for file logging
- Always include timing information for operations
- Use structured logging with context: `[COMPONENT] action: details`
- Suppress noisy third-party libraries (httpx, urllib3, chromadb)

### 3. Network & External Service Dependencies

**Problem**: AI failed to handle external service failures gracefully.

**Historical Issues**:
- PyTorch import errors (commit bc7f579)
- Model download failures (commit 8cda8a4)
- LLM service connectivity issues

**Guidelines**:
- Always check service availability before making requests
- Implement retry logic with exponential backoff
- Provide meaningful error messages for service unavailability
- Use fallback mechanisms where possible
- Include health check endpoints

### 4. Port & Process Management

**Problem**: AI didn't account for port conflicts in development environment.

**Historical Issues**:
- Port 8000/7860 conflicts (commit 63b894c)
- Process cleanup issues
- Auto-port detection requirements

**Guidelines**:
- Implement automatic port detection for development
- Use `find_available_port()` pattern when starting servers
- Include process cleanup in shutdown scripts
- Check for existing processes before starting

## Code Structure Guidelines

### 1. Main Application Structure (main.py)

**Pattern**: Follow the established pattern in main.py:164-276

```python
# Use structured error handling
try:
    logger.info(f"[CAPTURE] Step X/Y: Description")
    start_time = time.time()
    result = operation()
    duration = time.time() - start_time
    logger.info(f"[CAPTURE] Operation completed in {duration:.2f}s")
except SpecificException as e:
    logger.error(f"[CAPTURE] Operation failed: {type(e).__name__}: {str(e)}")
    raise HTTPException(status_code=appropriate_code, detail=user_friendly_message)
```

### 2. Module Organization

**Follow existing patterns**:
- Each module should have its own logger
- Use retry decorators for external service calls
- Include timing information for all operations
- Follow the import order: standard library → third-party → local imports

### 3. Error Handling Strategy

**Use specific exception types**:
- `ConnectionError, Timeout` → HTTP 503 (Service Unavailable)
- `ValidationError` → HTTP 422 (Unprocessable Entity)
- `ValueError, KeyError` → HTTP 400 (Bad Request)
- Generic `Exception` → HTTP 500 (Internal Server Error)

## Development Workflow Guidelines

### 1. Before Making Changes

1. **Understand the context**: Is this for local development or production?
2. **Check existing implementations**: Look for similar patterns in the codebase
3. **Consider dependencies**: Will this change affect external services?
4. **Think about failure modes**: What happens if external services are unavailable?

### 2. During Implementation

1. **Add logging**: Include structured logging with timing information
2. **Handle errors gracefully**: Use appropriate HTTP status codes
3. **Maintain backward compatibility**: Don't break existing functionality
4. **Test edge cases**: Consider what happens with missing files, network issues, etc.

### 3. After Implementation

1. **Run the application**: Ensure it starts without errors
2. **Test core functionality**: URL capture, search, health checks
3. **Check logs**: Verify appropriate logging levels and messages
4. **Test error conditions**: What happens with invalid URLs, missing services?

## Specific Implementation Patterns

### 1. API Endpoints

**Pattern for new endpoints**:
```python
@app.post("/new_endpoint")
async def new_endpoint(request: RequestModel):
    start_time = time.time()
    try:
        logger.info(f"[ENDPOINT] Starting operation")
        result = perform_operation()
        duration = time.time() - start_time
        logger.info(f"[ENDPOINT] Completed in {duration:.2f}s")
        return ResponseModel(**result)
    except (ConnectionError, Timeout) as e:
        duration = time.time() - start_time
        logger.error(f"[ENDPOINT] Network error after {duration:.2f}s: {str(e)}")
        raise HTTPException(status_code=503, detail="External service unavailable")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"[ENDPOINT] Unexpected error after {duration:.2f}s: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
```

### 2. External Service Calls

**Pattern for LLM/API calls**:
```python
@retry(max_attempts=3, delay=2)
def make_external_request(data):
    start_time = time.time()
    try:
        logger.info(f"[SERVICE] Making request to {service_name}")
        result = service.call(data)
        duration = time.time() - start_time
        logger.info(f"[SERVICE] Request completed in {duration:.2f}s")
        return result
    except SpecificException as e:
        duration = time.time() - start_time
        logger.error(f"[SERVICE] Request failed after {duration:.2f}s: {str(e)}")
        raise
```

### 3. File Operations

**Pattern for file operations**:
```python
def safe_file_operation(file_path, operation):
    try:
        logger.debug(f"[FILE] {operation}: {file_path}")
        result = perform_operation()
        logger.debug(f"[FILE] {operation} successful")
        return result
    except FileNotFoundError:
        logger.error(f"[FILE] {operation} failed: File not found {file_path}")
        raise ValueError(f"File not found: {file_path}")
    except PermissionError:
        logger.error(f"[FILE] {operation} failed: Permission denied {file_path}")
        raise ValueError(f"Permission denied: {file_path}")
```

## Testing Guidelines

### 1. Always Test Core Functionality

After any changes, verify:
- API server starts successfully
- Health check endpoint returns 200
- URL capture works with a test URL
- Search functionality returns results
- UI loads without JavaScript errors

### 2. Test Error Conditions

- Invalid URLs
- Missing environment variables
- Unavailable external services
- File permission issues

### 3. Performance Considerations

- Include timing information in logs
- Monitor memory usage for large operations
- Consider async operations for long-running tasks

## Common Pitfalls to Avoid

1. **Don't over-engineer authentication** for local development
2. **Don't assume external services are always available**
3. **Don't use generic exception handling** without specific error types
4. **Don't forget to include timing information** in operations
5. **Don't hardcode localhost ports** - use environment variables
6. **Don't ignore logging levels** - use DEBUG for development, INFO for production
7. **Don't break backward compatibility** without clear justification

## Environment Variables

Always use these for configuration:
- `OBSIDIAN_VAULT_PATH`: Path to Obsidian vault
- `CHROMA_DB_PATH`: Path to ChromaDB
- `OLLAMA_BASE_URL`: LLM service URL
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `ENVIRONMENT`: development | production

## Conclusion

The key lesson from the git history analysis is that **context matters more than technical complexity**. This is a personal local tool that should prioritize reliability and simplicity over enterprise-grade features. Always consider the actual usage pattern (single user, local development) when making implementation decisions.
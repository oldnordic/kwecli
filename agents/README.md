# Qwen Agent - Real Code Generation

A production-ready code generation agent that uses Ollama with the Qwen3-Coder model to generate high-quality, functional code in multiple programming languages.

## Features

- **Real Code Generation**: 100% functional implementation with actual Ollama integration
- **Multi-Language Support**: Python, Rust, JavaScript, TypeScript, Go, C++, Java, C#, PHP, Ruby, Swift, Kotlin, Scala, R, MATLAB, Shell, HTML, CSS, SQL, YAML, JSON, Markdown
- **Automatic Language Detection**: Intelligently detects programming language from prompts
- **Comprehensive Error Handling**: Robust error handling for network issues, timeouts, and model availability
- **Code Validation**: Validates generated code for syntax and structure
- **Async/Sync Support**: Both asynchronous and synchronous APIs
- **Context-Aware Generation**: Supports additional context, requirements, and style guides
- **Production Ready**: Full test coverage with unit and integration tests

## Installation

### Prerequisites

1. **Install Ollama**: Download and install from [https://ollama.ai/](https://ollama.ai/)
2. **Pull the Qwen3-Coder model**:
   ```bash
   ollama pull hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M
   ```

### Python Dependencies

```bash
pip install pytest pytest-asyncio
```

## Quick Start

### Basic Usage

```python
from agents.qwen_agent import CodeGenerationAgent, CodeGenerationRequest, Language

# Create agent
agent = CodeGenerationAgent()

# Generate Python code
request = CodeGenerationRequest(
    prompt="Create a function that calculates factorial",
    language=Language.PYTHON
)

result = await agent.generate_code(request)
if result.success:
    print(result.code)
else:
    print(f"Error: {result.error_message}")
```

### Synchronous Usage

```python
from agents.qwen_agent import generate_code

# Simple synchronous code generation
code = generate_code("Create a Python function that adds two numbers")
print(code)
```

### Asynchronous Usage

```python
from agents.qwen_agent import generate_code_async, Language

# Async code generation with language specification
result = await generate_code_async(
    "Create a calculator class",
    language=Language.PYTHON
)

if result.success:
    print(result.code)
    print(f"Metadata: {result.metadata}")
else:
    print(f"Error: {result.error_message}")
```

### Advanced Usage with Context

```python
from agents.qwen_agent import CodeGenerationAgent, CodeGenerationRequest, Language

agent = CodeGenerationAgent()

request = CodeGenerationRequest(
    prompt="Create an email validation function",
    language=Language.JAVASCRIPT,
    context="This is for a web application that needs email validation",
    requirements=[
        "Use regex for validation",
        "Handle edge cases",
        "Add proper error handling",
        "Include JSDoc comments"
    ],
    style_guide="Follow modern JavaScript best practices and ES6+ features"
)

result = await agent.generate_code(request)
if result.success:
    print("Generated code:")
    print(result.code)
    print(f"Warnings: {result.warnings}")
    print(f"Metadata: {result.metadata}")
```

## API Reference

### Classes

#### `CodeGenerationAgent`

Main agent class for code generation.

```python
agent = CodeGenerationAgent(default_model="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M")
```

**Methods:**
- `async generate_code(request: CodeGenerationRequest) -> CodeGenerationResult`
- `_check_ollama_available() -> bool`
- `_check_model_available(model: str) -> bool`
- `_detect_language(prompt: str) -> Language`
- `_build_prompt(request: CodeGenerationRequest) -> str`
- `_extract_code_blocks(response: str, language: Language) -> List[str]`
- `_validate_code(code: str, language: Language) -> Tuple[bool, List[str]]`

#### `CodeGenerationRequest`

Request object for code generation.

```python
request = CodeGenerationRequest(
    prompt="Create a function",
    language=Language.PYTHON,
    context="Optional context",
    requirements=["Requirement 1", "Requirement 2"],
    style_guide="Optional style guide",
    model="optional-model-name"
)
```

#### `CodeGenerationResult`

Result object containing generated code and metadata.

```python
result = CodeGenerationResult(
    code="Generated code string",
    language=Language.PYTHON,
    success=True,
    error_message=None,
    warnings=[],
    metadata={}
)
```

### Enums

#### `Language`

Supported programming languages:
- `PYTHON`, `RUST`, `JAVASCRIPT`, `TYPESCRIPT`, `GO`
- `CPP`, `JAVA`, `CSHARP`, `PHP`, `RUBY`
- `SWIFT`, `KOTLIN`, `SCALA`, `R`, `MATLAB`
- `SHELL`, `HTML`, `CSS`, `SQL`, `YAML`, `JSON`, `MARKDOWN`

### Functions

#### `generate_code(prompt: str, model: str = None) -> str`

Synchronous code generation for backward compatibility.

#### `async generate_code_async(prompt: str, language: Language = None, model: str = None) -> CodeGenerationResult`

Asynchronous code generation with full result object.

#### `test_ollama_connection() -> bool`

Test if Ollama is available and running.

#### `check_model_available(model: str) -> bool`

Check if a specific model is available.

## Error Handling

The agent provides comprehensive error handling for various scenarios:

### Common Error Scenarios

1. **Ollama Not Available**:
   ```
   Error: Ollama is not available. Please install and start Ollama first.
   ```

2. **Model Not Available**:
   ```
   Error: Model 'model-name' is not available. Please pull it first with 'ollama pull model-name'
   ```

3. **Generation Timeout**:
   ```
   Error: Code generation timed out after 120 seconds
   ```

4. **Invalid Code Generated**:
   ```
   Error: No code blocks found in Ollama response
   ```

### Error Response Structure

```python
result = CodeGenerationResult(
    code="",
    language=Language.PYTHON,
    success=False,
    error_message="Detailed error message",
    warnings=["Warning 1", "Warning 2"],
    metadata={"error": "error_type", "details": "..."}
)
```

## Testing

### Run Unit Tests

```bash
python -m pytest tests/test_qwen_agent.py -v
```

### Run Integration Tests

```bash
python -m pytest tests/test_qwen_agent_integration.py -v
```

### Run Demo

```bash
python demo_qwen_agent.py
```

## Language Detection

The agent automatically detects programming language from prompts using keyword matching:

```python
agent = CodeGenerationAgent()

# These will be detected correctly:
agent._detect_language("Create a Python function")  # -> Language.PYTHON
agent._detect_language("Write a Rust function with fn main()")  # -> Language.RUST
agent._detect_language("Create a JavaScript function with console.log")  # -> Language.JAVASCRIPT

# Default fallback:
agent._detect_language("Create a simple function")  # -> Language.PYTHON
```

## Code Validation

The agent validates generated code for:

- **Syntax Errors**: Basic syntax validation
- **Structure Issues**: Missing braces, colons, semicolons
- **Malformed Patterns**: Incomplete code blocks
- **Language-Specific Rules**: Language-specific validation

## Metadata

Each generation result includes comprehensive metadata:

```python
metadata = {
    "model": "model-name",
    "language": "python",
    "prompt_length": 123,
    "response_length": 456,
    "code_blocks_found": 2,
    "ollama_return_code": 0,
    "execution_time": 12.34
}
```

## Performance

- **Timeout**: 120 seconds per generation request
- **Model Loading**: Automatic model availability checking
- **Error Recovery**: Graceful handling of network issues and timeouts
- **Memory Usage**: Efficient code extraction and validation

## Examples

### Python Code Generation

```python
request = CodeGenerationRequest(
    prompt="Create a function that validates email addresses",
    language=Language.PYTHON,
    requirements=["Use regex", "Handle edge cases", "Add type hints"],
    style_guide="Follow PEP 8"
)
```

### Rust Code Generation

```python
request = CodeGenerationRequest(
    prompt="Create a Rust function that calculates fibonacci numbers",
    language=Language.RUST,
    context="For a performance-critical application"
)
```

### JavaScript Code Generation

```python
request = CodeGenerationRequest(
    prompt="Create a React hook for managing form state",
    language=Language.JAVASCRIPT,
    requirements=["Use useState", "Handle validation", "Return cleanup function"],
    style_guide="Follow modern React patterns"
)
```

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure all tests pass before submitting

## License

This project is part of the KWE CLI system and follows the same licensing terms.

## Support

For issues and questions:
1. Check the test files for usage examples
2. Run the demo script to verify functionality
3. Ensure Ollama is properly installed and the model is available
4. Check the error messages for specific guidance

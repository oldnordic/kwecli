#!/usr/bin/env python3
"""
Tool Interface - Modular Entry Point
====================================

Main entry point for the KWE CLI Tools System using modular components.
Provides all tool interface functionality through focused modules.

File: tools/core/tool_interface.py
Purpose: Tool interface entry point with modular imports (≤300 lines)
"""

# Import all tool interface components from modular structure
from .tool_types import (
    ToolInfo,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolExecutionError,
    ToolValidationError,
    ToolRegistrationError,
    ToolPermissionError,
    ToolStatus,
    ToolCapabilities,
    ToolPermissions,
    ResourceKeys,
    create_tool_info,
    validate_execution_result
)

from .base_tool import BaseTool

from .tool_validation import ToolValidationMixin

from .simple_tool import SimpleTool, CalculatorTool


# Re-export all components for backward compatibility
__all__ = [
    # Types and exceptions
    "ToolInfo",
    "ToolExecutionContext", 
    "ToolExecutionResult",
    "ToolExecutionError",
    "ToolValidationError",
    "ToolRegistrationError",
    "ToolPermissionError",
    
    # Constants
    "ToolStatus",
    "ToolCapabilities", 
    "ToolPermissions",
    "ResourceKeys",
    
    # Utility functions
    "create_tool_info",
    "validate_execution_result",
    
    # Base classes
    "BaseTool",
    "ToolValidationMixin",
    
    # Concrete implementations
    "SimpleTool",
    "CalculatorTool"
]


# Backward compatibility aliases
ToolExecutionException = ToolExecutionError
ToolValidationException = ToolValidationError


def get_available_tool_types():
    """Get list of available tool types for registration."""
    return {
        "simple_tool": SimpleTool,
        "calculator": CalculatorTool
    }


def create_tool_from_type(tool_type: str, **kwargs):
    """
    Factory function to create tools by type name.
    
    Args:
        tool_type: Type of tool to create
        **kwargs: Additional arguments for tool initialization
        
    Returns:
        Tool instance
        
    Raises:
        ToolRegistrationError: If tool type is unknown
    """
    available_types = get_available_tool_types()
    
    if tool_type not in available_types:
        raise ToolRegistrationError(
            f"Unknown tool type: {tool_type}. Available types: {list(available_types.keys())}"
        )
    
    tool_class = available_types[tool_type]
    try:
        return tool_class(**kwargs)
    except Exception as e:
        raise ToolRegistrationError(f"Failed to create tool of type {tool_type}: {e}")


def validate_tool_implementation(tool: BaseTool) -> dict:
    """
    Validate that a tool properly implements the BaseTool interface.
    
    Args:
        tool: Tool instance to validate
        
    Returns:
        Validation result with details
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "checks": {}
    }
    
    try:
        # Check required properties
        required_properties = ["name", "category", "capabilities"]
        for prop in required_properties:
            try:
                value = getattr(tool, prop)
                validation_result["checks"][f"has_{prop}"] = True
                
                if prop == "capabilities" and not isinstance(value, list):
                    validation_result["errors"].append(f"Property '{prop}' must be a list")
                    validation_result["valid"] = False
                elif prop in ["name", "category"] and not isinstance(value, str):
                    validation_result["errors"].append(f"Property '{prop}' must be a string")
                    validation_result["valid"] = False
                    
            except Exception as e:
                validation_result["errors"].append(f"Missing or invalid property '{prop}': {e}")
                validation_result["checks"][f"has_{prop}"] = False
                validation_result["valid"] = False
        
        # Check required methods
        required_methods = ["execute", "get_schema", "validate_parameters"]
        for method in required_methods:
            has_method = hasattr(tool, method) and callable(getattr(tool, method))
            validation_result["checks"][f"has_{method}"] = has_method
            
            if not has_method:
                validation_result["errors"].append(f"Missing required method: {method}")
                validation_result["valid"] = False
        
        # Check schema validity
        try:
            schema = tool.get_schema()
            if not isinstance(schema, dict):
                validation_result["errors"].append("get_schema() must return a dictionary")
                validation_result["valid"] = False
            else:
                validation_result["checks"]["schema_valid"] = True
        except Exception as e:
            validation_result["errors"].append(f"get_schema() failed: {e}")
            validation_result["checks"]["schema_valid"] = False
            validation_result["valid"] = False
        
        # Check tool info generation
        try:
            info = tool.get_info()
            if not isinstance(info, ToolInfo):
                validation_result["errors"].append("get_info() must return ToolInfo instance")
                validation_result["valid"] = False
            else:
                validation_result["checks"]["info_valid"] = True
        except Exception as e:
            validation_result["errors"].append(f"get_info() failed: {e}")
            validation_result["checks"]["info_valid"] = False
            validation_result["valid"] = False
        
        # Performance warnings
        if hasattr(tool, "resource_requirements"):
            reqs = tool.resource_requirements
            if reqs.get("memory_mb", 0) > 1000:
                validation_result["warnings"].append("High memory requirement (>1GB)")
            if reqs.get("timeout_seconds", 0) > 300:
                validation_result["warnings"].append("High timeout (>5 minutes)")
    
    except Exception as e:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Validation failed: {e}")
    
    return validation_result


def get_tool_interface_info():
    """Get information about the tool interface system."""
    return {
        "version": "2.0.0",
        "modular_architecture": True,
        "modules": [
            "tool_types",
            "base_tool", 
            "tool_validation",
            "simple_tool"
        ],
        "available_tool_types": list(get_available_tool_types().keys()),
        "capabilities": [
            "tool_registration",
            "parameter_validation",
            "lifecycle_management", 
            "health_checking",
            "usage_statistics",
            "schema_validation"
        ]
    }


# Test functionality if run directly
if __name__ == "__main__":
    import asyncio
    
    print("🧪 Testing Tool Interface System...")
    
    async def test_tool_interface():
        # Test tool creation
        simple_tool = create_tool_from_type("simple_tool")
        print(f"✅ Created tool: {simple_tool.name}")
        
        # Test tool validation
        validation = validate_tool_implementation(simple_tool)
        print(f"✅ Tool validation: {validation['valid']}")
        if validation["errors"]:
            print(f"   Errors: {validation['errors']}")
        
        # Test tool execution
        result = await simple_tool.execute({"text": "Hello", "operation": "uppercase"})
        print(f"✅ Tool execution: {result['success']}")
        
        # Test system info
        system_info = get_tool_interface_info()
        print(f"✅ System info: v{system_info['version']}, {len(system_info['modules'])} modules")
        
        # Test available types
        available = get_available_tool_types()
        print(f"✅ Available tools: {list(available.keys())}")
    
    asyncio.run(test_tool_interface())
    print("✅ Tool Interface System test complete")
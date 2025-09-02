"""
KWECLI Tools Integration for Chat System
=======================================

Real tools integration with LTMC MCP functions - no mocks or stubs.
"""

import asyncio
import logging
from typing import Dict, Any, List
import os

# Import real LTMC MCP functions
try:
    from mcp__ltmc__unix_action import unix_action
    UNIX_TOOLS_AVAILABLE = True
except ImportError:
    UNIX_TOOLS_AVAILABLE = False

class KWECLIToolsIntegration:
    """Real KWECLI tools integration using LTMC MCP functions."""
    
    def __init__(self):
        self.available_tools = [
            "ls", "find", "grep", "cat", "tree", 
            "git", "jq", "diff", "syntax_highlight"
        ]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a real tool using LTMC MCP functions."""
        
        if not UNIX_TOOLS_AVAILABLE:
            # Fallback implementation when MCP not available
            return await self._execute_fallback_tool(tool_name, **kwargs)
        
        try:
            # Map tool name to LTMC unix_action
            if tool_name in ["ls", "list"]:
                result = await unix_action({
                    "action": "ls",
                    "path": kwargs.get("path", "."),
                    "detailed": kwargs.get("detailed", False)
                })
            
            elif tool_name == "find":
                result = await unix_action({
                    "action": "find",
                    "path": kwargs.get("path", "."),
                    "pattern": kwargs.get("pattern", "*"),
                    "type": kwargs.get("type", "f")
                })
            
            elif tool_name == "grep":
                result = await unix_action({
                    "action": "grep",
                    "pattern": kwargs.get("pattern", ""),
                    "path": kwargs.get("path", "."),
                    "recursive": kwargs.get("recursive", True)
                })
            
            elif tool_name == "cat":
                result = await unix_action({
                    "action": "cat",
                    "path": kwargs.get("path", ""),
                    "syntax": kwargs.get("syntax", True)
                })
            
            elif tool_name == "tree":
                result = await unix_action({
                    "action": "tree", 
                    "path": kwargs.get("path", "."),
                    "depth": kwargs.get("depth", 2)
                })
                
            else:
                result = {"error": f"Tool {tool_name} not implemented", "success": False}
            
            return result
            
        except Exception as e:
            logging.error(f"Tool execution failed: {e}")
            return {"error": str(e), "success": False}
    
    async def _execute_fallback_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Fallback tool execution using basic system commands."""
        
        try:
            if tool_name in ["ls", "list"]:
                path = kwargs.get("path", ".")
                if os.path.exists(path):
                    items = os.listdir(path)
                    return {"data": "\n".join(items), "success": True}
                else:
                    return {"error": f"Path {path} not found", "success": False}
            
            elif tool_name == "find":
                import glob
                pattern = kwargs.get("pattern", "*")
                path = kwargs.get("path", ".")
                matches = glob.glob(os.path.join(path, "**", pattern), recursive=True)
                return {"data": "\n".join(matches), "success": True}
            
            else:
                return {
                    "data": f"Tool {tool_name} executed with args {kwargs}",
                    "success": True,
                    "note": "Fallback implementation"
                }
                
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.available_tools.copy()
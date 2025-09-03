#!/usr/bin/env python3
"""
Tool Registry - Modular Entry Point
===================================

Central tool registry using smart modular architecture.
Rebuilt from registry.py following CLAUDE.md ≤300 lines rule.

File: tools/core/registry_modular.py
Purpose: Main tool registry interface with modular imports (≤300 lines)
"""

import threading
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

# Import modular components
from .tool_discovery import ToolDiscoveryManager
from .tool_health_monitor import ToolHealthMonitor

# Import base types
from .tool_interface import BaseTool
from .models import ToolInfo

logger = logging.getLogger(__name__)


class ToolRegistrationError(Exception):
    """Raised when tool registration fails."""
    pass


class ToolRegistry:
    """
    Central registry for all KWE CLI tools with modular architecture.
    
    Modular Components:
    - ToolDiscoveryManager: Automatic tool discovery and registration
    - ToolHealthMonitor: Comprehensive health checking and monitoring
    
    Manages tool registration, discovery, categorization, and health monitoring
    with thread-safe operations for concurrent access in production environments.
    """
    
    def __init__(self):
        """Initialize tool registry with modular components and thread safety."""
        # Core registry storage
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._capabilities: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
        
        # Initialize modular components
        self.discovery_manager = ToolDiscoveryManager(registry_instance=self)
        self.health_monitor = ToolHealthMonitor()
        
        # Registry statistics
        self.registry_stats = {
            "total_registrations": 0,
            "failed_registrations": 0,
            "unregistrations": 0,
            "health_checks_requested": 0
        }
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info("ToolRegistry initialized with modular architecture")
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry with comprehensive validation.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ToolRegistrationError: If tool is already registered
            ValueError: If tool is invalid
        """
        # Validate tool instance
        if not isinstance(tool, BaseTool):
            self.registry_stats["failed_registrations"] += 1
            raise ValueError(f"Tool must be instance of BaseTool, got {type(tool)}")
        
        # Validate required attributes
        validation_errors = []
        if not hasattr(tool, 'name') or not tool.name:
            validation_errors.append("Tool must have a valid name")
        if not hasattr(tool, 'category') or not tool.category:
            validation_errors.append("Tool must have a valid category")
        if not hasattr(tool, 'capabilities') or not tool.capabilities:
            validation_errors.append("Tool must have capabilities")
        
        if validation_errors:
            self.registry_stats["failed_registrations"] += 1
            raise ValueError(f"Tool validation failed: {'; '.join(validation_errors)}")
        
        with self._lock:
            # Check for duplicate registration
            if tool.name in self._tools:
                self.registry_stats["failed_registrations"] += 1
                raise ToolRegistrationError(f"Tool '{tool.name}' is already registered")
            
            # Register the tool
            self._tools[tool.name] = tool
            
            # Update categories
            if tool.category not in self._categories:
                self._categories[tool.category] = set()
            self._categories[tool.category].add(tool.name)
            
            # Update capabilities
            for capability in tool.capabilities:
                if capability not in self._capabilities:
                    self._capabilities[capability] = set()
                self._capabilities[capability].add(tool.name)
            
            self.registry_stats["total_registrations"] += 1
            self.logger.info(f"Registered tool: {tool.name} ({tool.category}) with {len(tool.capabilities)} capabilities")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: Name of tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        with self._lock:
            if tool_name not in self._tools:
                return False
            
            tool = self._tools[tool_name]
            
            # Remove from tools
            del self._tools[tool_name]
            
            # Remove from categories
            category = tool.category
            if category in self._categories:
                self._categories[category].discard(tool_name)
                if not self._categories[category]:
                    del self._categories[category]
            
            # Remove from capabilities
            for capability in tool.capabilities:
                if capability in self._capabilities:
                    self._capabilities[capability].discard(tool_name)
                    if not self._capabilities[capability]:
                        del self._capabilities[capability]
            
            # Clear health status cache
            self.health_monitor.clear_health_cache(tool_name)
            
            self.registry_stats["unregistrations"] += 1
            self.logger.info(f"Unregistered tool: {tool_name}")
            return True
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get tool by name with thread-safe access.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        with self._lock:
            return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """Get list of all registered tool names."""
        with self._lock:
            return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """Get list of all tool categories."""
        with self._lock:
            return list(self._categories.keys())
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of tools in the category
        """
        with self._lock:
            if category not in self._categories:
                return []
            
            tool_names = self._categories[category]
            return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_tools_by_capability(self, capability: str) -> List[BaseTool]:
        """
        Get all tools with a specific capability.
        
        Args:
            capability: Capability name
            
        Returns:
            List of tools with the capability
        """
        with self._lock:
            if capability not in self._capabilities:
                return []
            
            tool_names = self._capabilities[capability]
            return [self._tools[name] for name in tool_names if name in self._tools]
    
    def check_tool_health(self, tool_name: str) -> Dict[str, Any]:
        """
        Check health status of a specific tool using health monitor.
        
        Args:
            tool_name: Name of tool to check
            
        Returns:
            Health status information
        """
        self.registry_stats["health_checks_requested"] += 1
        
        with self._lock:
            tool_instance = self._tools.get(tool_name)
            return self.health_monitor.check_tool_health(tool_name, tool_instance)
    
    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all registered tools."""
        with self._lock:
            return self.health_monitor.get_batch_health_status(self._tools)
    
    # Auto-discovery methods (delegated to discovery manager)
    def discover_filesystem_tools(self) -> int:
        """Discover and register filesystem tools."""
        return self.discovery_manager.discover_filesystem_tools()
    
    def discover_development_tools(self) -> int:
        """Discover and register development tools."""
        return self.discovery_manager.discover_development_tools()
    
    def discover_all_tools(self) -> Dict[str, int]:
        """Discover and register all available tools."""
        return self.discovery_manager.discover_all_tools()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize registry to dictionary with comprehensive information.
        
        Returns:
            Dictionary representation of registry
        """
        with self._lock:
            return {
                "tools": {
                    name: tool.get_info().to_dict() if hasattr(tool.get_info(), 'to_dict') else {
                        "name": tool.name,
                        "category": tool.category,
                        "capabilities": tool.capabilities
                    }
                    for name, tool in self._tools.items()
                },
                "categories": {
                    category: list(tool_names)
                    for category, tool_names in self._categories.items()
                },
                "capabilities": {
                    capability: list(tool_names)
                    for capability, tool_names in self._capabilities.items()
                },
                "statistics": self.get_comprehensive_stats(),
                "total_tools": len(self._tools),
                "last_updated": datetime.now().isoformat()
            }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            "registry_stats": self.registry_stats.copy(),
            "discovery_stats": self.discovery_manager.get_discovery_stats(),
            "health_monitor_stats": self.health_monitor.get_health_summary(),
            "current_counts": {
                "total_tools": len(self._tools),
                "total_categories": len(self._categories),
                "total_capabilities": len(self._capabilities)
            }
        }
    
    def clear(self) -> None:
        """Clear all registered tools and reset state."""
        with self._lock:
            cleared_count = len(self._tools)
            
            self._tools.clear()
            self._categories.clear()
            self._capabilities.clear()
            
            # Clear health monitor cache
            self.health_monitor.clear_health_cache()
            
            # Reset statistics
            for key in self.registry_stats:
                self.registry_stats[key] = 0
            
            self.logger.info(f"Cleared {cleared_count} tools from registry")
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        with self._lock:
            return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        with self._lock:
            return tool_name in self._tools


# Test functionality if run directly
if __name__ == "__main__":
    print("🧪 Testing Modular Tool Registry...")
    
    # Create mock tool for testing
    class MockTool(BaseTool):
        def __init__(self, name="mock_tool"):
            super().__init__()
            self._name = name
        
        @property
        def name(self) -> str: return self._name
        @property
        def category(self) -> str: return "testing"
        @property
        def capabilities(self) -> list: return ["testing", "mock"]
        
        async def execute(self, parameters): return {"success": True}
        def get_schema(self): return {"name": self._name}
    
    # Test registry
    registry = ToolRegistry()
    print("✅ Registry initialized")
    
    # Test tool registration
    mock_tool = MockTool()
    registry.register_tool(mock_tool)
    print(f"✅ Tool registration: {len(registry)} tools registered")
    
    # Test tool retrieval
    retrieved_tool = registry.get_tool("mock_tool")
    print(f"✅ Tool retrieval: {retrieved_tool is not None}")
    
    # Test health checking
    health = registry.check_tool_health("mock_tool")
    print(f"✅ Health check: {health.get('available', False)}")
    
    # Test comprehensive stats
    stats = registry.get_comprehensive_stats()
    print(f"✅ Comprehensive stats: {len(stats)} categories")
    
    print("✅ Modular Tool Registry test complete")
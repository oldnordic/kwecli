#!/usr/bin/env python3
"""
Core Native LTMC Bridge Implementation

Provides zero-latency access to LTMC tools without MCP protocol overhead.
Real functionality with comprehensive error handling and performance optimization.

File: bridge/bridge_core.py
Lines: < 300 (modular component)
Purpose: Main bridge class with direct LTMC integration
"""

import sys
import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add LTMC to Python path for native integration
LTMC_PATH = Path("/home/feanor/Projects/ltmc")
if str(LTMC_PATH) not in sys.path:
    sys.path.insert(0, str(LTMC_PATH))

# Set up LTMC data directory environment
os.environ.setdefault("LTMC_DATA_DIR", "/home/feanor/Projects/Data")
os.environ.setdefault("DB_PATH", "/home/feanor/Projects/Data/ltmc.db")
os.environ.setdefault("FAISS_INDEX_PATH", "/home/feanor/Projects/Data/faiss_index")

logger = logging.getLogger(__name__)


class NativeLTMCBridge:
    """
    Native LTMC integration bridge for KWE CLI digital body.
    
    Provides zero-latency access to LTMC tools without MCP protocol overhead.
    Implements direct tool integration as specified in strategic architecture.
    
    Features:
    - Direct tool imports and execution
    - Native performance optimization
    - Session continuity and memory persistence
    - Comprehensive error handling and recovery
    """
    
    def __init__(self):
        """Initialize native LTMC bridge."""
        self.initialized = False
        self.ltmc_tools = {}
        self.session_id = f"native_bridge_{int(asyncio.get_event_loop().time())}"
        
        # Performance tracking
        self.operation_count = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_operation_time = 0.0
        
        # Tool availability cache
        self.available_tools = set()
    
    async def initialize(self) -> bool:
        """Initialize native LTMC bridge with real tool imports."""
        if self.initialized:
            return True
        
        try:
            logger.info("🔧 Initializing native LTMC bridge...")
            
            # Import LTMC tools directly
            await self._import_ltmc_tools()
            
            # Validate tool functionality
            if not await self._validate_tool_functionality():
                logger.error("❌ Tool functionality validation failed")
                return False
            
            self.initialized = True
            logger.info("✅ Native LTMC bridge initialized successfully")
            logger.info(f"📊 Available tools: {len(self.available_tools)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Bridge initialization failed: {e}")
            return False
    
    async def _import_ltmc_tools(self):
        """Import LTMC tools for direct access."""
        try:
            # Import memory tools (class-based)
            from ltms.tools.memory.memory_actions import MemoryTools
            self.ltmc_tools['memory_tool'] = MemoryTools()
            self.available_tools.add('memory')
            
            # Import chat tools (class-based)
            from ltms.tools.memory.chat_actions import ChatTools
            self.ltmc_tools['chat_tool'] = ChatTools()
            self.available_tools.add('chat')
            
            # Import pattern tools (class-based)
            from ltms.tools.patterns.pattern_actions import PatternTools
            self.ltmc_tools['pattern_tool'] = PatternTools()
            self.available_tools.add('patterns')
            
            # Import graph tools (class-based)
            from ltms.tools.graph.graph_actions import GraphTools
            self.ltmc_tools['graph_tool'] = GraphTools()
            self.available_tools.add('graph')
            
            # Import blueprint tools (class-based)
            from ltms.tools.blueprints.blueprint_actions import BlueprintTools
            self.ltmc_tools['blueprint_tool'] = BlueprintTools()
            self.available_tools.add('blueprints')
            
            # Import todo tools (class-based)
            from ltms.tools.todos.todo_actions import TodoTools
            self.ltmc_tools['todo_tool'] = TodoTools()
            self.available_tools.add('todos')
            
            # Import sprint management tools (class-based)
            from ltms.tools.sprints.sprint_actions import SprintTools
            self.ltmc_tools['sprints_tool'] = SprintTools()
            self.available_tools.add('sprints')
            
            # Import sync/code drift detection tools (class-based)
            from ltms.tools.sync.sync_actions import SyncTools
            self.ltmc_tools['sync_tool'] = SyncTools()
            self.available_tools.add('sync')
            
            # Import coordination and workflow audit tools (class-based)
            from ltms.tools.coordination.coordination_actions import CoordinationTools
            self.ltmc_tools['coordination_tool'] = CoordinationTools()
            self.available_tools.add('coordination')
            
            logger.info(f"✅ Imported {len(self.available_tools)} LTMC tool categories")
            
        except Exception as e:
            logger.error(f"❌ Failed to import LTMC tools: {e}")
            raise
    
    async def _validate_tool_functionality(self) -> bool:
        """Validate that imported tools are functional."""
        try:
            # Test memory tools
            if 'memory_tool' in self.ltmc_tools:
                test_result = await self.ltmc_tools['memory_tool'].execute_action('store',
                    file_name='bridge_test.md',
                    content='# Bridge Test\nTesting native LTMC bridge functionality.',
                    conversation_id='bridge_test'
                )
                
                if not test_result.get('success'):
                    logger.error("❌ Memory tool validation failed")
                    return False
            
            logger.info("✅ Tool functionality validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Tool validation error: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if bridge is initialized."""
        return self.initialized
    
    async def store_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store memory using native LTMC tools."""
        if not self.initialized:
            return {'success': False, 'error': 'Bridge not initialized'}
        
        start_time = asyncio.get_event_loop().time()
        self.operation_count += 1
        
        try:
            result = await self.ltmc_tools['memory_tool'].execute_action('store', **data)
            
            # Track success
            if result.get('success'):
                self.successful_operations += 1
            else:
                self.failed_operations += 1
            
            # Track timing
            operation_time = asyncio.get_event_loop().time() - start_time
            self.total_operation_time += operation_time
            
            return result
            
        except Exception as e:
            self.failed_operations += 1
            operation_time = asyncio.get_event_loop().time() - start_time
            self.total_operation_time += operation_time
            
            logger.error(f"❌ Memory store failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def retrieve_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve memory using native LTMC tools."""
        if not self.initialized:
            return {'success': False, 'error': 'Bridge not initialized'}
        
        start_time = asyncio.get_event_loop().time()
        self.operation_count += 1
        
        try:
            result = await self.ltmc_tools['memory_tool'].execute_action('retrieve', **data)
            
            # Track success
            if result.get('success'):
                self.successful_operations += 1
            else:
                self.failed_operations += 1
            
            # Track timing
            operation_time = asyncio.get_event_loop().time() - start_time
            self.total_operation_time += operation_time
            
            return result
            
        except Exception as e:
            self.failed_operations += 1
            operation_time = asyncio.get_event_loop().time() - start_time
            self.total_operation_time += operation_time
            
            logger.error(f"❌ Memory retrieve failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_action(self, tool_category: str, action: str, **kwargs) -> Dict[str, Any]:
        """Execute action on specified tool category."""
        if not self.initialized:
            return {'success': False, 'error': 'Bridge not initialized'}
        
        # Check if tool category is available
        if tool_category not in self.available_tools:
            return {'success': False, 'error': f'Tool category {tool_category} not available'}
        
        start_time = asyncio.get_event_loop().time()
        self.operation_count += 1
        
        try:
            # Handle all tools using class-based approach (memory, chat, patterns, graph, blueprints, todos)
            tool_key = f"{tool_category}_tool"
            if tool_key not in self.ltmc_tools:
                return {'success': False, 'error': f'Tool {tool_key} not available'}
            
            tool_instance = self.ltmc_tools[tool_key]
            result = await tool_instance.execute_action(action, **kwargs)
            
            # Track success
            if result.get('success'):
                self.successful_operations += 1
            else:
                self.failed_operations += 1
            
            # Track timing
            operation_time = asyncio.get_event_loop().time() - start_time
            self.total_operation_time += operation_time
            
            return result
            
        except Exception as e:
            self.failed_operations += 1
            operation_time = asyncio.get_event_loop().time() - start_time
            self.total_operation_time += operation_time
            
            logger.error(f"❌ Action {tool_category}.{action} failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get bridge performance metrics."""
        success_rate = (self.successful_operations / max(self.operation_count, 1)) * 100
        avg_operation_time = self.total_operation_time / max(self.operation_count, 1)
        
        return {
            'session_id': self.session_id,
            'initialized': self.initialized,
            'available_tools': list(self.available_tools),
            'total_operations': self.operation_count,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'success_rate_percent': round(success_rate, 2),
            'total_operation_time_seconds': round(self.total_operation_time, 3),
            'average_operation_time_seconds': round(avg_operation_time, 6)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            # Test basic functionality
            test_data = {
                'file_name': f'health_check_{int(asyncio.get_event_loop().time())}.md',
                'content': '# Health Check\nTesting bridge health.',
                'conversation_id': 'health_check'
            }
            
            store_result = await self.store_memory(test_data)
            if not store_result.get('success'):
                return {
                    'healthy': False,
                    'error': 'Store operation failed',
                    'details': store_result
                }
            
            retrieve_result = await self.retrieve_memory({
                'query': 'health_check',
                'conversation_id': 'health_check',
                'k': 1
            })
            
            if not retrieve_result.get('success'):
                return {
                    'healthy': False,
                    'error': 'Retrieve operation failed',
                    'details': retrieve_result
                }
            
            return {
                'healthy': True,
                'timestamp': asyncio.get_event_loop().time(),
                'performance_metrics': self.get_performance_metrics()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': asyncio.get_event_loop().time()
            }
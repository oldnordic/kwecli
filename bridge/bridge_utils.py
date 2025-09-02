#!/usr/bin/env python3
"""
Bridge Utility Functions and Factory Methods

Provides utility functions for creating and managing LTMC bridge instances.
Real functionality with singleton pattern and connection pooling.

File: bridge/bridge_utils.py
Lines: < 300 (modular component) 
Purpose: Bridge factory and utility functions
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .bridge_core import NativeLTMCBridge

logger = logging.getLogger(__name__)

# Global bridge instance for singleton pattern
_bridge_instance: Optional[NativeLTMCBridge] = None
_bridge_lock = asyncio.Lock()


async def get_native_ltmc_bridge() -> NativeLTMCBridge:
    """
    Get or create native LTMC bridge instance (singleton pattern).
    
    Returns:
        NativeLTMCBridge: Initialized bridge instance
    """
    global _bridge_instance
    
    async with _bridge_lock:
        if _bridge_instance is None:
            logger.info("🏗️  Creating new native LTMC bridge instance...")
            _bridge_instance = await create_native_ltmc_bridge()
        elif not _bridge_instance.is_initialized():
            logger.info("🔄 Re-initializing existing bridge instance...")
            await _bridge_instance.initialize()
        
        return _bridge_instance


async def create_native_ltmc_bridge() -> NativeLTMCBridge:
    """
    Create and initialize a new native LTMC bridge instance.
    
    Returns:
        NativeLTMCBridge: New initialized bridge instance
    """
    try:
        logger.info("🚀 Creating new native LTMC bridge...")
        
        bridge = NativeLTMCBridge()
        
        # Initialize the bridge
        success = await bridge.initialize()
        
        if not success:
            raise Exception("Bridge initialization failed")
        
        logger.info("✅ Native LTMC bridge created and initialized")
        return bridge
        
    except Exception as e:
        logger.error(f"❌ Failed to create native LTMC bridge: {e}")
        raise


async def reset_bridge_connection():
    """Reset the global bridge connection (for troubleshooting)."""
    global _bridge_instance
    
    async with _bridge_lock:
        if _bridge_instance:
            logger.info("🔄 Resetting bridge connection...")
            _bridge_instance = None
        
        # Create new instance
        _bridge_instance = await create_native_ltmc_bridge()


# Convenience wrapper functions for direct tool access
async def memory_action(action: str, **kwargs) -> Dict[str, Any]:
    """Execute memory action through native bridge."""
    bridge = await get_native_ltmc_bridge()
    return await bridge.execute_action('memory', action, **kwargs)


async def chat_action(action: str, **kwargs) -> Dict[str, Any]:
    """Execute chat action through native bridge."""
    bridge = await get_native_ltmc_bridge()
    return await bridge.execute_action('chat', action, **kwargs)


async def pattern_action(action: str, **kwargs) -> Dict[str, Any]:
    """Execute pattern action through native bridge."""
    bridge = await get_native_ltmc_bridge()
    return await bridge.execute_action('patterns', action, **kwargs)


async def graph_action(action: str, **kwargs) -> Dict[str, Any]:
    """Execute graph action through native bridge."""
    bridge = await get_native_ltmc_bridge()
    return await bridge.execute_action('graph', action, **kwargs)


async def blueprint_action(action: str, **kwargs) -> Dict[str, Any]:
    """Execute blueprint action through native bridge."""
    bridge = await get_native_ltmc_bridge()
    return await bridge.execute_action('blueprints', action, **kwargs)


async def todo_action(action: str, **kwargs) -> Dict[str, Any]:
    """Execute todo action through native bridge."""
    bridge = await get_native_ltmc_bridge()
    return await bridge.execute_action('todos', action, **kwargs)


async def sprint_action(action: str, **kwargs) -> Dict[str, Any]:
    """Execute sprint management action through native bridge."""
    bridge = await get_native_ltmc_bridge()
    return await bridge.execute_action('sprints', action, **kwargs)


async def sync_action(action: str, **kwargs) -> Dict[str, Any]:
    """Execute sync/code drift detection action through native bridge."""
    bridge = await get_native_ltmc_bridge()
    return await bridge.execute_action('sync', action, **kwargs)


async def coordination_action(action: str, **kwargs) -> Dict[str, Any]:
    """Execute coordination and workflow audit action through native bridge."""
    bridge = await get_native_ltmc_bridge()
    return await bridge.execute_action('coordination', action, **kwargs)




async def get_bridge_status() -> Dict[str, Any]:
    """Get comprehensive bridge status and metrics."""
    try:
        bridge = await get_native_ltmc_bridge()
        
        # Get performance metrics
        metrics = bridge.get_performance_metrics()
        
        # Perform health check
        health_result = await bridge.health_check()
        
        return {
            'bridge_status': 'operational' if health_result.get('healthy') else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'health_check': health_result,
            'performance_metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting bridge status: {e}")
        return {
            'bridge_status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }


async def validate_bridge_functionality() -> bool:
    """Validate that bridge is working correctly."""
    try:
        bridge = await get_native_ltmc_bridge()
        
        if not bridge.is_initialized():
            logger.error("❌ Bridge not initialized")
            return False
        
        # Perform health check
        health_result = await bridge.health_check()
        
        if not health_result.get('healthy'):
            logger.error(f"❌ Bridge health check failed: {health_result.get('error')}")
            return False
        
        logger.info("✅ Bridge functionality validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Bridge validation failed: {e}")
        return False


# Tool availability checks
async def check_tool_availability() -> Dict[str, bool]:
    """Check availability of all LTMC tool categories."""
    try:
        bridge = await get_native_ltmc_bridge()
        metrics = bridge.get_performance_metrics()
        available_tools = metrics.get('available_tools', [])
        
        # Define all expected tools
        expected_tools = ['memory', 'chat', 'patterns', 'graph', 'blueprints', 'todos', 'sprints', 'sync', 'coordination']
        
        availability = {}
        for tool in expected_tools:
            availability[tool] = tool in available_tools
        
        return availability
        
    except Exception as e:
        logger.error(f"❌ Error checking tool availability: {e}")
        return {tool: False for tool in ['memory', 'chat', 'patterns', 'graph', 'blueprints', 'todos', 'sprints', 'sync', 'coordination']}


async def bridge_diagnostic() -> Dict[str, Any]:
    """Run comprehensive bridge diagnostic."""
    try:
        logger.info("🔍 Running bridge diagnostic...")
        
        # Check tool availability
        tool_availability = await check_tool_availability()
        
        # Get bridge status
        bridge_status = await get_bridge_status()
        
        # Validate functionality
        functionality_valid = await validate_bridge_functionality()
        
        diagnostic_result = {
            'timestamp': datetime.now().isoformat(),
            'tool_availability': tool_availability,
            'bridge_status': bridge_status,
            'functionality_valid': functionality_valid,
            'overall_health': (
                functionality_valid and
                bridge_status.get('bridge_status') == 'operational' and
                all(tool_availability.values())
            )
        }
        
        if diagnostic_result['overall_health']:
            logger.info("✅ Bridge diagnostic passed - all systems operational")
        else:
            logger.warning("⚠️ Bridge diagnostic found issues")
        
        return diagnostic_result
        
    except Exception as e:
        logger.error(f"❌ Bridge diagnostic failed: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'overall_health': False
        }
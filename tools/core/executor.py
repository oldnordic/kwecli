#!/usr/bin/env python3
"""
Tool Executor - Modular Entry Point
===================================

Safe and monitored tool execution with comprehensive modular architecture.
Streamlined version following CLAUDE.md ≤300 lines rule.

File: tools/core/executor_streamlined.py
Purpose: Main tool execution interface with modular imports (≤300 lines)
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Import modular components
from .execution_cache import ExecutionCacheManager
from .resource_monitor import ResourceMonitor
from .execution_manager import ExecutionManager

# Import base types
from .models import ToolResult, ExecutionStatus, ExecutionContext
from .registry import ToolRegistry
from .tool_interface import BaseTool

logger = logging.getLogger(__name__)


class ExecutionTimeoutError(Exception):
    """Raised when tool execution exceeds timeout."""
    pass


class ExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class ToolExecutor:
    """
    Comprehensive tool executor with modular architecture.
    
    Modular Components:
    - ExecutionManager: Active execution tracking, cancellation, and statistics
    - ExecutionCacheManager: TTL-based result caching and cache management
    - ResourceMonitor: System resource monitoring with psutil integration
    
    Provides async tool execution with timeout handling, retry mechanisms,
    resource monitoring, progress tracking, and result caching through
    enterprise-grade modular components.
    """
    
    def __init__(
        self,
        max_concurrent_executions: int = 10,
        default_timeout: float = 120.0,
        enable_resource_monitoring: bool = True,
        enable_caching: bool = False,
        cache_ttl: int = 3600
    ):
        """Initialize tool executor with modular components."""
        self.default_timeout = default_timeout
        
        # Initialize modular components
        self.execution_manager = ExecutionManager(max_concurrent_executions)
        self.cache_manager = ExecutionCacheManager(default_ttl=cache_ttl)
        self.resource_monitor = ResourceMonitor(enable_monitoring=enable_resource_monitoring)
        
        # Enable caching if requested
        if enable_caching:
            self.cache_manager.enable_caching(cache_ttl)
        
        # Execution semaphore for async coordination
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info("ToolExecutor initialized with modular architecture")
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        registry: ToolRegistry,
        timeout: Optional[float] = None,
        max_retries: int = 0,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        monitor_resources: bool = False
    ) -> ToolResult:
        """Execute a tool with comprehensive monitoring and error handling."""
        
        # Check cache first using cache manager
        if self.cache_manager.is_enabled():
            cached_result = self.cache_manager.get_cached_result(tool_name, parameters)
            if cached_result:
                self.logger.debug(f"Returning cached result for {tool_name}")
                return ToolResult.from_dict(cached_result)
        
        # Get tool from registry
        tool = registry.get_tool(tool_name)
        if not tool:
            raise ExecutionError(f"Tool '{tool_name}' not found in registry")
        
        # Create execution context
        execution_id = self.cache_manager.generate_cache_key(tool_name, parameters)
        context = ExecutionContext(
            execution_id=execution_id,
            tool_name=tool_name,
            parameters=parameters,
            timeout=timeout or self.default_timeout,
            max_retries=max_retries
        )
        
        # Progress callback setup
        if progress_callback:
            progress_callback(0.0, "Starting execution")
        
        # Execute with retries
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                context.attempt_count = attempt + 1
                
                async with self._execution_semaphore:
                    result = await self._execute_with_monitoring(
                        tool=tool,
                        context=context,
                        progress_callback=progress_callback,
                        monitor_resources=monitor_resources
                    )
                
                # Cache successful result using cache manager
                if self.cache_manager.is_enabled() and result.success:
                    self.cache_manager.store_result(tool_name, parameters, result.to_dict())
                
                if progress_callback:
                    progress_callback(1.0, "Execution completed")
                
                return result
                
            except ExecutionTimeoutError:
                # Don't retry timeout errors, re-raise immediately  
                raise
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Execution attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    if progress_callback:
                        progress_callback(0.1 * (attempt + 1), f"Retrying (attempt {attempt + 2})")
        
        # All retries failed
        execution_time = time.time() - context.created_at.timestamp()
        
        # Complete execution with failure status
        self.execution_manager.complete_execution(
            execution_id, ExecutionStatus.FAILED, execution_time
        )
        
        error_result = ToolResult(
            success=False,
            error_message=str(last_exception),
            status=ExecutionStatus.FAILED,
            execution_time=execution_time,
            execution_id=execution_id,
            metadata={
                "retry_count": max_retries,
                "final_attempt": max_retries + 1,
                "tool_name": tool_name,
                "exception_type": type(last_exception).__name__
            }
        )
        
        if progress_callback:
            progress_callback(1.0, f"Execution failed after {max_retries + 1} attempts")
        
        return error_result
    
    async def _execute_with_monitoring(
        self,
        tool: BaseTool,
        context: ExecutionContext,
        progress_callback: Optional[Callable[[float, str], None]],
        monitor_resources: bool
    ) -> ToolResult:
        """Execute tool with monitoring and timeout handling using modular components."""
        
        # Register execution with execution manager
        if not self.execution_manager.start_execution(context):
            raise ExecutionError("Execution rejected due to capacity limits")
        
        # Start resource monitoring session
        resource_session = None
        if monitor_resources and self.resource_monitor.is_available():
            resource_session = self.resource_monitor.start_monitoring_session()
        
        try:
            # Parameter validation
            if hasattr(tool, 'pre_execute'):
                try:
                    await tool.pre_execute(context.parameters)
                except Exception as e:
                    return self._create_error_result(
                        context, f"Parameter validation failed: {str(e)}", ExecutionStatus.FAILED
                    )
            
            # Progress update
            if progress_callback:
                progress_callback(0.2, "Parameters validated")
            
            # Execute with timeout
            try:
                if context.timeout:
                    result_data = await asyncio.wait_for(
                        tool.execute(context.parameters),
                        timeout=context.timeout
                    )
                else:
                    result_data = await tool.execute(context.parameters)
                    
            except asyncio.TimeoutError:
                # Complete execution with timeout status
                execution_time = time.time() - context.created_at.timestamp()
                self.execution_manager.complete_execution(
                    context.execution_id, ExecutionStatus.TIMEOUT, execution_time
                )
                raise ExecutionTimeoutError(
                    f"Tool '{tool.name}' execution timed out after {context.timeout} seconds"
                )
            
            # Progress update
            if progress_callback:
                progress_callback(0.8, "Execution completed, processing results")
            
            # Post-execution processing
            if hasattr(tool, 'post_execute'):
                try:
                    result_data = await tool.post_execute(context.parameters, result_data)
                except Exception as e:
                    self.logger.warning(f"Post-execution processing failed: {e}")
                    # Continue with original result
            
            # Calculate execution time and complete execution
            execution_time = time.time() - context.created_at.timestamp()
            self.execution_manager.complete_execution(
                context.execution_id, ExecutionStatus.COMPLETED, execution_time
            )
            
            # Get resource usage from resource monitor
            resource_usage = {}
            if resource_session and self.resource_monitor.is_available():
                usage_delta = self.resource_monitor.calculate_usage_delta(resource_session)
                if usage_delta.get("monitoring_available", False):
                    resource_usage = {
                        "memory_delta_mb": usage_delta.get("memory_delta_mb", 0),
                        "peak_memory_mb": usage_delta.get("peak_memory_mb", 0),
                        "cpu_utilization_percent": usage_delta.get("cpu_utilization_percent", 0),
                        "time_elapsed_seconds": usage_delta.get("time_elapsed_seconds", 0)
                    }
            
            # Create comprehensive result
            return self._create_success_result(context, result_data, execution_time, resource_usage)
            
        except ExecutionTimeoutError:
            # Let timeout errors propagate up
            raise
        except Exception as e:
            execution_time = time.time() - context.created_at.timestamp()
            self.execution_manager.complete_execution(
                context.execution_id, ExecutionStatus.FAILED, execution_time
            )
            return self._create_error_result(context, str(e), ExecutionStatus.FAILED, execution_time)
    
    def _create_success_result(
        self, context: ExecutionContext, result_data: Dict[str, Any], 
        execution_time: float, resource_usage: Dict[str, Any]
    ) -> ToolResult:
        """Create successful execution result with comprehensive metadata."""
        success = result_data.get("success", True)
        output = result_data.get("output", "")
        
        # Handle data field - preserve all result data
        data = {k: v for k, v in result_data.items() if k not in ["success", "output"]}
        if not data:
            data = result_data
        
        return ToolResult(
            success=success,
            output=output,
            data=data,
            status=ExecutionStatus.COMPLETED,
            execution_time=execution_time,
            execution_id=context.execution_id,
            metadata={
                "tool_name": context.tool_name,
                "resource_usage": resource_usage,
                "retry_count": context.attempt_count - 1,
                "context": result_data.get("context", {}),
                "monitoring_available": self.resource_monitor.is_available(),
                "cache_enabled": self.cache_manager.is_enabled()
            }
        )
    
    def _create_error_result(
        self, context: ExecutionContext, error_message: str, 
        status: ExecutionStatus, execution_time: float = 0.0
    ) -> ToolResult:
        """Create error execution result."""
        return ToolResult(
            success=False,
            error_message=error_message,
            status=status,
            execution_time=execution_time,
            execution_id=context.execution_id,
            metadata={
                "tool_name": context.tool_name,
                "attempt_count": context.attempt_count
            }
        )
    
    # Delegate methods to modular components
    def enable_caching(self, cache_ttl: Optional[int] = None):
        """Enable result caching using cache manager."""
        self.cache_manager.enable_caching(cache_ttl)
    
    def disable_caching(self):
        """Disable result caching using cache manager."""
        self.cache_manager.disable_caching()
    
    def clear_cache(self) -> int:
        """Clear all cached results using cache manager."""
        return self.cache_manager.clear_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information from cache manager."""
        return self.cache_manager.get_cache_info()
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system resource overview from resource monitor."""
        return self.resource_monitor.get_system_overview()
    
    def get_active_executions(self):
        """Get active executions from execution manager."""
        return self.execution_manager.get_active_executions()
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel execution using execution manager."""
        return self.execution_manager.cancel_execution(execution_id)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics from execution manager."""
        return self.execution_manager.get_execution_statistics()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            "execution_stats": self.get_execution_statistics(),
            "cache_info": self.get_cache_info(),
            "monitoring_stats": self.resource_monitor.get_monitoring_stats(),
            "system_overview": self.get_system_overview() if self.resource_monitor.is_available() else {},
            "component_status": {
                "execution_manager_active": len(self.execution_manager.get_active_execution_ids()) > 0,
                "cache_manager_enabled": self.cache_manager.is_enabled(),
                "resource_monitor_available": self.resource_monitor.is_available()
            }
        }


# Test functionality if run directly
if __name__ == "__main__":
    import asyncio
    
    print("🧪 Testing Streamlined Tool Executor...")
    
    async def test_executor():
        # Test executor initialization
        executor = ToolExecutor(enable_caching=True, enable_resource_monitoring=True)
        print("✅ Executor initialized with modular components")
        
        # Test comprehensive stats
        stats = executor.get_comprehensive_stats()
        print(f"✅ Stats components: {len(stats)} categories")
        
        return executor
    
    # Run test
    executor = asyncio.run(test_executor())
    print("✅ Streamlined Tool Executor test complete")
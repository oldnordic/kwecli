"""
Tool Executor for the KWE CLI Tools System.

This module provides safe and monitored tool execution with comprehensive
error handling, resource monitoring, timeout management, and caching capabilities.
"""

import asyncio
import time
import threading
import hashlib
import json
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
import logging
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

import os

from .models import ToolResult, ExecutionStatus, ExecutionContext
from .registry import ToolRegistry
from .tool_interface import BaseTool


class ExecutionTimeoutError(Exception):
    """Raised when tool execution exceeds timeout."""
    pass


class ExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class ToolExecutor:
    """
    Executes tools with comprehensive monitoring and safety features.
    
    Provides async tool execution with timeout handling, retry mechanisms,
    resource monitoring, progress tracking, and result caching.
    """
    
    def __init__(
        self,
        max_concurrent_executions: int = 10,
        default_timeout: float = 120.0,
        enable_resource_monitoring: bool = True
    ):
        self.max_concurrent_executions = max_concurrent_executions
        self.default_timeout = default_timeout
        self.enable_resource_monitoring = enable_resource_monitoring
        
        # Execution tracking
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self._execution_lock = threading.RLock()
        
        # Caching
        self._cache_enabled = False
        self._cache_ttl = 3600  # 1 hour default
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        
        # Resource monitoring
        self._resource_monitor = psutil.Process() if PSUTIL_AVAILABLE else None
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info("ToolExecutor initialized")
    
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
        """
        Execute a tool with comprehensive monitoring and error handling.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Parameters for tool execution
            registry: Tool registry to get tool from
            timeout: Execution timeout in seconds
            max_retries: Maximum retry attempts
            progress_callback: Optional progress monitoring callback
            monitor_resources: Enable resource usage monitoring
            
        Returns:
            ToolResult with execution details
            
        Raises:
            ExecutionTimeoutError: If execution times out
            ExecutionError: If tool not found or other errors
        """
        execution_id = self._generate_execution_id(tool_name, parameters)
        start_time = time.time()
        
        # Check cache first
        if self._cache_enabled:
            cached_result = self._get_cached_result(tool_name, parameters)
            if cached_result:
                cached_result["metadata"]["cached"] = True
                return ToolResult.from_dict(cached_result)
        
        # Get tool from registry
        tool = registry.get_tool(tool_name)
        if not tool:
            raise ExecutionError(f"Tool '{tool_name}' not found in registry")
        
        # Create execution context
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
                
                # Cache successful result
                if self._cache_enabled and result.success:
                    self._cache_result(tool_name, parameters, result)
                
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
        execution_time = time.time() - start_time
        
        error_result = ToolResult(
            success=False,
            error_message=str(last_exception),
            status=ExecutionStatus.FAILED,
            execution_time=execution_time,
            execution_id=execution_id,
            metadata={
                "retry_count": max_retries,
                "final_attempt": max_retries + 1,
                "tool_name": tool_name
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
        """Execute tool with monitoring and timeout handling."""
        
        # Track active execution
        with self._execution_lock:
            self._active_executions[context.execution_id] = context
        
        context.start_execution()
        
        try:
            # Resource monitoring setup
            initial_memory = None
            if (monitor_resources or self.enable_resource_monitoring) and self._resource_monitor:
                try:
                    initial_memory = self._resource_monitor.memory_info().rss
                except Exception:
                    initial_memory = None
            
            # Parameter validation
            try:
                await tool.pre_execute(context.parameters)
            except Exception as e:
                return ToolResult(
                    success=False,
                    error_message=f"Parameter validation failed: {str(e)}",
                    status=ExecutionStatus.FAILED,
                    execution_time=0.0,
                    execution_id=context.execution_id
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
                raise ExecutionTimeoutError(f"Tool '{tool.name}' execution timed out after {context.timeout} seconds")
            
            # Progress update
            if progress_callback:
                progress_callback(0.8, "Execution completed, processing results")
            
            # Post-execution processing
            try:
                result_data = await tool.post_execute(context.parameters, result_data)
            except Exception as e:
                self.logger.warning(f"Post-execution processing failed: {e}")
                # Continue with original result
            
            # Calculate execution time
            execution_time = time.time() - context.created_at.timestamp()
            context.complete_execution(ExecutionStatus.COMPLETED)
            
            # Resource monitoring
            resource_usage = {}
            if (monitor_resources or self.enable_resource_monitoring) and self._resource_monitor:
                try:
                    current_memory = self._resource_monitor.memory_info().rss
                    cpu_times = self._resource_monitor.cpu_times()
                    resource_usage = {
                        "memory_peak": max(initial_memory or 0, current_memory),
                        "memory_delta": current_memory - (initial_memory or 0),
                        "cpu_time": sum(cpu_times[:2])  # user + system time
                    }
                except Exception:
                    resource_usage = {"error": "Resource monitoring failed"}
            
            # Create result
            success = result_data.get("success", True)
            output = result_data.get("output", "")
            
            # Handle data field - use the entire result dict as data (excluding success/output for metadata)
            # But include all fields as they might be expected by the test
            data = {k: v for k, v in result_data.items() if k not in ["success", "output"]}
            
            # If there's no data after filtering, use the original result as data
            if not data:
                data = result_data
            
            result = ToolResult(
                success=success,
                output=output,
                data=data,
                status=ExecutionStatus.COMPLETED,
                execution_time=execution_time,
                execution_id=context.execution_id,
                metadata={
                    "tool_name": tool.name,
                    "resource_usage": resource_usage,
                    "retry_count": context.attempt_count - 1,
                    "context": result_data.get("context", {})
                }
            )
            
            return result
            
        except ExecutionTimeoutError:
            # Let timeout errors propagate up
            raise
        except Exception as e:
            context.complete_execution(ExecutionStatus.FAILED)
            execution_time = time.time() - context.created_at.timestamp()
            
            return ToolResult(
                success=False,
                error_message=str(e),
                status=ExecutionStatus.FAILED,
                execution_time=execution_time,
                execution_id=context.execution_id,
                metadata={
                    "tool_name": tool.name,
                    "exception_type": type(e).__name__
                }
            )
        
        finally:
            # Cleanup
            with self._execution_lock:
                self._active_executions.pop(context.execution_id, None)
            
            context.cleanup()
    
    def enable_caching(self, cache_ttl: int = 3600):
        """
        Enable result caching.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self._cache_enabled = True
        self._cache_ttl = cache_ttl
        self.logger.info(f"Caching enabled with TTL: {cache_ttl} seconds")
    
    def disable_caching(self):
        """Disable result caching."""
        self._cache_enabled = False
        with self._cache_lock:
            self._cache.clear()
        self.logger.info("Caching disabled")
    
    def _generate_execution_id(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate unique execution ID."""
        content = f"{tool_name}:{json.dumps(parameters, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for tool execution."""
        return self._generate_execution_id(tool_name, parameters)
    
    def _get_cached_result(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired."""
        if not self._cache_enabled:
            return None
        
        cache_key = self._get_cache_key(tool_name, parameters)
        
        with self._cache_lock:
            if cache_key not in self._cache:
                return None
            
            cached_item = self._cache[cache_key]
            
            # Check expiration
            cached_time = datetime.fromisoformat(cached_item["cached_at"])
            if datetime.now() - cached_time > timedelta(seconds=self._cache_ttl):
                del self._cache[cache_key]
                return None
            
            return cached_item["result"]
    
    def _cache_result(self, tool_name: str, parameters: Dict[str, Any], result: ToolResult):
        """Cache execution result."""
        if not self._cache_enabled:
            return
        
        cache_key = self._get_cache_key(tool_name, parameters)
        
        with self._cache_lock:
            self._cache[cache_key] = {
                "result": result.to_dict(),
                "cached_at": datetime.now().isoformat()
            }
    
    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs."""
        with self._execution_lock:
            return list(self._active_executions.keys())
    
    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active execution.
        
        Args:
            execution_id: ID of execution to cancel
            
        Returns:
            True if execution was cancelled
        """
        with self._execution_lock:
            if execution_id in self._active_executions:
                context = self._active_executions[execution_id]
                context.status = ExecutionStatus.CANCELLED
                # Note: Actual cancellation depends on tool implementation
                return True
            return False
    
    def clear_cache(self):
        """Clear all cached results."""
        with self._cache_lock:
            self._cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "enabled": self._cache_enabled,
                "ttl_seconds": self._cache_ttl,
                "cached_items": len(self._cache),
                "total_size_bytes": sum(
                    len(json.dumps(item).encode()) 
                    for item in self._cache.values()
                )
            }
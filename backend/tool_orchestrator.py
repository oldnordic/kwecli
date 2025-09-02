"""
Tool Orchestration Module for Backend Integration.

Coordinates execution of multiple tools based on prompt analysis,
handles dependencies, and manages error recovery.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from .prompt_analyzer import PromptAnalysis, ToolIntention
from tools.core.executor import ToolExecutor
from tools.core.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result of executing a single tool."""
    intention: ToolIntention
    result: Dict[str, Any]
    success: bool
    execution_time: float
    error_message: Optional[str] = None


@dataclass
class ToolWorkflowResult:
    """Result of executing a complete tool workflow."""
    execution_results: List[ToolExecutionResult]
    original_analysis: PromptAnalysis
    total_execution_time: float
    overall_success: bool


class ToolOrchestrator:
    """Orchestrates execution of tool workflows based on prompt analysis."""
    
    def __init__(self, tool_executor: ToolExecutor, tool_registry: ToolRegistry):
        self.tool_executor = tool_executor
        self.tool_registry = tool_registry
        self.default_timeout = 30.0
        self.max_concurrent_tools = 3
        self.retry_attempts = 2
    
    async def execute_tool_workflow(self, analysis: PromptAnalysis) -> ToolWorkflowResult:
        """
        Execute a complete tool workflow based on prompt analysis.
        
        Args:
            analysis: Result from PromptAnalyzer containing tool intentions
            
        Returns:
            ToolWorkflowResult with execution results and metadata
        """
        start_time = time.time()
        execution_results = []
        overall_success = True
        
        try:
            if not analysis.requires_tools or not analysis.tool_intentions:
                logger.info("No tools required for this prompt")
                return ToolWorkflowResult(
                    execution_results=[],
                    original_analysis=analysis,
                    total_execution_time=0.0,
                    overall_success=True
                )
            
            logger.info(f"Executing tool workflow with {len(analysis.tool_intentions)} intentions")
            
            # Group tools by priority for sequential execution within priority levels
            priority_groups = self._group_by_priority(analysis.tool_intentions)
            
            # Execute tools by priority group
            for priority, intentions in priority_groups.items():
                logger.debug(f"Executing priority {priority} tools: {[i.tool_name for i in intentions]}")
                
                # Execute tools in this priority group (can be concurrent within group)
                group_results = await self._execute_priority_group(intentions)
                execution_results.extend(group_results)
                
                # Check if any critical failures occurred
                critical_failures = [r for r in group_results if not r.success and r.intention.priority == 1]
                if critical_failures:
                    logger.warning(f"Critical tool failures in priority {priority}: {[r.intention.tool_name for r in critical_failures]}")
                    overall_success = False
                    # Continue with lower priority tools, but mark overall as failed
            
            # Determine overall success
            failed_critical = any(not r.success and r.intention.priority <= 2 for r in execution_results)
            overall_success = overall_success and not failed_critical
            
            total_time = time.time() - start_time
            
            logger.info(f"Tool workflow completed in {total_time:.2f}s, success: {overall_success}")
            
            return ToolWorkflowResult(
                execution_results=execution_results,
                original_analysis=analysis,
                total_execution_time=total_time,
                overall_success=overall_success
            )
            
        except Exception as e:
            logger.error(f"Tool workflow execution failed: {e}")
            total_time = time.time() - start_time
            
            return ToolWorkflowResult(
                execution_results=execution_results,
                original_analysis=analysis,
                total_execution_time=total_time,
                overall_success=False
            )
    
    def _group_by_priority(self, intentions: List[ToolIntention]) -> Dict[int, List[ToolIntention]]:
        """Group tool intentions by priority level."""
        priority_groups = {}
        
        for intention in intentions:
            priority = intention.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(intention)
        
        # Sort by priority (lower number = higher priority)
        return dict(sorted(priority_groups.items()))
    
    async def _execute_priority_group(self, intentions: List[ToolIntention]) -> List[ToolExecutionResult]:
        """Execute a group of tools with the same priority level."""
        if not intentions:
            return []
        
        # Determine if tools can be executed concurrently
        # For now, execute sequentially for safety, but can be optimized later
        results = []
        
        for intention in intentions:
            result = await self._execute_single_tool(intention)
            results.append(result)
            
            # If this is a critical tool and it failed, we might want to skip dependent tools
            if not result.success and intention.priority == 1:
                logger.warning(f"Critical tool {intention.tool_name} failed, continuing with remaining tools")
        
        return results
    
    async def _execute_single_tool(self, intention: ToolIntention) -> ToolExecutionResult:
        """
        Execute a single tool with error handling and retries.
        
        Args:
            intention: Tool intention with parameters
            
        Returns:
            ToolExecutionResult with execution details
        """
        start_time = time.time()
        
        for attempt in range(self.retry_attempts + 1):
            try:
                logger.debug(f"Executing tool {intention.tool_name} (attempt {attempt + 1})")
                
                # Execute the tool using the tool executor
                tool_result = await asyncio.wait_for(
                    self.tool_executor.execute_tool(
                        intention.tool_name,
                        intention.parameters,
                        self.tool_registry
                    ),
                    timeout=self.default_timeout
                )
                
                # Convert ToolResult to dict if needed
                if hasattr(tool_result, 'to_dict'):
                    result_dict = tool_result.to_dict()
                else:
                    result_dict = tool_result
                
                execution_time = time.time() - start_time
                
                # Check if execution was successful
                success = result_dict.get("success", False)
                
                if success:
                    logger.debug(f"Tool {intention.tool_name} executed successfully in {execution_time:.2f}s")
                    return ToolExecutionResult(
                        intention=intention,
                        result=result_dict,
                        success=True,
                        execution_time=execution_time
                    )
                else:
                    error_msg = result_dict.get("error_message", "Unknown error")
                    logger.warning(f"Tool {intention.tool_name} failed: {error_msg}")
                    
                    if attempt < self.retry_attempts:
                        logger.info(f"Retrying tool {intention.tool_name} (attempt {attempt + 2})")
                        await asyncio.sleep(0.5)  # Brief delay before retry
                        continue
                    
                    # Final attempt failed
                    return ToolExecutionResult(
                        intention=intention,
                        result=result_dict,
                        success=False,
                        execution_time=execution_time,
                        error_message=error_msg
                    )
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                error_msg = f"Tool {intention.tool_name} timed out after {self.default_timeout}s"
                logger.error(error_msg)
                
                if attempt < self.retry_attempts:
                    logger.info(f"Retrying tool {intention.tool_name} after timeout")
                    continue
                
                return ToolExecutionResult(
                    intention=intention,
                    result={"success": False, "error_message": error_msg},
                    success=False,
                    execution_time=execution_time,
                    error_message=error_msg
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Tool {intention.tool_name} execution error: {str(e)}"
                logger.error(error_msg)
                
                if attempt < self.retry_attempts:
                    logger.info(f"Retrying tool {intention.tool_name} after error: {e}")
                    await asyncio.sleep(0.5)
                    continue
                
                return ToolExecutionResult(
                    intention=intention,
                    result={"success": False, "error_message": error_msg},
                    success=False,
                    execution_time=execution_time,
                    error_message=error_msg
                )
        
        # This should never be reached due to the loop structure, but just in case
        return ToolExecutionResult(
            intention=intention,
            result={"success": False, "error_message": "Unexpected execution path"},
            success=False,
            execution_time=time.time() - start_time,
            error_message="Unexpected execution path"
        )
    
    def configure_timeouts(self, default_timeout: float, max_concurrent: int = None) -> None:
        """
        Configure execution timeouts and concurrency limits.
        
        Args:
            default_timeout: Default timeout for tool execution in seconds
            max_concurrent: Maximum number of concurrent tool executions
        """
        if default_timeout > 0:
            self.default_timeout = default_timeout
        
        if max_concurrent is not None and max_concurrent > 0:
            self.max_concurrent_tools = max_concurrent
    
    def get_execution_stats(self, workflow_result: ToolWorkflowResult) -> Dict[str, Any]:
        """
        Get execution statistics from a workflow result.
        
        Args:
            workflow_result: Result from execute_tool_workflow
            
        Returns:
            Dictionary with execution statistics
        """
        results = workflow_result.execution_results
        
        if not results:
            return {
                "total_tools": 0,
                "successful_tools": 0,
                "failed_tools": 0,
                "total_time": workflow_result.total_execution_time,
                "average_tool_time": 0.0,
                "success_rate": 1.0
            }
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_tool_time = sum(r.execution_time for r in results)
        average_time = total_tool_time / len(results) if results else 0.0
        success_rate = successful / len(results) if results else 0.0
        
        return {
            "total_tools": len(results),
            "successful_tools": successful,
            "failed_tools": failed,
            "total_time": workflow_result.total_execution_time,
            "average_tool_time": average_time,
            "success_rate": success_rate,
            "tool_names": [r.intention.tool_name for r in results],
            "failed_tools": [r.intention.tool_name for r in results if not r.success]
        }
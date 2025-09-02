#!/usr/bin/env python3
"""
KWE CLI Coordinator for ACP Bridge Integration

Production-ready coordinator class that enables Claude Code and other AI systems
to coordinate with KWE CLI agents through the ACP bridge. Implements the three
main coordination patterns:

1. Sequential Task Delegation - Chain tasks with dependency management
2. Parallel Task Distribution - Multi-agent simultaneous execution  
3. Iterative Collaboration - Multi-turn conversations with context preservation

This implements the coordination interface specified in ACP_COORDINATION_DESIGN.md
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Set, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum

from acp.acp_bridge_real import ACPBridge, acp_bridge_context
from acp.acp_client import ACPClient, ConnectionConfig
from acp.acp_models import (
    ACPMessage, FIPAPerformative, AgentProfile, AgentCapability,
    TaskRequest, TaskResult, CapabilityQuery, CapabilityResponse,
    StatusUpdate, ErrorInfo, ConversationContext, MessageMetrics,
    create_request_message, create_inform_message, create_error_message
)

# Import agent system
from agents.agent_registry import AgentRegistry
from agents.base_agent import SubAgent, AgentResult

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed" 
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class CoordinationPattern(Enum):
    """Coordination pattern types."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"  
    ITERATIVE = "iterative"


@dataclass
class TaskExecution:
    """Tracking data for task execution."""
    task_id: str
    conversation_id: str
    requester_id: str
    agent_id: str
    task_type: str
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 300
    context: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationContext:
    """Context for tracking multi-task coordination."""
    coordination_id: str
    pattern: CoordinationPattern
    requester_id: str
    tasks: List[TaskExecution] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None


class KWECLICoordinator:
    """
    Production coordinator for AI-to-AI coordination via ACP bridge.
    
    Enables Claude Code and other AI systems to coordinate with KWE CLI agents
    using enterprise-grade task delegation, monitoring, and result aggregation.
    """
    
    def __init__(
        self,
        acp_bridge: Optional[ACPBridge] = None,
        agent_registry: Optional[AgentRegistry] = None,
        server_host: str = "127.0.0.1",
        server_port: int = 8000,
        max_concurrent_tasks: int = 10,
        default_timeout: int = 300
    ):
        """Initialize the KWE CLI coordinator.
        
        Args:
            acp_bridge: ACP bridge instance (created if not provided)
            agent_registry: Agent registry instance (created if not provided)
            server_host: ACP server host
            server_port: ACP server port
            max_concurrent_tasks: Maximum concurrent task limit
            default_timeout: Default task timeout in seconds
        """
        self.acp_bridge = acp_bridge
        self.agent_registry = agent_registry or AgentRegistry()
        self.server_host = server_host
        self.server_port = server_port
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        
        # Task tracking
        self.active_tasks: Dict[str, TaskExecution] = {}
        self.coordination_contexts: Dict[str, CoordinationContext] = {}
        self.task_results: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.metrics = {
            "tasks_executed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "avg_execution_time": 0.0,
            "total_execution_time": 0.0,
            "concurrent_peak": 0
        }
        
        # Coordination patterns
        self._coordination_handlers = {
            CoordinationPattern.SEQUENTIAL: self._handle_sequential_coordination,
            CoordinationPattern.PARALLEL: self._handle_parallel_coordination,
            CoordinationPattern.ITERATIVE: self._handle_iterative_coordination
        }
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._running = False
    
    async def start(self) -> None:
        """Start the coordinator and initialize ACP bridge."""
        if self._running:
            return
            
        try:
            # Initialize ACP bridge if not provided
            if self.acp_bridge is None:
                client_config = ConnectionConfig(
                    server_host=self.server_host,
                    websocket_port=self.server_port + 1,
                    http_port=self.server_port + 2,
                    max_reconnect_attempts=3,
                    connection_timeout=10
                )
                
                self.acp_bridge = ACPBridge("kwe-coordinator", client_config)
                await self.acp_bridge.start()
            
            # Register available agents with ACP bridge
            await self._register_available_agents()
            
            self._running = True
            self.logger.info("KWE CLI Coordinator started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start coordinator: {e}")
            raise
    
    def start_standalone(self) -> None:
        """Start coordinator in standalone mode without ACP bridge (for HTTP API testing)."""
        if self._running:
            return
            
        # No ACP bridge initialization - just mark as running
        self._running = True
        self.logger.info("KWE CLI Coordinator started in standalone mode (HTTP API only)")
    
    async def stop(self) -> None:
        """Stop the coordinator and cleanup resources."""
        if not self._running:
            return
            
        try:
            # Cancel all active tasks
            for task_id in list(self.active_tasks.keys()):
                await self.cancel_task(task_id)
            
            # Stop ACP bridge
            if self.acp_bridge:
                await self.acp_bridge.stop()
            
            self._running = False
            self.logger.info("KWE CLI Coordinator stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping coordinator: {e}")
    
    async def delegate_task(
        self, 
        task_request: Dict[str, Any],
        requester_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Delegate a single task to an appropriate KWE CLI agent.
        
        Args:
            task_request: Task parameters including task_type, parameters, etc.
            requester_id: ID of the requesting system (e.g., "claude-code")
            conversation_id: Conversation context ID for tracking
            timeout: Task timeout in seconds
            
        Returns:
            Dict containing task_id, status, and delegation result
        """
        try:
            # Validate request
            if not isinstance(task_request, dict):
                return {
                    "success": False,
                    "error": "Task request must be a dictionary",
                    "task_id": None
                }
            
            task_type = task_request.get("task_type")
            if not task_type:
                return {
                    "success": False,
                    "error": "Task request must include 'task_type'",
                    "task_id": None
                }
            
            # Create task execution context
            task_id = str(uuid.uuid4())
            conversation_id = conversation_id or str(uuid.uuid4())
            requester_id = requester_id or "unknown"
            
            task_execution = TaskExecution(
                task_id=task_id,
                conversation_id=conversation_id,
                requester_id=requester_id,
                agent_id="",  # Will be assigned during routing
                task_type=task_type,
                parameters=task_request.get("parameters", {}),
                timeout_seconds=timeout or self.default_timeout,
                context=task_request.get("context", {})
            )
            
            # Find appropriate agent
            agent_id = await self._route_task_to_agent(task_execution)
            if not agent_id:
                return {
                    "success": False,
                    "error": f"No agent available for task type: {task_type}",
                    "task_id": task_id
                }
            
            task_execution.agent_id = agent_id
            self.active_tasks[task_id] = task_execution
            
            # Execute task asynchronously
            asyncio.create_task(self._execute_task(task_execution))
            
            return {
                "success": True,
                "task_id": task_id,
                "agent_id": agent_id,
                "status": TaskStatus.PENDING.value,
                "estimated_completion": (datetime.utcnow() + timedelta(seconds=task_execution.timeout_seconds)).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Task delegation failed: {e}")
            return {
                "success": False,
                "error": f"Task delegation failed: {str(e)}",
                "task_id": None
            }
    
    async def check_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Check the status of a delegated task.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            Dict containing task status, progress, and metadata
        """
        try:
            task = self.active_tasks.get(task_id)
            if not task:
                # Check if task is completed
                if task_id in self.task_results:
                    result = self.task_results[task_id]
                    return {
                        "success": True,
                        "task_id": task_id,
                        "status": TaskStatus.COMPLETED.value,
                        "completed": True,
                        "result": result.get("result"),
                        "execution_time": result.get("execution_time"),
                        "completed_at": result.get("completed_at")
                    }
                
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}",
                    "task_id": task_id
                }
            
            # Calculate progress indicators
            elapsed = (datetime.utcnow() - task.created_at).total_seconds()
            progress_percent = min((elapsed / task.timeout_seconds) * 100, 100)
            
            return {
                "success": True,
                "task_id": task_id,
                "status": task.status.value,
                "agent_id": task.agent_id,
                "task_type": task.task_type,
                "progress_percent": progress_percent,
                "elapsed_seconds": elapsed,
                "timeout_seconds": task.timeout_seconds,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed": task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT],
                "error": task.error
            }
            
        except Exception as e:
            self.logger.error(f"Status check failed for task {task_id}: {e}")
            return {
                "success": False,
                "error": f"Status check failed: {str(e)}",
                "task_id": task_id
            }
    
    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        Retrieve the result of a completed task.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            Dict containing task result, metadata, and execution statistics
        """
        try:
            # Check active tasks first
            task = self.active_tasks.get(task_id)
            if task and task.status == TaskStatus.COMPLETED:
                result_data = {
                    "success": True,
                    "task_id": task_id,
                    "status": task.status.value,
                    "result": task.result,
                    "agent_id": task.agent_id,
                    "task_type": task.task_type,
                    "execution_time": (task.completed_at - task.started_at).total_seconds() if (task.completed_at and task.started_at) else None,
                    "created_at": task.created_at.isoformat(),
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "metrics": task.metrics
                }
                
                # Move to results cache and remove from active
                self.task_results[task_id] = result_data
                del self.active_tasks[task_id]
                
                return result_data
            
            # Check results cache
            if task_id in self.task_results:
                return self.task_results[task_id]
            
            # Check if task is still running
            if task:
                return {
                    "success": False,
                    "error": f"Task {task_id} is still {task.status.value}",
                    "task_id": task_id,
                    "status": task.status.value
                }
            
            return {
                "success": False,
                "error": f"Task not found: {task_id}",
                "task_id": task_id
            }
            
        except Exception as e:
            self.logger.error(f"Result retrieval failed for task {task_id}: {e}")
            return {
                "success": False,
                "error": f"Result retrieval failed: {str(e)}",
                "task_id": task_id
            }
    
    async def parallel_delegation(
        self, 
        tasks: List[Dict[str, Any]],
        requester_id: Optional[str] = None,
        coordination_timeout: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tasks in parallel with result aggregation.
        
        Args:
            tasks: List of task requests to execute in parallel
            requester_id: ID of the requesting system
            coordination_timeout: Overall coordination timeout
            
        Returns:
            List of task results with coordination metadata
        """
        try:
            if not tasks:
                return []
            
            coordination_id = str(uuid.uuid4())
            conversation_id = str(uuid.uuid4())
            requester_id = requester_id or "unknown"
            
            # Create coordination context
            coordination = CoordinationContext(
                coordination_id=coordination_id,
                pattern=CoordinationPattern.PARALLEL,
                requester_id=requester_id
            )
            
            self.coordination_contexts[coordination_id] = coordination
            
            # Delegate all tasks
            delegation_tasks = []
            for i, task_request in enumerate(tasks):
                task_request["context"] = task_request.get("context", {})
                task_request["context"]["coordination_id"] = coordination_id
                task_request["context"]["task_index"] = i
                
                delegation_task = self.delegate_task(
                    task_request=task_request,
                    requester_id=requester_id,
                    conversation_id=conversation_id,
                    timeout=coordination_timeout
                )
                delegation_tasks.append(delegation_task)
            
            # Wait for all delegations to complete
            delegation_results = await asyncio.gather(*delegation_tasks, return_exceptions=True)
            
            # Extract task IDs and wait for completion
            task_ids = []
            for result in delegation_results:
                if isinstance(result, dict) and result.get("success") and result.get("task_id"):
                    task_ids.append(result["task_id"])
            
            # Wait for all tasks to complete or timeout
            timeout_duration = coordination_timeout or (self.default_timeout * 2)
            results = await self._wait_for_tasks_completion(task_ids, timeout_duration)
            
            # Update coordination context
            coordination.completed_at = datetime.utcnow()
            coordination.status = TaskStatus.COMPLETED
            coordination.result = {
                "total_tasks": len(tasks),
                "successful_tasks": len([r for r in results if r.get("success", False)]),
                "failed_tasks": len([r for r in results if not r.get("success", False)]),
                "results": results
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel delegation failed: {e}")
            return [{
                "success": False,
                "error": f"Parallel delegation failed: {str(e)}",
                "task_id": None
            } for _ in tasks]
    
    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel an active task.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            Dict containing cancellation result
        """
        try:
            task = self.active_tasks.get(task_id)
            if not task:
                return {
                    "success": False,
                    "error": f"Task not found or already completed: {task_id}",
                    "task_id": task_id
                }
            
            # Update task status
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            task.error = "Task cancelled by request"
            
            # Move to results cache
            self.task_results[task_id] = {
                "success": False,
                "task_id": task_id,
                "status": TaskStatus.CANCELLED.value,
                "error": task.error,
                "cancelled_at": task.completed_at.isoformat()
            }
            
            del self.active_tasks[task_id]
            
            return {
                "success": True,
                "task_id": task_id,
                "status": TaskStatus.CANCELLED.value,
                "message": "Task cancelled successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Task cancellation failed for {task_id}: {e}")
            return {
                "success": False,
                "error": f"Cancellation failed: {str(e)}",
                "task_id": task_id
            }
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get available agent capabilities and coordination patterns.
        
        Returns:
            Dict containing available capabilities and coordination information
        """
        try:
            # Get agent capabilities from registry
            agents_info = []
            for agent_name in self.agent_registry.get_agent_names():
                agent = self.agent_registry.get_agent(agent_name)
                if agent:
                    agents_info.append({
                        "name": agent_name,
                        "category": getattr(agent, 'category', 'unknown'),
                        "capabilities": getattr(agent, 'capabilities', []),
                        "description": getattr(agent, 'description', '')
                    })
            
            return {
                "success": True,
                "coordinator_info": {
                    "version": "1.0.0",
                    "max_concurrent_tasks": self.max_concurrent_tasks,
                    "default_timeout": self.default_timeout,
                    "supported_patterns": [pattern.value for pattern in CoordinationPattern]
                },
                "available_agents": agents_info,
                "coordination_patterns": {
                    "sequential": {
                        "description": "Chain tasks with dependency management",
                        "use_cases": ["Code generation -> Analysis -> Testing"]
                    },
                    "parallel": {
                        "description": "Execute multiple tasks simultaneously",
                        "use_cases": ["Multi-tool data processing", "Concurrent analysis"]
                    },
                    "iterative": {
                        "description": "Multi-turn conversations with context",
                        "use_cases": ["Collaborative development", "Iterative refinement"]
                    }
                },
                "metrics": self.metrics,
                "active_tasks": len(self.active_tasks)
            }
            
        except Exception as e:
            self.logger.error(f"Capability retrieval failed: {e}")
            return {
                "success": False,
                "error": f"Capability retrieval failed: {str(e)}"
            }
    
    # Private helper methods
    
    async def _register_available_agents(self) -> None:
        """Register available agents with the ACP bridge."""
        try:
            for agent_name in self.agent_registry.get_agent_names():
                agent = self.agent_registry.get_agent(agent_name)
                if agent:
                    capabilities = getattr(agent, 'capabilities', [])
                    await self.acp_bridge.register_agent(
                        agent=agent,
                        agent_id=agent_name,
                        capabilities=capabilities,
                        description=getattr(agent, 'description', f'{agent_name} agent')
                    )
                    
            self.logger.info(f"Registered {len(self.agent_registry.list_agents())} agents with ACP bridge")
            
        except Exception as e:
            self.logger.error(f"Agent registration failed: {e}")
            raise
    
    async def _route_task_to_agent(self, task: TaskExecution) -> Optional[str]:
        """Route task to appropriate agent based on capabilities."""
        try:
            task_type = task.task_type.lower()
            task_description = f"{task_type} task"  # Create task description for can_handle()
            
            # Method 1: Check for direct agent name match first
            try:
                if self.agent_registry.get_agent(task_type):
                    return task_type
            except Exception:
                # Agent not found by exact name, continue to capability-based routing
                pass
            
            # Method 2: Use agent's can_handle() method for proper task routing
            for agent_name in self.agent_registry.get_agent_names():
                try:
                    agent = self.agent_registry.get_agent(agent_name)
                    if agent and hasattr(agent, 'can_handle') and agent.can_handle(task_description):
                        self.logger.info(f"Routed {task_type} task to {agent_name} via can_handle()")
                        return agent_name
                except Exception as e:
                    self.logger.debug(f"Error checking agent {agent_name}: {e}")
                    continue
            
            # Method 3: Check agent tools/capabilities attribute
            for agent_name in self.agent_registry.get_agent_names():
                try:
                    agent = self.agent_registry.get_agent(agent_name)
                    if agent:
                        # Check tools list
                        tools = getattr(agent, 'tools', [])
                        if task_type in [tool.lower() for tool in tools]:
                            self.logger.info(f"Routed {task_type} task to {agent_name} via tools match")
                            return agent_name
                        
                        # Check capabilities (for backward compatibility)
                        capabilities = getattr(agent, 'capabilities', [])
                        if task_type in [cap.lower() for cap in capabilities]:
                            self.logger.info(f"Routed {task_type} task to {agent_name} via capabilities match")
                            return agent_name
                except Exception as e:
                    self.logger.debug(f"Error checking agent tools {agent_name}: {e}")
                    continue
            
            # Method 4: Enhanced fallback routing with common task types
            task_type_mappings = {
                'bash': ['bash_test_agent', 'bash_agent', 'command_agent', 'shell_agent'],
                'command': ['bash_test_agent', 'bash_agent', 'command_agent', 'shell_agent'],
                'shell': ['bash_test_agent', 'bash_agent', 'command_agent', 'shell_agent'],
                'analysis': ['analysis_test_agent', 'analysis_agent', 'code_analysis_agent'],
                'analyze': ['analysis_test_agent', 'analysis_agent', 'code_analysis_agent'],
                'code': ['code_generation_agent', 'code_agent'],
                'test': ['testing_agent', 'test_agent'],
                'build': ['build_agent', 'builder_agent'],
                'security': ['security_agent', 'sec_agent']
            }
            
            # Try to find agents by fallback mapping
            for keyword, candidate_names in task_type_mappings.items():
                if keyword in task_type:
                    for candidate_name in candidate_names:
                        try:
                            if self.agent_registry.get_agent(candidate_name):
                                self.logger.info(f"Routed {task_type} task to {candidate_name} via fallback mapping")
                                return candidate_name
                        except Exception:
                            continue
            
            self.logger.warning(f"No agent found for task type: {task_type}")
            return None
            
        except Exception as e:
            self.logger.error(f"Task routing failed: {e}")
            return None
    
    async def _execute_task(self, task: TaskExecution) -> None:
        """Execute a single task and update its status."""
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()
            
            # Get agent
            agent = self.agent_registry.get_agent(task.agent_id)
            if not agent:
                task.status = TaskStatus.FAILED
                task.error = f"Agent not found: {task.agent_id}"
                task.completed_at = datetime.utcnow()
                return
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    agent.execute_task(task.task_type, task.parameters),
                    timeout=task.timeout_seconds
                )
                
                if isinstance(result, AgentResult):
                    task.result = {
                        "success": result.success,
                        "output": result.output,
                        "metadata": result.metadata
                    }
                    task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                    task.error = result.error_message if not result.success else None
                else:
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                
            except asyncio.TimeoutError:
                task.status = TaskStatus.TIMEOUT
                task.error = f"Task timed out after {task.timeout_seconds} seconds"
            
            task.completed_at = datetime.utcnow()
            
            # Update metrics
            execution_time = (task.completed_at - task.started_at).total_seconds()
            task.metrics = {
                "execution_time": execution_time,
                "agent_id": task.agent_id,
                "task_type": task.task_type
            }
            
            self._update_performance_metrics(task)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"Task execution failed: {str(e)}"
            task.completed_at = datetime.utcnow()
            self.logger.error(f"Task execution failed: {e}")
    
    async def _wait_for_tasks_completion(self, task_ids: List[str], timeout: int) -> List[Dict[str, Any]]:
        """Wait for multiple tasks to complete and return their results."""
        start_time = time.time()
        results = []
        
        while task_ids and (time.time() - start_time) < timeout:
            completed_tasks = []
            
            for task_id in task_ids:
                result = await self.get_task_result(task_id)
                if result.get("success", False) or task_id not in self.active_tasks:
                    results.append(result)
                    completed_tasks.append(task_id)
            
            # Remove completed tasks from waiting list
            for task_id in completed_tasks:
                task_ids.remove(task_id)
            
            if task_ids:
                await asyncio.sleep(0.5)  # Check every 500ms
        
        # Handle remaining tasks (timeout)
        for task_id in task_ids:
            await self.cancel_task(task_id)
            results.append({
                "success": False,
                "error": "Task timed out during coordination",
                "task_id": task_id,
                "status": TaskStatus.TIMEOUT.value
            })
        
        return results
    
    def _update_performance_metrics(self, task: TaskExecution) -> None:
        """Update performance metrics based on completed task."""
        self.metrics["tasks_executed"] += 1
        
        if task.status == TaskStatus.COMPLETED:
            self.metrics["tasks_succeeded"] += 1
        else:
            self.metrics["tasks_failed"] += 1
        
        if task.started_at and task.completed_at:
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self.metrics["total_execution_time"] += execution_time
            self.metrics["avg_execution_time"] = (
                self.metrics["total_execution_time"] / self.metrics["tasks_executed"]
            )
        
        current_active = len(self.active_tasks)
        if current_active > self.metrics["concurrent_peak"]:
            self.metrics["concurrent_peak"] = current_active
    
    # Coordination pattern handlers (placeholders for future implementation)
    
    async def _handle_sequential_coordination(self, context: CoordinationContext) -> Dict[str, Any]:
        """Handle sequential task delegation pattern."""
        try:
            results = []
            shared_context = context.shared_context.copy()
            
            for i, task in enumerate(context.tasks):
                self.logger.info(f"Executing sequential task {i+1}/{len(context.tasks)}: {task.task_type}")
                
                # Add shared context from previous tasks
                task.context.update(shared_context)
                task.context["sequence_position"] = i
                task.context["total_tasks"] = len(context.tasks)
                
                # Execute task
                await self._execute_task(task)
                
                # Wait for completion
                while task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                    await asyncio.sleep(0.1)
                
                # Collect result and update shared context
                if task.status == TaskStatus.COMPLETED and task.result:
                    results.append(task.result)
                    # Add result to shared context for next task
                    shared_context[f"task_{i}_result"] = task.result
                    shared_context[f"task_{i}_output"] = task.result.get("output", "")
                else:
                    error_result = {"error": task.error or "Task failed", "task_index": i}
                    results.append(error_result)
                    # Decide whether to continue or stop on failure
                    if context.shared_context.get("stop_on_failure", True):
                        break
            
            context.result = {
                "pattern": "sequential",
                "status": "completed",
                "total_tasks": len(context.tasks),
                "completed_tasks": len([r for r in results if "error" not in r]),
                "failed_tasks": len([r for r in results if "error" in r]),
                "results": results,
                "shared_context": shared_context
            }
            
            return context.result
            
        except Exception as e:
            self.logger.error(f"Sequential coordination failed: {e}")
            return {
                "pattern": "sequential", 
                "status": "failed", 
                "error": str(e)
            }
    
    async def _handle_parallel_coordination(self, context: CoordinationContext) -> Dict[str, Any]:
        """Handle parallel task distribution pattern."""
        try:
            # Execute all tasks concurrently
            execution_tasks = []
            
            for i, task in enumerate(context.tasks):
                self.logger.info(f"Starting parallel task {i+1}/{len(context.tasks)}: {task.task_type}")
                
                # Add parallel context
                task.context.update(context.shared_context)
                task.context["parallel_index"] = i
                task.context["total_parallel_tasks"] = len(context.tasks)
                
                # Start task execution
                execution_tasks.append(self._execute_task(task))
            
            # Wait for all tasks to complete
            await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Collect results
            results = []
            successful_count = 0
            failed_count = 0
            
            for i, task in enumerate(context.tasks):
                if task.status == TaskStatus.COMPLETED and task.result:
                    results.append({
                        "task_index": i,
                        "task_type": task.task_type,
                        "status": "completed",
                        "result": task.result,
                        "execution_time": (task.completed_at - task.started_at).total_seconds() if (task.completed_at and task.started_at) else None
                    })
                    successful_count += 1
                else:
                    results.append({
                        "task_index": i,
                        "task_type": task.task_type,
                        "status": "failed",
                        "error": task.error or "Task failed",
                        "execution_time": (task.completed_at - task.started_at).total_seconds() if (task.completed_at and task.started_at) else None
                    })
                    failed_count += 1
            
            context.result = {
                "pattern": "parallel",
                "status": "completed",
                "total_tasks": len(context.tasks),
                "successful_tasks": successful_count,
                "failed_tasks": failed_count,
                "results": results,
                "coordination_id": context.coordination_id
            }
            
            return context.result
            
        except Exception as e:
            self.logger.error(f"Parallel coordination failed: {e}")
            return {
                "pattern": "parallel", 
                "status": "failed", 
                "error": str(e)
            }
    
    async def _handle_iterative_coordination(self, context: CoordinationContext) -> Dict[str, Any]:
        """Handle iterative collaboration pattern."""
        try:
            iterations = []
            shared_state = context.shared_context.copy()
            max_iterations = shared_state.get("max_iterations", 5)
            convergence_threshold = shared_state.get("convergence_threshold", 0.95)
            
            for iteration in range(max_iterations):
                self.logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")
                
                iteration_results = []
                iteration_start = datetime.utcnow()
                
                # Execute tasks in this iteration
                for i, task in enumerate(context.tasks):
                    # Add iteration context
                    task.context.update(shared_state)
                    task.context["iteration"] = iteration
                    task.context["max_iterations"] = max_iterations
                    task.context["previous_iterations"] = iterations.copy()
                    
                    # Execute task
                    await self._execute_task(task)
                    
                    # Wait for completion
                    while task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                        await asyncio.sleep(0.1)
                    
                    # Collect result
                    if task.status == TaskStatus.COMPLETED and task.result:
                        iteration_results.append(task.result)
                        # Update shared state with task output
                        shared_state[f"iteration_{iteration}_task_{i}_result"] = task.result
                    else:
                        iteration_results.append({"error": task.error or "Task failed"})
                
                iteration_end = datetime.utcnow()
                iteration_data = {
                    "iteration": iteration,
                    "results": iteration_results,
                    "shared_state_snapshot": shared_state.copy(),
                    "start_time": iteration_start.isoformat(),
                    "end_time": iteration_end.isoformat(),
                    "duration": (iteration_end - iteration_start).total_seconds()
                }
                
                iterations.append(iteration_data)
                
                # Check convergence criteria
                if self._check_iterative_convergence(iterations, convergence_threshold):
                    self.logger.info(f"Convergence achieved after {iteration + 1} iterations")
                    break
                
                # Update shared state for next iteration
                shared_state["current_iteration"] = iteration + 1
                shared_state["iteration_history"] = iterations
            
            context.result = {
                "pattern": "iterative",
                "status": "completed",
                "total_iterations": len(iterations),
                "max_iterations": max_iterations,
                "converged": len(iterations) < max_iterations,
                "iterations": iterations,
                "final_shared_state": shared_state,
                "coordination_id": context.coordination_id
            }
            
            return context.result
            
        except Exception as e:
            self.logger.error(f"Iterative coordination failed: {e}")
            return {
                "pattern": "iterative", 
                "status": "failed", 
                "error": str(e)
            }
    
    def _check_iterative_convergence(self, iterations: List[Dict[str, Any]], threshold: float) -> bool:
        """Check if iterative coordination has converged."""
        if len(iterations) < 2:
            return False
        
        # Simple convergence check: compare success rates of last two iterations
        try:
            current_iteration = iterations[-1]
            previous_iteration = iterations[-2]
            
            current_success_rate = self._calculate_iteration_success_rate(current_iteration["results"])
            previous_success_rate = self._calculate_iteration_success_rate(previous_iteration["results"])
            
            # If both iterations have high success rates, consider converged
            return current_success_rate >= threshold and previous_success_rate >= threshold
        
        except Exception:
            return False
    
    def _calculate_iteration_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate success rate for an iteration."""
        if not results:
            return 0.0
        
        successful = len([r for r in results if r.get("success", False) and "error" not in r])
        return successful / len(results)


@asynccontextmanager
async def kwe_coordinator_context(
    server_host: str = "127.0.0.1",
    server_port: int = 8000,
    max_concurrent_tasks: int = 10
):
    """Context manager for KWE CLI Coordinator with automatic cleanup."""
    coordinator = KWECLICoordinator(
        server_host=server_host,
        server_port=server_port,
        max_concurrent_tasks=max_concurrent_tasks
    )
    
    try:
        await coordinator.start()
        yield coordinator
    finally:
        await coordinator.stop()
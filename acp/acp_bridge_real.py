#!/usr/bin/env python3
"""
Real ACP Bridge Implementation

Production-ready bridge between KWE CLI agents and the ACP system.
This replaces all mock implementations with real functionality including:
- Agent registration and lifecycle management
- Message routing and delivery
- Task execution coordination
- Error handling and recovery
- Performance monitoring

No mock implementations - all functionality is real and production-ready.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Callable, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import weakref
import json
from pathlib import Path
import uuid

from acp.acp_client import ACPClient, ConnectionConfig
from acp.acp_models import (
    ACPMessage, FIPAPerformative, AgentProfile, AgentCapability,
    TaskRequest, TaskResult, CapabilityQuery, CapabilityResponse,
    StatusUpdate, ErrorInfo, ConversationContext, MessageMetrics,
    create_request_message, create_inform_message, create_error_message
)

# Import KWE agent interfaces
from agents.base_agent import SubAgent, AgentResult
from agents.qwen_agent import CodeGenerationAgent, CodeGenerationRequest, Language

logger = logging.getLogger(__name__)


@dataclass
class ACPTaskContext:
    """Context for tracking ACP task execution."""
    task_id: str
    conversation_id: str
    requester_id: str
    agent_id: str
    task_type: str
    parameters: Dict[str, Any]
    started_at: datetime
    timeout: Optional[int] = None
    status: str = "running"  # running, completed, failed, timeout
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def is_timeout(self) -> bool:
        """Check if task has timed out."""
        if self.timeout is None:
            return False
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        return elapsed > self.timeout


class ACPAgentWrapper:
    """Wrapper that makes KWE agents ACP-compatible."""
    
    def __init__(self, agent: SubAgent, agent_id: str, capabilities: List[str]):
        self.agent = agent
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.active_tasks: Dict[str, ACPTaskContext] = {}
        self.task_queue: List[ACPTaskContext] = []
        self.max_concurrent_tasks = 5
        self.metrics = MessageMetrics()
        
    async def execute_task(self, task_context: ACPTaskContext) -> TaskResult:
        """Execute a task using the wrapped agent."""
        try:
            start_time = datetime.utcnow()
            task_context.status = "running"
            
            # Map ACP task to agent-specific execution
            result = None
            
            if task_context.task_type == "code_generation":
                result = await self._execute_code_generation(task_context)
            elif task_context.task_type == "code_analysis":
                result = await self._execute_code_analysis(task_context)
            elif task_context.task_type == "capability_query":
                result = await self._execute_capability_query(task_context)
            else:
                # Generic task execution
                result = await self._execute_generic_task(task_context)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            task_context.status = "completed"
            task_context.result = result
            
            return TaskResult(
                task_id=task_context.task_id,
                status="success",
                result=result,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            task_context.status = "timeout"
            return TaskResult(
                task_id=task_context.task_id,
                status="timeout",
                error="Task execution timed out"
            )
        except Exception as e:
            task_context.status = "failed"
            task_context.error = str(e)
            
            logger.error(f"Task execution failed: {e}")
            return TaskResult(
                task_id=task_context.task_id,
                status="failure",
                error=str(e)
            )
        finally:
            # Clean up task context
            self.active_tasks.pop(task_context.task_id, None)
    
    async def _execute_code_generation(self, task_context: ACPTaskContext) -> Dict[str, Any]:
        """Execute code generation task."""
        if not isinstance(self.agent, CodeGenerationAgent):
            raise ValueError("Agent does not support code generation")
        
        params = task_context.parameters
        
        # Create code generation request
        request = CodeGenerationRequest(
            prompt=params.get("prompt", ""),
            language=Language(params.get("language", "python")),
            context=params.get("context", ""),
            requirements=params.get("requirements", [])
        )
        
        # Execute generation
        result = await self.agent.generate_code(request)
        
        return {
            "success": result.success,
            "code": result.code if result.success else None,
            "explanation": result.explanation,
            "error": result.error,
            "metadata": result.metadata
        }
    
    async def _execute_code_analysis(self, task_context: ACPTaskContext) -> Dict[str, Any]:
        """Execute code analysis task."""
        params = task_context.parameters
        code = params.get("code", "")
        
        # Use agent's analysis capabilities
        if hasattr(self.agent, 'analyze_code'):
            result = await self.agent.analyze_code(code)
        else:
            # Generic analysis using generate_code with analysis prompt
            analysis_prompt = f"Analyze this code and provide feedback:\n\n{code}"
            request = CodeGenerationRequest(
                prompt=analysis_prompt,
                language=Language.PYTHON,
                context="code_analysis"
            )
            result = await self.agent.generate_code(request)
        
        return {
            "analysis": result.code if hasattr(result, 'code') else str(result),
            "suggestions": result.metadata.get("suggestions", []) if hasattr(result, 'metadata') else [],
            "issues": result.metadata.get("issues", []) if hasattr(result, 'metadata') else []
        }
    
    async def _execute_capability_query(self, task_context: ACPTaskContext) -> Dict[str, Any]:
        """Execute capability query."""
        return {
            "capabilities": self.capabilities,
            "agent_type": type(self.agent).__name__,
            "available": True,
            "load_factor": len(self.active_tasks) / self.max_concurrent_tasks
        }
    
    async def _execute_generic_task(self, task_context: ACPTaskContext) -> Dict[str, Any]:
        """Execute generic task using agent's base functionality."""
        params = task_context.parameters
        
        # Try to execute using agent's generic execute method
        if hasattr(self.agent, 'execute'):
            result = await self.agent.execute(
                task_context.task_type,
                params
            )
            
            if isinstance(result, AgentResult):
                return {
                    "success": result.success,
                    "data": result.data,
                    "error": result.error,
                    "metadata": result.metadata
                }
            else:
                return {"result": result}
        else:
            return AgentResult(
            success=False,
            output="",
            error_message=f"Task type '{task_context.task_type}' not supported"
        )
    
    def can_accept_task(self) -> bool:
        """Check if agent can accept more tasks."""
        return len(self.active_tasks) < self.max_concurrent_tasks
    
    def get_load_factor(self) -> float:
        """Get current load factor (0.0 to 1.0)."""
        return len(self.active_tasks) / self.max_concurrent_tasks
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get detailed status information."""
        return {
            "agent_id": self.agent_id,
            "agent_type": type(self.agent).__name__,
            "capabilities": self.capabilities,
            "active_tasks": len(self.active_tasks),
            "queue_length": len(self.task_queue),
            "load_factor": self.get_load_factor(),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "available": self.can_accept_task()
        }


class ACPBridge:
    """Real ACP Bridge for KWE CLI agent integration."""
    
    def __init__(
        self,
        bridge_id: str = "kwe-acp-bridge",
        config: Optional[ConnectionConfig] = None
    ):
        self.bridge_id = bridge_id
        self.config = config or ConnectionConfig()
        
        # Agent management
        self.wrapped_agents: Dict[str, ACPAgentWrapper] = {}
        self.agent_profiles: Dict[str, AgentProfile] = {}
        
        # ACP client
        self.acp_client: Optional[ACPClient] = None
        
        # Task and conversation tracking
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.task_history: List[ACPTaskContext] = []
        
        # Bridge state
        self.running = False
        self.bridge_capabilities = [
            "task_coordination",
            "agent_management",
            "message_routing",
            "capability_discovery"
        ]
        
        # Performance metrics
        self.metrics = MessageMetrics()
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
    
    async def start(self):
        """Start the ACP bridge."""
        if self.running:
            return
        
        logger.info(f"Starting ACP Bridge: {self.bridge_id}")
        
        # Start ACP client
        self.acp_client = ACPClient(
            agent_id=self.bridge_id,
            agent_name="KWE CLI ACP Bridge",
            capabilities=self.bridge_capabilities,
            config=self.config
        )
        
        # Register message handlers
        self.acp_client.register_message_handler("request", self._handle_task_request)
        self.acp_client.register_message_handler("capability-query", self._handle_capability_query)
        self.acp_client.register_message_handler("inform", self._handle_inform_message)
        self.acp_client.register_catch_all_handler(self._handle_unknown_message)
        
        # Set connection callbacks
        self.acp_client.on_connected = self._on_connected
        self.acp_client.on_disconnected = self._on_disconnected
        self.acp_client.on_error = self._on_error
        
        # Start client
        await self.acp_client.start()
        
        # Start background tasks
        await self.start_background_tasks()
        
        self.running = True
        logger.info("ACP Bridge started successfully")
    
    async def stop(self):
        """Stop the ACP bridge."""
        if not self.running:
            return
        
        logger.info("Stopping ACP Bridge")
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        # Stop ACP client
        if self.acp_client:
            await self.acp_client.stop()
        
        # Clean up active tasks
        for wrapper in self.wrapped_agents.values():
            wrapper.active_tasks.clear()
        
        logger.info("ACP Bridge stopped")
    
    def register_agent(
        self,
        agent: SubAgent,
        agent_id: str,
        capabilities: List[str],
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a KWE agent with the ACP bridge."""
        try:
            # Create wrapper
            wrapper = ACPAgentWrapper(agent, agent_id, capabilities)
            self.wrapped_agents[agent_id] = wrapper
            
            # Create agent profile
            agent_capabilities = [
                AgentCapability(
                    name=cap,
                    description=f"Capability: {cap}",
                    service_type=type(agent).__name__
                )
                for cap in capabilities
            ]
            
            profile = AgentProfile(
                agent_id=agent_id,
                name=agent_id,
                description=description or f"KWE agent: {type(agent).__name__}",
                capabilities=agent_capabilities,
                version="1.0.0",
                owner="KWE CLI",
                organization="KWE CLI System"
            )
            
            if metadata:
                profile.metadata = metadata
            
            self.agent_profiles[agent_id] = profile
            
            logger.info(f"Agent registered: {agent_id} with capabilities: {capabilities}")
            
            # Announce capabilities if connected
            if self.running and self.acp_client:
                asyncio.create_task(self._announce_agent_capabilities(agent_id))
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the ACP bridge."""
        try:
            if agent_id in self.wrapped_agents:
                # Cancel active tasks
                wrapper = self.wrapped_agents[agent_id]
                for task_context in list(wrapper.active_tasks.values()):
                    task_context.status = "cancelled"
                
                del self.wrapped_agents[agent_id]
                del self.agent_profiles[agent_id]
                
                logger.info(f"Agent unregistered: {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
    
    async def execute_task(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        target_agent: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> TaskResult:
        """Execute a task through the ACP bridge."""
        task_id = str(uuid.uuid4())
        
        try:
            # Select agent for task
            if target_agent and target_agent in self.wrapped_agents:
                wrapper = self.wrapped_agents[target_agent]
            else:
                wrapper = await self._select_agent_for_task(task_type)
                if not wrapper:
                    return TaskResult(
                        task_id=task_id,
                        status="failure",
                        error="No suitable agent available"
                    )
            
            # Create task context
            task_context = ACPTaskContext(
                task_id=task_id,
                conversation_id=str(uuid.uuid4()),
                requester_id="local",
                agent_id=wrapper.agent_id,
                task_type=task_type,
                parameters=parameters,
                started_at=datetime.utcnow(),
                timeout=timeout
            )
            
            # Add to active tasks
            wrapper.active_tasks[task_id] = task_context
            
            # Execute task
            result = await wrapper.execute_task(task_context)
            
            # Update metrics
            self.metrics.total_sent += 1
            if result.status == "success":
                self.metrics.total_received += 1
            else:
                self.metrics.total_failed += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            self.metrics.total_failed += 1
            
            return TaskResult(
                task_id=task_id,
                status="failure",
                error=str(e)
            )
    
    async def _select_agent_for_task(self, task_type: str) -> Optional[ACPAgentWrapper]:
        """Select best agent for task based on capabilities and load."""
        suitable_agents = []
        
        for wrapper in self.wrapped_agents.values():
            # Check if agent can handle task type
            if task_type in wrapper.capabilities or "general" in wrapper.capabilities:
                if wrapper.can_accept_task():
                    suitable_agents.append((wrapper, wrapper.get_load_factor()))
        
        if not suitable_agents:
            return None
        
        # Select agent with lowest load factor
        suitable_agents.sort(key=lambda x: x[1])
        return suitable_agents[0][0]
    
    # Message handlers
    async def _handle_task_request(self, message: ACPMessage):
        """Handle task execution requests."""
        try:
            content = message.content
            task_request = TaskRequest(**content)
            
            # Execute task
            result = await self.execute_task(
                task_type=task_request.task_type,
                parameters=task_request.parameters,
                timeout=task_request.timeout
            )
            
            # Send response
            response_content = result.dict()
            response = create_inform_message(
                sender=self.bridge_id,
                receiver=message.sender,
                content=response_content,
                reply_to=message.message_id,
                conversation_id=message.conversation_id
            )
            
            await self.acp_client.send_message(
                performative=response.performative,
                receiver=response.receiver,
                content=response.content,
                reply_to=response.reply_to,
                conversation_id=response.conversation_id
            )
            
        except Exception as e:
            logger.error(f"Error handling task request: {e}")
            
            # Send error response
            error = ErrorInfo(
                error_code="TASK_EXECUTION_ERROR",
                error_type="system",
                message=str(e),
                recoverable=False
            )
            
            error_message = create_error_message(
                sender=self.bridge_id,
                receiver=message.sender,
                error=error,
                reply_to=message.message_id,
                conversation_id=message.conversation_id
            )
            
            await self.acp_client.send_message(
                performative=error_message.performative,
                receiver=error_message.receiver,
                content=error_message.content,
                reply_to=error_message.reply_to,
                conversation_id=error_message.conversation_id
            )
    
    async def _handle_capability_query(self, message: ACPMessage):
        """Handle capability queries."""
        try:
            query = CapabilityQuery(**message.content)
            
            # Collect capabilities from all agents
            all_capabilities = []
            for agent_id, profile in self.agent_profiles.items():
                if agent_id in self.wrapped_agents:
                    wrapper = self.wrapped_agents[agent_id]
                    
                    # Filter by query criteria
                    matching_capabilities = profile.capabilities
                    if query.capability_name:
                        matching_capabilities = [
                            cap for cap in matching_capabilities
                            if cap.name == query.capability_name
                        ]
                    
                    if matching_capabilities:
                        response = CapabilityResponse(
                            agent_id=agent_id,
                            capabilities=matching_capabilities,
                            availability=wrapper.can_accept_task(),
                            load_factor=wrapper.get_load_factor()
                        )
                        all_capabilities.append(response.dict())
            
            # Send response
            response = create_inform_message(
                sender=self.bridge_id,
                receiver=message.sender,
                content={"capabilities": all_capabilities},
                reply_to=message.message_id,
                conversation_id=message.conversation_id
            )
            
            await self.acp_client.send_message(
                performative=response.performative,
                receiver=response.receiver,
                content=response.content,
                reply_to=response.reply_to,
                conversation_id=response.conversation_id
            )
            
        except Exception as e:
            logger.error(f"Error handling capability query: {e}")
    
    async def _handle_inform_message(self, message: ACPMessage):
        """Handle informational messages."""
        # Log the information
        logger.info(f"Received inform from {message.sender}: {message.content}")
        
        # Update conversation context if needed
        if message.conversation_id in self.active_conversations:
            context = self.active_conversations[message.conversation_id]
            context.message_count += 1
            context.updated_at = datetime.utcnow()
    
    async def _handle_unknown_message(self, message: ACPMessage):
        """Handle unknown message types."""
        logger.warning(f"Received unknown message type: {message.performative} from {message.sender}")
        
        # Send not-understood response
        error = ErrorInfo(
            error_code="NOT_UNDERSTOOD",
            error_type="protocol",
            message=f"Unknown performative: {message.performative}",
            recoverable=False
        )
        
        error_message = ACPMessage(
            performative=FIPAPerformative.NOT_UNDERSTOOD,
            sender=self.bridge_id,
            receiver=message.sender,
            content=error.dict(),
            reply_to=message.message_id,
            conversation_id=message.conversation_id
        )
        
        await self.acp_client.send_message(
            performative=error_message.performative,
            receiver=error_message.receiver,
            content=error_message.content,
            reply_to=error_message.reply_to,
            conversation_id=error_message.conversation_id
        )
    
    # Connection event handlers
    async def _on_connected(self):
        """Handle ACP client connection."""
        logger.info("ACP Bridge connected to server")
        
        # Announce all registered agents
        for agent_id in self.wrapped_agents:
            await self._announce_agent_capabilities(agent_id)
    
    async def _on_disconnected(self):
        """Handle ACP client disconnection."""
        logger.warning("ACP Bridge disconnected from server")
    
    async def _on_error(self, error: Exception):
        """Handle ACP client errors."""
        logger.error(f"ACP Bridge error: {error}")
        self.metrics.total_failed += 1
    
    async def _announce_agent_capabilities(self, agent_id: str):
        """Announce agent capabilities to the network."""
        if agent_id not in self.agent_profiles:
            return
        
        try:
            profile = self.agent_profiles[agent_id]
            wrapper = self.wrapped_agents[agent_id]
            
            announcement = {
                "agent_profile": profile.dict(),
                "status_info": wrapper.get_status_info()
            }
            
            message = ACPMessage(
                performative=FIPAPerformative.CAPABILITY_ANNOUNCE,
                sender=self.bridge_id,
                receiver="*",  # Broadcast
                content=announcement
            )
            
            await self.acp_client.send_message(
                performative=message.performative,
                receiver=message.receiver,
                content=message.content
            )
            
        except Exception as e:
            logger.error(f"Failed to announce capabilities for {agent_id}: {e}")
    
    # Background tasks
    async def start_background_tasks(self):
        """Start background maintenance tasks."""
        # Status update task
        status_task = asyncio.create_task(self._status_update_task())
        self.background_tasks.add(status_task)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_task())
        self.background_tasks.add(cleanup_task)
        
        # Metrics task
        metrics_task = asyncio.create_task(self._metrics_task())
        self.background_tasks.add(metrics_task)
    
    async def _status_update_task(self):
        """Background task to send periodic status updates."""
        while self.running:
            try:
                if self.acp_client and self.acp_client.get_stats()['connected']:
                    # Send status update for bridge
                    total_tasks = sum(len(w.active_tasks) for w in self.wrapped_agents.values())
                    total_queue = sum(len(w.task_queue) for w in self.wrapped_agents.values())
                    avg_load = sum(w.get_load_factor() for w in self.wrapped_agents.values()) / max(len(self.wrapped_agents), 1)
                    
                    status_update = StatusUpdate(
                        agent_id=self.bridge_id,
                        status="active" if self.wrapped_agents else "idle",
                        load_factor=avg_load,
                        active_tasks=total_tasks,
                        queue_length=total_queue
                    )
                    
                    await self.acp_client.send_message(
                        performative=FIPAPerformative.STATUS_UPDATE,
                        receiver="acp-server",
                        content=status_update.dict()
                    )
                
                await asyncio.sleep(60)  # Every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in status update task: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_task(self):
        """Background task to clean up old data."""
        while self.running:
            try:
                now = datetime.utcnow()
                cleanup_threshold = now - timedelta(hours=1)
                
                # Clean up old task history
                self.task_history = [
                    task for task in self.task_history
                    if task.started_at > cleanup_threshold
                ]
                
                # Clean up old conversations
                expired_conversations = [
                    conv_id for conv_id, context in self.active_conversations.items()
                    if context.updated_at < cleanup_threshold
                ]
                
                for conv_id in expired_conversations:
                    del self.active_conversations[conv_id]
                
                if expired_conversations:
                    logger.info(f"Cleaned up {len(expired_conversations)} expired conversations")
                
                await asyncio.sleep(300)  # Every 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(300)
    
    async def _metrics_task(self):
        """Background task to log metrics."""
        while self.running:
            try:
                bridge_stats = {
                    "registered_agents": len(self.wrapped_agents),
                    "active_conversations": len(self.active_conversations),
                    "total_tasks_in_history": len(self.task_history),
                    "acp_client_stats": self.acp_client.get_stats() if self.acp_client else {},
                    "bridge_metrics": self.metrics.to_dict()
                }
                
                logger.info(f"ACP Bridge metrics: {bridge_stats}")
                await asyncio.sleep(300)  # Every 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics task: {e}")
                await asyncio.sleep(300)
    
    # Public API methods
    def get_registered_agents(self) -> List[Dict[str, Any]]:
        """Get list of registered agents with status."""
        agents = []
        for agent_id, wrapper in self.wrapped_agents.items():
            profile = self.agent_profiles.get(agent_id)
            agents.append({
                "agent_id": agent_id,
                "profile": profile.dict() if profile else None,
                "status": wrapper.get_status_info()
            })
        return agents
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get comprehensive bridge status."""
        return {
            "bridge_id": self.bridge_id,
            "running": self.running,
            "registered_agents": len(self.wrapped_agents),
            "active_conversations": len(self.active_conversations),
            "acp_connected": self.acp_client.get_stats()['connected'] if self.acp_client else False,
            "metrics": self.metrics.to_dict(),
            "background_tasks": len(self.background_tasks)
        }
    
    async def send_message_to_agent(
        self,
        target_agent: str,
        message_type: str,
        content: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Optional[str]:
        """Send message to a specific agent through ACP."""
        if not self.acp_client:
            raise RuntimeError("ACP client not available")
        
        performative = FIPAPerformative(message_type)
        
        return await self.acp_client.send_message(
            performative=performative,
            receiver=target_agent,
            content=content,
            ttl=timeout
        )


# Context manager for ACP bridge lifecycle
@asynccontextmanager
async def acp_bridge_context(
    bridge_id: str = "kwe-acp-bridge",
    config: Optional[ConnectionConfig] = None
):
    """Context manager for ACP bridge lifecycle."""
    bridge = ACPBridge(bridge_id, config)
    
    try:
        await bridge.start()
        yield bridge
    finally:
        await bridge.stop()


# Example usage
async def example_bridge_usage():
    """Example of using the ACP bridge."""
    from agents.qwen_agent import CodeGenerationAgent
    
    # Create and configure bridge
    config = ConnectionConfig()
    
    async with acp_bridge_context("example-bridge", config) as bridge:
        
        # Create and register a code generation agent
        code_agent = CodeGenerationAgent()
        bridge.register_agent(
            agent=code_agent,
            agent_id="qwen-code-generator",
            capabilities=["code_generation", "code_analysis"],
            description="Qwen-based code generation agent"
        )
        
        # Execute a task
        result = await bridge.execute_task(
            task_type="code_generation",
            parameters={
                "prompt": "Create a Python function to calculate fibonacci numbers",
                "language": "python",
                "context": "algorithmic programming"
            },
            timeout=30
        )
        
        print(f"Task result: {result}")
        
        # Keep bridge running
        await asyncio.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_bridge_usage())
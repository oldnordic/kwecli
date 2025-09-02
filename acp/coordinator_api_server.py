#!/usr/bin/env python3
"""
KWE CLI Coordinator HTTP API Server

FastAPI server that exposes the KWECLICoordinator functionality via REST API endpoints.
This enables Claude Code and other AI systems to coordinate with KWE CLI through HTTP.

Implements the coordination endpoints specified in ACP_COORDINATION_DESIGN.md:
- POST /api/acp/delegate - Single task delegation
- GET /api/acp/task/{task_id}/status - Task status monitoring  
- GET /api/acp/task/{task_id}/result - Task result retrieval
- POST /api/acp/task/{task_id}/cancel - Task cancellation
- POST /api/acp/parallel - Parallel task coordination
- GET /api/acp/capabilities - Capability discovery
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from acp.kwe_coordinator import KWECLICoordinator, TaskStatus, CoordinationPattern, kwe_coordinator_context

# Import agent system for registration
from agents.agent_registry import AgentRegistry
from agents.base_agent import SubAgent, AgentResult, AgentStatus, AgentExpertise

logger = logging.getLogger(__name__)


# Test agents for HTTP API server coordination testing
class SimpleBashAgent(SubAgent):
    """Simple test agent that can handle bash commands."""
    
    def __init__(self):
        super().__init__(
            name="bash_test_agent",
            description="Simple agent for testing bash command execution",
            expertise=[AgentExpertise.DEVOPS, AgentExpertise.INFRASTRUCTURE],
            tools=["bash", "command_execution"]
        )
    
    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle the task."""
        task_lower = task.lower()
        return any(keyword in task_lower for keyword in ["bash", "command", "shell", "execute", "echo"])
    
    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a bash command (simplified for testing)."""
        try:
            # Extract command from context/parameters
            command = context.get("command") or context.get("parameters", {}).get("command", task)
            
            # Simulate successful bash execution
            if "echo" in command.lower():
                output = command.replace("echo ", "").strip("'\"")
                return AgentResult(
                    success=True,
                    output=output,
                    metadata={"command": command, "agent": self.name, "return_code": 0}
                )
            else:
                return AgentResult(
                    success=True,
                    output=f"Simulated execution of: {command}",
                    metadata={"command": command, "agent": self.name, "simulated": True, "return_code": 0}
                )
                
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                error_message=f"Command execution failed: {str(e)}",
                metadata={"command": command, "agent": self.name}
            )


class SimpleAnalysisAgent(SubAgent):
    """Simple test agent that can handle analysis tasks."""
    
    def __init__(self):
        super().__init__(
            name="analysis_test_agent",
            description="Simple agent for testing code/file analysis",
            expertise=[AgentExpertise.CODE_ANALYSIS, AgentExpertise.TESTING],
            tools=["analysis", "inspection"]
        )
    
    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle the task."""
        task_lower = task.lower()
        return any(keyword in task_lower for keyword in ["analyze", "analysis", "review", "inspect", "examine"])
    
    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Perform simple analysis (simulated for testing)."""
        try:
            # Extract parameters from context
            target = context.get("target") or context.get("parameters", {}).get("target", "unknown")
            analysis_type = context.get("analysis_type") or context.get("parameters", {}).get("analysis_type", "general")
            
            # Simulate analysis result
            analysis_output = f"""
Analysis Results for: {target}
Analysis Type: {analysis_type}
Agent: {self.name}

Key Findings:
- Structure appears well-organized
- No critical issues identified
- Follows standard conventions
- Recommendations: Consider adding more documentation

Analysis completed successfully by {self.name}
"""
            
            return AgentResult(
                success=True,
                output=analysis_output,
                metadata={
                    "target": target,
                    "analysis_type": analysis_type,
                    "agent": self.name,
                    "findings_count": 4,
                    "score": 8.5
                }
            )
                
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                error_message=f"Analysis failed: {str(e)}",
                metadata={"target": target, "agent": self.name}
            )


def create_test_agent_registry() -> AgentRegistry:
    """Create a test agent registry with sample agents for API server."""
    registry = AgentRegistry(enforce_quality=False)  # Disable quality for testing
    
    # Register test agents
    bash_agent = SimpleBashAgent()
    analysis_agent = SimpleAnalysisAgent()
    
    registry.register_agent(bash_agent)
    registry.register_agent(analysis_agent)
    
    logger.info(f"Created test agent registry with {registry.get_agent_count()} agents: {registry.get_agent_names()}")
    return registry


def create_real_agent_registry() -> AgentRegistry:
    """Create a real agent registry with functional agents for production use."""
    from agents.real_file_agent import RealFileAgent
    from agents.real_subprocess_agent import RealSubprocessAgent
    
    registry = AgentRegistry(enforce_quality=True)  # Enable quality enforcement for production
    
    # Register real agents - no mocks, no stubs, no placeholders
    file_agent = RealFileAgent()
    subprocess_agent = RealSubprocessAgent()
    
    registry.register_agent(file_agent)
    registry.register_agent(subprocess_agent)
    
    logger.info(f"Created real agent registry with {registry.get_agent_count()} agents: {registry.get_agent_names()}")
    return registry


# Request/Response Models
class TaskDelegationRequest(BaseModel):
    """Request model for task delegation."""
    task_type: str = Field(..., description="Type of task to delegate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    requester_id: Optional[str] = Field(default="unknown", description="ID of requesting system")
    conversation_id: Optional[str] = Field(default=None, description="Conversation context ID")
    timeout: Optional[int] = Field(default=300, description="Task timeout in seconds")


class ParallelCoordinationRequest(BaseModel):
    """Request model for parallel task coordination."""
    tasks: List[Dict[str, Any]] = Field(..., description="List of tasks to execute in parallel")
    requester_id: Optional[str] = Field(default="unknown", description="ID of requesting system")
    coordination_timeout: Optional[int] = Field(default=600, description="Overall coordination timeout")


class TaskResponse(BaseModel):
    """Response model for task operations."""
    success: bool
    task_id: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    success: bool
    task_id: str
    status: str
    progress_percent: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    completed: bool = False
    error: Optional[str] = None


class TaskResultResponse(BaseModel):
    """Response model for task results."""
    success: bool
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None


class CapabilitiesResponse(BaseModel):
    """Response model for capabilities."""
    success: bool
    coordinator_info: Dict[str, Any]
    available_agents: List[Dict[str, Any]]
    coordination_patterns: Dict[str, Any]
    metrics: Dict[str, Any]
    error: Optional[str] = None


# Global coordinator instance
coordinator_instance: Optional[KWECLICoordinator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for coordinator setup/teardown."""
    global coordinator_instance
    try:
        # Create real agent registry with functional agents
        real_registry = create_real_agent_registry()
        
        # Initialize coordinator with real agents
        coordinator_instance = KWECLICoordinator(
            acp_bridge=None,  # No ACP bridge - standalone HTTP API mode
            agent_registry=real_registry,  # Use real agents
            server_host="127.0.0.1",
            server_port=8000,
            max_concurrent_tasks=10
        )
        # Start in standalone mode
        coordinator_instance.start_standalone()
        
        logger.info("KWE CLI Coordinator API Server started")
        yield
        
    finally:
        # Cleanup coordinator
        if coordinator_instance:
            await coordinator_instance.stop()
            coordinator_instance = None
        logger.info("KWE CLI Coordinator API Server stopped")


def get_coordinator() -> KWECLICoordinator:
    """Dependency to get coordinator instance."""
    if coordinator_instance is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    return coordinator_instance


# Create FastAPI application
app = FastAPI(
    title="KWE CLI Coordinator API",
    description="REST API for AI-to-AI coordination via KWE CLI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "coordinator_running": coordinator_instance is not None and coordinator_instance._running
    }


@app.post("/api/acp/delegate", response_model=TaskResponse)
async def delegate_task(
    request: TaskDelegationRequest,
    coordinator: KWECLICoordinator = Depends(get_coordinator)
):
    """Delegate a single task to KWE CLI."""
    try:
        task_request = {
            "task_type": request.task_type,
            "parameters": request.parameters
        }
        
        result = await coordinator.delegate_task(
            task_request=task_request,
            requester_id=request.requester_id,
            conversation_id=request.conversation_id,
            timeout=request.timeout
        )
        
        return TaskResponse(**result)
        
    except Exception as e:
        logger.error(f"Task delegation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task delegation failed: {str(e)}")


@app.get("/api/acp/task/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    coordinator: KWECLICoordinator = Depends(get_coordinator)
):
    """Get status of a delegated task."""
    try:
        result = await coordinator.check_task_status(task_id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result.get("error", "Task not found"))
        
        return TaskStatusResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.get("/api/acp/task/{task_id}/result", response_model=TaskResultResponse)
async def get_task_result(
    task_id: str,
    coordinator: KWECLICoordinator = Depends(get_coordinator)
):
    """Get result of a completed task."""
    try:
        result = await coordinator.get_task_result(task_id)
        
        if not result["success"]:
            if "not found" in result.get("error", "").lower():
                raise HTTPException(status_code=404, detail=result.get("error"))
            elif "still" in result.get("error", "").lower():
                raise HTTPException(status_code=202, detail=result.get("error"))
            else:
                raise HTTPException(status_code=500, detail=result.get("error"))
        
        return TaskResultResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task result retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Result retrieval failed: {str(e)}")


@app.post("/api/acp/task/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(
    task_id: str,
    coordinator: KWECLICoordinator = Depends(get_coordinator)
):
    """Cancel an active task."""
    try:
        result = await coordinator.cancel_task(task_id)
        
        if not result["success"]:
            if "not found" in result.get("error", "").lower():
                raise HTTPException(status_code=404, detail=result.get("error"))
            else:
                raise HTTPException(status_code=500, detail=result.get("error"))
        
        return TaskResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task cancellation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")


@app.post("/api/acp/parallel")
async def parallel_coordination(
    request: ParallelCoordinationRequest,
    coordinator: KWECLICoordinator = Depends(get_coordinator)
):
    """Execute multiple tasks in parallel."""
    try:
        results = await coordinator.parallel_delegation(
            tasks=request.tasks,
            requester_id=request.requester_id,
            coordination_timeout=request.coordination_timeout
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Parallel coordination failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parallel coordination failed: {str(e)}")


@app.get("/api/acp/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities(
    coordinator: KWECLICoordinator = Depends(get_coordinator)
):
    """Get KWE CLI capabilities and coordination information."""
    try:
        result = await coordinator.get_capabilities()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return CapabilitiesResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Capability retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Capability retrieval failed: {str(e)}")


@app.get("/api/acp/metrics")
async def get_metrics(
    coordinator: KWECLICoordinator = Depends(get_coordinator)
):
    """Get coordination performance metrics."""
    try:
        return {
            "success": True,
            "metrics": coordinator.metrics,
            "active_tasks": len(coordinator.active_tasks),
            "coordination_contexts": len(coordinator.coordination_contexts),
            "cached_results": len(coordinator.task_results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@app.get("/api/acp/tasks")
async def list_active_tasks(
    coordinator: KWECLICoordinator = Depends(get_coordinator)
):
    """List all active tasks."""
    try:
        active_tasks = []
        for task_id, task in coordinator.active_tasks.items():
            active_tasks.append({
                "task_id": task_id,
                "status": task.status.value,
                "task_type": task.task_type,
                "agent_id": task.agent_id,
                "requester_id": task.requester_id,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None
            })
        
        return {
            "success": True,
            "active_tasks": active_tasks,
            "total_active": len(active_tasks)
        }
        
    except Exception as e:
        logger.error(f"Task listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task listing failed: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )


async def start_coordinator_api_server(
    host: str = "0.0.0.0",
    port: int = 18103,
    log_level: str = "info"
):
    """Start the coordinator API server."""
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start server
    asyncio.run(start_coordinator_api_server())
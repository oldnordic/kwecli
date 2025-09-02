#!/usr/bin/env python3
"""
KWE CLI Base Agent Architecture

This module provides the foundational architecture for all autonomous agents
in the KWE CLI ecosystem. Each agent integrates sequential thinking, LTMC 
coordination, and specialized capabilities.

Key Features:
- Native sequential reasoning integration
- LTMC-based persistent memory and coordination
- Tool system integration for real operations
- Comprehensive activity tracking
- Agent coordination protocol
- Performance monitoring and optimization
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from .sequential_thinking import (
    KWECLISequentialThinking, Problem, ReasoningResult, 
    ReasoningType, ThoughtType
)

# Configure logging
logger = logging.getLogger(__name__)

class AgentCapability(Enum):
    """Available agent capabilities."""
    SEQUENTIAL_REASONING = "sequential_reasoning"
    CODE_GENERATION = "code_generation" 
    FILE_OPERATIONS = "file_operations"
    TESTING = "testing"
    SECURITY_ANALYSIS = "security_analysis"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    PLANNING = "planning"
    MONITORING = "monitoring"
    COORDINATION = "coordination"

class AgentRole(Enum):
    """Agent roles in the autonomous development ecosystem."""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    IMPLEMENTER = "implementer" 
    TESTER = "tester"
    SECURITY_SPECIALIST = "security_specialist"
    DOCUMENTATION_SPECIALIST = "documentation_specialist"
    RESEARCH_SPECIALIST = "research_specialist"
    MONITOR = "monitor"

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"

@dataclass
class AgentTask:
    """Represents a task assigned to an agent."""
    task_id: str
    title: str
    description: str
    priority: TaskPriority
    assigned_agent: str
    requester_agent: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass
class AgentResult:
    """Result of agent task execution."""
    task_id: str
    agent_id: str
    success: bool
    result_data: Any
    reasoning_session: Optional[ReasoningResult] = None
    execution_time_ms: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    artifacts_created: List[str] = field(default_factory=list)
    learning_insights: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass
class AgentMetrics:
    """Performance and activity metrics for an agent."""
    agent_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time_ms: float = 0.0
    average_task_time_ms: float = 0.0
    success_rate: float = 0.0
    reasoning_sessions_completed: int = 0
    tool_calls_executed: int = 0
    ltmc_operations: int = 0
    last_activity: Optional[str] = None
    performance_score: float = 0.0
    
    def update_metrics(self, result: AgentResult):
        """Update metrics based on task result."""
        if result.success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        self.total_execution_time_ms += result.execution_time_ms
        total_tasks = self.tasks_completed + self.tasks_failed
        
        if total_tasks > 0:
            self.average_task_time_ms = self.total_execution_time_ms / total_tasks
            self.success_rate = self.tasks_completed / total_tasks
        
        if result.reasoning_session:
            self.reasoning_sessions_completed += 1
        
        self.last_activity = datetime.now().isoformat()
        self.performance_score = self.calculate_performance_score()
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        # Weight different factors
        success_weight = 0.4
        speed_weight = 0.3
        reliability_weight = 0.3
        
        # Success rate component (0-1)
        success_component = self.success_rate
        
        # Speed component (inverse of average time, normalized)
        speed_component = max(0, 1 - (self.average_task_time_ms / 60000))  # 1 minute baseline
        
        # Reliability component (based on consistency)
        reliability_component = min(1.0, self.tasks_completed / 10)  # Full reliability after 10 tasks
        
        return (success_component * success_weight + 
                speed_component * speed_weight + 
                reliability_component * reliability_weight)

class AgentCoordinationProtocol:
    """Protocol for inter-agent coordination."""
    
    def __init__(self, ltmc_integration=None):
        self.ltmc_integration = ltmc_integration
        self.coordination_channels: Dict[str, List[Callable]] = {}
        self.agent_registry: Dict[str, 'BaseKWECLIAgent'] = {}
    
    def register_agent(self, agent: 'BaseKWECLIAgent'):
        """Register agent for coordination."""
        self.agent_registry[agent.agent_id] = agent
        logger.info(f"Agent registered for coordination: {agent.agent_id}")
    
    async def send_message(self, from_agent: str, to_agent: str, 
                          message_type: str, content: Dict[str, Any]):
        """Send message between agents via LTMC."""
        if not self.ltmc_integration:
            logger.warning("LTMC integration not available for agent coordination")
            return
        
        try:
            message_doc = f"AGENT_MESSAGE_{from_agent}_to_{to_agent}_{int(time.time())}.md"
            message_content = f"""# Inter-Agent Communication
## From: {from_agent}
## To: {to_agent}  
## Type: {message_type}
## Timestamp: {datetime.now().isoformat()}

### Message Content:
```json
{json.dumps(content, indent=2)}
```

This message is part of autonomous agent coordination in KWE CLI.
"""
            
            await self.ltmc_integration.store_document(
                file_name=message_doc,
                content=message_content,
                conversation_id="agent_coordination",
                resource_type="agent_message"
            )
            
            logger.debug(f"Agent message sent: {from_agent} -> {to_agent} ({message_type})")
            
        except Exception as e:
            logger.error(f"Failed to send agent message: {e}")
    
    async def broadcast_message(self, from_agent: str, message_type: str, 
                              content: Dict[str, Any]):
        """Broadcast message to all registered agents."""
        for agent_id in self.agent_registry:
            if agent_id != from_agent:
                await self.send_message(from_agent, agent_id, message_type, content)

class BaseKWECLIAgent(ABC):
    """
    Base class for all KWE CLI autonomous agents.
    
    Provides:
    - Sequential reasoning integration
    - LTMC coordination and memory
    - Tool system integration
    - Performance monitoring
    - Agent coordination protocols
    """
    
    def __init__(self, agent_id: str, role: AgentRole, 
                 capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        
        # Core systems
        self.sequential_thinking = KWECLISequentialThinking()
        self.ltmc_integration = None  # Initialized when available
        self.coordination_protocol = AgentCoordinationProtocol()
        self.tool_system = None  # Initialized when available
        
        # State management
        self.current_tasks: List[AgentTask] = []
        self.task_history: List[AgentTask] = []
        self.metrics = AgentMetrics(agent_id=agent_id)
        self.is_active = False
        self.configuration: Dict[str, Any] = {}
        
        # Performance optimization
        self.task_cache: Dict[str, Any] = {}
        self.learning_patterns: Dict[str, Any] = {}
        
        logger.info(f"Initialized agent: {agent_id} with role {role.value}")
    
    def initialize_integrations(self, ltmc_client=None, tool_system=None):
        """Initialize external system integrations."""
        if ltmc_client:
            self.ltmc_integration = ltmc_client
            self.sequential_thinking.initialize_ltmc_integration(ltmc_client)
            self.coordination_protocol = AgentCoordinationProtocol(ltmc_client)
            logger.info(f"LTMC integration initialized for agent {self.agent_id}")
        
        if tool_system:
            self.tool_system = tool_system
            logger.info(f"Tool system integration initialized for agent {self.agent_id}")
    
    async def start(self):
        """Start the agent's main execution loop."""
        self.is_active = True
        logger.info(f"Agent {self.agent_id} started")
        
        # Register with coordination protocol
        self.coordination_protocol.register_agent(self)
        
        # Start main execution loop
        asyncio.create_task(self.main_execution_loop())
        
        # Store startup event in LTMC
        if self.ltmc_integration:
            await self.store_agent_event("startup", {
                "agent_id": self.agent_id,
                "role": self.role.value,
                "capabilities": [c.value for c in self.capabilities],
                "startup_time": datetime.now().isoformat()
            })
    
    async def stop(self):
        """Stop the agent gracefully."""
        self.is_active = False
        
        # Complete current tasks
        for task in self.current_tasks:
            if task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.CANCELLED
        
        # Store shutdown event
        if self.ltmc_integration:
            await self.store_agent_event("shutdown", {
                "agent_id": self.agent_id,
                "shutdown_time": datetime.now().isoformat(),
                "final_metrics": asdict(self.metrics)
            })
        
        logger.info(f"Agent {self.agent_id} stopped")
    
    async def main_execution_loop(self):
        """Main agent execution loop."""
        while self.is_active:
            try:
                # Check for new tasks
                await self.check_for_new_tasks()
                
                # Process current tasks
                await self.process_current_tasks()
                
                # Perform maintenance operations
                await self.perform_maintenance()
                
                # Brief pause before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in agent {self.agent_id} execution loop: {e}")
                await asyncio.sleep(5)  # Longer pause on error
    
    async def check_for_new_tasks(self):
        """Check for new tasks assigned to this agent."""
        if not self.ltmc_integration:
            return
        
        try:
            # Query LTMC for tasks assigned to this agent
            tasks = await self.ltmc_integration.retrieve_documents(
                query=f"AGENT_TASK_{self.agent_id}",
                conversation_id="agent_tasks",
                k=10
            )
            
            for task_doc in tasks.get("results", []):
                # Parse task from document
                task = self.parse_task_from_document(task_doc)
                if task and task.status == TaskStatus.PENDING:
                    if not any(t.task_id == task.task_id for t in self.current_tasks):
                        self.current_tasks.append(task)
                        logger.info(f"New task assigned to {self.agent_id}: {task.title}")
                        
        except Exception as e:
            logger.error(f"Failed to check for new tasks: {e}")
    
    def parse_task_from_document(self, task_doc: Dict[str, Any]) -> Optional[AgentTask]:
        """Parse task from LTMC document."""
        try:
            content = task_doc.get("content", "")
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                task_json = content[json_start:json_end].strip()
                task_data = json.loads(task_json)
                
                return AgentTask(
                    task_id=task_data.get("task_id", str(uuid.uuid4())),
                    title=task_data.get("title", "Untitled Task"),
                    description=task_data.get("description", ""),
                    priority=TaskPriority(task_data.get("priority", "normal")),
                    assigned_agent=self.agent_id,
                    requirements=task_data.get("requirements", []),
                    success_criteria=task_data.get("success_criteria", [])
                )
        except Exception as e:
            logger.error(f"Failed to parse task from document: {e}")
            return None
    
    async def process_current_tasks(self):
        """Process all current tasks."""
        for task in self.current_tasks.copy():
            if task.status == TaskStatus.PENDING:
                await self.execute_task(task)
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a specific task using sequential reasoning."""
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now().isoformat()
            
            logger.info(f"Agent {self.agent_id} executing task: {task.title}")
            
            start_time = time.time()
            
            # Convert task to problem for sequential reasoning
            problem = Problem(
                description=task.description,
                context={
                    "task_id": task.task_id,
                    "requirements": task.requirements,
                    "agent_capabilities": [c.value for c in self.capabilities]
                },
                success_criteria=task.success_criteria,
                priority=task.priority.value
            )
            
            # Execute sequential reasoning
            reasoning_result = await self.sequential_thinking.sequential_reason(
                problem, self.get_preferred_reasoning_type(task)
            )
            
            # Execute the specialized task logic
            execution_result = await self.execute_specialized_task(task, reasoning_result)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000
            
            # Create result
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=execution_result.get("success", False),
                result_data=execution_result.get("data"),
                reasoning_session=reasoning_result,
                execution_time_ms=execution_time,
                resource_usage=execution_result.get("resource_usage", {}),
                error_message=execution_result.get("error"),
                warnings=execution_result.get("warnings", []),
                artifacts_created=execution_result.get("artifacts", []),
                learning_insights=reasoning_result.learning_insights if reasoning_result else []
            )
            
            # Update task status
            if result.success:
                task.status = TaskStatus.COMPLETED
            else:
                task.status = TaskStatus.FAILED
            
            task.completed_at = datetime.now().isoformat()
            
            # Update metrics
            self.metrics.update_metrics(result)
            
            # Store result in LTMC
            if self.ltmc_integration:
                await self.store_task_result(task, result)
            
            # Move to history
            self.current_tasks.remove(task)
            self.task_history.append(task)
            
            logger.info(f"Task completed by {self.agent_id}: {task.title} (Success: {result.success})")
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            
            # Update task as failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now().isoformat()
            
            # Create error result
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                result_data=None,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            )
            
            self.metrics.update_metrics(result)
            
            return result
    
    @abstractmethod
    async def execute_specialized_task(self, task: AgentTask, 
                                     reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """
        Execute the specialized logic for this agent type.
        
        Must be implemented by each specific agent subclass.
        
        Args:
            task: The task to execute
            reasoning_result: Result of sequential reasoning about the task
            
        Returns:
            Dictionary with execution results
        """
        pass
    
    def get_preferred_reasoning_type(self, task: AgentTask) -> ReasoningType:
        """Get preferred reasoning type for a task."""
        if task.priority == TaskPriority.CRITICAL:
            return ReasoningType.ANALYTICAL
        elif len(task.requirements) > 5:
            return ReasoningType.BRANCHING
        elif "creative" in task.description.lower():
            return ReasoningType.CREATIVE
        else:
            return ReasoningType.LINEAR
    
    async def perform_maintenance(self):
        """Perform regular maintenance operations."""
        try:
            # Clean up old task cache entries
            current_time = time.time()
            self.task_cache = {
                k: v for k, v in self.task_cache.items()
                if current_time - v.get("timestamp", 0) < 3600  # 1 hour
            }
            
            # Update learning patterns based on recent performance
            await self.update_learning_patterns()
            
            # Store periodic metrics in LTMC
            if len(self.task_history) % 10 == 0 and self.ltmc_integration:
                await self.store_agent_metrics()
            
        except Exception as e:
            logger.error(f"Maintenance operation failed for {self.agent_id}: {e}")
    
    async def update_learning_patterns(self):
        """Update learning patterns based on task execution history."""
        if len(self.task_history) < 2:
            return
        
        # Analyze recent task patterns
        recent_tasks = self.task_history[-10:]
        
        # Pattern: Task types that succeed more often
        success_by_type = {}
        for task in recent_tasks:
            task_type = self.classify_task_type(task)
            if task_type not in success_by_type:
                success_by_type[task_type] = {"success": 0, "total": 0}
            
            success_by_type[task_type]["total"] += 1
            if task.status == TaskStatus.COMPLETED:
                success_by_type[task_type]["success"] += 1
        
        # Store learning patterns
        self.learning_patterns["task_success_patterns"] = {
            task_type: data["success"] / data["total"]
            for task_type, data in success_by_type.items()
            if data["total"] > 0
        }
    
    def classify_task_type(self, task: AgentTask) -> str:
        """Classify task type for learning pattern analysis."""
        description = task.description.lower()
        
        if "code" in description or "implement" in description:
            return "implementation"
        elif "test" in description or "validate" in description:
            return "testing"
        elif "analyze" in description or "research" in description:
            return "analysis"
        elif "document" in description or "write" in description:
            return "documentation"
        else:
            return "general"
    
    async def store_agent_event(self, event_type: str, event_data: Dict[str, Any]):
        """Store agent event in LTMC."""
        if not self.ltmc_integration:
            return
        
        try:
            event_doc = f"AGENT_EVENT_{self.agent_id}_{event_type}_{int(time.time())}.md"
            content = f"""# Agent Event: {event_type}
## Agent: {self.agent_id}
## Role: {self.role.value}
## Timestamp: {datetime.now().isoformat()}

### Event Data:
```json
{json.dumps(event_data, indent=2)}
```

This event is part of autonomous agent activity tracking in KWE CLI.
"""
            
            await self.ltmc_integration.store_document(
                file_name=event_doc,
                content=content,
                conversation_id="agent_events",
                resource_type="agent_event"
            )
            
        except Exception as e:
            logger.error(f"Failed to store agent event: {e}")
    
    async def store_task_result(self, task: AgentTask, result: AgentResult):
        """Store task result in LTMC."""
        if not self.ltmc_integration:
            return
        
        try:
            result_doc = f"AGENT_TASK_RESULT_{task.task_id}_{self.agent_id}.md"
            content = f"""# Agent Task Result
## Task: {task.title}
## Agent: {self.agent_id}
## Success: {result.success}
## Execution Time: {result.execution_time_ms:.2f}ms
## Timestamp: {datetime.now().isoformat()}

### Task Details:
```json
{json.dumps(task.to_dict(), indent=2)}
```

### Result Details:
```json
{json.dumps(result.to_dict(), indent=2)}
```

This result is part of autonomous agent task execution in KWE CLI.
"""
            
            await self.ltmc_integration.store_document(
                file_name=result_doc,
                content=content,
                conversation_id="agent_results",
                resource_type="task_result"
            )
            
        except Exception as e:
            logger.error(f"Failed to store task result: {e}")
    
    async def store_agent_metrics(self):
        """Store current agent metrics in LTMC."""
        if not self.ltmc_integration:
            return
        
        try:
            metrics_doc = f"AGENT_METRICS_{self.agent_id}_{int(time.time())}.md"
            content = f"""# Agent Performance Metrics
## Agent: {self.agent_id}
## Role: {self.role.value}
## Report Time: {datetime.now().isoformat()}

### Current Metrics:
```json
{json.dumps(asdict(self.metrics), indent=2)}
```

### Learning Patterns:
```json
{json.dumps(self.learning_patterns, indent=2)}
```

This metrics report is part of autonomous agent performance monitoring in KWE CLI.
"""
            
            await self.ltmc_integration.store_document(
                file_name=metrics_doc,
                content=content,
                conversation_id="agent_metrics",
                resource_type="performance_metrics"
            )
            
        except Exception as e:
            logger.error(f"Failed to store agent metrics: {e}")

# Export main classes
__all__ = [
    'BaseKWECLIAgent',
    'AgentTask', 
    'AgentResult',
    'AgentRole',
    'AgentCapability',
    'TaskPriority',
    'TaskStatus',
    'AgentCoordinationProtocol',
    'AgentMetrics'
]
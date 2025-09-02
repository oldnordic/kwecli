#!/usr/bin/env python3
"""
Advanced Agent Operations

This module provides advanced utility functions for agent operations including
timing execution, serialization, and complex agent management tasks.
"""

import time
from typing import Dict, Any, TYPE_CHECKING, List, Optional

from .agent_result import AgentStatus, AgentExpertise, AgentResult

if TYPE_CHECKING:
    from .agent_base import SubAgent


async def execute_agent_with_timing(
    agent: 'SubAgent', task: str, context: Dict[str, Any]
) -> AgentResult:
    """Execute an agent task with timing and history tracking.
    
    Args:
        agent: Agent to execute
        task: Description of the task to execute
        context: Additional context information
        
    Returns:
        AgentResult containing the execution result
    """
    start_time = time.time()
    
    try:
        agent.update_status(AgentStatus.BUSY)
        result = await agent.execute_task(task, context)
        
        # Add timing information
        execution_time = time.time() - start_time
        result.metadata["execution_time"] = execution_time
        result.metadata["agent"] = agent.name
        result.execution_time = execution_time
        
        # Add to work history
        agent.add_work_history(task, result)
        
        # Update status based on result
        if result.success:
            agent.update_status(AgentStatus.COMPLETED)
        else:
            agent.update_status(AgentStatus.FAILED)
        
        return result
        
    except Exception as e:
        # Handle errors gracefully
        execution_time = time.time() - start_time
        error_result = AgentResult(
            success=False,
            output="",
            error_message=str(e),
            quality_score=0,
            metadata={
                "execution_time": execution_time,
                "agent": agent.name,
                "error_type": type(e).__name__
            },
            execution_time=execution_time
        )
        
        agent.add_work_history(task, error_result)
        agent.update_status(AgentStatus.ERROR)
        
        return error_result
        
    finally:
        if agent.status == AgentStatus.BUSY:
            agent.update_status(AgentStatus.IDLE)


def agent_to_dict(agent: 'SubAgent') -> Dict[str, Any]:
    """Convert an agent to a dictionary for serialization.
    
    Args:
        agent: Agent to serialize
        
    Returns:
        Dictionary representation of the agent
    """
    return {
        "name": agent.name,
        "expertise": [exp.value for exp in agent.expertise],
        "tools": agent.tools,
        "description": agent.description,
        "status": agent.status.value,
        "work_history": [
            {
                "task": entry["task"],
                "timestamp": entry["timestamp"],
                "result": {
                    "success": entry["result"].success,
                    "output": entry["result"].output,
                    "metadata": entry["result"].metadata,
                    "error_message": entry["result"].error_message,
                    "quality_score": entry["result"].quality_score,
                    "recommendations": entry["result"].recommendations
                }
            }
            for entry in agent.work_history
        ]
    }


def agent_from_dict(cls, data: Dict[str, Any]) -> 'SubAgent':
    """Create an agent from a dictionary.
    
    Args:
        cls: Agent class to instantiate
        data: Dictionary containing agent data
        
    Returns:
        Agent instance
    """
    # This is a base implementation - subclasses should override
    # to properly reconstruct their specific state
    expertise = [AgentExpertise(exp) for exp in data["expertise"]]
    agent = cls(
        name=data["name"],
        expertise=expertise,
        tools=data["tools"],
        description=data.get("description", "")
    )
    
    agent.status = AgentStatus(data["status"])
    
    # Restore work history
    for entry in data.get("work_history", []):
        result = AgentResult(
            success=entry["result"]["success"],
            output=entry["result"]["output"],
            metadata=entry["result"]["metadata"],
            error_message=entry["result"]["error_message"],
            quality_score=entry["result"]["quality_score"],
            recommendations=entry["result"]["recommendations"]
        )
        agent.add_work_history(entry["task"], result)
    
    return agent


def create_agent_pool(agents: List['SubAgent']) -> Dict[str, 'SubAgent']:
    """Create a pool of agents indexed by name for quick lookup.
    
    Args:
        agents: List of agents
        
    Returns:
        Dictionary mapping agent names to agent instances
    """
    return {agent.name: agent for agent in agents}


def get_agents_by_expertise(
    agents: List['SubAgent'], 
    expertise: AgentExpertise
) -> List['SubAgent']:
    """Get all agents that have a specific expertise.
    
    Args:
        agents: List of agents to filter
        expertise: Expertise to filter by
        
    Returns:
        List of agents with the specified expertise
    """
    return [
        agent for agent in agents 
        if expertise in agent.get_expertise()
    ]


def get_most_experienced_agent(
    agents: List['SubAgent']
) -> Optional['SubAgent']:
    """Get the agent with the most work history.
    
    Args:
        agents: List of agents to evaluate
        
    Returns:
        Agent with most experience, or None if no agents
    """
    if not agents:
        return None
    
    return max(agents, key=lambda agent: len(agent.work_history))


def get_highest_performing_agent(
    agents: List['SubAgent']
) -> Optional['SubAgent']:
    """Get the agent with the highest success rate.
    
    Args:
        agents: List of agents to evaluate
        
    Returns:
        Highest performing agent, or None if no agents
    """
    if not agents:
        return None
    
    agents_with_history = [
        agent for agent in agents 
        if agent.work_history
    ]
    
    if not agents_with_history:
        return agents[0]  # Return first agent if none have history
    
    return max(agents_with_history, key=lambda agent: agent.get_success_rate())


def balance_agent_workload(agents: List['SubAgent']) -> List['SubAgent']:
    """Sort agents by their current workload for balanced task distribution.
    
    Args:
        agents: List of agents to sort
        
    Returns:
        List of agents sorted by workload (least busy first)
    """
    from .agent_utils import get_agent_load_factor
    
    return sorted(agents, key=lambda agent: get_agent_load_factor(agent))


def reset_all_agent_status(agents: List['SubAgent']) -> None:
    """Reset all agents to IDLE status.
    
    Args:
        agents: List of agents to reset
    """
    for agent in agents:
        agent.update_status(AgentStatus.IDLE)


def cleanup_agent_histories(
    agents: List['SubAgent'], 
    max_history_size: int = 100
) -> None:
    """Clean up agent work histories to prevent memory bloat.
    
    Args:
        agents: List of agents to clean up
        max_history_size: Maximum number of history entries to keep
    """
    for agent in agents:
        if len(agent.work_history) > max_history_size:
            # Keep only the most recent entries
            agent.work_history = agent.work_history[-max_history_size:]
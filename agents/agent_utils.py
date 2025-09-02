#!/usr/bin/env python3
"""
Agent Utility Functions

This module provides utility functions for agent operations including
availability checking, expertise matching algorithms, and advanced agent operations.
"""

import time
from typing import Dict, Any, Optional, List, Union

from .agent_result import AgentStatus, AgentExpertise, AgentResult


def is_agent_available(agent: 'SubAgent') -> bool:
    """Check if an agent is available for work.
    
    Args:
        agent: Agent to check
        
    Returns:
        True if agent is available, False otherwise
    """
    return agent.get_status() in [AgentStatus.IDLE]


def is_agent_busy(agent: 'SubAgent') -> bool:
    """Check if an agent is currently busy.
    
    Args:
        agent: Agent to check
        
    Returns:
        True if agent is busy, False otherwise
    """
    return agent.get_status() == AgentStatus.BUSY


def is_agent_failed(agent: 'SubAgent') -> bool:
    """Check if an agent is in a failed state.
    
    Args:
        agent: Agent to check
        
    Returns:
        True if agent has failed, False otherwise
    """
    return agent.get_status() in [AgentStatus.FAILED, AgentStatus.ERROR]


def get_agent_expertise_match(
    agent: 'SubAgent', task: str
) -> float:
    """Calculate how well an agent's expertise matches a task.
    
    Args:
        agent: Agent to evaluate
        task: Task description
        
    Returns:
        Match score between 0.0 and 1.0
    """
    # Simple keyword matching - can be enhanced with NLP
    task_lower = task.lower()
    expertise_keywords = {
        AgentExpertise.AI_ML: ["ai", "ml", "machine learning", "neural", "model"],
        AgentExpertise.BACKEND_ARCHITECTURE: [
            "backend", "api", "database", "server"
        ],
        AgentExpertise.FRONTEND_DEVELOPMENT: [
            "frontend", "ui", "ux", "react", "vue"
        ],
        AgentExpertise.DEVOPS: [
            "devops", "deployment", "ci/cd", "docker", "kubernetes"
        ],
        AgentExpertise.MOBILE_DEVELOPMENT: [
            "mobile", "ios", "android", "app"
        ],
        AgentExpertise.TESTING: ["test", "testing", "qa", "quality"],
        AgentExpertise.UX_RESEARCH: [
            "ux", "user experience", "research"
        ],
        AgentExpertise.UI_DESIGN: [
            "ui", "design", "visual", "interface"
        ],
        AgentExpertise.PRODUCT_MANAGEMENT: [
            "product", "feature", "roadmap"
        ],
        AgentExpertise.MARKETING: [
            "marketing", "growth", "acquisition"
        ],
        AgentExpertise.CONTENT_CREATION: [
            "content", "copy", "writing"
        ],
        AgentExpertise.ANALYTICS: [
            "analytics", "data", "metrics", "reporting"
        ],
        AgentExpertise.INFRASTRUCTURE: [
            "infrastructure", "system", "monitoring"
        ],
        AgentExpertise.SUPPORT: [
            "support", "help", "customer", "issue"
        ]
    }
    
    match_score = 0.0
    total_keywords = 0
    
    for expertise in agent.get_expertise():
        if expertise in expertise_keywords:
            keywords = expertise_keywords[expertise]
            total_keywords += len(keywords)
            
            for keyword in keywords:
                if keyword in task_lower:
                    match_score += 1.0
    
    if total_keywords == 0:
        return 0.0
    
    return match_score / total_keywords


def get_agent_load_factor(agent: 'SubAgent') -> float:
    """Calculate the current load factor of an agent based on work history.
    
    Args:
        agent: Agent to evaluate
        
    Returns:
        Load factor between 0.0 (no load) and 1.0 (maximum load)
    """
    # Simple implementation based on status and recent work
    status = agent.get_status()
    
    if status == AgentStatus.BUSY:
        return 1.0
    elif status in [AgentStatus.ERROR, AgentStatus.FAILED]:
        return 0.8  # High load due to error state
    elif status == AgentStatus.OFFLINE:
        return 1.0  # Cannot take work
    else:  # IDLE or COMPLETED
        # Check recent work frequency
        recent_work = agent.get_recent_work(limit=5)
        if len(recent_work) >= 3:
            return 0.3  # Some recent activity
        elif len(recent_work) >= 1:
            return 0.1  # Light recent activity
        else:
            return 0.0  # No recent activity


def calculate_agent_score(agent: 'SubAgent', task: str) -> float:
    """Calculate a comprehensive score for how suitable an agent is for a task.
    
    Combines expertise matching, availability, and performance history.
    
    Args:
        agent: Agent to evaluate
        task: Task description
        
    Returns:
        Overall suitability score between 0.0 and 1.0
    """
    # Base score from expertise matching
    expertise_score = get_agent_expertise_match(agent, task)
    
    # Availability factor
    if not is_agent_available(agent):
        availability_factor = 0.0
    else:
        load_factor = get_agent_load_factor(agent)
        availability_factor = 1.0 - load_factor
    
    # Performance factor from success rate
    success_rate = agent.get_success_rate()
    performance_factor = success_rate if success_rate > 0 else 0.5  # Neutral for new agents
    
    # Weighted combination
    # Expertise is most important, then availability, then past performance
    score = (
        expertise_score * 0.5 +
        availability_factor * 0.3 +
        performance_factor * 0.2
    )
    
    return min(1.0, max(0.0, score))


def get_agent_status_summary(agents: List['SubAgent']) -> Dict[str, int]:
    """Get a summary of agent statuses.
    
    Args:
        agents: List of agents to analyze
        
    Returns:
        Dictionary with status counts
    """
    status_counts = {}
    
    for status in AgentStatus:
        status_counts[status.value] = 0
    
    for agent in agents:
        status = agent.get_status()
        status_counts[status.value] += 1
    
    return status_counts


def filter_available_agents(agents: List['SubAgent']) -> List['SubAgent']:
    """Filter agents to only those that are available for work.
    
    Args:
        agents: List of agents to filter
        
    Returns:
        List of available agents
    """
    return [agent for agent in agents if is_agent_available(agent)]


def find_best_agent_for_task(agents: List['SubAgent'], task: str) -> Optional['SubAgent']:
    """Find the best available agent for a given task.
    
    Args:
        agents: List of agents to consider
        task: Task description
        
    Returns:
        Best matching agent, or None if no suitable agent found
    """
    available_agents = filter_available_agents(agents)
    
    if not available_agents:
        return None
    
    # Score all available agents
    scored_agents = [
        (agent, calculate_agent_score(agent, task))
        for agent in available_agents
    ]
    
    # Sort by score (highest first)
    scored_agents.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best agent if it has a reasonable score
    best_agent, best_score = scored_agents[0]
    if best_score > 0.1:  # Minimum threshold
        return best_agent
    
    return None


# Forward reference resolution - import SubAgent at the end to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent_base import SubAgent
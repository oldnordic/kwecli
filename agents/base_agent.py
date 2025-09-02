#!/usr/bin/env python3
"""
Base SubAgent Infrastructure - Main Module

This module provides the foundational classes for the sub-agent orchestration system.
All sub-agents must inherit from the SubAgent abstract class and implement its methods.

This is the main entry point that imports from the modularized components for 
backward compatibility while maintaining clean separation of concerns.
"""

# Import all components from the modularized structure
from .agent_result import (
    AgentResult,
    AgentStatus, 
    AgentExpertise,
    create_agent_result,
    create_success_result,
    create_error_result
)

from .agent_base import SubAgent

# Backward compatibility alias for tests and legacy code
BaseAgent = SubAgent

from .agent_utils import (
    is_agent_available,
    is_agent_busy,
    is_agent_failed,
    get_agent_expertise_match,
    get_agent_load_factor,
    calculate_agent_score,
    get_agent_status_summary,
    filter_available_agents,
    find_best_agent_for_task
)

from .agent_advanced import (
    execute_agent_with_timing,
    agent_to_dict,
    agent_from_dict,
    create_agent_pool,
    get_agents_by_expertise,
    get_most_experienced_agent,
    get_highest_performing_agent,
    balance_agent_workload,
    reset_all_agent_status,
    cleanup_agent_histories
)

# Re-export all components for backward compatibility
__all__ = [
    # Core classes
    "SubAgent",
    "BaseAgent",  # Backward compatibility alias
    "AgentResult",
    
    # Enums
    "AgentStatus",
    "AgentExpertise",
    
    # Result creation functions
    "create_agent_result",
    "create_success_result", 
    "create_error_result",
    
    # Utility functions
    "is_agent_available",
    "is_agent_busy",
    "is_agent_failed",
    "get_agent_expertise_match",
    "get_agent_load_factor",
    "calculate_agent_score",
    "get_agent_status_summary",
    "filter_available_agents",
    "find_best_agent_for_task",
    
    # Advanced functions
    "execute_agent_with_timing",
    "agent_to_dict",
    "agent_from_dict",
    "create_agent_pool",
    "get_agents_by_expertise",
    "get_most_experienced_agent",
    "get_highest_performing_agent",
    "balance_agent_workload",
    "reset_all_agent_status",
    "cleanup_agent_histories"
]

# Maintain backward compatibility - all original functionality is available
# through imports from the modularized structure
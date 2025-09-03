#!/usr/bin/env python3
"""
KWECLI Planner Module
====================

Planning agent for goal-to-task graph conversion with LTMC integration.

Core Components:
- PlannerAgent: Main planning orchestrator
- Goal parsing and task graph generation
- Neo4j DAG storage and Redis runtime state
- Progress tracking and execution coordination

Usage:
    from planner.agent import PlannerAgent
    
    planner = PlannerAgent(project_path=".")
    goal_data = planner.parse_goal("Build a REST API with FastAPI")
    graph_data = planner.create_task_graph(goal_data)
    execution_result = planner.execute_plan(goal_data["id"], dry_run=True)
"""

from .agent import PlannerAgent

__all__ = ["PlannerAgent"]
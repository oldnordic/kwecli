#!/usr/bin/env python3
"""
KWECLI Planner Agent - Goal → Plan → Task Graph
===============================================

Main orchestrator for autonomous planning using modular components.
Uses Neo4j for DAG storage, Redis for runtime state, and LTMC for memory integration.

Core Functions:
- parse_goal: Convert natural language goal to structured plan
- create_task_graph: Build DAG of tasks in Neo4j
- execute_plan: Coordinate task execution
- track_progress: Monitor plan completion

File: planner/agent.py
Purpose: Planning orchestration with modular LTMC integration
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bridge.ltmc_native import get_ltmc_native, save_thought, log_artifact
from .goal_parser import GoalParser
from .task_generator import TaskGenerator

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Autonomous planning agent with modular LTMC integration."""
    
    def __init__(self, project_path: str = "."):
        """Initialize planner agent with modular components."""
        self.project_path = Path(project_path).resolve()
        self.ltmc = get_ltmc_native()
        self.goal_parser = GoalParser()
        self.task_generator = TaskGenerator()
        
        # Ensure LTMC health
        health = self.ltmc.health_check()
        if not health.get("healthy", False):
            logger.warning(f"LTMC not fully healthy: {health}")
        
        logger.info(f"PlannerAgent initialized for project: {self.project_path}")
    
    def parse_goal(self, goal_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse natural language goal into structured plan components.
        
        Args:
            goal_text: Natural language description of goal
            context: Additional context (project type, constraints, etc.)
        
        Returns:
            Structured goal with objectives, constraints, and success criteria
        """
        # Add project context
        if context is None:
            context = {}
        context["project_path"] = str(self.project_path)
        
        # Use modular goal parser
        goal_data = self.goal_parser.parse_goal(goal_text, context)
        
        # Save goal to LTMC
        save_result = save_thought(
            kind="goal",
            content=f"Goal: {goal_text}",
            metadata=goal_data
        )
        
        if save_result.get("success"):
            logger.info(f"Goal parsed and saved: {goal_data['id']}")
        else:
            logger.error(f"Failed to save goal: {save_result}")
        
        return goal_data
    
    def create_task_graph(self, goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create task dependency graph in Neo4j from parsed goal.
        
        Args:
            goal_data: Structured goal from parse_goal
        
        Returns:
            Task graph with nodes and relationships
        """
        goal_id = goal_data["id"]
        
        # Generate tasks using modular task generator
        tasks = self.task_generator.generate_tasks(goal_data)
        dependencies = self.task_generator.build_dependencies(tasks, goal_data)
        
        # Create task graph structure
        graph_data = {
            "goal_id": goal_id,
            "tasks": tasks,
            "dependencies": dependencies,
            "created_at": datetime.now().isoformat(),
            "status": "planned",
            "total_estimated_time": self._calculate_total_time(tasks)
        }
        
        # Save task graph to Neo4j via LTMC operations
        try:
            self._save_graph_to_neo4j(goal_data, tasks, dependencies)
            logger.info(f"Task graph created with {len(tasks)} tasks")
            
        except Exception as e:
            logger.error(f"Failed to create task graph in Neo4j: {e}")
            graph_data["error"] = str(e)
        
        # Save plan summary to LTMC
        save_thought(
            kind="plan",
            content=f"Task graph for goal: {goal_data['text']} ({len(tasks)} tasks)",
            metadata=graph_data
        )
        
        return graph_data
    
    def execute_plan(self, goal_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """Execute task graph plan with progress tracking.
        
        Args:
            goal_id: Goal ID to execute
            dry_run: If True, only simulate execution
        
        Returns:
            Execution results and progress
        """
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        execution_data = {
            "execution_id": execution_id,
            "goal_id": goal_id,
            "started_at": datetime.now().isoformat(),
            "dry_run": dry_run,
            "status": "executing",
            "completed_tasks": [],
            "failed_tasks": [],
            "current_task": None,
            "progress_percentage": 0.0
        }
        
        try:
            # Get task execution order from Neo4j
            tasks = self._get_execution_order(goal_id)
            
            if not tasks:
                execution_data["status"] = "no_tasks"
                execution_data["error"] = "No tasks found for goal"
                return execution_data
            
            # Execute tasks in dependency order
            for i, task in enumerate(tasks):
                task_id = task["id"]
                execution_data["current_task"] = task_id
                execution_data["progress_percentage"] = (i / len(tasks)) * 100
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would execute task: {task['description']}")
                    execution_data["completed_tasks"].append(task_id)
                else:
                    # Simulate task execution
                    logger.info(f"Simulating execution of task: {task['description']}")
                    execution_data["completed_tasks"].append(task_id)
                
                # Update task status in Neo4j
                self._update_task_status(task_id, "completed")
            
            execution_data["status"] = "completed"
            execution_data["completed_at"] = datetime.now().isoformat()
            execution_data["progress_percentage"] = 100.0
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            execution_data["status"] = "failed"
            execution_data["error"] = str(e)
            execution_data["failed_at"] = datetime.now().isoformat()
        
        # Save execution results
        save_thought(
            kind="execution",
            content=f"Plan execution for goal {goal_id}: {execution_data['status']}",
            metadata=execution_data
        )
        
        return execution_data
    
    def track_progress(self, goal_id: str) -> Dict[str, Any]:
        """Track progress of goal execution.
        
        Args:
            goal_id: Goal ID to track
        
        Returns:
            Progress summary with completion metrics
        """
        try:
            connections = self.ltmc.connections
            if not connections.neo4j_conn:
                return {"error": "Neo4j connection not available"}
            
            # Query task completion status from Neo4j
            # This would use actual Neo4j queries in production
            progress_data = {
                "goal_id": goal_id,
                "checked_at": datetime.now().isoformat(),
                "total_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "in_progress_tasks": 0,
                "completion_percentage": 0.0,
                "estimated_remaining_time": "0h",
                "status": "tracking"
            }
            
            # Placeholder for real Neo4j progress queries
            progress_data["message"] = "Progress tracking ready (awaiting task execution)"
            
            return progress_data
            
        except Exception as e:
            logger.error(f"Progress tracking failed: {e}")
            return {"error": str(e)}
    
    def _calculate_total_time(self, tasks: List[Dict[str, Any]]) -> str:
        """Calculate total estimated time for all tasks."""
        total_minutes = 0
        
        for task in tasks:
            duration = task.get("estimated_duration", "1h")
            # Parse duration string (e.g., "2h", "30m")
            if "h" in duration:
                hours = int(duration.replace("h", ""))
                total_minutes += hours * 60
            elif "m" in duration:
                minutes = int(duration.replace("m", ""))
                total_minutes += minutes
        
        hours = total_minutes // 60
        minutes = total_minutes % 60
        
        if hours > 0:
            return f"{hours}h{minutes}m" if minutes > 0 else f"{hours}h"
        else:
            return f"{minutes}m"
    
    def _save_graph_to_neo4j(self, goal_data: Dict[str, Any], tasks: List[Dict[str, Any]], dependencies: List[Dict[str, Any]]):
        """Save task graph to Neo4j database."""
        connections = self.ltmc.connections
        if not connections.neo4j_conn:
            raise Exception("Neo4j connection not available")
        
        goal_id = goal_data["id"]
        
        # Create goal node
        connections.neo4j_conn.store_document_node(
            doc_id=goal_id,
            content=goal_data["text"],
            tags=["Goal", goal_data["type"]],
            metadata=goal_data
        )
        
        # Create task nodes and link to goal
        for task in tasks:
            task_id = task["id"]
            connections.neo4j_conn.store_document_node(
                doc_id=task_id,
                content=task["description"],
                tags=["Task", task["type"]],
                metadata=task
            )
            
            # Link task to goal
            connections.neo4j_conn.create_relationship(
                source_id=goal_id,
                target_id=task_id,
                relationship_type="CONTAINS",
                properties={"order": task["order"], "priority": task["priority"]}
            )
        
        # Create task dependencies
        for dep in dependencies:
            connections.neo4j_conn.create_relationship(
                source_id=dep["prerequisite_id"],
                target_id=dep["task_id"],
                relationship_type="PRECEDES",
                properties={"dependency_type": dep["type"], "reason": dep.get("reason", "")}
            )
    
    def _get_execution_order(self, goal_id: str) -> List[Dict[str, Any]]:
        """Get tasks in execution order from Neo4j (placeholder)."""
        # In real implementation, this would query Neo4j for topologically sorted tasks
        return []
    
    def _update_task_status(self, task_id: str, status: str):
        """Update task status in Neo4j (placeholder)."""
        # In real implementation, this would update the task node in Neo4j
        pass


# CLI interface for direct testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KWECLI Planner Agent")
    parser.add_argument("command", choices=["parse", "plan", "execute", "track"], help="Command to run")
    parser.add_argument("--goal", help="Goal text for parse/plan commands")
    parser.add_argument("--goal-id", help="Goal ID for execute/track commands")
    parser.add_argument("--project", default=".", help="Project directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run execution")
    
    args = parser.parse_args()
    
    # Initialize planner
    planner = PlannerAgent(args.project)
    
    if args.command == "parse":
        if not args.goal:
            print("Error: --goal required for parse command")
            sys.exit(1)
        
        result = planner.parse_goal(args.goal)
        print(f"Goal parsed: {result['id']}")
        print(f"Type: {result['type']}, Complexity: {result['complexity']}")
        print(f"Estimated tasks: {result['estimated_tasks']}")
        print(f"Priority: {result['priority']}")
        print(f"Keywords: {', '.join(result['keywords'])}")
    
    elif args.command == "plan":
        if not args.goal:
            print("Error: --goal required for plan command")
            sys.exit(1)
        
        goal_data = planner.parse_goal(args.goal)
        graph_data = planner.create_task_graph(goal_data)
        print(f"Task graph created for goal: {goal_data['id']}")
        print(f"Tasks: {len(graph_data['tasks'])}")
        print(f"Dependencies: {len(graph_data['dependencies'])}")
        print(f"Total estimated time: {graph_data['total_estimated_time']}")
        
        print("\nTask breakdown:")
        for task in graph_data['tasks']:
            deps = f" (depends on {len(task['dependencies'])} tasks)" if task['dependencies'] else ""
            print(f"  {task['order']}. {task['description']} ({task['priority']}, {task['estimated_duration']}){deps}")
    
    elif args.command == "execute":
        if not args.goal_id:
            print("Error: --goal-id required for execute command")
            sys.exit(1)
        
        result = planner.execute_plan(args.goal_id, args.dry_run)
        print(f"Execution {result['status']}: {result['execution_id']}")
        if result.get("error"):
            print(f"Error: {result['error']}")
        elif result.get("completed_tasks"):
            print(f"Completed {len(result['completed_tasks'])} tasks")
    
    elif args.command == "track":
        if not args.goal_id:
            print("Error: --goal-id required for track command")
            sys.exit(1)
        
        progress = planner.track_progress(args.goal_id)
        if progress.get("error"):
            print(f"Error: {progress['error']}")
        else:
            print(f"Progress for {args.goal_id}: {progress.get('completion_percentage', 0):.1f}%")
            print(f"Status: {progress.get('message', 'No status available')}")
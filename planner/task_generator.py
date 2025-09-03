#!/usr/bin/env python3
"""
Task Generator - Task Breakdown and Graph Creation
=================================================

Focused module for generating task breakdowns and dependency graphs.
Creates structured task lists with dependencies and priorities.

File: planner/task_generator.py
Purpose: Task generation and dependency management
"""

import logging
import uuid
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskGenerator:
    """Focused task generation and dependency management."""
    
    def __init__(self):
        """Initialize task generator with templates."""
        self.task_templates = {
            "development": [
                {"description": "Analyze requirements for {goal}", "type": "analysis", "priority": "high", "duration": "1h"},
                {"description": "Design architecture for {goal}", "type": "design", "priority": "high", "duration": "2h"},
                {"description": "Set up development environment for {goal}", "type": "setup", "priority": "medium", "duration": "1h"},
                {"description": "Implement core functionality for {goal}", "type": "implementation", "priority": "high", "duration": "4h"},
                {"description": "Implement user interface for {goal}", "type": "implementation", "priority": "medium", "duration": "3h"},
                {"description": "Write unit tests for {goal}", "type": "testing", "priority": "high", "duration": "2h"},
                {"description": "Write integration tests for {goal}", "type": "testing", "priority": "medium", "duration": "2h"},
                {"description": "Create documentation for {goal}", "type": "documentation", "priority": "low", "duration": "1h"},
                {"description": "Deploy and validate {goal}", "type": "deployment", "priority": "high", "duration": "1h"},
                {"description": "Monitor and optimize {goal}", "type": "monitoring", "priority": "low", "duration": "2h"}
            ],
            "maintenance": [
                {"description": "Investigate issue for {goal}", "type": "investigation", "priority": "high", "duration": "1h"},
                {"description": "Reproduce problem for {goal}", "type": "investigation", "priority": "high", "duration": "30m"},
                {"description": "Identify root cause of {goal}", "type": "analysis", "priority": "high", "duration": "1h"},
                {"description": "Design fix approach for {goal}", "type": "design", "priority": "high", "duration": "30m"},
                {"description": "Implement fix for {goal}", "type": "implementation", "priority": "high", "duration": "2h"},
                {"description": "Test fix for {goal}", "type": "testing", "priority": "high", "duration": "1h"},
                {"description": "Verify resolution of {goal}", "type": "verification", "priority": "medium", "duration": "30m"},
                {"description": "Document fix for {goal}", "type": "documentation", "priority": "low", "duration": "30m"}
            ],
            "analysis": [
                {"description": "Define analysis scope for {goal}", "type": "planning", "priority": "high", "duration": "30m"},
                {"description": "Gather data for {goal}", "type": "research", "priority": "high", "duration": "2h"},
                {"description": "Process and clean data for {goal}", "type": "processing", "priority": "medium", "duration": "1h"},
                {"description": "Analyze findings for {goal}", "type": "analysis", "priority": "high", "duration": "3h"},
                {"description": "Validate analysis results for {goal}", "type": "validation", "priority": "medium", "duration": "1h"},
                {"description": "Create visualizations for {goal}", "type": "visualization", "priority": "low", "duration": "2h"},
                {"description": "Create report for {goal}", "type": "documentation", "priority": "medium", "duration": "1h"},
                {"description": "Present findings for {goal}", "type": "presentation", "priority": "low", "duration": "1h"}
            ],
            "improvement": [
                {"description": "Assess current state for {goal}", "type": "assessment", "priority": "high", "duration": "2h"},
                {"description": "Identify improvement opportunities for {goal}", "type": "analysis", "priority": "high", "duration": "1h"},
                {"description": "Plan improvement approach for {goal}", "type": "planning", "priority": "high", "duration": "1h"},
                {"description": "Backup current implementation for {goal}", "type": "backup", "priority": "high", "duration": "30m"},
                {"description": "Implement improvements for {goal}", "type": "implementation", "priority": "high", "duration": "4h"},
                {"description": "Validate improvements for {goal}", "type": "validation", "priority": "high", "duration": "2h"},
                {"description": "Performance test improvements for {goal}", "type": "testing", "priority": "medium", "duration": "1h"},
                {"description": "Document improvements for {goal}", "type": "documentation", "priority": "low", "duration": "1h"},
                {"description": "Monitor improvement impact for {goal}", "type": "monitoring", "priority": "low", "duration": "2h"}
            ]
        }
    
    def generate_tasks(self, goal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate task breakdown from goal data.
        
        Args:
            goal_data: Parsed goal data with type, complexity, etc.
        
        Returns:
            List of task dictionaries with metadata
        """
        tasks = []
        goal_type = goal_data["type"]
        estimated_count = goal_data["estimated_tasks"]
        goal_text = goal_data["text"]
        
        # Get appropriate templates for goal type
        templates = self.task_templates.get(goal_type, self.task_templates["development"])
        
        # Select tasks based on estimated count and goal specifics
        selected_templates = self._select_templates(templates, estimated_count, goal_data)
        
        # Generate actual tasks from templates
        for i, template in enumerate(selected_templates):
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            task = {
                "id": task_id,
                "description": template["description"].format(goal=goal_text),
                "type": template["type"],
                "order": i + 1,
                "priority": template["priority"],
                "estimated_duration": template["duration"],
                "status": "pending",
                "goal_id": goal_data["id"],
                "created_at": datetime.now().isoformat(),
                "dependencies": [],  # Will be filled by build_dependencies
                "keywords": goal_data.get("keywords", [])
            }
            
            tasks.append(task)
        
        return tasks
    
    def build_dependencies(self, tasks: List[Dict[str, Any]], goal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build task dependency relationships based on task types and order.
        
        Args:
            tasks: List of task dictionaries
            goal_data: Original goal data for context
        
        Returns:
            List of dependency relationships
        """
        dependencies = []
        goal_type = goal_data["type"]
        
        # Build dependencies based on goal type patterns
        if goal_type == "development":
            dependencies.extend(self._build_development_dependencies(tasks))
        elif goal_type == "maintenance":
            dependencies.extend(self._build_maintenance_dependencies(tasks))
        elif goal_type == "analysis":
            dependencies.extend(self._build_analysis_dependencies(tasks))
        elif goal_type == "improvement":
            dependencies.extend(self._build_improvement_dependencies(tasks))
        else:
            # Default sequential dependencies
            dependencies.extend(self._build_sequential_dependencies(tasks))
        
        # Add dependencies to task objects
        for dep in dependencies:
            for task in tasks:
                if task["id"] == dep["task_id"]:
                    task["dependencies"].append(dep["prerequisite_id"])
        
        return dependencies
    
    def _select_templates(self, templates: List[Dict[str, Any]], count: int, goal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select appropriate templates based on goal characteristics."""
        complexity = goal_data["complexity"]
        keywords = goal_data.get("keywords", [])
        
        # Priority order for template selection
        priority_order = ["high", "medium", "low"]
        
        # Start with high priority tasks
        selected = []
        for priority in priority_order:
            for template in templates:
                if len(selected) >= count:
                    break
                if template["priority"] == priority:
                    # Check if template is relevant to goal keywords
                    if self._is_template_relevant(template, keywords) or len(selected) < count // 2:
                        selected.append(template)
        
        # Ensure minimum tasks for complexity
        min_tasks = {"low": 2, "medium": 3, "high": 4}
        if len(selected) < min_tasks.get(complexity, 2):
            # Add more templates if needed
            for template in templates:
                if len(selected) >= count:
                    break
                if template not in selected:
                    selected.append(template)
        
        return selected[:count]
    
    def _is_template_relevant(self, template: Dict[str, Any], keywords: List[str]) -> bool:
        """Check if template is relevant to goal keywords."""
        template_text = template["description"].lower() + " " + template["type"].lower()
        
        relevance_keywords = {
            "api": ["implementation", "testing"],
            "database": ["design", "analysis", "implementation"],
            "frontend": ["implementation", "testing", "design"],
            "backend": ["implementation", "testing", "design"],
            "security": ["analysis", "testing", "implementation"],
            "performance": ["analysis", "testing", "monitoring"]
        }
        
        for keyword in keywords:
            if keyword.lower() in relevance_keywords:
                relevant_types = relevance_keywords[keyword.lower()]
                if any(rt in template_text for rt in relevant_types):
                    return True
        
        return False
    
    def _build_development_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build dependencies for development projects."""
        dependencies = []
        task_map = {task["type"]: task for task in tasks}
        
        # Standard development workflow dependencies
        workflows = [
            ("analysis", "design"),
            ("design", "setup"),
            ("setup", "implementation"),
            ("implementation", "testing"),
            ("testing", "documentation"),
            ("testing", "deployment"),
            ("deployment", "monitoring")
        ]
        
        for prereq_type, task_type in workflows:
            prereq_task = task_map.get(prereq_type)
            dependent_task = task_map.get(task_type)
            
            if prereq_task and dependent_task:
                dependencies.append({
                    "prerequisite_id": prereq_task["id"],
                    "task_id": dependent_task["id"],
                    "type": "workflow",
                    "reason": f"{task_type} depends on {prereq_type}"
                })
        
        return dependencies
    
    def _build_maintenance_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build dependencies for maintenance projects."""
        dependencies = []
        task_map = {task["type"]: task for task in tasks}
        
        # Maintenance workflow
        workflows = [
            ("investigation", "analysis"),
            ("analysis", "design"),
            ("design", "implementation"),
            ("implementation", "testing"),
            ("testing", "verification"),
            ("verification", "documentation")
        ]
        
        for prereq_type, task_type in workflows:
            prereq_task = task_map.get(prereq_type)
            dependent_task = task_map.get(task_type)
            
            if prereq_task and dependent_task:
                dependencies.append({
                    "prerequisite_id": prereq_task["id"],
                    "task_id": dependent_task["id"],
                    "type": "workflow",
                    "reason": f"{task_type} depends on {prereq_type}"
                })
        
        return dependencies
    
    def _build_analysis_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build dependencies for analysis projects."""
        dependencies = []
        task_map = {task["type"]: task for task in tasks}
        
        # Analysis workflow
        workflows = [
            ("planning", "research"),
            ("research", "processing"),
            ("processing", "analysis"),
            ("analysis", "validation"),
            ("validation", "visualization"),
            ("validation", "documentation"),
            ("documentation", "presentation")
        ]
        
        for prereq_type, task_type in workflows:
            prereq_task = task_map.get(prereq_type)
            dependent_task = task_map.get(task_type)
            
            if prereq_task and dependent_task:
                dependencies.append({
                    "prerequisite_id": prereq_task["id"],
                    "task_id": dependent_task["id"],
                    "type": "workflow",
                    "reason": f"{task_type} depends on {prereq_type}"
                })
        
        return dependencies
    
    def _build_improvement_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build dependencies for improvement projects."""
        dependencies = []
        task_map = {task["type"]: task for task in tasks}
        
        # Improvement workflow
        workflows = [
            ("assessment", "analysis"),
            ("analysis", "planning"),
            ("planning", "backup"),
            ("backup", "implementation"),
            ("implementation", "validation"),
            ("validation", "testing"),
            ("testing", "documentation"),
            ("documentation", "monitoring")
        ]
        
        for prereq_type, task_type in workflows:
            prereq_task = task_map.get(prereq_type)
            dependent_task = task_map.get(task_type)
            
            if prereq_task and dependent_task:
                dependencies.append({
                    "prerequisite_id": prereq_task["id"],
                    "task_id": dependent_task["id"],
                    "type": "workflow",
                    "reason": f"{task_type} depends on {prereq_type}"
                })
        
        return dependencies
    
    def _build_sequential_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build simple sequential dependencies."""
        dependencies = []
        
        for i in range(1, len(tasks)):
            dependencies.append({
                "prerequisite_id": tasks[i-1]["id"],
                "task_id": tasks[i]["id"],
                "type": "sequential",
                "reason": "Sequential task order"
            })
        
        return dependencies


# Test functionality if run directly
if __name__ == "__main__":
    print("🧪 Testing TaskGenerator...")
    
    generator = TaskGenerator()
    
    # Test goal data
    test_goal = {
        "id": "goal_test123",
        "text": "Create a REST API with FastAPI for user management",
        "type": "development",
        "complexity": "medium",
        "estimated_tasks": 5,
        "keywords": ["api", "fastapi", "user", "management"]
    }
    
    # Generate tasks
    tasks = generator.generate_tasks(test_goal)
    print(f"Generated {len(tasks)} tasks:")
    for task in tasks:
        print(f"  {task['order']}. {task['description']} ({task['priority']}, {task['estimated_duration']})")
    
    # Build dependencies
    dependencies = generator.build_dependencies(tasks, test_goal)
    print(f"\nCreated {len(dependencies)} dependencies:")
    for dep in dependencies:
        prereq = next(t for t in tasks if t["id"] == dep["prerequisite_id"])
        task = next(t for t in tasks if t["id"] == dep["task_id"])
        print(f"  {prereq['description']} -> {task['description']}")
    
    print("\n✅ TaskGenerator test complete")
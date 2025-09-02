#!/usr/bin/env python3
"""
Studio Producer Agent Utils Module

This module contains utility and helper methods for the StudioProducer agent.
These methods handle formatting, data processing, and generation tasks.
"""

from typing import Dict, List, Any


class StudioProducerUtils:
    """Utility methods for Studio Producer agent."""

    def format_resource_constraints(self, constraints: Dict[str, Any]) -> str:
        """Format resource constraints for output."""
        if not constraints:
            return "- No specific resource constraints identified"
        
        formatted = []
        for resource, count in constraints.items():
            formatted.append(f"- **{resource.replace('_', ' ').title()}**: {count}")
        
        return "\n".join(formatted)

    def format_project_priorities(self, priorities: List[str]) -> str:
        """Format project priorities for output."""
        if not priorities:
            return "- No specific project priorities identified"
        
        formatted = []
        for i, priority in enumerate(priorities, 1):
            formatted.append(f"- **{i}**: {priority}")
        
        return "\n".join(formatted)

    def generate_allocation_plan(self, constraints: Dict[str, Any], priorities: List[str]) -> str:
        """Generate resource allocation plan."""
        if not constraints or not priorities:
            return "- Insufficient data for detailed allocation plan"
        
        plan = []
        for i, priority in enumerate(priorities[:3], 1):  # Top 3 priorities
            plan.append(f"**Priority {i}**: {priority}")
            plan.append("  - Allocate senior resources for critical path")
            plan.append("  - Ensure adequate support resources")
            plan.append("  - Plan for 20% buffer time")
            plan.append("")
        
        return "\n".join(plan)

    def format_team_health(self, health: Dict[str, str]) -> str:
        """Format team health data for output."""
        if not health:
            return "- No specific team health data available"
        
        formatted = []
        for team, status in health.items():
            status_emoji = {
                "good": "🟢",
                "stressed": "🟡", 
                "overloaded": "🟠",
                "blocked": "🔴"
            }.get(status.lower(), "⚪")
            formatted.append(f"- **{team.title()}**: {status_emoji} {status}")
        
        return "\n".join(formatted)

    def generate_workflow_improvements(self, bottlenecks: List[str]) -> str:
        """Generate workflow improvement suggestions."""
        if not bottlenecks:
            return "- No specific bottlenecks identified for improvement"
        
        improvements = []
        for bottleneck in bottlenecks:
            improvements.append(f"**{bottleneck.replace('_', ' ').title()}**")
            improvements.append("  - Analyze root causes")
            improvements.append("  - Implement process improvements")
            improvements.append("  - Add automation where possible")
            improvements.append("  - Monitor improvement effectiveness")
            improvements.append("")
        
        return "\n".join(improvements)

    def generate_process_improvements(self, processes: List[str]) -> str:
        """Generate process improvement suggestions."""
        if not processes:
            return "- Comprehensive process improvement across all areas"
        
        improvements = []
        for process in processes:
            improvements.append(f"**{process.replace('_', ' ').title()}**")
            improvements.append("  - Streamline workflow steps")
            improvements.append("  - Automate repetitive tasks")
            improvements.append("  - Standardize procedures")
            improvements.append("  - Improve quality controls")
            improvements.append("")
        
        return "\n".join(improvements)
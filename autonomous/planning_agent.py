#!/usr/bin/env python3
"""
KWE CLI Planning Agent

This agent specializes in project planning, requirements analysis, architecture 
design, and task decomposition. It uses sequential reasoning to break down complex
projects into executable tasks and coordinates with other agents.

Key Capabilities:
- Requirements analysis and validation
- Architecture design and planning
- Task decomposition and dependency mapping
- Resource estimation and timeline planning
- Risk assessment and mitigation strategies
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from .base_agent import (
    BaseKWECLIAgent, AgentTask, AgentResult, AgentRole, 
    AgentCapability, TaskPriority, TaskStatus
)
from .sequential_thinking import ReasoningResult, ReasoningType

# Configure logging
logger = logging.getLogger(__name__)

class PlanningAgent(BaseKWECLIAgent):
    """
    Specialized agent for project planning and architecture design.
    
    Uses sequential reasoning to:
    - Analyze project requirements systematically
    - Design scalable and maintainable architectures
    - Break down complex projects into manageable tasks
    - Estimate resources and timelines accurately
    - Identify potential risks and mitigation strategies
    """
    
    def __init__(self):
        super().__init__(
            agent_id="planning_agent",
            role=AgentRole.PLANNER,
            capabilities=[
                AgentCapability.SEQUENTIAL_REASONING,
                AgentCapability.PLANNING,
                AgentCapability.RESEARCH,
                AgentCapability.COORDINATION
            ]
        )
        
        # Planning-specific configuration
        self.planning_templates = {
            "web_application": self.plan_web_application,
            "api_service": self.plan_api_service,
            "data_pipeline": self.plan_data_pipeline,
            "cli_tool": self.plan_cli_tool,
            "library": self.plan_library,
            "automation_script": self.plan_automation_script
        }
        
        self.architecture_patterns = {
            "microservices": self.design_microservices_architecture,
            "monolithic": self.design_monolithic_architecture,
            "serverless": self.design_serverless_architecture,
            "event_driven": self.design_event_driven_architecture
        }
        
        logger.info("Planning Agent initialized with comprehensive project planning capabilities")
    
    async def execute_specialized_task(self, task: AgentTask, 
                                     reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Execute planning-specific tasks."""
        try:
            task_type = self.classify_planning_task(task)
            
            if task_type == "requirements_analysis":
                return await self.analyze_requirements(task, reasoning_result)
            elif task_type == "architecture_design":
                return await self.design_architecture(task, reasoning_result)
            elif task_type == "task_decomposition":
                return await self.decompose_project(task, reasoning_result)
            elif task_type == "resource_planning":
                return await self.plan_resources(task, reasoning_result)
            elif task_type == "risk_assessment":
                return await self.assess_risks(task, reasoning_result)
            elif task_type == "project_planning":
                return await self.create_comprehensive_project_plan(task, reasoning_result)
            else:
                return await self.handle_general_planning_task(task, reasoning_result)
                
        except Exception as e:
            logger.error(f"Planning task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    def classify_planning_task(self, task: AgentTask) -> str:
        """Classify the type of planning task."""
        description = task.description.lower()
        
        if "requirements" in description or "analyze requirements" in description:
            return "requirements_analysis"
        elif "architecture" in description or "design" in description:
            return "architecture_design"
        elif "decompose" in description or "break down" in description:
            return "task_decomposition"
        elif "resource" in description or "estimate" in description:
            return "resource_planning"
        elif "risk" in description or "assess" in description:
            return "risk_assessment"
        elif "project plan" in description or "comprehensive plan" in description:
            return "project_planning"
        else:
            return "general_planning"
    
    async def analyze_requirements(self, task: AgentTask, 
                                 reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Analyze project requirements using sequential reasoning insights."""
        try:
            logger.info(f"Analyzing requirements for: {task.title}")
            
            # Extract project description from task
            project_description = task.description
            
            # Use reasoning insights to structure analysis
            requirements_analysis = {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "technical_requirements": [],
                "business_requirements": [],
                "constraints": [],
                "assumptions": []
            }
            
            # Analyze reasoning thoughts for requirement extraction
            for thought in reasoning_result.thought_sequence:
                content = thought.content.lower()
                
                # Extract functional requirements
                if "must" in content or "shall" in content or "requirement" in content:
                    requirements_analysis["functional_requirements"].append({
                        "id": f"FR-{len(requirements_analysis['functional_requirements']) + 1}",
                        "description": thought.content,
                        "priority": "high" if thought.confidence > 0.7 else "medium",
                        "source": f"reasoning_thought_{thought.thought_number}"
                    })
                
                # Extract non-functional requirements
                if any(nfr in content for nfr in ["performance", "security", "scalability", "reliability"]):
                    requirements_analysis["non_functional_requirements"].append({
                        "id": f"NFR-{len(requirements_analysis['non_functional_requirements']) + 1}",
                        "type": self.extract_nfr_type(content),
                        "description": thought.content,
                        "priority": "high" if thought.confidence > 0.8 else "medium"
                    })
                
                # Extract technical requirements
                if any(tech in content for tech in ["database", "api", "framework", "language", "technology"]):
                    requirements_analysis["technical_requirements"].append({
                        "id": f"TR-{len(requirements_analysis['technical_requirements']) + 1}",
                        "technology": self.extract_technology(content),
                        "description": thought.content,
                        "justification": f"Confidence: {thought.confidence:.2f}"
                    })
            
            # Generate comprehensive requirements document
            requirements_document = await self.generate_requirements_document(
                project_description, requirements_analysis, reasoning_result
            )
            
            # Store requirements in LTMC
            if self.ltmc_integration:
                await self.store_requirements_analysis(task, requirements_analysis, requirements_document)
            
            return {
                "success": True,
                "data": {
                    "requirements_analysis": requirements_analysis,
                    "requirements_document": requirements_document,
                    "functional_count": len(requirements_analysis["functional_requirements"]),
                    "non_functional_count": len(requirements_analysis["non_functional_requirements"]),
                    "technical_count": len(requirements_analysis["technical_requirements"])
                },
                "artifacts": [f"requirements_analysis_{task.task_id}.md"]
            }
            
        except Exception as e:
            logger.error(f"Requirements analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def design_architecture(self, task: AgentTask, 
                                reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Design system architecture based on requirements and reasoning."""
        try:
            logger.info(f"Designing architecture for: {task.title}")
            
            # Determine project type from reasoning insights
            project_type = self.determine_project_type(reasoning_result)
            
            # Select appropriate architecture pattern
            architecture_pattern = self.select_architecture_pattern(reasoning_result, project_type)
            
            # Generate architecture design
            if project_type in self.planning_templates:
                architecture_design = await self.planning_templates[project_type](
                    task, reasoning_result, architecture_pattern
                )
            else:
                architecture_design = await self.design_generic_architecture(
                    task, reasoning_result, architecture_pattern
                )
            
            # Generate architecture documentation
            architecture_document = await self.generate_architecture_document(
                project_type, architecture_pattern, architecture_design, reasoning_result
            )
            
            # Store architecture design in LTMC
            if self.ltmc_integration:
                await self.store_architecture_design(task, architecture_design, architecture_document)
            
            return {
                "success": True,
                "data": {
                    "project_type": project_type,
                    "architecture_pattern": architecture_pattern,
                    "architecture_design": architecture_design,
                    "architecture_document": architecture_document,
                    "components_count": len(architecture_design.get("components", [])),
                    "services_count": len(architecture_design.get("services", []))
                },
                "artifacts": [f"architecture_design_{task.task_id}.md"]
            }
            
        except Exception as e:
            logger.error(f"Architecture design failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def decompose_project(self, task: AgentTask, 
                              reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Decompose project into manageable tasks with dependencies."""
        try:
            logger.info(f"Decomposing project: {task.title}")
            
            # Extract project scope from reasoning
            project_scope = self.extract_project_scope(reasoning_result)
            
            # Generate task breakdown structure
            task_breakdown = {
                "phases": [],
                "tasks": [],
                "dependencies": [],
                "milestones": []
            }
            
            # Phase 1: Setup and Planning
            setup_tasks = await self.generate_setup_tasks(project_scope, reasoning_result)
            task_breakdown["phases"].append({
                "name": "Setup and Planning",
                "description": "Initial project setup and detailed planning",
                "tasks": setup_tasks,
                "estimated_duration": "1-2 weeks"
            })
            
            # Phase 2: Core Implementation
            implementation_tasks = await self.generate_implementation_tasks(project_scope, reasoning_result)
            task_breakdown["phases"].append({
                "name": "Core Implementation",
                "description": "Main development work and feature implementation",
                "tasks": implementation_tasks,
                "estimated_duration": "3-6 weeks"
            })
            
            # Phase 3: Testing and Quality Assurance
            testing_tasks = await self.generate_testing_tasks(project_scope, reasoning_result)
            task_breakdown["phases"].append({
                "name": "Testing and QA",
                "description": "Comprehensive testing and quality assurance",
                "tasks": testing_tasks,
                "estimated_duration": "1-2 weeks"
            })
            
            # Phase 4: Deployment and Documentation
            deployment_tasks = await self.generate_deployment_tasks(project_scope, reasoning_result)
            task_breakdown["phases"].append({
                "name": "Deployment and Documentation",
                "description": "Production deployment and documentation completion",
                "tasks": deployment_tasks,
                "estimated_duration": "1 week"
            })
            
            # Generate dependencies between tasks
            task_breakdown["dependencies"] = self.generate_task_dependencies(task_breakdown["phases"])
            
            # Generate milestones
            task_breakdown["milestones"] = self.generate_project_milestones(task_breakdown["phases"])
            
            # Create individual agent tasks
            agent_tasks = await self.create_agent_tasks(task_breakdown, task.task_id)
            
            # Store task decomposition in LTMC
            if self.ltmc_integration:
                await self.store_task_decomposition(task, task_breakdown, agent_tasks)
            
            return {
                "success": True,
                "data": {
                    "task_breakdown": task_breakdown,
                    "agent_tasks": agent_tasks,
                    "total_tasks": sum(len(phase["tasks"]) for phase in task_breakdown["phases"]),
                    "total_phases": len(task_breakdown["phases"]),
                    "estimated_timeline": self.calculate_total_timeline(task_breakdown["phases"])
                },
                "artifacts": [f"task_decomposition_{task.task_id}.md"]
            }
            
        except Exception as e:
            logger.error(f"Project decomposition failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_comprehensive_project_plan(self, task: AgentTask, 
                                              reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Create a comprehensive project plan combining all planning activities."""
        try:
            logger.info(f"Creating comprehensive project plan for: {task.title}")
            
            # Execute all planning phases in sequence
            requirements_result = await self.analyze_requirements(task, reasoning_result)
            architecture_result = await self.design_architecture(task, reasoning_result)
            decomposition_result = await self.decompose_project(task, reasoning_result)
            resource_result = await self.plan_resources(task, reasoning_result)
            risk_result = await self.assess_risks(task, reasoning_result)
            
            # Combine all results into comprehensive plan
            comprehensive_plan = {
                "project_overview": {
                    "title": task.title,
                    "description": task.description,
                    "created_at": datetime.now().isoformat(),
                    "planning_agent": self.agent_id
                },
                "requirements": requirements_result.get("data", {}),
                "architecture": architecture_result.get("data", {}),
                "task_breakdown": decomposition_result.get("data", {}),
                "resources": resource_result.get("data", {}),
                "risks": risk_result.get("data", {}),
                "timeline": self.generate_master_timeline(decomposition_result.get("data", {})),
                "success_metrics": self.define_success_metrics(requirements_result.get("data", {}))
            }
            
            # Generate master project document
            master_document = await self.generate_master_project_document(comprehensive_plan)
            
            # Store comprehensive plan in LTMC
            if self.ltmc_integration:
                await self.store_comprehensive_plan(task, comprehensive_plan, master_document)
            
            return {
                "success": True,
                "data": {
                    "comprehensive_plan": comprehensive_plan,
                    "master_document": master_document,
                    "planning_completeness": self.calculate_planning_completeness(comprehensive_plan)
                },
                "artifacts": [f"comprehensive_project_plan_{task.task_id}.md"]
            }
            
        except Exception as e:
            logger.error(f"Comprehensive planning failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Helper methods for specific planning functions
    
    def extract_nfr_type(self, content: str) -> str:
        """Extract non-functional requirement type from content."""
        if "performance" in content:
            return "performance"
        elif "security" in content:
            return "security"
        elif "scalability" in content:
            return "scalability"
        elif "reliability" in content:
            return "reliability"
        elif "usability" in content:
            return "usability"
        else:
            return "general"
    
    def extract_technology(self, content: str) -> str:
        """Extract technology from content."""
        technologies = ["python", "javascript", "react", "node.js", "postgresql", "redis", "docker"]
        for tech in technologies:
            if tech in content:
                return tech
        return "unspecified"
    
    def determine_project_type(self, reasoning_result: ReasoningResult) -> str:
        """Determine project type from reasoning insights."""
        content = " ".join([thought.content for thought in reasoning_result.thought_sequence]).lower()
        
        if "web application" in content or "webapp" in content:
            return "web_application"
        elif "api" in content or "service" in content:
            return "api_service"
        elif "data pipeline" in content or "etl" in content:
            return "data_pipeline"
        elif "cli" in content or "command line" in content:
            return "cli_tool"
        elif "library" in content or "package" in content:
            return "library"
        elif "automation" in content or "script" in content:
            return "automation_script"
        else:
            return "general_application"
    
    def select_architecture_pattern(self, reasoning_result: ReasoningResult, project_type: str) -> str:
        """Select appropriate architecture pattern."""
        content = " ".join([thought.content for thought in reasoning_result.thought_sequence]).lower()
        
        if "microservices" in content:
            return "microservices"
        elif "serverless" in content:
            return "serverless"
        elif "event driven" in content:
            return "event_driven"
        else:
            return "monolithic"
    
    async def plan_web_application(self, task: AgentTask, reasoning_result: ReasoningResult, 
                                 architecture_pattern: str) -> Dict[str, Any]:
        """Plan web application architecture."""
        return {
            "type": "web_application",
            "pattern": architecture_pattern,
            "components": [
                {"name": "frontend", "type": "react_spa", "responsibilities": ["user_interface", "client_state"]},
                {"name": "backend", "type": "rest_api", "responsibilities": ["business_logic", "data_processing"]},
                {"name": "database", "type": "postgresql", "responsibilities": ["data_persistence", "transactions"]},
                {"name": "cache", "type": "redis", "responsibilities": ["session_storage", "performance_cache"]}
            ],
            "services": [
                {"name": "user_service", "responsibilities": ["authentication", "user_management"]},
                {"name": "api_service", "responsibilities": ["core_api", "business_logic"]},
                {"name": "notification_service", "responsibilities": ["email", "push_notifications"]}
            ],
            "technologies": {
                "frontend": ["React", "TypeScript", "Tailwind CSS"],
                "backend": ["Python", "FastAPI", "SQLAlchemy"],
                "database": ["PostgreSQL", "Redis"],
                "deployment": ["Docker", "Kubernetes", "CI/CD"]
            }
        }
    
    async def plan_api_service(self, task: AgentTask, reasoning_result: ReasoningResult, 
                             architecture_pattern: str) -> Dict[str, Any]:
        """Plan API service architecture."""
        return {
            "type": "api_service",
            "pattern": architecture_pattern,
            "components": [
                {"name": "api_gateway", "type": "fastapi", "responsibilities": ["routing", "middleware"]},
                {"name": "service_layer", "type": "business_logic", "responsibilities": ["operations", "validation"]},
                {"name": "data_layer", "type": "repository", "responsibilities": ["data_access", "persistence"]},
                {"name": "cache_layer", "type": "redis", "responsibilities": ["caching", "session_management"]}
            ],
            "endpoints": [
                {"path": "/health", "method": "GET", "purpose": "health_check"},
                {"path": "/api/v1/resources", "method": "GET", "purpose": "list_resources"},
                {"path": "/api/v1/resources", "method": "POST", "purpose": "create_resource"},
                {"path": "/api/v1/resources/{id}", "method": "GET", "purpose": "get_resource"},
                {"path": "/api/v1/resources/{id}", "method": "PUT", "purpose": "update_resource"},
                {"path": "/api/v1/resources/{id}", "method": "DELETE", "purpose": "delete_resource"}
            ],
            "technologies": {
                "framework": "FastAPI",
                "database": "PostgreSQL",
                "cache": "Redis",
                "authentication": "JWT",
                "documentation": "OpenAPI"
            }
        }
    
    # Continue with other planning template methods...
    
    async def generate_setup_tasks(self, project_scope: Dict[str, Any], 
                                 reasoning_result: ReasoningResult) -> List[Dict[str, Any]]:
        """Generate setup phase tasks."""
        return [
            {
                "id": "SETUP-001",
                "title": "Initialize project repository",
                "description": "Create Git repository and initial project structure",
                "assigned_to": "implementation_agent",
                "estimated_hours": 2,
                "priority": TaskPriority.HIGH.value
            },
            {
                "id": "SETUP-002", 
                "title": "Setup development environment",
                "description": "Configure development tools and dependencies",
                "assigned_to": "implementation_agent",
                "estimated_hours": 4,
                "priority": TaskPriority.HIGH.value
            },
            {
                "id": "SETUP-003",
                "title": "Create project documentation structure",
                "description": "Setup README, docs folder, and basic documentation",
                "assigned_to": "documentation_specialist",
                "estimated_hours": 3,
                "priority": TaskPriority.NORMAL.value
            }
        ]
    
    async def generate_implementation_tasks(self, project_scope: Dict[str, Any],
                                          reasoning_result: ReasoningResult) -> List[Dict[str, Any]]:
        """Generate implementation phase tasks."""
        return [
            {
                "id": "IMPL-001",
                "title": "Implement core data models",
                "description": "Create database schemas and data models",
                "assigned_to": "implementation_agent",
                "estimated_hours": 8,
                "priority": TaskPriority.HIGH.value
            },
            {
                "id": "IMPL-002",
                "title": "Implement business logic layer",
                "description": "Core business logic and service layer implementation",
                "assigned_to": "implementation_agent",
                "estimated_hours": 16,
                "priority": TaskPriority.HIGH.value
            },
            {
                "id": "IMPL-003",
                "title": "Implement API endpoints",
                "description": "REST API endpoints with validation and error handling",
                "assigned_to": "implementation_agent",
                "estimated_hours": 12,
                "priority": TaskPriority.HIGH.value
            }
        ]
    
    async def generate_testing_tasks(self, project_scope: Dict[str, Any],
                                   reasoning_result: ReasoningResult) -> List[Dict[str, Any]]:
        """Generate testing phase tasks.""" 
        return [
            {
                "id": "TEST-001",
                "title": "Write unit tests",
                "description": "Comprehensive unit test coverage for all modules",
                "assigned_to": "tester",
                "estimated_hours": 10,
                "priority": TaskPriority.HIGH.value
            },
            {
                "id": "TEST-002",
                "title": "Write integration tests",
                "description": "Integration tests for API endpoints and database operations",
                "assigned_to": "tester",
                "estimated_hours": 8,
                "priority": TaskPriority.HIGH.value
            },
            {
                "id": "TEST-003",
                "title": "Security testing",
                "description": "Security vulnerability assessment and penetration testing",
                "assigned_to": "security_specialist",
                "estimated_hours": 6,
                "priority": TaskPriority.HIGH.value
            }
        ]
    
    async def generate_deployment_tasks(self, project_scope: Dict[str, Any],
                                      reasoning_result: ReasoningResult) -> List[Dict[str, Any]]:
        """Generate deployment phase tasks."""
        return [
            {
                "id": "DEPLOY-001",
                "title": "Setup CI/CD pipeline",
                "description": "Configure automated testing and deployment pipeline",
                "assigned_to": "implementation_agent",
                "estimated_hours": 6,
                "priority": TaskPriority.NORMAL.value
            },
            {
                "id": "DEPLOY-002",
                "title": "Create production configuration",
                "description": "Production environment configuration and secrets management",
                "assigned_to": "implementation_agent", 
                "estimated_hours": 4,
                "priority": TaskPriority.HIGH.value
            },
            {
                "id": "DEPLOY-003",
                "title": "Complete project documentation",
                "description": "Finalize all documentation including API docs and user guides",
                "assigned_to": "documentation_specialist",
                "estimated_hours": 8,
                "priority": TaskPriority.NORMAL.value
            }
        ]
    
    # Additional helper methods would continue here...
    
    async def store_requirements_analysis(self, task: AgentTask, 
                                        requirements_analysis: Dict[str, Any],
                                        requirements_document: str):
        """Store requirements analysis in LTMC."""
        if not self.ltmc_integration:
            return
        
        try:
            doc_name = f"REQUIREMENTS_ANALYSIS_{task.task_id}.md"
            await self.ltmc_integration.store_document(
                file_name=doc_name,
                content=requirements_document,
                conversation_id="project_planning",
                resource_type="requirements_analysis"
            )
            logger.info(f"Requirements analysis stored for task {task.task_id}")
        except Exception as e:
            logger.error(f"Failed to store requirements analysis: {e}")

# Export the PlanningAgent
__all__ = ['PlanningAgent']
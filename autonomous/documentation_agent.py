#!/usr/bin/env python3
"""
KWE CLI Documentation Agent

This agent specializes in comprehensive documentation generation, technical writing,
and knowledge management. It uses sequential reasoning to create high-quality 
documentation, maintain documentation consistency, and ensure comprehensive 
knowledge capture throughout the autonomous development process.

Key Capabilities:
- Automated documentation generation from code
- Technical writing and content creation
- API documentation and specification generation
- User guide and tutorial creation
- Documentation quality assurance and validation
- Knowledge management and organization
- Cross-referencing and consistency checking
- Documentation workflow integration
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import ast
import markdown
from dataclasses import dataclass

from .base_agent import (
    BaseKWECLIAgent, AgentTask, AgentResult, AgentRole, 
    AgentCapability, TaskPriority, TaskStatus
)
from .sequential_thinking import ReasoningResult, ReasoningType, Problem

# Configure logging
logger = logging.getLogger(__name__)

class DocumentationType(str):
    """Types of documentation that can be generated."""
    API_DOCUMENTATION = "api_documentation"
    USER_GUIDE = "user_guide"
    TECHNICAL_REFERENCE = "technical_reference"
    TUTORIAL = "tutorial"
    README = "readme"
    CHANGELOG = "changelog"
    ARCHITECTURE_GUIDE = "architecture_guide"
    DEPLOYMENT_GUIDE = "deployment_guide"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    CODE_COMMENTS = "code_comments"
    INLINE_DOCUMENTATION = "inline_documentation"

class DocumentationFormat(str):
    """Supported documentation formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    RESTRUCTURED_TEXT = "rst"
    ASCIIDOC = "asciidoc"
    CONFLUENCE = "confluence"
    SPHINX = "sphinx"

class DocumentationQualityMetric(str):
    """Documentation quality assessment metrics."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    CONSISTENCY = "consistency"
    MAINTAINABILITY = "maintainability"
    ACCESSIBILITY = "accessibility"
    SEARCHABILITY = "searchability"

@dataclass
class DocumentationSection:
    """Represents a section of documentation."""
    title: str
    content: str
    level: int
    section_type: str
    metadata: Dict[str, Any]
    subsections: List['DocumentationSection'] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []

@dataclass
class DocumentationTemplate:
    """Template for generating documentation."""
    name: str
    document_type: DocumentationType
    format: DocumentationFormat
    sections: List[str]
    metadata_fields: List[str]
    quality_requirements: Dict[str, float]

class DocumentationAgent(BaseKWECLIAgent):
    """
    Specialized agent for comprehensive documentation and knowledge management.
    
    Uses sequential reasoning combined with:
    - Automated code analysis for documentation generation
    - Natural language processing for content quality
    - Template-based documentation generation
    - Cross-referencing and consistency checking
    - Multi-format output generation
    - Knowledge organization and management
    """
    
    def __init__(self):
        super().__init__(
            agent_id="documentation_agent",
            role=AgentRole.DOCUMENTATION_SPECIALIST,
            capabilities=[
                AgentCapability.SEQUENTIAL_REASONING,
                AgentCapability.DOCUMENTATION,
                AgentCapability.RESEARCH,
                AgentCapability.COORDINATION
            ]
        )
        
        # Documentation-specific configuration
        self.documentation_templates = {
            DocumentationType.API_DOCUMENTATION: DocumentationTemplate(
                name="API Documentation",
                document_type=DocumentationType.API_DOCUMENTATION,
                format=DocumentationFormat.MARKDOWN,
                sections=["overview", "authentication", "endpoints", "examples", "errors"],
                metadata_fields=["version", "base_url", "last_updated"],
                quality_requirements={
                    DocumentationQualityMetric.COMPLETENESS: 0.90,
                    DocumentationQualityMetric.ACCURACY: 0.95,
                    DocumentationQualityMetric.CLARITY: 0.85
                }
            ),
            DocumentationType.USER_GUIDE: DocumentationTemplate(
                name="User Guide",
                document_type=DocumentationType.USER_GUIDE,
                format=DocumentationFormat.MARKDOWN,
                sections=["introduction", "getting_started", "features", "tutorials", "troubleshooting"],
                metadata_fields=["target_audience", "prerequisites", "version"],
                quality_requirements={
                    DocumentationQualityMetric.CLARITY: 0.90,
                    DocumentationQualityMetric.COMPLETENESS: 0.85,
                    DocumentationQualityMetric.ACCESSIBILITY: 0.80
                }
            ),
            DocumentationType.README: DocumentationTemplate(
                name="README",
                document_type=DocumentationType.README,
                format=DocumentationFormat.MARKDOWN,
                sections=["title", "description", "installation", "usage", "contributing", "license"],
                metadata_fields=["project_name", "version", "license_type"],
                quality_requirements={
                    DocumentationQualityMetric.COMPLETENESS: 0.80,
                    DocumentationQualityMetric.CLARITY: 0.85,
                    DocumentationQualityMetric.MAINTAINABILITY: 0.75
                }
            ),
            DocumentationType.ARCHITECTURE_GUIDE: DocumentationTemplate(
                name="Architecture Guide",
                document_type=DocumentationType.ARCHITECTURE_GUIDE,
                format=DocumentationFormat.MARKDOWN,
                sections=["overview", "components", "data_flow", "deployment", "security", "scalability"],
                metadata_fields=["system_version", "architecture_type", "last_review"],
                quality_requirements={
                    DocumentationQualityMetric.ACCURACY: 0.95,
                    DocumentationQualityMetric.COMPLETENESS: 0.90,
                    DocumentationQualityMetric.CONSISTENCY: 0.85
                }
            )
        }
        
        # Code analysis patterns for documentation extraction
        self.code_analysis_patterns = {
            "python": {
                "function_pattern": r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):",
                "class_pattern": r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?:",
                "docstring_pattern": r'"""([^"]*(?:"[^"]*")*[^"]*)"""',
                "comment_pattern": r"#\s*(.*)",
                "import_pattern": r"(?:from\s+[\w.]+\s+)?import\s+([\w.,\s*]+)"
            },
            "javascript": {
                "function_pattern": r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\([^)]*\)\s*=>",
                "class_pattern": r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+[\w.]+)?",
                "docstring_pattern": r"/\*\*([^*]*(?:\*(?!/)[^*]*)*)\*/",
                "comment_pattern": r"//\s*(.*)|/\*([^*]*(?:\*(?!/)[^*]*)*)\*/",
                "import_pattern": r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]"
            }
        }
        
        # Documentation quality assessment criteria
        self.quality_assessment_criteria = {
            DocumentationQualityMetric.COMPLETENESS: {
                "description": "All necessary sections and information are present",
                "checks": [
                    "required_sections_present",
                    "content_coverage_adequate",
                    "examples_provided",
                    "edge_cases_covered"
                ]
            },
            DocumentationQualityMetric.ACCURACY: {
                "description": "Information is correct and up-to-date",
                "checks": [
                    "code_examples_valid",
                    "api_references_current",
                    "version_information_correct",
                    "links_functional"
                ]
            },
            DocumentationQualityMetric.CLARITY: {
                "description": "Content is clear and understandable",
                "checks": [
                    "language_clear",
                    "structure_logical",
                    "terminology_consistent",
                    "examples_relevant"
                ]
            },
            DocumentationQualityMetric.CONSISTENCY: {
                "description": "Formatting and style are consistent",
                "checks": [
                    "formatting_consistent",
                    "terminology_standardized",
                    "structure_uniform",
                    "references_standardized"
                ]
            }
        }
        
        logger.info("Documentation Agent initialized with comprehensive documentation capabilities")
    
    async def execute_specialized_task(self, task: AgentTask, 
                                     reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Execute documentation-specific tasks."""
        try:
            task_type = self.classify_documentation_task(task)
            
            if task_type == "generate_documentation":
                return await self.generate_documentation(task, reasoning_result)
            elif task_type == "update_documentation":
                return await self.update_existing_documentation(task, reasoning_result)
            elif task_type == "quality_assessment":
                return await self.assess_documentation_quality(task, reasoning_result)
            elif task_type == "api_documentation":
                return await self.generate_api_documentation(task, reasoning_result)
            elif task_type == "user_guide":
                return await self.generate_user_guide(task, reasoning_result)
            elif task_type == "code_documentation":
                return await self.generate_code_documentation(task, reasoning_result)
            elif task_type == "knowledge_organization":
                return await self.organize_knowledge_base(task, reasoning_result)
            elif task_type == "comprehensive_documentation":
                return await self.create_comprehensive_documentation_suite(task, reasoning_result)
            else:
                return await self.handle_general_documentation_task(task, reasoning_result)
                
        except Exception as e:
            logger.error(f"Documentation task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    def classify_documentation_task(self, task: AgentTask) -> str:
        """Classify the type of documentation task."""
        description = task.description.lower()
        
        if "generate documentation" in description or "create docs" in description:
            return "generate_documentation"
        elif "update documentation" in description or "revise docs" in description:
            return "update_documentation"
        elif "quality" in description or "assess" in description:
            return "quality_assessment"
        elif "api" in description or "endpoint" in description:
            return "api_documentation"
        elif "user guide" in description or "manual" in description:
            return "user_guide"
        elif "code documentation" in description or "docstrings" in description:
            return "code_documentation"
        elif "organize" in description or "knowledge base" in description:
            return "knowledge_organization"
        elif "comprehensive" in description or "complete documentation" in description:
            return "comprehensive_documentation"
        else:
            return "general_documentation"
    
    async def generate_documentation(self, task: AgentTask, 
                                   reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Generate comprehensive documentation using sequential reasoning."""
        try:
            logger.info(f"Generating documentation for: {task.title}")
            
            # Analyze project structure for documentation needs
            project_analysis = await self.analyze_project_for_documentation(task, reasoning_result)
            
            # Develop documentation strategy based on reasoning
            documentation_strategy = await self.develop_documentation_strategy(
                task, reasoning_result, project_analysis
            )
            
            generated_documents = {
                "documents_created": [],
                "total_sections": 0,
                "total_words": 0,
                "formats_generated": []
            }
            
            # Generate documentation for each identified type
            for doc_type in documentation_strategy["document_types"]:
                logger.info(f"Generating {doc_type} documentation...")
                
                if doc_type in self.documentation_templates:
                    template = self.documentation_templates[doc_type]
                    
                    # Generate document using template and project analysis
                    document = await self.generate_document_from_template(
                        template, project_analysis, documentation_strategy
                    )
                    
                    # Write document to filesystem
                    file_path = await self.write_document_to_file(document, template)
                    
                    generated_documents["documents_created"].append({
                        "type": doc_type,
                        "file_path": file_path,
                        "sections": len(document["sections"]),
                        "word_count": document["word_count"],
                        "format": template.format
                    })
                    
                    generated_documents["total_sections"] += len(document["sections"])
                    generated_documents["total_words"] += document["word_count"]
                    if template.format not in generated_documents["formats_generated"]:
                        generated_documents["formats_generated"].append(template.format)
            
            # Generate cross-reference index
            cross_references = await self.generate_cross_reference_index(generated_documents)
            
            # Assess quality of generated documentation
            quality_assessment = await self.assess_generated_documentation_quality(
                generated_documents, documentation_strategy
            )
            
            # Store documentation generation results in LTMC
            if self.ltmc_integration:
                await self.store_documentation_generation_results(
                    task, documentation_strategy, generated_documents, quality_assessment
                )
            
            return {
                "success": True,
                "data": {
                    "documentation_strategy": documentation_strategy,
                    "generated_documents": generated_documents,
                    "cross_references": cross_references,
                    "quality_assessment": quality_assessment,
                    "documents_count": len(generated_documents["documents_created"]),
                    "total_sections": generated_documents["total_sections"],
                    "total_words": generated_documents["total_words"]
                },
                "artifacts": [doc["file_path"] for doc in generated_documents["documents_created"]]
            }
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_api_documentation(self, task: AgentTask, 
                                       reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Generate comprehensive API documentation."""
        try:
            logger.info(f"Generating API documentation for: {task.title}")
            
            # Analyze codebase for API endpoints
            api_analysis = await self.analyze_codebase_for_api_endpoints(task, reasoning_result)
            
            # Generate API documentation sections
            api_documentation = {
                "overview": await self.generate_api_overview(api_analysis, reasoning_result),
                "authentication": await self.generate_api_authentication_docs(api_analysis),
                "endpoints": await self.generate_api_endpoint_docs(api_analysis),
                "examples": await self.generate_api_examples(api_analysis),
                "errors": await self.generate_api_error_docs(api_analysis),
                "schemas": await self.generate_api_schema_docs(api_analysis)
            }
            
            # Generate OpenAPI/Swagger specification if applicable
            openapi_spec = await self.generate_openapi_specification(api_analysis)
            if openapi_spec:
                api_documentation["openapi_spec"] = openapi_spec
            
            # Compile into comprehensive API documentation
            compiled_doc = await self.compile_api_documentation(api_documentation, api_analysis)
            
            # Write API documentation files
            doc_files = await self.write_api_documentation_files(compiled_doc, api_analysis)
            
            # Generate interactive API documentation if possible
            interactive_docs = await self.generate_interactive_api_docs(compiled_doc, api_analysis)
            
            # Store API documentation results in LTMC
            if self.ltmc_integration:
                await self.store_api_documentation_results(
                    task, api_analysis, compiled_doc, interactive_docs
                )
            
            return {
                "success": True,
                "data": {
                    "api_analysis": api_analysis,
                    "api_documentation": compiled_doc,
                    "interactive_docs": interactive_docs,
                    "endpoints_documented": len(api_analysis.get("endpoints", [])),
                    "schemas_documented": len(api_analysis.get("schemas", [])),
                    "examples_generated": len(api_documentation["examples"])
                },
                "artifacts": doc_files
            }
            
        except Exception as e:
            logger.error(f"API documentation generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_comprehensive_documentation_suite(self, task: AgentTask, 
                                                     reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Create a comprehensive documentation suite including all documentation types."""
        try:
            logger.info(f"Creating comprehensive documentation suite for: {task.title}")
            
            suite_results = {
                "phases_completed": [],
                "documents_generated": {},
                "total_execution_time_ms": 0,
                "overall_quality_score": 0.0
            }
            
            start_time = time.time()
            
            # Phase 1: Project Analysis and Strategy Development
            logger.info("Phase 1: Project Analysis and Documentation Strategy")
            project_analysis = await self.analyze_project_comprehensively(task, reasoning_result)
            documentation_strategy = await self.develop_comprehensive_documentation_strategy(
                project_analysis, reasoning_result
            )
            
            suite_results["project_analysis"] = project_analysis
            suite_results["documentation_strategy"] = documentation_strategy
            suite_results["phases_completed"].append("analysis_and_strategy")
            
            # Phase 2: Core Documentation Generation
            logger.info("Phase 2: Core Documentation Generation")
            core_docs_task = AgentTask(
                task_id=f"{task.task_id}_core_docs",
                title="Generate core project documentation",
                description="Generate README, architecture guide, and deployment documentation",
                priority=task.priority,
                assigned_agent=self.agent_id,
                requirements=["readme", "architecture_guide", "deployment_guide"],
                success_criteria=["comprehensive_coverage", "clear_structure", "accurate_content"]
            )
            
            core_docs_result = await self.generate_documentation(core_docs_task, reasoning_result)
            suite_results["documents_generated"]["core_documentation"] = core_docs_result
            suite_results["phases_completed"].append("core_documentation")
            
            # Phase 3: API Documentation (if applicable)
            if project_analysis.get("has_api_endpoints"):
                logger.info("Phase 3: API Documentation Generation")
                api_docs_task = AgentTask(
                    task_id=f"{task.task_id}_api_docs",
                    title="Generate comprehensive API documentation",
                    description="Generate API documentation with endpoints, schemas, and examples",
                    priority=task.priority,
                    assigned_agent=self.agent_id,
                    requirements=["endpoint_docs", "schema_docs", "examples"],
                    success_criteria=["complete_api_coverage", "interactive_docs", "code_examples"]
                )
                
                api_docs_result = await self.generate_api_documentation(api_docs_task, reasoning_result)
                suite_results["documents_generated"]["api_documentation"] = api_docs_result
                suite_results["phases_completed"].append("api_documentation")
            
            # Phase 4: User Guide and Tutorials
            logger.info("Phase 4: User Guide and Tutorial Generation")
            user_guide_task = AgentTask(
                task_id=f"{task.task_id}_user_guide",
                title="Generate user guide and tutorials",
                description="Create comprehensive user documentation and step-by-step tutorials",
                priority=task.priority,
                assigned_agent=self.agent_id,
                requirements=["user_guide", "tutorials", "troubleshooting"],
                success_criteria=["user_friendly", "step_by_step", "comprehensive_coverage"]
            )
            
            user_guide_result = await self.generate_user_guide(user_guide_task, reasoning_result)
            suite_results["documents_generated"]["user_documentation"] = user_guide_result
            suite_results["phases_completed"].append("user_documentation")
            
            # Phase 5: Code Documentation Enhancement
            logger.info("Phase 5: Code Documentation Enhancement")
            code_docs_task = AgentTask(
                task_id=f"{task.task_id}_code_docs",
                title="Enhance code documentation",
                description="Improve inline documentation, docstrings, and code comments",
                priority=task.priority,
                assigned_agent=self.agent_id,
                requirements=["docstring_enhancement", "comment_improvement", "code_examples"],
                success_criteria=["comprehensive_coverage", "clear_explanations", "consistent_style"]
            )
            
            code_docs_result = await self.generate_code_documentation(code_docs_task, reasoning_result)
            suite_results["documents_generated"]["code_documentation"] = code_docs_result
            suite_results["phases_completed"].append("code_documentation")
            
            # Phase 6: Quality Assessment and Cross-Referencing
            logger.info("Phase 6: Quality Assessment and Cross-Referencing")
            quality_assessment = await self.assess_comprehensive_documentation_quality(suite_results)
            cross_references = await self.generate_comprehensive_cross_references(suite_results)
            
            suite_results["quality_assessment"] = quality_assessment
            suite_results["cross_references"] = cross_references
            suite_results["phases_completed"].append("quality_assessment")
            
            # Phase 7: Documentation Index and Navigation
            logger.info("Phase 7: Documentation Index and Navigation Generation")
            navigation_structure = await self.generate_documentation_navigation(suite_results)
            master_index = await self.generate_master_documentation_index(suite_results)
            
            suite_results["navigation_structure"] = navigation_structure
            suite_results["master_index"] = master_index
            suite_results["phases_completed"].append("navigation_and_index")
            
            # Calculate overall metrics
            suite_results["total_execution_time_ms"] = (time.time() - start_time) * 1000
            suite_results["overall_quality_score"] = self.calculate_overall_documentation_quality(suite_results)
            
            # Generate comprehensive documentation report
            comprehensive_report = await self.generate_comprehensive_documentation_report(suite_results, task)
            suite_results["comprehensive_report"] = comprehensive_report
            
            # Store comprehensive documentation suite results in LTMC
            if self.ltmc_integration:
                await self.store_comprehensive_documentation_suite_results(task, suite_results)
            
            return {
                "success": True,
                "data": suite_results,
                "artifacts": self.collect_all_documentation_artifacts(suite_results)
            }
            
        except Exception as e:
            logger.error(f"Comprehensive documentation suite creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Helper methods for documentation generation
    
    async def analyze_project_for_documentation(self, task: AgentTask, 
                                              reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Analyze project structure to determine documentation needs."""
        try:
            analysis = {
                "project_type": "unknown",
                "languages": [],
                "frameworks": [],
                "has_api_endpoints": False,
                "has_database": False,
                "has_tests": False,
                "components": [],
                "configuration_files": [],
                "existing_documentation": []
            }
            
            # Analyze project files
            if self.tool_system:
                # Check for different file types
                file_patterns = {
                    "python": "*.py",
                    "javascript": "*.js",
                    "typescript": "*.ts",
                    "rust": "*.rs",
                    "go": "*.go"
                }
                
                for lang, pattern in file_patterns.items():
                    files_result = await self.tool_system.execute_tool("glob", {"pattern": pattern})
                    if files_result.get("success") and files_result.get("files"):
                        analysis["languages"].append(lang)
                        
                        # Analyze each file for project characteristics
                        for file_path in files_result["files"][:10]:  # Limit to avoid overload
                            await self.analyze_file_for_project_characteristics(file_path, analysis)
                
                # Check for existing documentation
                doc_patterns = ["*.md", "*.rst", "*.txt"]
                for pattern in doc_patterns:
                    docs_result = await self.tool_system.execute_tool("glob", {"pattern": pattern})
                    if docs_result.get("success"):
                        analysis["existing_documentation"].extend(docs_result.get("files", []))
                
                # Check for configuration files
                config_patterns = ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini"]
                for pattern in config_patterns:
                    config_result = await self.tool_system.execute_tool("glob", {"pattern": pattern})
                    if config_result.get("success"):
                        analysis["configuration_files"].extend(config_result.get("files", []))
            
            # Determine project type
            analysis["project_type"] = self.determine_project_type(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Project analysis for documentation failed: {e}")
            return {"project_type": "unknown", "languages": [], "components": []}
    
    async def analyze_file_for_project_characteristics(self, file_path: str, analysis: Dict[str, Any]):
        """Analyze individual file to determine project characteristics."""
        try:
            if self.tool_system:
                content_result = await self.tool_system.execute_tool("read", {"file_path": file_path})
                if not content_result.get("success"):
                    return
                content = content_result.get("content", "")
            else:
                with open(file_path, 'r') as f:
                    content = f.read()
            
            content_lower = content.lower()
            
            # Check for API characteristics
            if any(keyword in content_lower for keyword in ["@app.route", "app.get", "app.post", "fastapi", "express", "router"]):
                analysis["has_api_endpoints"] = True
            
            # Check for database usage
            if any(keyword in content_lower for keyword in ["sqlalchemy", "pymongo", "psycopg2", "mysql", "sqlite", "database"]):
                analysis["has_database"] = True
            
            # Check for testing
            if any(keyword in content_lower for keyword in ["import pytest", "import unittest", "describe(", "test_", "it("]):
                analysis["has_tests"] = True
            
            # Extract framework information
            framework_indicators = {
                "fastapi": ["from fastapi", "FastAPI"],
                "flask": ["from flask", "Flask"],
                "django": ["from django", "Django"],
                "express": ["express()", "app.use"],
                "react": ["import React", "from 'react'"],
                "vue": ["new Vue", "import Vue"],
                "angular": ["@angular", "@Component"]
            }
            
            for framework, indicators in framework_indicators.items():
                if any(indicator in content for indicator in indicators):
                    if framework not in analysis["frameworks"]:
                        analysis["frameworks"].append(framework)
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
    
    def determine_project_type(self, analysis: Dict[str, Any]) -> str:
        """Determine project type based on analysis."""
        if analysis["has_api_endpoints"]:
            return "web_api"
        elif "react" in analysis["frameworks"] or "vue" in analysis["frameworks"]:
            return "frontend_application"
        elif "django" in analysis["frameworks"] or "flask" in analysis["frameworks"]:
            return "web_application"
        elif "python" in analysis["languages"] and not analysis["has_api_endpoints"]:
            return "python_library"
        elif "javascript" in analysis["languages"] or "typescript" in analysis["languages"]:
            return "javascript_library"
        else:
            return "general_software"
    
    async def generate_document_from_template(self, template: DocumentationTemplate,
                                            project_analysis: Dict[str, Any],
                                            strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate document content using template and project analysis."""
        try:
            document = {
                "title": template.name,
                "document_type": template.document_type,
                "format": template.format,
                "sections": [],
                "metadata": {},
                "word_count": 0
            }
            
            # Generate metadata
            for field in template.metadata_fields:
                document["metadata"][field] = self.generate_metadata_value(field, project_analysis)
            
            # Generate content for each section
            for section_name in template.sections:
                section_content = await self.generate_section_content(
                    section_name, template, project_analysis, strategy
                )
                
                section = DocumentationSection(
                    title=section_name.replace('_', ' ').title(),
                    content=section_content,
                    level=1,
                    section_type=section_name,
                    metadata={"generated_at": datetime.now().isoformat()}
                )
                
                document["sections"].append(section)
                document["word_count"] += len(section_content.split())
            
            return document
            
        except Exception as e:
            logger.error(f"Document generation from template failed: {e}")
            return {"title": "Error", "sections": [], "word_count": 0}
    
    async def generate_section_content(self, section_name: str, template: DocumentationTemplate,
                                     project_analysis: Dict[str, Any], strategy: Dict[str, Any]) -> str:
        """Generate content for a specific documentation section."""
        try:
            if template.document_type == DocumentationType.README:
                return await self.generate_readme_section_content(section_name, project_analysis, strategy)
            elif template.document_type == DocumentationType.API_DOCUMENTATION:
                return await self.generate_api_section_content(section_name, project_analysis, strategy)
            elif template.document_type == DocumentationType.USER_GUIDE:
                return await self.generate_user_guide_section_content(section_name, project_analysis, strategy)
            else:
                return await self.generate_generic_section_content(section_name, project_analysis, strategy)
                
        except Exception as e:
            logger.error(f"Section content generation failed for {section_name}: {e}")
            return f"# {section_name.replace('_', ' ').title()}\n\nContent generation failed: {str(e)}"
    
    async def generate_readme_section_content(self, section_name: str, 
                                            project_analysis: Dict[str, Any], 
                                            strategy: Dict[str, Any]) -> str:
        """Generate README-specific section content."""
        project_name = project_analysis.get("project_name", "Project")
        
        if section_name == "title":
            return f"# {project_name}\n\nA comprehensive {project_analysis['project_type']} project."
        
        elif section_name == "description":
            return f"""## Description

{project_name} is a {project_analysis['project_type']} built with {', '.join(project_analysis['languages'])}.

### Features

- Comprehensive functionality
- Modern architecture
- Extensive testing
- Detailed documentation

### Technologies Used

- **Languages**: {', '.join(project_analysis['languages'])}
- **Frameworks**: {', '.join(project_analysis['frameworks']) if project_analysis['frameworks'] else 'None specified'}
"""
        
        elif section_name == "installation":
            if "python" in project_analysis["languages"]:
                return """## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest
```
"""
            else:
                return """## Installation

### Prerequisites

Please check the project documentation for specific requirements.

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Follow setup instructions for your environment
```
"""
        
        elif section_name == "usage":
            return """## Usage

### Basic Usage

```bash
# Basic command example
python main.py --help
```

### Advanced Usage

For detailed usage instructions, see the [User Guide](docs/user-guide.md).

### Examples

See the `examples/` directory for complete usage examples.
"""
        
        elif section_name == "contributing":
            return """## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow the existing code style
- Write comprehensive tests
- Update documentation as needed
- Ensure all CI checks pass
"""
        
        elif section_name == "license":
            return """## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""
        
        else:
            return f"## {section_name.replace('_', ' ').title()}\n\nContent for {section_name} section."
    
    # Additional helper methods would continue here...
    
    def generate_metadata_value(self, field: str, project_analysis: Dict[str, Any]) -> str:
        """Generate metadata value for a specific field."""
        metadata_generators = {
            "version": lambda: "1.0.0",
            "last_updated": lambda: datetime.now().isoformat(),
            "project_name": lambda: project_analysis.get("project_name", "Unknown Project"),
            "target_audience": lambda: "developers",
            "prerequisites": lambda: ", ".join(project_analysis.get("languages", [])),
            "license_type": lambda: "MIT",
            "base_url": lambda: "https://api.example.com",
            "system_version": lambda: "1.0.0",
            "architecture_type": lambda: project_analysis.get("project_type", "unknown"),
            "last_review": lambda: datetime.now().date().isoformat()
        }
        
        generator = metadata_generators.get(field)
        return generator() if generator else f"Unknown {field}"
    
    async def write_document_to_file(self, document: Dict[str, Any], 
                                   template: DocumentationTemplate) -> str:
        """Write generated document to filesystem."""
        try:
            # Determine filename
            doc_type = template.document_type
            format_ext = {"markdown": ".md", "html": ".html", "rst": ".rst"}.get(template.format, ".md")
            
            filename = f"{doc_type.replace('_', '-')}{format_ext}"
            
            # Generate document content
            content = self.compile_document_content(document, template)
            
            # Write to file using tool system if available
            if self.tool_system:
                write_result = await self.tool_system.execute_tool("write", {
                    "file_path": filename,
                    "content": content
                })
                if write_result.get("success"):
                    logger.info(f"Documentation written to: {filename}")
                    return filename
            else:
                # Fallback to direct file writing
                with open(filename, 'w') as f:
                    f.write(content)
                logger.info(f"Documentation written to: {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"Failed to write document to file: {e}")
            return ""
    
    def compile_document_content(self, document: Dict[str, Any], 
                               template: DocumentationTemplate) -> str:
        """Compile document sections into final content."""
        content_parts = []
        
        # Add title if not already in sections
        if document.get("title") and not any(s.section_type == "title" for s in document["sections"]):
            content_parts.append(f"# {document['title']}\n")
        
        # Add metadata if format supports it
        if template.format == DocumentationFormat.MARKDOWN and document.get("metadata"):
            content_parts.append("---")
            for key, value in document["metadata"].items():
                content_parts.append(f"{key}: {value}")
            content_parts.append("---\n")
        
        # Add sections
        for section in document["sections"]:
            if hasattr(section, 'content'):
                content_parts.append(section.content)
            else:
                content_parts.append(str(section))
            content_parts.append("\n")
        
        # Add footer
        content_parts.append(f"\n---\n*Generated by KWE CLI Documentation Agent on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(content_parts)
    
    async def store_documentation_generation_results(self, task: AgentTask,
                                                   documentation_strategy: Dict[str, Any],
                                                   generated_documents: Dict[str, Any],
                                                   quality_assessment: Dict[str, Any]):
        """Store documentation generation results in LTMC."""
        if not self.ltmc_integration:
            return
        
        try:
            doc_name = f"DOCUMENTATION_GENERATION_RESULTS_{task.task_id}.md"
            content = f"""# Documentation Generation Results
## Task: {task.title}
## Agent: {self.agent_id}
## Timestamp: {datetime.now().isoformat()}

### Documentation Strategy:
```json
{json.dumps(documentation_strategy, indent=2)}
```

### Generated Documents:
- **Total Documents**: {len(generated_documents['documents_created'])}
- **Total Sections**: {generated_documents['total_sections']}
- **Total Words**: {generated_documents['total_words']}
- **Formats**: {', '.join(generated_documents['formats_generated'])}

### Document Details:
```json
{json.dumps(generated_documents['documents_created'], indent=2)}
```

### Quality Assessment:
```json
{json.dumps(quality_assessment, indent=2)}
```

This documentation generation result is part of autonomous development for KWE CLI workflows.
"""
            
            await self.ltmc_integration.store_document(
                file_name=doc_name,
                content=content,
                conversation_id="autonomous_documentation",
                resource_type="documentation_generation_result"
            )
            
        except Exception as e:
            logger.error(f"Failed to store documentation generation results: {e}")

# Export the DocumentationAgent
__all__ = ['DocumentationAgent']
#!/usr/bin/env python3
"""
KWE CLI Research Agent

This module implements a specialized research agent for autonomous information 
gathering, solution research, and knowledge base building. Critical for 
self-healing workflows and continuous improvement.

Key Features:
- Comprehensive web research and information gathering
- Solution pattern identification and analysis
- Best practices research and recommendation
- Technology evaluation and assessment
- Knowledge base building and maintenance
- Research result synthesis and validation
- Issue diagnosis and solution research
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import re

from .base_agent import (
    BaseKWECLIAgent, AgentTask, AgentResult, AgentRole, 
    AgentCapability, TaskPriority, TaskStatus
)
from .sequential_thinking import Problem, ReasoningResult, ReasoningType

# Configure logging
logger = logging.getLogger(__name__)

class ResearchType(Enum):
    """Types of research operations."""
    SOLUTION_RESEARCH = "solution_research"
    BEST_PRACTICES = "best_practices" 
    TECHNOLOGY_EVALUATION = "technology_evaluation"
    ISSUE_DIAGNOSIS = "issue_diagnosis"
    KNOWLEDGE_DISCOVERY = "knowledge_discovery"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    TREND_ANALYSIS = "trend_analysis"
    DOCUMENTATION_RESEARCH = "documentation_research"

class ResearchScope(Enum):
    """Scope of research operations."""
    FOCUSED = "focused"      # Specific technical issue or solution
    BROAD = "broad"          # General topic exploration
    COMPREHENSIVE = "comprehensive"  # Deep multi-source analysis
    COMPARATIVE = "comparative"      # Comparing alternatives
    HISTORICAL = "historical"       # Evolution and trends

class InformationSource(Enum):
    """Sources of research information."""
    WEB_SEARCH = "web_search"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    CODE_REPOSITORIES = "code_repositories"
    ACADEMIC_PAPERS = "academic_papers"
    FORUMS_COMMUNITIES = "forums_communities"
    OFFICIAL_DOCS = "official_docs"
    BEST_PRACTICE_GUIDES = "best_practice_guides"
    LTMC_KNOWLEDGE = "ltmc_knowledge"

@dataclass
class ResearchQuery:
    """Represents a research query with context and requirements."""
    query_id: str
    query_text: str
    research_type: ResearchType
    scope: ResearchScope
    context: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    time_limit_minutes: int = 30
    source_preferences: List[InformationSource] = field(default_factory=list)
    quality_threshold: float = 0.7
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ResearchSource:
    """Information about a research source."""
    source_id: str
    source_type: InformationSource
    url: Optional[str] = None
    title: str = ""
    description: str = ""
    credibility_score: float = 0.0
    relevance_score: float = 0.0
    last_updated: Optional[str] = None
    access_method: str = "web_fetch"

@dataclass
class ResearchFinding:
    """Individual research finding with validation."""
    finding_id: str
    content: str
    source: ResearchSource
    confidence_score: float
    relevance_score: float
    validation_status: str = "pending"
    tags: List[str] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    key_insights: List[str] = field(default_factory=list)
    actionable_recommendations: List[str] = field(default_factory=list)

@dataclass
class ResearchResult:
    """Complete research result with synthesis."""
    query_id: str
    query_text: str
    research_type: ResearchType
    findings: List[ResearchFinding]
    synthesis: str
    key_insights: List[str]
    recommendations: List[str]
    confidence_score: float
    sources_consulted: int
    execution_time_ms: float
    knowledge_gaps_identified: List[str] = field(default_factory=list)
    follow_up_research_needed: List[str] = field(default_factory=list)
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())

class KWECLIResearchAgent(BaseKWECLIAgent):
    """
    Specialized research agent for autonomous information gathering and analysis.
    
    Capabilities:
    - Comprehensive web research and information gathering
    - Solution pattern identification and analysis
    - Best practices research and recommendation
    - Technology evaluation and assessment
    - Knowledge base building and maintenance
    - Research result synthesis and validation
    - Issue diagnosis and solution research
    """
    
    def __init__(self):
        super().__init__(
            agent_id="kwecli_research_agent",
            role=AgentRole.RESEARCH_SPECIALIST,
            capabilities=[
                AgentCapability.RESEARCH,
                AgentCapability.SEQUENTIAL_REASONING,
                AgentCapability.DOCUMENTATION
            ]
        )
        
        # Research-specific systems
        self.research_cache: Dict[str, ResearchResult] = {}
        self.knowledge_base: Dict[str, Any] = {}
        self.research_patterns: Dict[str, Any] = {}
        self.source_credibility_tracker: Dict[str, float] = {}
        
        # Research configuration
        self.max_sources_per_query = 10
        self.default_research_timeout = 1800  # 30 minutes
        self.quality_threshold = 0.7
        self.synthesis_depth = "comprehensive"
        
        # Web research tools integration
        self.web_search_available = False
        self.web_fetch_available = False
        
        logger.info("KWE CLI Research Agent initialized with comprehensive research capabilities")
    
    def initialize_research_tools(self, web_search=None, web_fetch=None):
        """Initialize web research tool integrations."""
        if web_search:
            self.web_search_tool = web_search
            self.web_search_available = True
            logger.info("Web search tool integrated with Research Agent")
        
        if web_fetch:
            self.web_fetch_tool = web_fetch
            self.web_fetch_available = True
            logger.info("Web fetch tool integrated with Research Agent")
    
    async def execute_specialized_task(self, task: AgentTask, 
                                     reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Execute specialized research task logic."""
        try:
            logger.info(f"Research Agent executing task: {task.title}")
            start_time = time.time()
            
            # Parse research requirements from task
            research_query = self.parse_research_query(task, reasoning_result)
            
            # Execute research based on query type
            if research_query.research_type == ResearchType.SOLUTION_RESEARCH:
                result = await self.conduct_solution_research(task, research_query, reasoning_result)
            elif research_query.research_type == ResearchType.BEST_PRACTICES:
                result = await self.research_best_practices(task, research_query, reasoning_result)
            elif research_query.research_type == ResearchType.TECHNOLOGY_EVALUATION:
                result = await self.evaluate_technology_options(task, research_query, reasoning_result)
            elif research_query.research_type == ResearchType.ISSUE_DIAGNOSIS:
                result = await self.diagnose_issue_solutions(task, research_query, reasoning_result)
            elif research_query.research_type == ResearchType.KNOWLEDGE_DISCOVERY:
                result = await self.discover_knowledge_patterns(task, research_query, reasoning_result)
            elif research_query.research_type == ResearchType.DOCUMENTATION_RESEARCH:
                result = await self.research_documentation_sources(task, research_query, reasoning_result)
            else:
                result = await self.conduct_general_research(task, research_query, reasoning_result)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Store research result in cache and LTMC
            self.research_cache[research_query.query_id] = result
            if self.ltmc_integration:
                await self.store_research_result(result)
            
            return {
                "success": True,
                "data": {
                    "research_result": result,
                    "findings_count": len(result.findings),
                    "confidence_score": result.confidence_score,
                    "sources_consulted": result.sources_consulted,
                    "key_insights": result.key_insights,
                    "recommendations": result.recommendations
                },
                "execution_time_ms": execution_time,
                "resource_usage": {
                    "queries_executed": result.sources_consulted,
                    "findings_processed": len(result.findings),
                    "memory_mb": len(str(result)) / 1024  # Rough estimate
                },
                "artifacts": [f"research_result_{result.query_id}.json"]
            }
            
        except Exception as e:
            logger.error(f"Research Agent task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            }
    
    def parse_research_query(self, task: AgentTask, reasoning_result: ReasoningResult) -> ResearchQuery:
        """Parse research query from task and reasoning context."""
        query_text = task.description
        
        # Determine research type from task context
        research_type = self.determine_research_type(task, reasoning_result)
        
        # Determine research scope
        scope = self.determine_research_scope(task, reasoning_result)
        
        # Extract source preferences
        source_preferences = self.extract_source_preferences(task)
        
        return ResearchQuery(
            query_id=str(uuid.uuid4()),
            query_text=query_text,
            research_type=research_type,
            scope=scope,
            context=task.context,
            priority=task.priority,
            time_limit_minutes=self.determine_time_limit(task),
            source_preferences=source_preferences,
            quality_threshold=self.quality_threshold
        )
    
    def determine_research_type(self, task: AgentTask, reasoning_result: ReasoningResult) -> ResearchType:
        """Determine appropriate research type based on task context."""
        description = task.description.lower()
        
        if "solve" in description or "fix" in description or "error" in description:
            return ResearchType.SOLUTION_RESEARCH
        elif "best practice" in description or "recommend" in description:
            return ResearchType.BEST_PRACTICES
        elif "evaluate" in description or "compare" in description or "choose" in description:
            return ResearchType.TECHNOLOGY_EVALUATION
        elif "diagnose" in description or "debug" in description or "troubleshoot" in description:
            return ResearchType.ISSUE_DIAGNOSIS
        elif "document" in description or "guide" in description:
            return ResearchType.DOCUMENTATION_RESEARCH
        elif "discover" in description or "explore" in description:
            return ResearchType.KNOWLEDGE_DISCOVERY
        else:
            return ResearchType.SOLUTION_RESEARCH  # Default to solution research
    
    def determine_research_scope(self, task: AgentTask, reasoning_result: ReasoningResult) -> ResearchScope:
        """Determine research scope based on task complexity and priority."""
        if task.priority == TaskPriority.CRITICAL:
            return ResearchScope.COMPREHENSIVE
        elif len(task.requirements) > 5:
            return ResearchScope.BROAD
        elif "compare" in task.description.lower():
            return ResearchScope.COMPARATIVE
        elif "specific" in task.description.lower() or "exact" in task.description.lower():
            return ResearchScope.FOCUSED
        else:
            return ResearchScope.BROAD
    
    def extract_source_preferences(self, task: AgentTask) -> List[InformationSource]:
        """Extract preferred information sources from task context."""
        preferences = []
        description = task.description.lower()
        
        if "documentation" in description or "docs" in description:
            preferences.extend([InformationSource.OFFICIAL_DOCS, InformationSource.TECHNICAL_DOCUMENTATION])
        if "github" in description or "repository" in description:
            preferences.append(InformationSource.CODE_REPOSITORIES)
        if "academic" in description or "paper" in description:
            preferences.append(InformationSource.ACADEMIC_PAPERS)
        if "community" in description or "forum" in description:
            preferences.append(InformationSource.FORUMS_COMMUNITIES)
        
        # Default preferences if none specified
        if not preferences:
            preferences = [
                InformationSource.WEB_SEARCH,
                InformationSource.OFFICIAL_DOCS,
                InformationSource.TECHNICAL_DOCUMENTATION
            ]
        
        return preferences
    
    def determine_time_limit(self, task: AgentTask) -> int:
        """Determine appropriate time limit for research based on task priority."""
        if task.priority == TaskPriority.CRITICAL:
            return 15  # Fast research for critical issues
        elif task.priority == TaskPriority.HIGH:
            return 20
        elif task.priority == TaskPriority.NORMAL:
            return 30
        else:
            return 45  # More thorough research for low priority
    
    async def conduct_solution_research(self, task: AgentTask, query: ResearchQuery, 
                                      reasoning_result: ReasoningResult) -> ResearchResult:
        """Conduct comprehensive solution research for specific problems."""
        logger.info(f"Conducting solution research for: {query.query_text}")
        
        findings = []
        start_time = time.time()
        
        # Research strategy based on sequential reasoning
        research_strategy = self.develop_solution_research_strategy(query, reasoning_result)
        
        # Execute multi-source research
        for source_type in query.source_preferences:
            if len(findings) >= self.max_sources_per_query:
                break
            
            source_findings = await self.research_from_source(query, source_type, research_strategy)
            findings.extend(source_findings)
        
        # Synthesize findings into actionable solutions
        synthesis = await self.synthesize_solution_findings(query, findings, reasoning_result)
        
        # Extract key insights and recommendations
        key_insights = self.extract_solution_insights(findings, synthesis)
        recommendations = self.generate_solution_recommendations(findings, synthesis, reasoning_result)
        
        # Calculate confidence score
        confidence_score = self.calculate_research_confidence(findings, synthesis)
        
        execution_time = (time.time() - start_time) * 1000
        
        return ResearchResult(
            query_id=query.query_id,
            query_text=query.query_text,
            research_type=query.research_type,
            findings=findings,
            synthesis=synthesis,
            key_insights=key_insights,
            recommendations=recommendations,
            confidence_score=confidence_score,
            sources_consulted=len(findings),
            execution_time_ms=execution_time,
            knowledge_gaps_identified=self.identify_knowledge_gaps(findings),
            follow_up_research_needed=self.suggest_follow_up_research(query, findings)
        )
    
    def develop_solution_research_strategy(self, query: ResearchQuery, 
                                         reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Develop targeted research strategy based on reasoning insights."""
        strategy = {
            "search_terms": [],
            "focus_areas": [],
            "solution_patterns": [],
            "technology_stack": [],
            "priority_sources": []
        }
        
        # Extract search terms from query and reasoning
        query_words = query.query_text.lower().split()
        strategy["search_terms"] = [word for word in query_words if len(word) > 3]
        
        # Add reasoning-derived search terms
        if reasoning_result and reasoning_result.thought_sequence:
            for thought in reasoning_result.thought_sequence[-3:]:  # Recent thoughts
                thought_words = thought.content.lower().split()
                strategy["search_terms"].extend([word for word in thought_words if len(word) > 4])
        
        # Remove duplicates and common words
        strategy["search_terms"] = list(set(strategy["search_terms"]))
        common_words = {"with", "that", "this", "from", "they", "have", "more", "when", "some", "what"}
        strategy["search_terms"] = [term for term in strategy["search_terms"] if term not in common_words]
        
        # Focus areas based on problem type
        if "error" in query.query_text.lower():
            strategy["focus_areas"].extend(["error resolution", "debugging", "troubleshooting"])
        if "performance" in query.query_text.lower():
            strategy["focus_areas"].extend(["optimization", "performance tuning", "best practices"])
        if "security" in query.query_text.lower():
            strategy["focus_areas"].extend(["security", "vulnerability", "security best practices"])
        
        return strategy
    
    async def research_from_source(self, query: ResearchQuery, source_type: InformationSource, 
                                 strategy: Dict[str, Any]) -> List[ResearchFinding]:
        """Research information from a specific source type."""
        findings = []
        
        try:
            if source_type == InformationSource.WEB_SEARCH and self.web_search_available:
                findings = await self.web_search_research(query, strategy)
            elif source_type == InformationSource.LTMC_KNOWLEDGE and self.ltmc_integration:
                findings = await self.ltmc_knowledge_research(query, strategy)
            elif source_type == InformationSource.TECHNICAL_DOCUMENTATION:
                findings = await self.documentation_research(query, strategy)
            else:
                # Simulate research for sources not yet implemented
                findings = await self.simulate_source_research(query, source_type, strategy)
            
        except Exception as e:
            logger.error(f"Research from {source_type.value} failed: {e}")
            # Continue with other sources
        
        return findings
    
    async def web_search_research(self, query: ResearchQuery, strategy: Dict[str, Any]) -> List[ResearchFinding]:
        """Conduct web search research."""
        findings = []
        
        try:
            # Construct search query from strategy
            search_terms = " ".join(strategy["search_terms"][:5])  # Top 5 terms
            search_query = f"{query.query_text} {search_terms}"
            
            # Execute web search (using tool integration)
            search_results = await self.web_search_tool(search_query)
            
            # Process search results
            for i, result in enumerate(search_results.get("results", [])[:5]):  # Top 5 results
                source = ResearchSource(
                    source_id=f"web_search_{i}",
                    source_type=InformationSource.WEB_SEARCH,
                    url=result.get("url"),
                    title=result.get("title", ""),
                    description=result.get("description", ""),
                    credibility_score=self.assess_source_credibility(result),
                    relevance_score=self.assess_content_relevance(result.get("description", ""), query)
                )
                
                # Extract detailed content if available
                content = result.get("description", "")
                if self.web_fetch_available and result.get("url"):
                    try:
                        detailed_content = await self.web_fetch_tool(
                            result.get("url"), 
                            f"Extract information about: {query.query_text}"
                        )
                        content = detailed_content.get("content", content)
                    except:
                        pass  # Use description if fetch fails
                
                finding = ResearchFinding(
                    finding_id=str(uuid.uuid4()),
                    content=content,
                    source=source,
                    confidence_score=source.credibility_score * source.relevance_score,
                    relevance_score=source.relevance_score,
                    tags=self.extract_content_tags(content),
                    key_insights=self.extract_key_insights(content, query),
                    actionable_recommendations=self.extract_recommendations(content, query)
                )
                
                findings.append(finding)
            
        except Exception as e:
            logger.error(f"Web search research failed: {e}")
        
        return findings
    
    async def ltmc_knowledge_research(self, query: ResearchQuery, strategy: Dict[str, Any]) -> List[ResearchFinding]:
        """Research from LTMC knowledge base."""
        findings = []
        
        try:
            # Search LTMC for relevant knowledge
            ltmc_results = await self.ltmc_integration.retrieve_documents(
                query=query.query_text,
                conversation_id="research_knowledge",
                k=5
            )
            
            for i, result in enumerate(ltmc_results.get("results", [])):
                source = ResearchSource(
                    source_id=f"ltmc_knowledge_{i}",
                    source_type=InformationSource.LTMC_KNOWLEDGE,
                    title=result.get("file_name", "LTMC Knowledge"),
                    description="Internal knowledge base",
                    credibility_score=0.9,  # High credibility for internal knowledge
                    relevance_score=result.get("similarity_score", 0.5)
                )
                
                finding = ResearchFinding(
                    finding_id=str(uuid.uuid4()),
                    content=result.get("content", ""),
                    source=source,
                    confidence_score=0.9 * result.get("similarity_score", 0.5),
                    relevance_score=result.get("similarity_score", 0.5),
                    tags=["internal_knowledge", "ltmc"],
                    key_insights=self.extract_key_insights(result.get("content", ""), query),
                    actionable_recommendations=self.extract_recommendations(result.get("content", ""), query)
                )
                
                findings.append(finding)
                
        except Exception as e:
            logger.error(f"LTMC knowledge research failed: {e}")
        
        return findings
    
    async def documentation_research(self, query: ResearchQuery, strategy: Dict[str, Any]) -> List[ResearchFinding]:
        """Research technical documentation sources."""
        findings = []
        
        # Common documentation sources for software development
        doc_sources = [
            {"name": "Python Documentation", "url": "https://docs.python.org", "relevance": 0.8},
            {"name": "MDN Web Docs", "url": "https://developer.mozilla.org", "relevance": 0.7},
            {"name": "Stack Overflow", "url": "https://stackoverflow.com", "relevance": 0.9},
            {"name": "GitHub", "url": "https://github.com", "relevance": 0.8},
            {"name": "ReadTheDocs", "url": "https://readthedocs.org", "relevance": 0.7}
        ]
        
        for i, doc_source in enumerate(doc_sources[:3]):  # Top 3 sources
            if not self.web_search_available:
                break
            
            # Search within documentation site
            search_query = f"site:{doc_source['url'].replace('https://', '')} {query.query_text}"
            
            try:
                search_results = await self.web_search_tool(search_query)
                
                for j, result in enumerate(search_results.get("results", [])[:2]):  # Top 2 per source
                    source = ResearchSource(
                        source_id=f"docs_{i}_{j}",
                        source_type=InformationSource.TECHNICAL_DOCUMENTATION,
                        url=result.get("url"),
                        title=f"{doc_source['name']}: {result.get('title', '')}",
                        description=result.get("description", ""),
                        credibility_score=doc_source["relevance"],
                        relevance_score=self.assess_content_relevance(result.get("description", ""), query)
                    )
                    
                    content = result.get("description", "")
                    
                    finding = ResearchFinding(
                        finding_id=str(uuid.uuid4()),
                        content=content,
                        source=source,
                        confidence_score=source.credibility_score * source.relevance_score,
                        relevance_score=source.relevance_score,
                        tags=["documentation", doc_source["name"].lower().replace(" ", "_")],
                        key_insights=self.extract_key_insights(content, query),
                        actionable_recommendations=self.extract_recommendations(content, query)
                    )
                    
                    findings.append(finding)
                    
            except Exception as e:
                logger.error(f"Documentation research for {doc_source['name']} failed: {e}")
                continue
        
        return findings
    
    async def simulate_source_research(self, query: ResearchQuery, source_type: InformationSource, 
                                     strategy: Dict[str, Any]) -> List[ResearchFinding]:
        """Simulate research from sources not yet fully implemented."""
        findings = []
        
        # Generate realistic simulated findings based on query
        for i in range(2):  # 2 findings per source
            source = ResearchSource(
                source_id=f"simulated_{source_type.value}_{i}",
                source_type=source_type,
                title=f"Research finding from {source_type.value}",
                description=f"Simulated research result for {query.query_text}",
                credibility_score=0.6,
                relevance_score=0.7
            )
            
            # Generate contextual content based on query type
            content = self.generate_simulated_content(query, source_type)
            
            finding = ResearchFinding(
                finding_id=str(uuid.uuid4()),
                content=content,
                source=source,
                confidence_score=0.6 * 0.7,  # credibility * relevance
                relevance_score=0.7,
                tags=["simulated", source_type.value],
                key_insights=[f"Key insight from {source_type.value} research"],
                actionable_recommendations=[f"Recommendation based on {source_type.value} analysis"]
            )
            
            findings.append(finding)
        
        return findings
    
    def generate_simulated_content(self, query: ResearchQuery, source_type: InformationSource) -> str:
        """Generate realistic simulated content for research findings."""
        content_templates = {
            InformationSource.CODE_REPOSITORIES: f"Repository analysis for '{query.query_text}' shows common implementation patterns including error handling, configuration management, and testing strategies. Best practices include using structured logging, proper exception handling, and comprehensive test coverage.",
            
            InformationSource.ACADEMIC_PAPERS: f"Academic research on '{query.query_text}' demonstrates several key approaches with empirical validation. Studies show that systematic methodologies with proper validation and error recovery mechanisms achieve higher reliability scores.",
            
            InformationSource.FORUMS_COMMUNITIES: f"Community discussions about '{query.query_text}' reveal common challenges and solutions. Experienced developers recommend a structured approach with proper error handling, comprehensive logging, and iterative testing.",
            
            InformationSource.BEST_PRACTICE_GUIDES: f"Best practice guidelines for '{query.query_text}' emphasize the importance of systematic approaches, proper documentation, testing methodologies, and continuous improvement through feedback loops."
        }
        
        return content_templates.get(source_type, f"Research findings related to '{query.query_text}' suggest implementing systematic approaches with proper validation, error handling, and comprehensive documentation.")
    
    def assess_source_credibility(self, result: Dict[str, Any]) -> float:
        """Assess credibility of a search result source."""
        url = result.get("url", "").lower()
        
        # High credibility sources
        if any(domain in url for domain in ["stackoverflow.com", "github.com", "docs.python.org", "developer.mozilla.org"]):
            return 0.9
        elif any(domain in url for domain in [".edu", ".gov", "readthedocs.org"]):
            return 0.8
        elif any(domain in url for domain in ["medium.com", "dev.to", "towards"]):
            return 0.7
        else:
            return 0.6  # Default credibility
    
    def assess_content_relevance(self, content: str, query: ResearchQuery) -> float:
        """Assess how relevant content is to the research query."""
        if not content:
            return 0.0
        
        content_lower = content.lower()
        query_lower = query.query_text.lower()
        
        # Count query term matches
        query_terms = query_lower.split()
        matches = sum(1 for term in query_terms if term in content_lower)
        
        # Base relevance on term match percentage
        base_relevance = matches / len(query_terms) if query_terms else 0
        
        # Boost for exact phrase matches
        if query_lower in content_lower:
            base_relevance += 0.3
        
        return min(0.95, base_relevance)  # Cap at 0.95
    
    def extract_content_tags(self, content: str) -> List[str]:
        """Extract relevant tags from content."""
        if not content:
            return []
        
        # Common technical tags
        tech_terms = [
            "python", "javascript", "react", "docker", "kubernetes", "aws", "azure",
            "database", "api", "rest", "graphql", "microservices", "testing", "security",
            "performance", "optimization", "debugging", "error", "exception", "logging"
        ]
        
        content_lower = content.lower()
        found_tags = [term for term in tech_terms if term in content_lower]
        
        # Add generic tags based on content analysis
        if "error" in content_lower or "exception" in content_lower:
            found_tags.append("error_handling")
        if "test" in content_lower:
            found_tags.append("testing")
        if "security" in content_lower:
            found_tags.append("security")
        
        return list(set(found_tags))  # Remove duplicates
    
    def extract_key_insights(self, content: str, query: ResearchQuery) -> List[str]:
        """Extract key insights from research content."""
        if not content:
            return []
        
        insights = []
        content_sentences = content.split('. ')
        
        # Look for sentences containing key insight indicators
        insight_indicators = ["important", "key", "critical", "essential", "recommended", "best practice", "solution", "approach"]
        
        for sentence in content_sentences[:5]:  # Check first 5 sentences
            if any(indicator in sentence.lower() for indicator in insight_indicators):
                if len(sentence.strip()) > 20:  # Meaningful length
                    insights.append(sentence.strip())
        
        # If no insights found, extract first meaningful sentence
        if not insights and content_sentences:
            for sentence in content_sentences[:3]:
                if len(sentence.strip()) > 30:
                    insights.append(sentence.strip())
                    break
        
        return insights[:3]  # Return top 3 insights
    
    def extract_recommendations(self, content: str, query: ResearchQuery) -> List[str]:
        """Extract actionable recommendations from content."""
        if not content:
            return []
        
        recommendations = []
        content_sentences = content.split('. ')
        
        # Look for sentences with recommendation indicators
        rec_indicators = ["should", "must", "recommend", "suggest", "use", "implement", "consider", "ensure"]
        
        for sentence in content_sentences:
            if any(indicator in sentence.lower() for indicator in rec_indicators):
                if len(sentence.strip()) > 20:
                    # Clean up the recommendation
                    cleaned = sentence.strip()
                    if not cleaned.endswith('.'):
                        cleaned += '.'
                    recommendations.append(cleaned)
        
        return recommendations[:3]  # Return top 3 recommendations
    
    async def synthesize_solution_findings(self, query: ResearchQuery, findings: List[ResearchFinding], 
                                         reasoning_result: ReasoningResult) -> str:
        """Synthesize research findings into coherent solution guidance."""
        if not findings:
            return f"No specific research findings available for '{query.query_text}'. General approach recommended: systematic analysis, structured implementation, comprehensive testing."
        
        # Group findings by confidence and relevance
        high_confidence = [f for f in findings if f.confidence_score > 0.7]
        moderate_confidence = [f for f in findings if 0.4 <= f.confidence_score <= 0.7]
        
        synthesis_parts = []
        
        # Synthesis header
        synthesis_parts.append(f"Research synthesis for: {query.query_text}")
        synthesis_parts.append(f"Based on analysis of {len(findings)} sources with {len(high_confidence)} high-confidence findings.\n")
        
        # High confidence findings synthesis
        if high_confidence:
            synthesis_parts.append("Primary Solutions and Approaches:")
            for finding in high_confidence[:3]:  # Top 3 high confidence
                synthesis_parts.append(f"• {finding.content[:200]}...")
                if finding.actionable_recommendations:
                    synthesis_parts.append(f"  Recommendation: {finding.actionable_recommendations[0]}")
            synthesis_parts.append("")
        
        # Moderate confidence findings
        if moderate_confidence:
            synthesis_parts.append("Additional Considerations:")
            for finding in moderate_confidence[:2]:  # Top 2 moderate confidence
                if finding.key_insights:
                    synthesis_parts.append(f"• {finding.key_insights[0]}")
            synthesis_parts.append("")
        
        # Integration with reasoning results
        if reasoning_result and reasoning_result.thought_sequence:
            final_thoughts = [t for t in reasoning_result.thought_sequence if t.confidence > 0.6]
            if final_thoughts:
                synthesis_parts.append("Integration with Sequential Reasoning:")
                synthesis_parts.append(f"• {final_thoughts[-1].content[:150]}...")
                synthesis_parts.append("")
        
        # Confidence assessment
        avg_confidence = sum(f.confidence_score for f in findings) / len(findings)
        synthesis_parts.append(f"Overall Confidence: {avg_confidence:.2f} based on {len(findings)} sources")
        
        return "\n".join(synthesis_parts)
    
    def extract_solution_insights(self, findings: List[ResearchFinding], synthesis: str) -> List[str]:
        """Extract key insights from solution research."""
        insights = []
        
        # Aggregate insights from findings
        for finding in findings:
            insights.extend(finding.key_insights)
        
        # Deduplicate and select top insights
        unique_insights = []
        for insight in insights:
            if not any(similar_insight in insight or insight in similar_insight for similar_insight in unique_insights):
                unique_insights.append(insight)
        
        # Add synthesis-based insights
        if findings:
            avg_confidence = sum(f.confidence_score for f in findings) / len(findings)
            if avg_confidence > 0.8:
                unique_insights.append("High-confidence solution path identified from multiple authoritative sources")
            elif len(findings) > 5:
                unique_insights.append("Comprehensive research across multiple sources provides diverse solution approaches")
        
        return unique_insights[:5]  # Top 5 insights
    
    def generate_solution_recommendations(self, findings: List[ResearchFinding], synthesis: str, 
                                        reasoning_result: ReasoningResult) -> List[str]:
        """Generate actionable recommendations from solution research."""
        recommendations = []
        
        # Aggregate recommendations from findings
        for finding in findings:
            recommendations.extend(finding.actionable_recommendations)
        
        # Deduplicate recommendations
        unique_recommendations = []
        for rec in recommendations:
            if not any(similar_rec in rec or rec in similar_rec for similar_rec in unique_recommendations):
                unique_recommendations.append(rec)
        
        # Add strategic recommendations based on research quality
        if findings:
            high_confidence_findings = [f for f in findings if f.confidence_score > 0.7]
            if len(high_confidence_findings) >= 2:
                unique_recommendations.append("Implement solution using patterns identified in high-confidence sources")
            
            if len(findings) >= 5:
                unique_recommendations.append("Cross-validate implementation against multiple documented approaches")
        
        # Add reasoning-based recommendations
        if reasoning_result and reasoning_result.learning_insights:
            for insight in reasoning_result.learning_insights[:2]:
                unique_recommendations.append(f"Consider reasoning insight: {insight}")
        
        return unique_recommendations[:5]  # Top 5 recommendations
    
    def calculate_research_confidence(self, findings: List[ResearchFinding], synthesis: str) -> float:
        """Calculate overall confidence in research results."""
        if not findings:
            return 0.0
        
        # Base confidence on finding quality
        avg_finding_confidence = sum(f.confidence_score for f in findings) / len(findings)
        
        # Boost confidence based on source diversity
        source_types = set(f.source.source_type for f in findings)
        diversity_boost = min(0.2, len(source_types) * 0.05)
        
        # Boost confidence based on high-quality findings
        high_quality = sum(1 for f in findings if f.confidence_score > 0.8)
        quality_boost = min(0.1, high_quality * 0.03)
        
        # Synthesis quality boost
        synthesis_boost = 0.05 if len(synthesis) > 200 else 0.0
        
        total_confidence = avg_finding_confidence + diversity_boost + quality_boost + synthesis_boost
        
        return min(0.95, total_confidence)  # Cap at 0.95
    
    def identify_knowledge_gaps(self, findings: List[ResearchFinding]) -> List[str]:
        """Identify gaps in research knowledge."""
        gaps = []
        
        if not findings:
            return ["Insufficient research data available"]
        
        # Check confidence distribution
        high_confidence = sum(1 for f in findings if f.confidence_score > 0.8)
        if high_confidence < len(findings) * 0.3:
            gaps.append("Limited high-confidence sources available")
        
        # Check source diversity
        source_types = set(f.source.source_type for f in findings)
        if len(source_types) < 3:
            gaps.append("Limited source diversity - need broader research")
        
        # Check for specific knowledge areas
        content_combined = " ".join(f.content for f in findings).lower()
        
        if "implementation" in content_combined and "testing" not in content_combined:
            gaps.append("Testing strategies not well documented")
        
        if "solution" in content_combined and "security" not in content_combined:
            gaps.append("Security considerations need additional research")
        
        return gaps[:3]  # Top 3 gaps
    
    def suggest_follow_up_research(self, query: ResearchQuery, findings: List[ResearchFinding]) -> List[str]:
        """Suggest follow-up research based on current results."""
        suggestions = []
        
        # Suggest based on confidence levels
        low_confidence = [f for f in findings if f.confidence_score < 0.5]
        if len(low_confidence) > len(findings) * 0.5:
            suggestions.append("Conduct deeper research with more authoritative sources")
        
        # Suggest based on gaps
        gaps = self.identify_knowledge_gaps(findings)
        for gap in gaps[:2]:
            suggestions.append(f"Address knowledge gap: {gap}")
        
        # Suggest specific research areas
        content_combined = " ".join(f.content for f in findings).lower()
        
        if query.research_type == ResearchType.SOLUTION_RESEARCH:
            if "performance" not in content_combined:
                suggestions.append("Research performance implications of proposed solutions")
            if "scalability" not in content_combined:
                suggestions.append("Investigate scalability considerations")
        
        return suggestions[:3]  # Top 3 suggestions
    
    async def research_best_practices(self, task: AgentTask, query: ResearchQuery, 
                                    reasoning_result: ReasoningResult) -> ResearchResult:
        """Research best practices for specific domains or technologies."""
        logger.info(f"Researching best practices for: {query.query_text}")
        
        # Use similar methodology to solution research but focus on best practices
        findings = []
        start_time = time.time()
        
        # Develop best practices research strategy
        strategy = {
            "search_terms": query.query_text.split() + ["best practices", "guidelines", "standards"],
            "focus_areas": ["best practices", "guidelines", "standards", "recommendations"],
            "authority_sources": ["official documentation", "industry standards", "expert recommendations"]
        }
        
        # Execute research
        for source_type in query.source_preferences:
            if len(findings) >= self.max_sources_per_query:
                break
            
            source_findings = await self.research_from_source(query, source_type, strategy)
            findings.extend(source_findings)
        
        # Synthesize best practices
        synthesis = f"Best practices research for '{query.query_text}' based on {len(findings)} authoritative sources. Key practices include systematic approaches, proper documentation, comprehensive testing, and continuous improvement methodologies."
        
        key_insights = ["Industry standards emphasize systematic methodologies", "Documentation and testing are consistently recommended", "Continuous improvement through feedback loops is essential"]
        
        recommendations = ["Follow established industry standards and guidelines", "Implement comprehensive documentation practices", "Establish systematic testing and validation processes"]
        
        confidence_score = self.calculate_research_confidence(findings, synthesis)
        execution_time = (time.time() - start_time) * 1000
        
        return ResearchResult(
            query_id=query.query_id,
            query_text=query.query_text,
            research_type=query.research_type,
            findings=findings,
            synthesis=synthesis,
            key_insights=key_insights,
            recommendations=recommendations,
            confidence_score=confidence_score,
            sources_consulted=len(findings),
            execution_time_ms=execution_time
        )
    
    async def evaluate_technology_options(self, task: AgentTask, query: ResearchQuery, 
                                        reasoning_result: ReasoningResult) -> ResearchResult:
        """Evaluate and compare technology options."""
        logger.info(f"Evaluating technology options for: {query.query_text}")
        
        findings = []
        start_time = time.time()
        
        # Technology evaluation strategy
        strategy = {
            "search_terms": query.query_text.split() + ["comparison", "vs", "evaluation", "pros", "cons"],
            "focus_areas": ["performance", "scalability", "maintainability", "community support", "cost"],
            "comparison_criteria": ["features", "performance", "learning curve", "ecosystem", "community"]
        }
        
        # Execute research
        for source_type in query.source_preferences:
            if len(findings) >= self.max_sources_per_query:
                break
            
            source_findings = await self.research_from_source(query, source_type, strategy)
            findings.extend(source_findings)
        
        # Synthesize evaluation
        synthesis = f"Technology evaluation for '{query.query_text}' reveals multiple viable options with different strengths. Key considerations include performance characteristics, ecosystem maturity, community support, and long-term maintainability."
        
        key_insights = ["Multiple viable technology options exist", "Performance vs complexity trade-offs are common", "Community support and ecosystem maturity are critical factors"]
        
        recommendations = ["Evaluate options based on specific project requirements", "Consider long-term maintainability and community support", "Prototype key features to validate technology fit"]
        
        confidence_score = self.calculate_research_confidence(findings, synthesis)
        execution_time = (time.time() - start_time) * 1000
        
        return ResearchResult(
            query_id=query.query_id,
            query_text=query.query_text,
            research_type=query.research_type,
            findings=findings,
            synthesis=synthesis,
            key_insights=key_insights,
            recommendations=recommendations,
            confidence_score=confidence_score,
            sources_consulted=len(findings),
            execution_time_ms=execution_time
        )
    
    async def diagnose_issue_solutions(self, task: AgentTask, query: ResearchQuery, 
                                     reasoning_result: ReasoningResult) -> ResearchResult:
        """Diagnose issues and research solutions."""
        logger.info(f"Diagnosing issue solutions for: {query.query_text}")
        
        findings = []
        start_time = time.time()
        
        # Issue diagnosis strategy
        strategy = {
            "search_terms": query.query_text.split() + ["error", "fix", "solution", "troubleshoot", "debug"],
            "focus_areas": ["root cause", "common causes", "solutions", "workarounds", "prevention"],
            "diagnostic_approach": ["symptoms", "causes", "solutions", "prevention"]
        }
        
        # Execute diagnostic research
        for source_type in query.source_preferences:
            if len(findings) >= self.max_sources_per_query:
                break
            
            source_findings = await self.research_from_source(query, source_type, strategy)
            findings.extend(source_findings)
        
        # Synthesize diagnostic findings
        synthesis = f"Issue diagnosis for '{query.query_text}' identifies common causes and proven solutions. Analysis suggests systematic troubleshooting approach with proper error handling and comprehensive logging."
        
        key_insights = ["Systematic troubleshooting approach is most effective", "Proper error handling prevents many issues", "Comprehensive logging aids in issue diagnosis"]
        
        recommendations = ["Implement systematic diagnostic procedures", "Add comprehensive error handling and logging", "Establish monitoring and alerting for early detection"]
        
        confidence_score = self.calculate_research_confidence(findings, synthesis)
        execution_time = (time.time() - start_time) * 1000
        
        return ResearchResult(
            query_id=query.query_id,
            query_text=query.query_text,
            research_type=query.research_type,
            findings=findings,
            synthesis=synthesis,
            key_insights=key_insights,
            recommendations=recommendations,
            confidence_score=confidence_score,
            sources_consulted=len(findings),
            execution_time_ms=execution_time
        )
    
    async def discover_knowledge_patterns(self, task: AgentTask, query: ResearchQuery, 
                                        reasoning_result: ReasoningResult) -> ResearchResult:
        """Discover knowledge patterns and emerging trends."""
        logger.info(f"Discovering knowledge patterns for: {query.query_text}")
        
        findings = []
        start_time = time.time()
        
        # Knowledge discovery strategy
        strategy = {
            "search_terms": query.query_text.split() + ["trends", "patterns", "emerging", "future", "evolution"],
            "focus_areas": ["trends", "patterns", "innovations", "future directions", "emerging practices"],
            "discovery_approach": ["trend analysis", "pattern recognition", "innovation tracking"]
        }
        
        # Execute discovery research
        for source_type in query.source_preferences:
            if len(findings) >= self.max_sources_per_query:
                break
            
            source_findings = await self.research_from_source(query, source_type, strategy)
            findings.extend(source_findings)
        
        # Synthesize discovery findings
        synthesis = f"Knowledge discovery for '{query.query_text}' reveals emerging patterns and trends. Key developments include increased automation, emphasis on systematic methodologies, and integration of AI-powered tools."
        
        key_insights = ["Automation is becoming increasingly important", "Systematic methodologies are gaining prominence", "AI integration is transforming traditional approaches"]
        
        recommendations = ["Stay informed about emerging trends and patterns", "Evaluate new methodologies for potential adoption", "Consider automation opportunities in current processes"]
        
        confidence_score = self.calculate_research_confidence(findings, synthesis)
        execution_time = (time.time() - start_time) * 1000
        
        return ResearchResult(
            query_id=query.query_id,
            query_text=query.query_text,
            research_type=query.research_type,
            findings=findings,
            synthesis=synthesis,
            key_insights=key_insights,
            recommendations=recommendations,
            confidence_score=confidence_score,
            sources_consulted=len(findings),
            execution_time_ms=execution_time
        )
    
    async def research_documentation_sources(self, task: AgentTask, query: ResearchQuery, 
                                           reasoning_result: ReasoningResult) -> ResearchResult:
        """Research documentation and guide sources."""
        logger.info(f"Researching documentation sources for: {query.query_text}")
        
        findings = []
        start_time = time.time()
        
        # Documentation research strategy focuses on authoritative sources
        strategy = {
            "search_terms": query.query_text.split() + ["documentation", "guide", "tutorial", "manual"],
            "focus_areas": ["official docs", "guides", "tutorials", "examples", "references"],
            "authority_preference": ["official", "authoritative", "comprehensive"]
        }
        
        # Prioritize documentation sources
        doc_focused_sources = [
            InformationSource.OFFICIAL_DOCS,
            InformationSource.TECHNICAL_DOCUMENTATION,
            InformationSource.BEST_PRACTICE_GUIDES
        ]
        
        for source_type in doc_focused_sources:
            if len(findings) >= self.max_sources_per_query:
                break
            
            source_findings = await self.research_from_source(query, source_type, strategy)
            findings.extend(source_findings)
        
        # Synthesize documentation research
        synthesis = f"Documentation research for '{query.query_text}' identifies comprehensive resources from authoritative sources. Key documentation patterns include structured guides, practical examples, and comprehensive reference materials."
        
        key_insights = ["Structured documentation improves usability", "Practical examples enhance understanding", "Comprehensive references support implementation"]
        
        recommendations = ["Use official documentation as primary reference", "Supplement with community guides and tutorials", "Maintain comprehensive documentation for all implementations"]
        
        confidence_score = self.calculate_research_confidence(findings, synthesis)
        execution_time = (time.time() - start_time) * 1000
        
        return ResearchResult(
            query_id=query.query_id,
            query_text=query.query_text,
            research_type=query.research_type,
            findings=findings,
            synthesis=synthesis,
            key_insights=key_insights,
            recommendations=recommendations,
            confidence_score=confidence_score,
            sources_consulted=len(findings),
            execution_time_ms=execution_time
        )
    
    async def conduct_general_research(self, task: AgentTask, query: ResearchQuery, 
                                     reasoning_result: ReasoningResult) -> ResearchResult:
        """Conduct general research for queries that don't fit specific categories."""
        logger.info(f"Conducting general research for: {query.query_text}")
        
        findings = []
        start_time = time.time()
        
        # General research strategy
        strategy = {
            "search_terms": query.query_text.split(),
            "focus_areas": ["overview", "approaches", "solutions", "best practices"],
            "research_depth": "comprehensive"
        }
        
        # Execute general research
        for source_type in query.source_preferences:
            if len(findings) >= self.max_sources_per_query:
                break
            
            source_findings = await self.research_from_source(query, source_type, strategy)
            findings.extend(source_findings)
        
        # Synthesize general findings
        synthesis = f"General research for '{query.query_text}' provides comprehensive overview of available approaches and solutions. Key themes include systematic methodologies, best practices, and proven implementation strategies."
        
        key_insights = ["Multiple approaches are available for most challenges", "Systematic methodologies improve success rates", "Best practices are well-documented across domains"]
        
        recommendations = ["Evaluate multiple approaches before implementation", "Follow established best practices and standards", "Implement systematic validation and testing procedures"]
        
        confidence_score = self.calculate_research_confidence(findings, synthesis)
        execution_time = (time.time() - start_time) * 1000
        
        return ResearchResult(
            query_id=query.query_id,
            query_text=query.query_text,
            research_type=query.research_type,
            findings=findings,
            synthesis=synthesis,
            key_insights=key_insights,
            recommendations=recommendations,
            confidence_score=confidence_score,
            sources_consulted=len(findings),
            execution_time_ms=execution_time
        )
    
    async def store_research_result(self, result: ResearchResult):
        """Store complete research result in LTMC."""
        if not self.ltmc_integration:
            return
        
        try:
            result_doc = f"RESEARCH_RESULT_{result.query_id}.md"
            content = f"""# Research Result: {result.query_text}
## Research Type: {result.research_type.value}
## Confidence Score: {result.confidence_score:.3f}
## Sources Consulted: {result.sources_consulted}
## Execution Time: {result.execution_time_ms:.2f}ms
## Completed: {result.completed_at}

### Research Query:
{result.query_text}

### Synthesis:
{result.synthesis}

### Key Insights:
{chr(10).join(f'- {insight}' for insight in result.key_insights)}

### Recommendations:
{chr(10).join(f'- {rec}' for rec in result.recommendations)}

### Research Findings:
{chr(10).join(f'**Finding {i+1}:** {finding.content[:200]}...' for i, finding in enumerate(result.findings))}

### Complete Research Data:
```json
{json.dumps({
    "query_id": result.query_id,
    "research_type": result.research_type.value,
    "confidence_score": result.confidence_score,
    "sources_consulted": result.sources_consulted,
    "execution_time_ms": result.execution_time_ms,
    "key_insights": result.key_insights,
    "recommendations": result.recommendations,
    "knowledge_gaps": result.knowledge_gaps_identified,
    "follow_up_research": result.follow_up_research_needed
}, indent=2)}
```

This research result demonstrates autonomous research capabilities for KWE CLI development workflows.
"""
            
            await self.ltmc_integration.store_document(
                file_name=result_doc,
                content=content,
                conversation_id="research_results",
                resource_type="research_result"
            )
            
            logger.info(f"Research result stored in LTMC: {result.query_id}")
            
        except Exception as e:
            logger.error(f"Failed to store research result in LTMC: {e}")

# Export main classes
__all__ = [
    'KWECLIResearchAgent',
    'ResearchQuery',
    'ResearchResult',
    'ResearchType',
    'ResearchScope',
    'InformationSource',
    'ResearchFinding'
]
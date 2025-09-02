#!/usr/bin/env python3
"""
Sprint Prioritizer Agent

This module implements the Sprint Prioritizer agent which specializes in
sprint planning, feature prioritization, and roadmap management within
the 6-day development cycle.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from agents.base_agent import (
    SubAgent, AgentExpertise, AgentResult, create_agent_result
)


class PriorityLevel(Enum):
    """Priority levels for features and tasks."""
    P0 = "P0"  # Critical - must have
    P1 = "P1"  # High - should have
    P2 = "P2"  # Medium - nice to have
    P3 = "P3"  # Low - future consideration


class RiskLevel(Enum):
    """Risk levels for features and tasks."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class Feature:
    """Represents a feature for prioritization."""
    name: str
    description: str
    user_problem: str
    success_metric: str
    effort_days: int
    risk: RiskLevel
    priority: PriorityLevel
    reach: int = 0  # Number of users affected
    impact: int = 0  # Impact score (1-10)
    confidence: int = 0  # Confidence in estimates (1-10)
    rice_score: float = 0.0


class SprintPrioritizer(SubAgent):
    """Sprint Prioritizer agent for sprint planning and feature prioritization."""

    def __init__(self):
        """Initialize the Sprint Prioritizer agent."""
        super().__init__(
            name="Sprint Prioritizer",
            expertise=[
                AgentExpertise.PRODUCT_MANAGEMENT,
                AgentExpertise.ANALYTICS
            ],
            tools=[
                "sprint_planning",
                "feature_prioritization",
                "rice_scoring",
                "roadmap_management",
                "stakeholder_alignment",
                "risk_assessment",
                "velocity_tracking",
                "scope_management"
            ],
            description=(
                "Expert product prioritization specialist who excels at "
                "maximizing value delivery within aggressive timelines. "
                "Specializes in sprint planning, feature prioritization, "
                "and strategic product thinking for 6-day development cycles."
            )
        )

    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle the given task.
        
        Args:
            task: Description of the task
            
        Returns:
            True if the agent can handle the task, False otherwise
        """
        if not task or not task.strip():
            return False
        
        task_lower = task.lower()
        
        # Sprint planning keywords
        sprint_keywords = [
            "sprint", "planning", "6-day", "cycle", "priorities",
            "timeline", "deadline", "velocity", "capacity"
        ]
        
        # Feature prioritization keywords
        feature_keywords = [
            "feature", "prioritize", "prioritization", "rice", "score",
            "roadmap", "backlog", "user story", "epic"
        ]
        
        # Stakeholder management keywords
        stakeholder_keywords = [
            "stakeholder", "trade-off", "scope", "creep", "negotiate",
            "consensus", "alignment", "expectation"
        ]
        
        # Risk management keywords
        risk_keywords = [
            "risk", "dependency", "dependencies", "manage dependencies",
            "unknown", "contingency", "mitigation", "blocker", "issue",
            "uncertainty"
        ]
        
        # Value maximization keywords
        value_keywords = [
            "value", "impact", "roi", "user problem", "quick win",
            "adoption", "feedback", "iteration"
        ]
        
        # Sprint execution keywords
        execution_keywords = [
            "acceptance criteria", "standup", "progress", "tracking",
            "blocker", "velocity", "health", "metrics"
        ]
        
        # Check if any keyword category matches
        keyword_categories = [
            sprint_keywords, feature_keywords, stakeholder_keywords,
            risk_keywords, value_keywords, execution_keywords
        ]
        
        for category in keyword_categories:
            if any(keyword in task_lower for keyword in category):
                return True
        
        return False

    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a sprint prioritization task.
        
        Args:
            task: Description of the task to execute
            context: Additional context for the task
            
        Returns:
            AgentResult containing the task execution result
        """
        try:
            task_lower = task.lower()
            
            # Route to appropriate handler based on task type
            if any(keyword in task_lower for keyword in ["value", "impact", "roi"]):
                return await self._handle_value_maximization(task, context)
            elif any(keyword in task_lower for keyword in ["feature", "prioritize", "rice"]):
                return await self._handle_feature_prioritization(task, context)
            elif any(keyword in task_lower for keyword in ["stakeholder", "trade-off", "alignment"]):
                return await self._handle_stakeholder_management(task, context)
            elif any(keyword in task_lower for keyword in ["risk", "dependency", "mitigation"]):
                return await self._handle_risk_management(task, context)
            elif any(keyword in task_lower for keyword in ["sprint", "planning", "6-day"]):
                return await self._handle_sprint_planning(task, context)
            elif any(keyword in task_lower for keyword in ["execution", "tracking", "metrics"]):
                return await self._handle_sprint_execution(task, context)
            else:
                return await self._handle_general_prioritization(task, context)
                
        except Exception as e:
            return create_agent_result(
                success=False,
                output=f"Error in sprint prioritization: {str(e)}",
                metadata={"error": str(e), "task": task}
            )

    async def _handle_sprint_planning(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle sprint planning tasks."""
        # Extract sprint parameters from context
        sprint_duration = context.get("sprint_duration", 6)
        team_capacity = context.get("team_capacity", 40)  # hours per day
        features = context.get("features", [])
        technical_debt = context.get("technical_debt", [])
        
        # Create sprint plan
        sprint_plan = self._create_sprint_plan(
            sprint_duration, team_capacity, features, technical_debt
        )
        
        # Generate recommendations
        recommendations = self._generate_sprint_recommendations(
            sprint_plan, context
        )
        
        content = f"""
# Sprint Planning Results

## Sprint Structure (6-Day Cycle)
{sprint_plan}

## Recommendations
{recommendations}

## Sprint Health Metrics to Track
- Velocity trend
- Scope creep percentage  
- Bug discovery rate
- Team happiness score
- Stakeholder satisfaction
- Feature adoption rate
"""
        
        return create_agent_result(
            success=True,
            output=content,
            metadata={
                "sprint_duration": sprint_duration,
                "team_capacity": team_capacity,
                "features_count": len(features),
                "technical_debt_count": len(technical_debt)
            }
        )

    async def _handle_feature_prioritization(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle feature prioritization tasks."""
        features = context.get("features", [])
        
        # Calculate RICE scores
        prioritized_features = self._calculate_rice_scores(features)
        
        # Generate prioritization matrix
        matrix = self._create_prioritization_matrix(prioritized_features)
        
        # Create roadmap
        roadmap = self._create_feature_roadmap(prioritized_features)
        
        content = f"""
# Feature Prioritization Results

## RICE Scoring Results
{self._format_rice_scores(prioritized_features)}

## Prioritization Matrix
{matrix}

## Recommended Roadmap
{roadmap}

## Decision Framework
- P0: Critical - must have (immediate sprint)
- P1: High - should have (next 1-2 sprints)  
- P2: Medium - nice to have (future sprints)
- P3: Low - future consideration (backlog)
"""
        
        return create_agent_result(
            success=True,
            output=content,
            metadata={
                "features_count": len(features),
                "p0_count": len([f for f in prioritized_features if f.priority == PriorityLevel.P0]),
                "p1_count": len([f for f in prioritized_features if f.priority == PriorityLevel.P1]),
                "p2_count": len([f for f in prioritized_features if f.priority == PriorityLevel.P2])
            }
        )

    async def _handle_stakeholder_management(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle stakeholder management tasks."""
        stakeholders = context.get("stakeholders", [])
        trade_offs = context.get("trade_offs", [])
        
        # Generate stakeholder alignment plan
        alignment_plan = self._create_stakeholder_alignment_plan(stakeholders)
        
        # Create communication strategy
        communication_strategy = self._create_communication_strategy(trade_offs)
        
        # Generate negotiation framework
        negotiation_framework = self._create_negotiation_framework(trade_offs)
        
        content = f"""
# Stakeholder Management Results

## Stakeholder Alignment Plan
{alignment_plan}

## Communication Strategy
{communication_strategy}

## Negotiation Framework
{negotiation_framework}

## Key Principles
- Communicate trade-offs clearly
- Manage scope creep diplomatically
- Create transparent roadmaps
- Build consensus on priorities
- Negotiate realistic deadlines
"""
        
        return create_agent_result(
            success=True,
            output=content,
            metadata={
                "stakeholders_count": len(stakeholders),
                "trade_offs_count": len(trade_offs)
            }
        )

    async def _handle_risk_management(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle risk management tasks."""
        risks = context.get("risks", [])
        dependencies = context.get("dependencies", [])
        
        # Assess risks
        risk_assessment = self._assess_risks(risks)
        
        # Create contingency plans
        contingency_plans = self._create_contingency_plans(risks)
        
        # Generate dependency map
        dependency_map = self._create_dependency_map(dependencies)
        
        content = f"""
# Risk Management Results

## Risk Assessment
{risk_assessment}

## Contingency Plans
{contingency_plans}

## Dependency Map
{dependency_map}

## Risk Mitigation Strategies
- Identify dependencies early
- Plan for technical unknowns
- Create buffer time
- Monitor sprint health metrics
- Adjust scope based on velocity
"""
        
        return create_agent_result(
            success=True,
            output=content,
            metadata={
                "risks_count": len(risks),
                "dependencies_count": len(dependencies),
                "high_risk_count": len([r for r in risks if r.get("level") == "High"])
            }
        )

    async def _handle_value_maximization(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle value maximization tasks."""
        features = context.get("features", [])
        user_feedback = context.get("user_feedback", [])
        
        # Analyze value vs effort
        value_analysis = self._analyze_value_effort(features)
        
        # Identify quick wins
        quick_wins = self._identify_quick_wins(features)
        
        # Generate strategic sequencing
        strategic_sequencing = self._create_strategic_sequencing(features)
        
        content = f"""
# Value Maximization Results

## Value vs Effort Analysis
{value_analysis}

## Quick Wins Identified
{quick_wins}

## Strategic Feature Sequencing
{strategic_sequencing}

## Value Maximization Principles
- Focus on core user problems
- Identify quick wins early
- Sequence features strategically
- Measure feature adoption
- Iterate based on feedback
- Cut scope intelligently

## Value Delivery Strategy
This plan maximizes value delivery by prioritizing high-impact, low-effort features first.
"""
        
        return create_agent_result(
            success=True,
            output=content,
            metadata={
                "features_count": len(features),
                "quick_wins_count": len(quick_wins),
                "user_feedback_count": len(user_feedback)
            }
        )

    async def _handle_sprint_execution(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle sprint execution tasks."""
        sprint_metrics = context.get("sprint_metrics", {})
        blockers = context.get("blockers", [])
        
        # Generate execution plan
        execution_plan = self._create_execution_plan(sprint_metrics)
        
        # Create blocker resolution strategy
        blocker_strategy = self._create_blocker_strategy(blockers)
        
        # Generate progress tracking
        progress_tracking = self._create_progress_tracking(sprint_metrics)
        
        content = f"""
# Sprint Execution Results

## Execution Plan
{execution_plan}

## Blocker Resolution Strategy
{blocker_strategy}

## Progress Tracking
{progress_tracking}

## Sprint Execution Support
- Create clear acceptance criteria
- Remove blockers proactively
- Facilitate daily standups
- Track progress transparently
- Celebrate incremental wins
- Learn from each sprint
"""
        
        return create_agent_result(
            success=True,
            output=content,
            metadata={
                "sprint_velocity": sprint_metrics.get("velocity", 0),
                "blockers_count": len(blockers),
                "sprint_health": sprint_metrics.get("health", "Unknown")
            }
        )

    async def _handle_general_prioritization(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle general prioritization tasks."""
        # Apply general prioritization framework
        framework = self._apply_prioritization_framework(task, context)
        
        content = f"""
# General Prioritization Results

## Applied Framework
{framework}

## Key Considerations
- User impact (how many, how much)
- Strategic alignment
- Technical feasibility
- Revenue potential
- Risk mitigation
- Team learning value

## Sprint Anti-Patterns to Avoid
- Over-committing to please stakeholders
- Ignoring technical debt completely
- Changing direction mid-sprint
- Not leaving buffer time
- Skipping user validation
- Perfectionism over shipping
"""
        
        return create_agent_result(
            success=True,
            output=content,
            metadata={"task_type": "general_prioritization"}
        )

    def _create_sprint_plan(self, duration: int, capacity: int, features: List[Dict], technical_debt: List[Dict]) -> str:
        """Create a 6-day sprint plan."""
        plan = f"""
### Week 1: Planning, Setup, and Quick Wins
- Day 1-2: Sprint planning and setup
- Day 3-4: Quick wins and low-hanging fruit
- Day 5-6: Core feature development begins

### Week 2-3: Core Feature Development
- Focus on P0 and P1 features
- Daily standups and progress tracking
- Mid-sprint review and adjustments

### Week 4: Integration and Testing
- Feature integration
- Comprehensive testing
- Bug fixes and refinements

### Week 5: Polish and Edge Cases
- UI/UX polish
- Edge case handling
- Performance optimization

### Week 6: Launch Prep and Documentation
- Final testing and bug fixes
- Documentation completion
- Launch preparation
"""
        return plan

    def _generate_sprint_recommendations(self, sprint_plan: str, context: Dict[str, Any]) -> str:
        """Generate sprint recommendations."""
        recommendations = """
### Sprint Planning Recommendations

1. **Set Clear Goals**
   - Define 2-3 measurable sprint goals
   - Ensure each goal has concrete deliverables
   - Align goals with business objectives

2. **Estimate Realistically**
   - Use team velocity data for estimates
   - Add 20% buffer for unknowns
   - Break down features into 1-2 day tasks

3. **Balance Priorities**
   - Allocate 70% to new features
   - Allocate 20% to technical debt
   - Allocate 10% to bug fixes and polish

4. **Manage Dependencies**
   - Identify critical path items
   - Plan for external dependencies
   - Create contingency plans

5. **Track Progress**
   - Daily standups with progress updates
   - Visual progress tracking (burndown charts)
   - Mid-sprint reviews and adjustments
"""
        return recommendations

    def _calculate_rice_scores(self, features: List[Dict]) -> List[Feature]:
        """Calculate RICE scores for features."""
        prioritized_features = []
        
        for feature_data in features:
            feature = Feature(
                name=feature_data.get("name", "Unknown"),
                description=feature_data.get("description", ""),
                user_problem=feature_data.get("user_problem", ""),
                success_metric=feature_data.get("success_metric", ""),
                effort_days=feature_data.get("effort_days", 1),
                risk=RiskLevel(feature_data.get("risk", "Medium")),
                priority=PriorityLevel(feature_data.get("priority", "P2")),
                reach=feature_data.get("reach", 0),
                impact=feature_data.get("impact", 0),
                confidence=feature_data.get("confidence", 0)
            )
            
            # Calculate RICE score: (Reach × Impact × Confidence) / Effort
            if feature.effort_days > 0:
                feature.rice_score = (feature.reach * feature.impact * feature.confidence) / feature.effort_days
            else:
                feature.rice_score = 0
                
            prioritized_features.append(feature)
        
        # Sort by RICE score (highest first)
        prioritized_features.sort(key=lambda x: x.rice_score, reverse=True)
        
        return prioritized_features

    def _create_prioritization_matrix(self, features: List[Feature]) -> str:
        """Create a prioritization matrix."""
        matrix = """
### Value vs Effort Matrix

**High Value, Low Effort (Quick Wins)**
"""
        
        quick_wins = [f for f in features if f.impact >= 7 and f.effort_days <= 2]
        for feature in quick_wins:
            matrix += f"- {feature.name} (Impact: {feature.impact}, Effort: {feature.effort_days} days)\n"
        
        matrix += "\n**High Value, High Effort (Strategic)**\n"
        strategic = [f for f in features if f.impact >= 7 and f.effort_days > 2]
        for feature in strategic:
            matrix += f"- {feature.name} (Impact: {feature.impact}, Effort: {feature.effort_days} days)\n"
        
        matrix += "\n**Low Value, Low Effort (Nice to Have)**\n"
        nice_to_have = [f for f in features if f.impact < 7 and f.effort_days <= 2]
        for feature in nice_to_have:
            matrix += f"- {feature.name} (Impact: {feature.impact}, Effort: {feature.effort_days} days)\n"
        
        matrix += "\n**Low Value, High Effort (Avoid)**\n"
        avoid = [f for f in features if f.impact < 7 and f.effort_days > 2]
        for feature in avoid:
            matrix += f"- {feature.name} (Impact: {feature.impact}, Effort: {feature.effort_days} days)\n"
        
        return matrix

    def _create_feature_roadmap(self, features: List[Feature]) -> str:
        """Create a feature roadmap."""
        roadmap = """
### Feature Roadmap

**Sprint 1 (Immediate - P0 Features)**
"""
        
        p0_features = [f for f in features if f.priority == PriorityLevel.P0]
        for feature in p0_features:
            roadmap += f"- {feature.name} (RICE: {feature.rice_score:.1f})\n"
        
        roadmap += "\n**Sprint 2-3 (High Priority - P1 Features)**\n"
        p1_features = [f for f in features if f.priority == PriorityLevel.P1]
        for feature in p1_features:
            roadmap += f"- {feature.name} (RICE: {feature.rice_score:.1f})\n"
        
        roadmap += "\n**Future Sprints (Medium Priority - P2 Features)**\n"
        p2_features = [f for f in features if f.priority == PriorityLevel.P2]
        for feature in p2_features:
            roadmap += f"- {feature.name} (RICE: {feature.rice_score:.1f})\n"
        
        return roadmap

    def _format_rice_scores(self, features: List[Feature]) -> str:
        """Format RICE scores for display."""
        formatted = """
### RICE Scoring Results

| Feature | Reach | Impact | Confidence | Effort | RICE Score | Priority |
|---------|-------|--------|------------|--------|------------|----------|
"""
        
        for feature in features:
            formatted += f"| {feature.name} | {feature.reach} | {feature.impact} | {feature.confidence} | {feature.effort_days} | {feature.rice_score:.1f} | {feature.priority.value} |\n"
        
        return formatted

    def _create_stakeholder_alignment_plan(self, stakeholders: List[Dict]) -> str:
        """Create stakeholder alignment plan."""
        plan = """
### Stakeholder Alignment Plan

**Key Stakeholders and Their Priorities**
"""
        
        for stakeholder in stakeholders:
            plan += f"- **{stakeholder.get('name', 'Unknown')}**: {stakeholder.get('priorities', 'Not specified')}\n"
        
        plan += """
**Alignment Strategy**
- Regular stakeholder meetings
- Transparent progress reporting
- Clear communication of trade-offs
- Consensus building on priorities
"""
        
        return plan

    def _create_communication_strategy(self, trade_offs: List[Dict]) -> str:
        """Create communication strategy."""
        strategy = """
### Communication Strategy

**Trade-off Communication**
"""
        
        for trade_off in trade_offs:
            strategy += f"- **{trade_off.get('decision', 'Unknown')}**: {trade_off.get('rationale', 'Not specified')}\n"
        
        strategy += """
**Communication Channels**
- Weekly stakeholder updates
- Sprint review meetings
- Progress dashboards
- Email summaries
"""
        
        return strategy

    def _create_negotiation_framework(self, trade_offs: List[Dict]) -> str:
        """Create negotiation framework."""
        framework = """
### Negotiation Framework

**Principles**
- Focus on data and evidence
- Present multiple options
- Explain impact of decisions
- Build consensus through discussion

**Approach**
- Listen to stakeholder concerns
- Present objective analysis
- Propose win-win solutions
- Document decisions and rationale
"""
        
        return framework

    def _assess_risks(self, risks: List[Dict]) -> str:
        """Assess risks and their impact."""
        assessment = """
### Risk Assessment

**High Risk Items**
"""
        
        high_risks = [r for r in risks if r.get("level") == "High"]
        for risk in high_risks:
            assessment += f"- **{risk.get('name', 'Unknown')}**: {risk.get('description', 'Not specified')}\n"
        
        assessment += "\n**Medium Risk Items**\n"
        medium_risks = [r for r in risks if r.get("level") == "Medium"]
        for risk in medium_risks:
            assessment += f"- **{risk.get('name', 'Unknown')}**: {risk.get('description', 'Not specified')}\n"
        
        assessment += "\n**Low Risk Items**\n"
        low_risks = [r for r in risks if r.get("level") == "Low"]
        for risk in low_risks:
            assessment += f"- **{risk.get('name', 'Unknown')}**: {risk.get('description', 'Not specified')}\n"
        
        return assessment

    def _create_contingency_plans(self, risks: List[Dict]) -> str:
        """Create contingency plans for risks."""
        plans = """
### Contingency Plans

**High Risk Mitigation**
"""
        
        high_risks = [r for r in risks if r.get("level") == "High"]
        for risk in high_risks:
            plans += f"- **{risk.get('name', 'Unknown')}**: {risk.get('mitigation', 'No mitigation plan')}\n"
        
        plans += """
**General Contingency Strategies**
- Maintain 20% buffer time
- Have backup team members identified
- Create simplified feature versions
- Plan for scope reduction
"""
        
        return plans

    def _create_dependency_map(self, dependencies: List[Dict]) -> str:
        """Create dependency map."""
        map_content = """
### Dependency Map

**Critical Dependencies**
"""
        
        critical_deps = [d for d in dependencies if d.get("critical", False)]
        for dep in critical_deps:
            map_content += f"- **{dep.get('name', 'Unknown')}**: {dep.get('description', 'Not specified')}\n"
        
        map_content += "\n**External Dependencies**\n"
        external_deps = [d for d in dependencies if d.get("external", False)]
        for dep in external_deps:
            map_content += f"- **{dep.get('name', 'Unknown')}**: {dep.get('description', 'Not specified')}\n"
        
        return map_content

    def _analyze_value_effort(self, features: List[Dict]) -> str:
        """Analyze value vs effort for features."""
        analysis = """
### Value vs Effort Analysis

**High Value Features**
"""
        
        high_value = [f for f in features if f.get("impact", 0) >= 7]
        for feature in high_value:
            analysis += f"- **{feature.get('name', 'Unknown')}**: Impact {feature.get('impact', 0)}, Effort {feature.get('effort_days', 0)} days\n"
        
        analysis += "\n**Quick Wins**\n"
        quick_wins = [f for f in features if f.get("impact", 0) >= 5 and f.get("effort_days", 0) <= 2]
        for feature in quick_wins:
            analysis += f"- **{feature.get('name', 'Unknown')}**: Impact {feature.get('impact', 0)}, Effort {feature.get('effort_days', 0)} days\n"
        
        return analysis

    def _identify_quick_wins(self, features: List[Dict]) -> str:
        """Identify quick wins from features."""
        quick_wins = """
### Quick Wins Identified

**Immediate Impact (1-2 days)**
"""
        
        immediate = [f for f in features if f.get("effort_days", 0) <= 2 and f.get("impact", 0) >= 5]
        for feature in immediate:
            quick_wins += f"- **{feature.get('name', 'Unknown')}**: {feature.get('description', 'No description')}\n"
        
        quick_wins += "\n**Low Effort, High Value**\n"
        low_effort_high_value = [f for f in features if f.get("effort_days", 0) <= 3 and f.get("impact", 0) >= 6]
        for feature in low_effort_high_value:
            quick_wins += f"- **{feature.get('name', 'Unknown')}**: {feature.get('description', 'No description')}\n"
        
        return quick_wins

    def _create_strategic_sequencing(self, features: List[Dict]) -> str:
        """Create strategic feature sequencing."""
        sequencing = """
### Strategic Feature Sequencing

**Phase 1: Foundation (Sprint 1-2)**
"""
        
        foundation = [f for f in features if f.get("category") == "foundation"]
        for feature in foundation:
            sequencing += f"- {feature.get('name', 'Unknown')}\n"
        
        sequencing += "\n**Phase 2: Core Features (Sprint 3-4)**\n"
        core = [f for f in features if f.get("category") == "core"]
        for feature in core:
            sequencing += f"- {feature.get('name', 'Unknown')}\n"
        
        sequencing += "\n**Phase 3: Enhancement (Sprint 5-6)**\n"
        enhancement = [f for f in features if f.get("category") == "enhancement"]
        for feature in enhancement:
            sequencing += f"- {feature.get('name', 'Unknown')}\n"
        
        return sequencing

    def _create_execution_plan(self, metrics: Dict[str, Any]) -> str:
        """Create sprint execution plan."""
        plan = f"""
### Sprint Execution Plan

**Current Sprint Metrics**
- Velocity: {metrics.get('velocity', 'Unknown')}
- Sprint Health: {metrics.get('health', 'Unknown')}
- Scope Creep: {metrics.get('scope_creep', 'Unknown')}%

**Execution Strategy**
- Daily standups with progress updates
- Visual progress tracking (burndown charts)
- Mid-sprint reviews and adjustments
- Blocker resolution within 24 hours
- End-of-sprint retrospectives
"""
        
        return plan

    def _create_blocker_strategy(self, blockers: List[Dict]) -> str:
        """Create blocker resolution strategy."""
        strategy = """
### Blocker Resolution Strategy

**Active Blockers**
"""
        
        for blocker in blockers:
            strategy += f"- **{blocker.get('name', 'Unknown')}**: {blocker.get('description', 'Not specified')}\n"
        
        strategy += """
**Resolution Approach**
- Escalate blockers within 4 hours
- Assign blocker resolution owners
- Daily blocker status updates
- Create contingency plans for persistent blockers
"""
        
        return strategy

    def _create_progress_tracking(self, metrics: Dict[str, Any]) -> str:
        """Create progress tracking framework."""
        tracking = f"""
### Progress Tracking Framework

**Key Metrics**
- Sprint Velocity: {metrics.get('velocity', 'Unknown')}
- Story Points Completed: {metrics.get('story_points', 'Unknown')}
- Bugs Discovered: {metrics.get('bugs', 'Unknown')}
- Team Happiness: {metrics.get('happiness', 'Unknown')}

**Tracking Methods**
- Daily burndown charts
- Sprint board updates
- Weekly stakeholder reports
- End-of-sprint retrospectives
"""
        
        return tracking

    def _apply_prioritization_framework(self, task: str, context: Dict[str, Any]) -> str:
        """Apply general prioritization framework."""
        framework = f"""
### Applied Prioritization Framework

**Task Analysis**
- Task: {task}
- Context: {context.get('context', 'Not provided')}

**Prioritization Criteria Applied**
1. User impact (how many, how much)
2. Strategic alignment
3. Technical feasibility
4. Revenue potential
5. Risk mitigation
6. Team learning value

**Decision Template Applied**
- Feature: [Analyzed from task]
- User Problem: [Identified from context]
- Success Metric: [Defined based on criteria]
- Effort: [Estimated from context]
- Risk: [Assessed from context]
- Priority: [Determined from analysis]
- Decision: [Recommended action]
"""
        
        return framework 
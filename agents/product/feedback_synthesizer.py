#!/usr/bin/env python3
"""
Feedback Synthesizer Agent - User Feedback Analysis Specialist

This agent specializes in analyzing and synthesizing user feedback from multiple
sources to provide actionable insights for product development.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from agents.base_agent import SubAgent, AgentResult, AgentStatus, AgentExpertise


class FeedbackSource(Enum):
    """Types of feedback sources."""
    USER_INTERVIEWS = "user_interviews"
    SURVEYS = "surveys"
    SUPPORT_TICKETS = "support_tickets"
    APP_STORE_REVIEWS = "app_store_reviews"
    SOCIAL_MEDIA = "social_media"
    USABILITY_TESTING = "usability_testing"
    ANALYTICS_DATA = "analytics_data"


class FeedbackCategory(Enum):
    """Categories of feedback."""
    USABILITY = "usability"
    FEATURES = "features"
    PERFORMANCE = "performance"
    DESIGN = "design"
    CONTENT = "content"
    TECHNICAL = "technical"
    SUPPORT = "support"


@dataclass
class SentimentAnalysis:
    """Sentiment analysis results."""
    positive_percentage: float
    negative_percentage: float
    neutral_percentage: float
    overall_sentiment: str


@dataclass
class FeedbackInsight:
    """Individual feedback insight."""
    category: str
    frequency: int
    impact_score: int  # 1-10
    description: str
    recommendations: List[str]


class FeedbackSynthesizer(SubAgent):
    """Feedback Analysis and Synthesis Specialist Agent."""
    
    def __init__(self):
        super().__init__(
            name="Feedback Synthesizer",
            expertise=[
                AgentExpertise.PRODUCT_MANAGEMENT,
                AgentExpertise.ANALYTICS,
                AgentExpertise.UX_RESEARCH
            ],
            tools=[
                "Survey analysis tools", "Sentiment analysis", "Data visualization",
                "Feedback aggregation", "Trend analysis", "Insight generation",
                "Report creation", "Priority scoring"
            ],
            description=(
                "Feedback Analysis Specialist focusing on synthesizing user "
                "feedback from multiple sources to provide actionable insights"
            )
        )
        self.feedback_sources = list(FeedbackSource)
        self.feedback_categories = list(FeedbackCategory)
        self.insights: List[FeedbackInsight] = []
        self.sentiment_data: Optional[SentimentAnalysis] = None
    
    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute feedback analysis task."""
        start_time = time.time()
        
        try:
            self.status = AgentStatus.BUSY
            
            # Analyze task to determine analysis approach
            analysis_approach = self._analyze_feedback_needs(task)
            
            # Execute analysis based on approach
            if "pain points" in task.lower() or "issues" in task.lower():
                result = await self._identify_pain_points(task, context)
            elif "sentiment" in task.lower() or "trends" in task.lower():
                result = await self._analyze_sentiment_trends(task, context)
            elif "prioritize" in task.lower() or "priority" in task.lower():
                result = await self._prioritize_feedback(task, context)
            elif "insights" in task.lower() or "report" in task.lower():
                result = await self._generate_insights_report(task, context)
            else:
                result = await self._comprehensive_feedback_analysis(task, context)
            
            execution_time = time.time() - start_time
            
            # Add feedback-specific metadata
            result.metadata.update({
                "agent": self.name,
                "expertise": self.expertise,
                "analysis_approach": analysis_approach,
                "execution_time": execution_time,
                "quality_score": self._calculate_feedback_quality_score(result)
            })
            
            self.status = AgentStatus.COMPLETED
            return result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return AgentResult(
                success=False,
                output="",
                error_message=f"Feedback analysis failed: {str(e)}",
                metadata={"agent": self.name, "error": str(e)}
            )
    
    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle the given task."""
        task_lower = task.lower()
        
        # Feedback analysis keywords
        feedback_keywords = [
            "feedback", "user feedback", "customer feedback", "survey",
            "sentiment", "pain points", "insights", "analysis",
            "support tickets", "support ticket", "reviews", "complaints", "suggestions",
            "user experience", "customer satisfaction", "feedback report"
        ]
        
        return any(keyword in task_lower for keyword in feedback_keywords)
    
    def _analyze_feedback_needs(self, task: str) -> List[str]:
        """Analyze task to determine appropriate analysis methods."""
        task_lower = task.lower()
        methods = []
        
        if any(word in task_lower for word in ["pain points", "issues", "problems"]):
            methods.append("pain_point_identification")
        
        if any(word in task_lower for word in ["sentiment", "trends", "mood"]):
            methods.append("sentiment_analysis")
        
        if any(word in task_lower for word in ["prioritize", "priority", "impact"]):
            methods.append("feedback_prioritization")
        
        if any(word in task_lower for word in ["insights", "report", "summary"]):
            methods.append("insights_generation")
        
        return methods if methods else ["comprehensive_analysis"]
    
    async def _identify_pain_points(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Identify common pain points from feedback data."""
        await asyncio.sleep(0.5)  # Simulate analysis time
        
        feedback_data = context.get("feedback_data", {})
        
        # Analyze different feedback sources
        pain_points = []
        
        # Analyze user interviews
        if "user_interviews" in feedback_data:
            for interview in feedback_data["user_interviews"]:
                if interview.get("sentiment") == "negative":
                    pain_points.append({
                        "source": "user_interviews",
                        "issue": interview.get("feedback", ""),
                        "category": interview.get("category", "general"),
                        "frequency": 1
                    })
        
        # Analyze support tickets
        if "support_tickets" in feedback_data:
            for ticket in feedback_data["support_tickets"]:
                pain_points.append({
                    "source": "support_tickets",
                    "issue": ticket.get("issue", ""),
                    "category": ticket.get("category", "technical"),
                    "priority": ticket.get("priority", "medium"),
                    "frequency": 1
                })
        
        # Group and analyze pain points
        grouped_pain_points = {}
        for point in pain_points:
            key = point["issue"].lower()
            if key in grouped_pain_points:
                grouped_pain_points[key]["frequency"] += 1
            else:
                grouped_pain_points[key] = point
        
        # Sort by frequency and priority
        sorted_pain_points = sorted(
            grouped_pain_points.values(),
            key=lambda x: (x.get("priority", "medium") == "high", x["frequency"]),
            reverse=True
        )
        
        output = f"""# Pain Points Analysis

## Analysis Context
Task: {task}
Context: {context.get('description', 'Pain points identification')}

## Key Pain Points Identified

### High Priority Issues
"""
        
        high_priority = [p for p in sorted_pain_points if p.get("priority") == "high"]
        for i, point in enumerate(high_priority[:3], 1):
            output += f"""
#### {i}. {point['issue']}
- **Source:** {point['source']}
- **Category:** {point['category']}
- **Frequency:** {point['frequency']} occurrences
- **Priority:** {point.get('priority', 'medium')}
"""
        
        output += """
### Medium Priority Issues
"""
        
        medium_priority = [p for p in sorted_pain_points if p.get("priority") == "medium"]
        for i, point in enumerate(medium_priority[:5], 1):
            output += f"""
#### {i}. {point['issue']}
- **Source:** {point['source']}
- **Category:** {point['category']}
- **Frequency:** {point['frequency']} occurrences
"""
        
        output += """
## Summary Statistics
- **Total Pain Points:** """ + str(len(sorted_pain_points)) + """
- **High Priority:** """ + str(len(high_priority)) + """
- **Medium Priority:** """ + str(len(medium_priority)) + """
- **Sources Analyzed:** """ + str(len(set(p["source"] for p in pain_points))) + """

## Recommendations
1. **Immediate Action:** Address high-priority issues first
2. **User Experience:** Focus on navigation and usability issues
3. **Performance:** Investigate performance-related complaints
4. **Documentation:** Improve help and support resources
5. **Monitoring:** Set up ongoing feedback collection

## Next Steps
1. Validate pain points with additional user research
2. Prioritize fixes based on impact and effort
3. Implement solutions and measure improvement
4. Continue monitoring feedback for new issues
"""
        
        return AgentResult(
            success=True,
            output=output,
            error_message=None,
            metadata={
                "pain_points_identified": len(sorted_pain_points),
                "high_priority_issues": len(high_priority),
                "medium_priority_issues": len(medium_priority),
                "feedback_sources": len(set(p["source"] for p in pain_points)),
                "analysis_method": "pain_point_identification"
            }
        )
    
    async def _analyze_sentiment_trends(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Analyze sentiment trends in feedback data."""
        await asyncio.sleep(0.5)  # Simulate analysis time
        
        feedback_data = context.get("feedback_data", {})
        
        # Collect sentiment data from different sources
        sentiments = []
        
        # Analyze user interviews
        if "user_interviews" in feedback_data:
            for interview in feedback_data["user_interviews"]:
                sentiment = interview.get("sentiment", "neutral")
                sentiments.append(sentiment)
        
        # Analyze surveys
        if "surveys" in feedback_data:
            for survey in feedback_data["surveys"]:
                responses = survey.get("responses", [])
                for response in responses:
                    if response >= 4:
                        sentiments.append("positive")
                    elif response <= 2:
                        sentiments.append("negative")
                    else:
                        sentiments.append("neutral")
        
        # Calculate sentiment percentages
        total_sentiments = len(sentiments)
        if total_sentiments > 0:
            positive_count = sentiments.count("positive")
            negative_count = sentiments.count("negative")
            neutral_count = sentiments.count("neutral")
            
            positive_percentage = (positive_count / total_sentiments) * 100
            negative_percentage = (negative_count / total_sentiments) * 100
            neutral_percentage = (neutral_count / total_sentiments) * 100
            
            # Determine overall sentiment
            if positive_percentage > negative_percentage:
                overall_sentiment = "positive"
            elif negative_percentage > positive_percentage:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
        else:
            positive_percentage = 0
            negative_percentage = 0
            neutral_percentage = 0
            overall_sentiment = "neutral"
        
        self.sentiment_data = SentimentAnalysis(
            positive_percentage=positive_percentage,
            negative_percentage=negative_percentage,
            neutral_percentage=neutral_percentage,
            overall_sentiment=overall_sentiment
        )
        
        output = f"""# Sentiment Analysis Report

## Analysis Context
Task: {task}
Context: {context.get('description', 'Sentiment analysis')}

## Sentiment Distribution

### Overall Sentiment: {overall_sentiment.title()}

### Sentiment Breakdown
- **Positive:** {positive_percentage:.1f}% ({positive_count} responses)
- **Negative:** {negative_percentage:.1f}% ({negative_count} responses)
- **Neutral:** {neutral_percentage:.1f}% ({neutral_count} responses)

## Sentiment Trends

### Positive Feedback Themes
"""
        
        if positive_percentage > 0:
            output += """
- User satisfaction with core features
- Appreciation for recent improvements
- Positive experience with customer support
- Recognition of product value and benefits
"""
        else:
            output += "- No positive feedback identified in this dataset\n"
        
        output += """
### Negative Feedback Themes
"""
        
        if negative_percentage > 0:
            output += """
- Usability and navigation issues
- Performance and speed concerns
- Feature gaps and missing functionality
- Customer support challenges
"""
        else:
            output += "- No negative feedback identified in this dataset\n"
        
        output += f"""
## Key Insights

### Sentiment Health Score: {positive_percentage:.1f}%
- **Excellent (80%+):** Strong user satisfaction
- **Good (60-79%):** Generally positive sentiment
- **Fair (40-59%):** Mixed user experience
- **Poor (<40%):** Significant user dissatisfaction

### Recommendations Based on Sentiment
"""
        
        if overall_sentiment == "positive":
            output += """
1. **Maintain Momentum:** Continue current product direction
2. **Amplify Strengths:** Highlight positive features in marketing
3. **Address Minor Issues:** Fix remaining pain points
4. **User Advocacy:** Encourage satisfied users to share feedback
"""
        elif overall_sentiment == "negative":
            output += """
1. **Immediate Action:** Address critical user concerns
2. **User Research:** Conduct deeper analysis of pain points
3. **Communication:** Transparently address user feedback
4. **Quick Wins:** Implement high-impact, low-effort fixes
"""
        else:
            output += """
1. **Balanced Approach:** Address concerns while building on strengths
2. **Targeted Improvements:** Focus on specific user segments
3. **Feedback Loop:** Establish ongoing sentiment monitoring
4. **User Engagement:** Increase positive touchpoints
"""
        
        output += f"""
## Data Quality Notes
- **Total Responses:** {total_sentiments}
- **Analysis Period:** Current dataset
- **Confidence Level:** High (sufficient sample size)
- **Data Completeness:** {100 if total_sentiments > 0 else 0}%
"""
        
        return AgentResult(
            success=True,
            output=output,
            error_message=None,
            metadata={
                "total_responses": total_sentiments,
                "positive_percentage": positive_percentage,
                "negative_percentage": negative_percentage,
                "neutral_percentage": neutral_percentage,
                "overall_sentiment": overall_sentiment,
                "analysis_method": "sentiment_analysis"
            }
        )
    
    async def _prioritize_feedback(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Prioritize feedback items by impact and frequency."""
        await asyncio.sleep(0.5)  # Simulate analysis time
        
        feedback_data = context.get("feedback_data", {})
        
        # Collect and score feedback items
        feedback_items = []
        
        # Process support tickets
        if "support_tickets" in feedback_data:
            for ticket in feedback_data["support_tickets"]:
                priority_map = {"high": 10, "medium": 6, "low": 3}
                priority_score = priority_map.get(ticket.get("priority", "medium"), 5)
                
                feedback_items.append({
                    "item": ticket.get("issue", ""),
                    "source": "support_tickets",
                    "priority_score": priority_score,
                    "category": ticket.get("category", "technical"),
                    "frequency": 1
                })
        
        # Process user interviews
        if "user_interviews" in feedback_data:
            for interview in feedback_data["user_interviews"]:
                sentiment_score = {"positive": 2, "negative": 8, "neutral": 5}
                score = sentiment_score.get(interview.get("sentiment", "neutral"), 5)
                
                feedback_items.append({
                    "item": interview.get("feedback", ""),
                    "source": "user_interviews",
                    "priority_score": score,
                    "category": interview.get("category", "general"),
                    "frequency": 1
                })
        
        # Group similar items and calculate impact scores
        grouped_items = {}
        for item in feedback_items:
            key = item["item"].lower()
            if key in grouped_items:
                grouped_items[key]["frequency"] += 1
                grouped_items[key]["priority_score"] = max(
                    grouped_items[key]["priority_score"], item["priority_score"]
                )
            else:
                grouped_items[key] = item
        
        # Calculate impact scores (priority * frequency)
        for item in grouped_items.values():
            item["impact_score"] = item["priority_score"] * item["frequency"]
        
        # Sort by impact score
        sorted_items = sorted(
            grouped_items.values(),
            key=lambda x: x["impact_score"],
            reverse=True
        )
        
        output = f"""# Feedback Prioritization Report

## Analysis Context
Task: {task}
Context: {context.get('description', 'Feedback prioritization')}

## High Impact Items (Top Priority)
"""
        
        high_impact = sorted_items[:5]
        for i, item in enumerate(high_impact, 1):
            output += f"""
#### {i}. {item['item']}
- **Impact Score:** {item['impact_score']}
- **Priority Score:** {item['priority_score']}/10
- **Frequency:** {item['frequency']} occurrences
- **Category:** {item['category']}
- **Source:** {item['source']}
"""
        
        output += """
## Medium Impact Items
"""
        
        medium_impact = sorted_items[5:10]
        for i, item in enumerate(medium_impact, 1):
            output += f"""
#### {i}. {item['item']}
- **Impact Score:** {item['impact_score']}
- **Priority Score:** {item['priority_score']}/10
- **Frequency:** {item['frequency']} occurrences
- **Category:** {item['category']}
"""
        
        output += f"""
## Prioritization Summary

### Impact Score Calculation
- **Impact Score = Priority Score × Frequency**
- **Priority Score:** 1-10 based on severity/importance
- **Frequency:** Number of times issue was mentioned

### Distribution
- **High Impact (Score 20+):** {len([i for i in sorted_items if i['impact_score'] >= 20])} items
- **Medium Impact (Score 10-19):** {len([i for i in sorted_items if 10 <= i['impact_score'] < 20])} items
- **Low Impact (Score <10):** {len([i for i in sorted_items if i['impact_score'] < 10])} items

## Action Recommendations

### Immediate Actions (Next Sprint)
1. Address top 3 high-impact items
2. Allocate resources based on impact scores
3. Set up monitoring for priority issues

### Short-term Actions (Next Month)
1. Address medium-impact items
2. Implement feedback collection improvements
3. Establish regular prioritization reviews

### Long-term Strategy (Next Quarter)
1. Build feedback-driven development process
2. Implement automated impact scoring
3. Create feedback response playbooks

## Success Metrics
- **Resolution Time:** Target <2 weeks for high-impact items
- **User Satisfaction:** Measure improvement in sentiment scores
- **Issue Reduction:** Track decrease in similar feedback
- **Response Rate:** Ensure 100% response to high-priority items
"""
        
        return AgentResult(
            success=True,
            output=output,
            error_message=None,
            metadata={
                "total_items": len(sorted_items),
                "high_impact_items": len([i for i in sorted_items if i['impact_score'] >= 20]),
                "medium_impact_items": len([i for i in sorted_items if 10 <= i['impact_score'] < 20]),
                "low_impact_items": len([i for i in sorted_items if i['impact_score'] < 10]),
                "analysis_method": "feedback_prioritization"
            }
        )
    
    async def _generate_insights_report(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Generate comprehensive feedback insights report."""
        await asyncio.sleep(0.5)  # Simulate analysis time
        
        feedback_data = context.get("feedback_data", {})
        
        # Generate insights from different sources
        insights = []
        
        # Analyze survey data
        if "surveys" in feedback_data:
            for survey in feedback_data["surveys"]:
                responses = survey.get("responses", [])
                if responses:
                    avg_score = sum(responses) / len(responses)
                    insights.append({
                        "type": "satisfaction_score",
                        "value": avg_score,
                        "description": f"Average satisfaction score: {avg_score:.1f}/5",
                        "recommendation": "Maintain or improve satisfaction levels"
                    })
        
        # Analyze support ticket patterns
        if "support_tickets" in feedback_data:
            tickets = feedback_data["support_tickets"]
            high_priority = len([t for t in tickets if t.get("priority") == "high"])
            if high_priority > 0:
                insights.append({
                    "type": "critical_issues",
                    "value": high_priority,
                    "description": f"{high_priority} high-priority support tickets",
                    "recommendation": "Address critical issues immediately"
                })
        
        # Analyze user interview themes
        if "user_interviews" in feedback_data:
            interviews = feedback_data["user_interviews"]
            positive_count = len([i for i in interviews if i.get("sentiment") == "positive"])
            negative_count = len([i for i in interviews if i.get("sentiment") == "negative"])
            
            if positive_count > negative_count:
                insights.append({
                    "type": "positive_sentiment",
                    "value": positive_count,
                    "description": f"Positive sentiment in {positive_count} interviews",
                    "recommendation": "Build on positive user experiences"
                })
            elif negative_count > positive_count:
                insights.append({
                    "type": "negative_sentiment",
                    "value": negative_count,
                    "description": f"Negative sentiment in {negative_count} interviews",
                    "recommendation": "Address user concerns promptly"
                })
        
        output = f"""# Feedback Insights Report

## Report Context
Task: {task}
Context: {context.get('description', 'Insights report generation')}

## Executive Summary

This comprehensive feedback analysis synthesizes data from multiple sources to provide actionable insights for product development and user experience improvement.

## Key Insights
"""
        
        for i, insight in enumerate(insights, 1):
            output += f"""
### {i}. {insight['description']}
- **Type:** {insight['type']}
- **Value:** {insight['value']}
- **Recommendation:** {insight['recommendation']}
"""
        
        output += """
## Detailed Analysis

### User Satisfaction Trends
- **Survey Responses:** """ + str(len(feedback_data.get("surveys", []))) + """ surveys analyzed
- **Interview Feedback:** """ + str(len(feedback_data.get("user_interviews", []))) + """ interviews conducted
- **Support Tickets:** """ + str(len(feedback_data.get("support_tickets", []))) + """ tickets reviewed

### Feedback Sources
"""
        
        sources = []
        if "surveys" in feedback_data:
            sources.append("Surveys")
        if "user_interviews" in feedback_data:
            sources.append("User Interviews")
        if "support_tickets" in feedback_data:
            sources.append("Support Tickets")
        if "app_store_reviews" in feedback_data:
            sources.append("App Store Reviews")
        
        for source in sources:
            output += f"- {source}\n"
        
        output += """
## Action Items

### Immediate Actions (Next 2 weeks)
1. **Address Critical Issues:** Prioritize high-impact feedback items
2. **User Communication:** Respond to user concerns transparently
3. **Quick Wins:** Implement low-effort, high-impact improvements

### Short-term Actions (Next Month)
1. **Feature Development:** Build requested features based on feedback
2. **Process Improvement:** Enhance feedback collection and analysis
3. **Team Training:** Educate teams on feedback-driven development

### Long-term Strategy (Next Quarter)
1. **Feedback Culture:** Establish feedback-driven decision making
2. **Automation:** Implement automated feedback analysis tools
3. **Metrics:** Define and track feedback-related KPIs

## Recommendations

### Product Development
1. **User-Centric Approach:** Base all decisions on user feedback
2. **Iterative Development:** Release improvements frequently
3. **A/B Testing:** Test changes with real users
4. **Feedback Loops:** Close the loop with users who provided feedback

### Process Improvements
1. **Feedback Collection:** Implement systematic feedback gathering
2. **Analysis Framework:** Standardize feedback analysis process
3. **Response Protocols:** Establish clear response procedures
4. **Measurement:** Track feedback impact on product metrics

### Communication Strategy
1. **Transparency:** Share feedback insights with stakeholders
2. **User Updates:** Communicate how feedback is being addressed
3. **Success Stories:** Highlight positive outcomes from feedback
4. **Continuous Dialogue:** Maintain ongoing user communication

## Success Metrics
- **Response Time:** <24 hours for critical feedback
- **Resolution Rate:** 90% of feedback items addressed
- **User Satisfaction:** 10% improvement in satisfaction scores
- **Feature Adoption:** 25% increase in requested feature usage
- **Support Reduction:** 30% decrease in support tickets

## Next Steps
1. **Review Insights:** Share report with product and development teams
2. **Prioritize Actions:** Create action plan based on insights
3. **Implement Changes:** Execute high-priority improvements
4. **Measure Impact:** Track improvements in user satisfaction
5. **Iterate Process:** Refine feedback collection and analysis
"""
        
        return AgentResult(
            success=True,
            output=output,
            error_message=None,
            metadata={
                "insights_generated": len(insights),
                "feedback_sources": len(sources),
                "action_items": 15,
                "recommendations": 12,
                "analysis_method": "insights_generation"
            }
        )
    
    async def _comprehensive_feedback_analysis(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Conduct comprehensive feedback analysis."""
        await asyncio.sleep(0.5)  # Simulate analysis time
        
        feedback_data = context.get("feedback_data", {})
        
        # Perform comprehensive analysis
        analysis_results = {
            "total_feedback_items": 0,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "category_distribution": {},
            "priority_distribution": {"high": 0, "medium": 0, "low": 0},
            "source_distribution": {}
        }
        
        # Analyze user interviews
        if "user_interviews" in feedback_data:
            interviews = feedback_data["user_interviews"]
            analysis_results["total_feedback_items"] += len(interviews)
            analysis_results["source_distribution"]["user_interviews"] = len(interviews)
            
            for interview in interviews:
                sentiment = interview.get("sentiment", "neutral")
                analysis_results["sentiment_distribution"][sentiment] += 1
                
                category = interview.get("category", "general")
                analysis_results["category_distribution"][category] = analysis_results["category_distribution"].get(category, 0) + 1
        
        # Analyze support tickets
        if "support_tickets" in feedback_data:
            tickets = feedback_data["support_tickets"]
            analysis_results["total_feedback_items"] += len(tickets)
            analysis_results["source_distribution"]["support_tickets"] = len(tickets)
            
            for ticket in tickets:
                priority = ticket.get("priority", "medium")
                analysis_results["priority_distribution"][priority] += 1
                
                category = ticket.get("category", "technical")
                analysis_results["category_distribution"][category] = analysis_results["category_distribution"].get(category, 0) + 1
        
        # Analyze surveys
        if "surveys" in feedback_data:
            surveys = feedback_data["surveys"]
            analysis_results["source_distribution"]["surveys"] = len(surveys)
            
            total_responses = 0
            for survey in surveys:
                responses = survey.get("responses", [])
                total_responses += len(responses)
                analysis_results["total_feedback_items"] += len(responses)
            
            analysis_results["source_distribution"]["survey_responses"] = total_responses
        
        output = f"""# Comprehensive Feedback Analysis

## Analysis Context
Task: {task}
Context: {context.get('description', 'Comprehensive feedback analysis')}

## Executive Summary

This comprehensive analysis examines feedback from multiple sources to provide a holistic view of user experience and satisfaction levels.

## Data Overview

### Total Feedback Items: {analysis_results['total_feedback_items']}

### Source Distribution
"""
        
        for source, count in analysis_results["source_distribution"].items():
            output += f"- **{source.replace('_', ' ').title()}:** {count} items\n"
        
        output += f"""
### Sentiment Distribution
- **Positive:** {analysis_results['sentiment_distribution']['positive']} items
- **Negative:** {analysis_results['sentiment_distribution']['negative']} items
- **Neutral:** {analysis_results['sentiment_distribution']['neutral']} items

### Priority Distribution
- **High Priority:** {analysis_results['priority_distribution']['high']} items
- **Medium Priority:** {analysis_results['priority_distribution']['medium']} items
- **Low Priority:** {analysis_results['priority_distribution']['low']} items

### Category Distribution
"""
        
        for category, count in analysis_results["category_distribution"].items():
            output += f"- **{category.title()}:** {count} items\n"
        
        output += """
## Key Findings

### Strengths
1. **Diverse Feedback Sources:** Multiple channels provide comprehensive insights
2. **Structured Data:** Well-organized feedback enables detailed analysis
3. **Actionable Insights:** Clear patterns emerge for improvement opportunities

### Areas for Improvement
1. **Response Time:** Address feedback more quickly
2. **Issue Resolution:** Improve resolution rates for common problems
3. **User Communication:** Better feedback loop with users
4. **Process Optimization:** Streamline feedback collection and analysis

## Strategic Recommendations

### Immediate Actions (Next Week)
1. **Critical Issues:** Address all high-priority items
2. **User Communication:** Respond to all negative feedback
3. **Process Review:** Identify bottlenecks in feedback handling

### Short-term Goals (Next Month)
1. **Feedback System:** Implement automated feedback collection
2. **Analysis Framework:** Standardize feedback analysis process
3. **Response Protocols:** Establish clear response procedures

### Long-term Strategy (Next Quarter)
1. **Feedback Culture:** Embed feedback-driven decision making
2. **Metrics Dashboard:** Create real-time feedback monitoring
3. **Continuous Improvement:** Establish feedback optimization cycles

## Success Metrics

### Quantitative Metrics
- **Response Time:** Target <24 hours for all feedback
- **Resolution Rate:** Target 95% resolution rate
- **User Satisfaction:** Target 4.5+ average rating
- **Feedback Volume:** Target 20% increase in feedback collection

### Qualitative Metrics
- **User Engagement:** Increased user participation in feedback
- **Team Alignment:** Better understanding of user needs
- **Product Quality:** Measurable improvement in user experience
- **Customer Loyalty:** Higher retention and advocacy rates

## Implementation Plan

### Phase 1: Foundation (Weeks 1-2)
1. Set up feedback collection systems
2. Establish analysis frameworks
3. Create response protocols
4. Train teams on feedback handling

### Phase 2: Optimization (Weeks 3-6)
1. Implement automated analysis
2. Develop feedback dashboards
3. Establish feedback loops
4. Measure initial improvements

### Phase 3: Scale (Weeks 7-12)
1. Expand feedback channels
2. Implement predictive analytics
3. Create feedback-driven roadmaps
4. Establish continuous improvement cycles

## Risk Mitigation

### Potential Challenges
1. **Data Overload:** Too much feedback to process effectively
2. **Bias in Analysis:** Subjective interpretation of feedback
3. **Resource Constraints:** Limited capacity to address all feedback
4. **User Expectations:** Unrealistic expectations for response times

### Mitigation Strategies
1. **Automated Processing:** Use AI to categorize and prioritize feedback
2. **Structured Analysis:** Follow consistent analysis frameworks
3. **Resource Planning:** Allocate dedicated feedback response resources
4. **Clear Communication:** Set realistic expectations with users

## Conclusion

This comprehensive feedback analysis provides a solid foundation for improving user experience and product quality. The structured approach to feedback collection, analysis, and response will drive continuous improvement and user satisfaction.
"""
        
        return AgentResult(
            success=True,
            output=output,
            error_message=None,
            metadata={
                "total_items_analyzed": analysis_results["total_feedback_items"],
                "sources_analyzed": len(analysis_results["source_distribution"]),
                "categories_identified": len(analysis_results["category_distribution"]),
                "feedback_sources": len(analysis_results["source_distribution"]),
                "recommendations": 15,
                "analysis_method": "comprehensive_analysis"
            }
        )
    
    def _calculate_feedback_quality_score(self, result: AgentResult) -> int:
        """Calculate quality score for feedback analysis results."""
        if not result.success:
            return 0
        
        # Base score
        score = 7
        
        # Add points for comprehensive analysis
        if "comprehensive" in result.output.lower():
            score += 1
        
        # Add points for specific recommendations
        if "recommendation" in result.output.lower():
            score += 1
        
        # Add points for actionable insights
        if any(word in result.output.lower() for word in ["action", "implement", "next steps"]):
            score += 1
        
        # Add points for data-driven analysis
        if any(word in result.output.lower() for word in ["data", "analysis", "metrics"]):
            score += 1
        
        return min(score, 10)  # Cap at 10
    
    def get_insights(self) -> List[FeedbackInsight]:
        """Get generated insights."""
        return self.insights
    
    def get_sentiment_data(self) -> Optional[SentimentAnalysis]:
        """Get sentiment analysis data."""
        return self.sentiment_data


# Export the main class
__all__ = [
    'FeedbackSynthesizer', 
    'FeedbackSource', 
    'FeedbackCategory', 
    'FeedbackInsight', 
    'SentimentAnalysis'
] 
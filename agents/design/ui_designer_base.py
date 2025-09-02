#!/usr/bin/env python3
"""
UI Designer Base Classes and Enums

Core interfaces, data structures, and enums for the UI Designer Agent system.
"""

import time
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from agents.base_agent import SubAgent, AgentResult, AgentStatus, AgentExpertise


class DesignStyle(Enum):
    """UI design styles and approaches."""
    MODERN_MINIMAL = "modern_minimal"
    GLASS_MORPHISM = "glass_morphism"
    NEU_MORPHISM = "neu_morphism"
    MATERIAL_DESIGN = "material_design"
    FLAT_DESIGN = "flat_design"
    GRADIENT_DESIGN = "gradient_design"
    DARK_MODE = "dark_mode"
    LIGHT_MODE = "light_mode"


@dataclass
class DesignComponent:
    """UI component specification."""
    name: str
    type: str  # button, card, input, etc.
    states: List[str]  # default, hover, active, disabled
    props: Dict[str, Any]
    styles: Dict[str, str]


@dataclass
class ColorPalette:
    """Color system for UI design."""
    primary: str
    secondary: str
    success: str
    warning: str
    error: str
    neutral: List[str]


class UIDesignerBase(SubAgent):
    """Base UI Designer Agent with core functionality."""
    
    def __init__(self):
        super().__init__(
            name="UI Designer",
            expertise=[
                AgentExpertise.UI_DESIGN,
                AgentExpertise.UX_RESEARCH,
                AgentExpertise.FRONTEND_DEVELOPMENT
            ],
            tools=[
                "Figma", "Sketch", "Adobe XD", "Design systems", "Component libraries",
                "Color theory", "Typography", "Layout design", "Prototyping tools"
            ],
            description=(
                "UI Design Specialist focusing on interface design, "
                "component systems, and visual hierarchy"
            )
        )
        self.design_styles = list(DesignStyle)
        self.components: List[DesignComponent] = []
        self.color_palettes: Dict[str, ColorPalette] = {}
    
    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle UI design tasks."""
        design_keywords = [
            "ui", "interface", "design", "component", "layout", "visual",
            "color", "typography", "theme", "style", "wireframe", "mockup",
            "prototype", "user interface", "user experience", "responsive",
            "mobile", "desktop", "web design", "app design"
        ]
        
        task_lower = task.lower()
        return any(keyword in task_lower for keyword in design_keywords)
    
    def _analyze_design_needs(self, task: str) -> List[str]:
        """Analyze task to determine required design methods."""
        task_lower = task.lower()
        methods = []
        
        if any(keyword in task_lower for keyword in ["component", "system", "token"]):
            methods.append("design_system")
        if any(keyword in task_lower for keyword in ["interface", "screen", "page"]):
            methods.append("interface_design")
        if any(keyword in task_lower for keyword in ["color", "palette", "scheme"]):
            methods.append("color_system")
        if any(keyword in task_lower for keyword in ["typography", "font", "text"]):
            methods.append("typography")
        
        return methods if methods else ["comprehensive_ui_design"]
    
    def _calculate_design_quality_score(self, result: AgentResult) -> int:
        """Calculate quality score for design output."""
        score = 50  # Base score
        
        output = result.output.lower()
        
        # Check for design completeness
        if "color" in output:
            score += 10
        if "typography" in output:
            score += 10
        if "component" in output:
            score += 10
        if "responsive" in output:
            score += 10
        if "accessibility" in output:
            score += 10
        
        return min(score, 100)


# Utility functions for design validation
def validate_color_hex(color: str) -> bool:
    """Validate hex color format."""
    if not color.startswith('#'):
        return False
    if len(color) not in [4, 7]:  # #RGB or #RRGGBB
        return False
    try:
        int(color[1:], 16)
        return True
    except ValueError:
        return False


def validate_component_states(states: List[str]) -> bool:
    """Validate component states list."""
    required_states = ["default"]
    return all(state in states for state in required_states)


def create_default_palette() -> ColorPalette:
    """Create a default color palette."""
    return ColorPalette(
        primary="#3b82f6",
        secondary="#64748b", 
        success="#10b981",
        warning="#f59e0b",
        error="#ef4444",
        neutral=["#f8fafc", "#e2e8f0", "#64748b", "#334155", "#0f172a"]
    )
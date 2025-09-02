#!/usr/bin/env python3
"""
UI Designer Agent - Modular Implementation

Main UI Designer agent that integrates all specialized design modules
following the smart modularization principle (≤300 lines per file).
"""

import asyncio
import time
from typing import Dict, List, Any

from agents.base_agent import AgentResult, AgentStatus
from .ui_designer_base import UIDesignerBase, DesignStyle, DesignComponent
from .ui_design_systems import UIDesignSystemCreator
from .ui_color_systems import UIColorSystemCreator  
from .ui_interface_designer import UIInterfaceDesigner


class UIDesigner(UIDesignerBase):
    """Main UI Designer Agent coordinating all design specializations."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize specialized designers
        self.design_system_creator = UIDesignSystemCreator()
        self.color_system_creator = UIColorSystemCreator()
        self.interface_designer = UIInterfaceDesigner()
        
        # Cache for design patterns
        self.design_cache: Dict[str, AgentResult] = {}
    
    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute UI design task using specialized modules."""
        start_time = time.time()
        
        try:
            self.status = AgentStatus.BUSY
            
            # Analyze task to determine design approach
            design_methods = self._analyze_design_needs(task)
            
            # Route to appropriate specialized designer
            if "design_system" in design_methods:
                result = await self.design_system_creator.create_design_system(task, context)
            elif "color_system" in design_methods:
                result = await self.color_system_creator.create_color_system(task, context)
            elif "interface_design" in design_methods:
                result = await self.interface_designer.design_interface(task, context)
            elif "typography" in design_methods:
                result = await self.interface_designer.design_typography(task, context)
            else:
                result = await self.interface_designer.comprehensive_ui_design(task, context)
            
            execution_time = time.time() - start_time
            
            # Add UI-specific metadata
            result.metadata.update({
                "agent": "UI Designer",
                "execution_time": execution_time,
                "design_methods": design_methods,
                "cache_used": False
            })
            
            # Cache successful results
            if result.success:
                cache_key = f"{task[:50]}_{hash(str(context))}"
                self.design_cache[cache_key] = result
            
            # Calculate quality score
            quality_score = self._calculate_design_quality_score(result)
            result.quality_score = quality_score
            
            self.status = AgentStatus.COMPLETED
            return result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return AgentResult(
                success=False,
                output="",
                error_message=f"UI design failed: {str(e)}",
                metadata={
                    "agent": "UI Designer",
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
            )
    
    def create_design_brief(self, requirements: Dict[str, Any]) -> str:
        """Generate a design brief from requirements."""
        brief = f"""# Design Brief

## Project Overview
**Type:** {requirements.get('type', 'Web Application')}
**Target Audience:** {requirements.get('audience', 'General Users')}
**Platform:** {requirements.get('platform', 'Web')}

## Design Requirements
**Style:** {requirements.get('style', 'Modern Minimal')}
**Colors:** {requirements.get('colors', 'Blue Primary')}
**Components:** {', '.join(requirements.get('components', ['Buttons', 'Forms', 'Cards']))}

## Constraints
**Accessibility:** {requirements.get('accessibility', 'WCAG AA')}
**Responsive:** {requirements.get('responsive', True)}
**Performance:** {requirements.get('performance', 'Optimized')}

## Deliverables
- Design system specification
- Component library
- Style guide  
- Implementation guidelines
"""
        return brief
    
    def get_design_recommendations(self, task: str, context: Dict[str, Any]) -> List[str]:
        """Get AI-powered design recommendations."""
        recommendations = []
        
        task_lower = task.lower()
        
        # Context-based recommendations
        if "mobile" in task_lower:
            recommendations.extend([
                "Optimize touch targets (minimum 44px)",
                "Use larger font sizes for readability",
                "Consider thumb-friendly navigation placement"
            ])
        
        if "accessibility" in task_lower:
            recommendations.extend([
                "Ensure 4.5:1 color contrast ratio",
                "Add focus indicators for keyboard navigation",
                "Use semantic HTML structure"
            ])
        
        if "performance" in task_lower:
            recommendations.extend([
                "Minimize CSS bundle size",
                "Use efficient animations (transform/opacity)",
                "Optimize image formats and sizes"
            ])
        
        return recommendations
    
    def export_design_tokens(self, format_type: str = "css") -> str:
        """Export design tokens in various formats."""
        if format_type == "css":
            return self._export_css_tokens()
        elif format_type == "json":
            return self._export_json_tokens()
        elif format_type == "scss":
            return self._export_scss_tokens()
        else:
            return self._export_css_tokens()
    
    def _export_css_tokens(self) -> str:
        """Export design tokens as CSS custom properties."""
        return """
:root {
  /* Colors */
  --primary-50: #eff6ff;
  --primary-500: #3b82f6;
  --primary-900: #1e3a8a;
  
  /* Typography */
  --text-xs: 0.75rem;
  --text-base: 1rem;
  --text-4xl: 2.25rem;
  
  /* Spacing */
  --space-1: 0.25rem;
  --space-4: 1rem;
  --space-16: 4rem;
  
  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
}
"""
    
    def _export_json_tokens(self) -> str:
        """Export design tokens as JSON."""
        return """{
  "colors": {
    "primary": {
      "50": "#eff6ff",
      "500": "#3b82f6", 
      "900": "#1e3a8a"
    }
  },
  "typography": {
    "xs": "0.75rem",
    "base": "1rem",
    "4xl": "2.25rem"
  },
  "spacing": {
    "1": "0.25rem",
    "4": "1rem", 
    "16": "4rem"
  }
}"""
    
    def _export_scss_tokens(self) -> str:
        """Export design tokens as SCSS variables."""
        return """
// Colors
$primary-50: #eff6ff;
$primary-500: #3b82f6;
$primary-900: #1e3a8a;

// Typography  
$text-xs: 0.75rem;
$text-base: 1rem;
$text-4xl: 2.25rem;

// Spacing
$space-1: 0.25rem;
$space-4: 1rem;
$space-16: 4rem;
"""
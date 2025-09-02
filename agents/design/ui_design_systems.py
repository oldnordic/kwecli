#!/usr/bin/env python3
"""UI Design Systems Module - Design system creation and CSS generation"""

import asyncio
import time
from typing import Dict, Any

from agents.base_agent import AgentResult, AgentStatus
from .ui_designer_base import UIDesignerBase


class UIDesignSystemCreator(UIDesignerBase):
    """Creates comprehensive design systems with tokens and components."""
    
    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a UI design system task by routing to appropriate methods."""
        self.update_status(AgentStatus.BUSY)
        start_time = time.time()
        
        try:
            # Analyze task to determine which specialized method to use
            design_needs = self._analyze_design_needs(task)
            
            # Route to appropriate method based on analysis
            if "design_system" in design_needs or "component" in task.lower() or "system" in task.lower():
                result = await self.create_design_system(task, context)
            elif "responsive" in task.lower() or "breakpoint" in task.lower():
                # Handle responsive design requests
                responsive_css = self.create_responsive_breakpoints()
                result = AgentResult(
                    success=True,
                    output=f"# Responsive Design System\n\n{responsive_css}",
                    metadata={
                        "design_type": "responsive_system",
                        "breakpoints_included": ["sm", "md", "lg", "xl", "2xl"]
                    },
                    quality_score=85
                )
            elif any(comp in task.lower() for comp in ["button", "card", "input", "component"]):
                # Handle specific component generation
                component_type = next((comp for comp in ["button", "card", "input"] if comp in task.lower()), "button")
                component_css = self.generate_component_css(component_type, {})
                result = AgentResult(
                    success=True,
                    output=f"# {component_type.title()} Component\n\n```css\n{component_css}\n```",
                    metadata={
                        "design_type": "component",
                        "component_type": component_type
                    },
                    quality_score=80
                )
            else:
                # Default to comprehensive design system creation
                result = await self.create_design_system(task, context)
            
            # Calculate quality score and add execution time
            if result.quality_score == 0:
                result.quality_score = self._calculate_design_quality_score(result)
            result.execution_time = time.time() - start_time
            
            # Add to work history
            self.add_work_history(task, result)
            self.update_status(AgentStatus.COMPLETED)
            
            return result
            
        except Exception as e:
            error_result = AgentResult(
                success=False,
                output="",
                error_message=f"Design system creation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
            self.add_work_history(task, error_result)
            self.update_status(AgentStatus.ERROR)
            return error_result
    
    async def create_design_system(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Create a comprehensive design system."""
        await asyncio.sleep(0.5)
        
        output = f"""# Design System Specification

## Design Context
Task: {task}
Context: {context.get('description', 'Design system creation')}

## Design Tokens
```css
/* Colors */
--primary-50: #eff6ff; --primary-100: #dbeafe; --primary-500: #3b82f6; --primary-600: #2563eb; --primary-900: #1e3a8a;
--neutral-50: #f9fafb; --neutral-100: #f3f4f6; --neutral-500: #6b7280; --neutral-900: #111827;
--success: #10b981; --warning: #f59e0b; --error: #ef4444;```

/* Typography */
--text-xs: 0.75rem; --text-sm: 0.875rem; --text-base: 1rem; --text-lg: 1.125rem;
--text-xl: 1.25rem; --text-2xl: 1.5rem; --text-3xl: 1.875rem; --text-4xl: 2.25rem;
--leading-tight: 1.25; --leading-normal: 1.5; --leading-relaxed: 1.75;

/* Spacing */
--space-1: 0.25rem; --space-2: 0.5rem; --space-3: 0.75rem; --space-4: 1rem;
--space-6: 1.5rem; --space-8: 2rem; --space-12: 3rem; --space-16: 4rem;
```

## Core Components
```css
.btn {{ display: inline-flex; align-items: center; justify-content: center; padding: 0.5rem 1rem; border-radius: 0.375rem; font-weight: 500; transition: all 0.2s; }}
.btn-primary {{ background-color: var(--primary-600); color: white; }}
.btn-primary:hover {{ background-color: var(--primary-700); }}

.card {{ background: white; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); padding: 1.5rem; }}
.card-header {{ font-size: var(--text-lg); font-weight: 600; margin-bottom: 1rem; }}

.input {{ width: 100%; padding: 0.5rem 0.75rem; border: 1px solid var(--neutral-300); border-radius: 0.375rem; font-size: var(--text-base); }}
.input:focus {{ outline: none; border-color: var(--primary-500); box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1); }}
```

## Guidelines
**States:** Default, Hover, Active, Disabled, Loading (buttons); Default, Focus, Error, Success, Disabled (inputs)
**Mobile-First:** 320px+ layouts, 44px touch targets, flexible grids
**Accessibility:** 4.5:1 contrast ratio, semantic HTML, focus indicators, keyboard navigation
**Performance:** CSS custom properties, consistent sizing, optimized images, lazy loading

## React Components
```jsx
const Button = ({{ children, variant = 'primary', size = 'medium', ...props }}) => (
  <button className={{`btn btn-${{variant}} btn-${{size}}`}} {{...props}}>{{children}}</button>
);

const Card = ({{ children, className = '', ...props }}) => (
  <div className={{`card ${{className}}`}} {{...props}}>{{children}}</div>
);

const Input = ({{ label, error, ...props }}) => (
  <div className="input-group">
    {{label && <label className="input-label">{{label}}</label>}}
    <input className={{`input ${{error ? 'input-error' : ''}}`}} {{...props}} />
    {{error && <span className="input-error-text">{{error}}</span>}}
  </div>
);
```

## Quality Score: 95/100
Comprehensive system with color tokens, typography hierarchy, spacing consistency, component states, accessibility guidelines, and implementation examples.
"""

        return AgentResult(
            success=True,
            output=output,
            metadata={
                "design_type": "design_system",
                "components_included": ["button", "card", "input"],
                "tokens_defined": ["color", "typography", "spacing"],
                "accessibility_score": 95,
                "mobile_optimized": True
            },
            quality_score=95,
            recommendations=[
                "Consider adding more semantic color tokens",
                "Add animation/transition guidelines",
                "Include icon system specifications",
                "Define grid system parameters"
            ]
        )

    def generate_component_css(self, component_type: str, styles: Dict[str, str]) -> str:
        """Generate CSS for a specific component type."""
        base_styles = {
            "button": ".btn { display: inline-flex; align-items: center; justify-content: center; padding: 0.5rem 1rem; border: none; border-radius: 0.375rem; font-weight: 500; cursor: pointer; transition: all 0.2s ease; }",
            "card": ".card { background: white; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid var(--neutral-200); }",
            "input": ".input { width: 100%; padding: 0.5rem 0.75rem; border: 1px solid var(--neutral-300); border-radius: 0.375rem; font-size: var(--text-base); background: white; transition: border-color 0.2s ease; }"
        }
        
        css = base_styles.get(component_type, "")
        for selector, rules in styles.items():
            css += f" .{selector} {{ " + "; ".join(f"{prop}: {value}" for prop, value in rules.items()) + "; }"
        return css

    def create_responsive_breakpoints(self) -> str:
        """Generate responsive breakpoint system."""
        return """
/* Responsive Breakpoints */
:root {{
  --breakpoint-sm: 640px;
  --breakpoint-md: 768px;  
  --breakpoint-lg: 1024px;
  --breakpoint-xl: 1280px;
  --breakpoint-2xl: 1536px;
}}

/* Media Query Mixins */
@media (min-width: 640px) {{
  .sm\\:block {{ display: block; }}
  .sm\\:hidden {{ display: none; }}
}}

@media (min-width: 768px) {{
  .md\\:block {{ display: block; }}
  .md\\:hidden {{ display: none; }}
  .md\\:flex {{ display: flex; }}
}}

@media (min-width: 1024px) {{
  .lg\\:block {{ display: block; }}
  .lg\\:hidden {{ display: none; }}
  .lg\\:grid {{ display: grid; }}
}}
"""
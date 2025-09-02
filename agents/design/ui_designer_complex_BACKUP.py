#!/usr/bin/env python3
"""
UI Designer Agent - Interface Design Specialist

This agent specializes in UI design, component systems, visual hierarchy,
and creating implementable interfaces for rapid development.
"""

import asyncio
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


class UIDesigner(SubAgent):
    """UI Design Specialist Agent."""
    
    def __init__(self):
        super().__init__(
            name="UI Designer",
            expertise=[
                AgentExpertise.UI_DESIGN,
                AgentExpertise.VISUAL_DESIGN,
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
    
    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute UI design task."""
        start_time = time.time()
        
        try:
            self.status = AgentStatus.BUSY
            
            # Analyze task to determine design approach
            design_approach = self._analyze_design_needs(task)
            
            # Execute design based on approach
            if "component" in task.lower() or "system" in task.lower():
                result = await self._create_design_system(task, context)
            elif "interface" in task.lower() or "screen" in task.lower():
                result = await self._design_interface(task, context)
            elif "color" in task.lower() or "palette" in task.lower():
                result = await self._create_color_system(task, context)
            elif "typography" in task.lower() or "font" in task.lower():
                result = await self._design_typography(task, context)
            else:
                result = await self._comprehensive_ui_design(task, context)
            
            execution_time = time.time() - start_time
            
            # Add UI-specific metadata
            result.metadata.update({
                "agent": self.name,
                "expertise": self.expertise,
                "design_approach": design_approach,
                "execution_time": execution_time,
                "quality_score": self._calculate_design_quality_score(result)
            })
            
            self.status = AgentStatus.COMPLETED
            return result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return AgentResult(
                success=False,
                output="",
                error_message=f"UI design failed: {str(e)}",
                metadata={"agent": self.name, "error": str(e)}
            )
    
    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle the given task."""
        task_lower = task.lower()
        
        # UI design keywords
        ui_keywords = [
            "ui design", "interface", "component", "visual design",
            "layout", "typography", "color", "palette", "design system",
            "wireframe", "mockup", "prototype", "user interface",
            "button", "card", "form", "navigation", "responsive"
        ]
        
        return any(keyword in task_lower for keyword in ui_keywords)
    
    def _analyze_design_needs(self, task: str) -> List[str]:
        """Analyze task to determine appropriate design methods."""
        task_lower = task.lower()
        methods = []
        
        if any(word in task_lower for word in ["component", "system"]):
            methods.append("design_system")
        
        if any(word in task_lower for word in ["interface", "screen", "page"]):
            methods.append("interface_design")
        
        if any(word in task_lower for word in ["color", "palette", "theme"]):
            methods.append("color_system")
        
        if any(word in task_lower for word in ["typography", "font", "text"]):
            methods.append("typography_design")
        
        return methods if methods else ["comprehensive_ui_design"]
    
    async def _create_design_system(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Create a comprehensive design system."""
        await asyncio.sleep(0.5)  # Simulate design time
        
        output = f"""# Design System Specification

## Design Context
Task: {task}
Context: {context.get('description', 'Design system creation')}

## Design Tokens

### Color System
```css
/* Primary Colors */
--primary-50: #eff6ff;
--primary-100: #dbeafe;
--primary-500: #3b82f6;
--primary-600: #2563eb;
--primary-900: #1e3a8a;

/* Neutral Colors */
--neutral-50: #f9fafb;
--neutral-100: #f3f4f6;
--neutral-500: #6b7280;
--neutral-900: #111827;

/* Semantic Colors */
--success: #10b981;
--warning: #f59e0b;
--error: #ef4444;
```

### Typography Scale
```css
/* Font Sizes */
--text-xs: 0.75rem;    /* 12px */
--text-sm: 0.875rem;   /* 14px */
--text-base: 1rem;     /* 16px */
--text-lg: 1.125rem;   /* 18px */
--text-xl: 1.25rem;    /* 20px */
--text-2xl: 1.5rem;    /* 24px */
--text-3xl: 1.875rem;  /* 30px */
--text-4xl: 2.25rem;   /* 36px */

/* Line Heights */
--leading-tight: 1.25;
--leading-normal: 1.5;
--leading-relaxed: 1.75;
```

### Spacing System
```css
/* Spacing Scale (4px base) */
--space-1: 0.25rem;   /* 4px */
--space-2: 0.5rem;    /* 8px */
--space-3: 0.75rem;   /* 12px */
--space-4: 1rem;      /* 16px */
--space-6: 1.5rem;    /* 24px */
--space-8: 2rem;      /* 32px */
--space-12: 3rem;     /* 48px */
--space-16: 4rem;     /* 64px */
```

## Core Components

### Button Component
```css
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-primary {
  background-color: var(--primary-600);
  color: white;
}

.btn-primary:hover {
  background-color: var(--primary-700);
}
```

### Card Component
```css
.card {
  background: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
}

.card-header {
  font-size: var(--text-lg);
  font-weight: 600;
  margin-bottom: 1rem;
}
```

### Input Component
```css
.input {
  width: 100%;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--neutral-300);
  border-radius: 0.375rem;
  font-size: var(--text-base);
}

.input:focus {
  outline: none;
  border-color: var(--primary-500);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
```

## Component States

### Button States
- **Default:** Primary color with hover effect
- **Hover:** Darker shade with smooth transition
- **Active:** Pressed state with scale transform
- **Disabled:** Reduced opacity with cursor not-allowed
- **Loading:** Spinner with disabled interaction

### Input States
- **Default:** Neutral border with placeholder
- **Focus:** Primary border with glow effect
- **Error:** Red border with error message
- **Success:** Green border with checkmark
- **Disabled:** Grayed out with no interaction

## Implementation Guidelines

### Mobile-First Approach
- Start with mobile layouts (320px+)
- Use flexible grids and responsive units
- Optimize touch targets (44px minimum)
- Consider thumb reach zones

### Accessibility
- Maintain 4.5:1 contrast ratio minimum
- Use semantic HTML elements
- Provide focus indicators
- Support keyboard navigation

### Performance
- Use CSS custom properties for theming
- Optimize images and assets
- Minimize CSS bundle size
- Use efficient selectors

## Design Principles
1. **Consistency:** Use design tokens throughout
2. **Simplicity:** Clear visual hierarchy
3. **Accessibility:** WCAG 2.1 AA compliance
4. **Performance:** Optimize for speed
5. **Scalability:** Flexible component system
"""
        
        return AgentResult(
            success=True,
            output=output,
            error_message=None,
            metadata={
                "components_defined": 3,
                "design_tokens": 15,
                "implementation_guidelines": 4,
                "design_method": "design_system"
            }
        )
    
    def _calculate_design_quality_score(self, result: AgentResult) -> int:
        """Calculate quality score for UI design results."""
        if not result.success:
            return 0
        
        # Base score
        score = 7
        
        # Add points for comprehensive design
        if "design system" in result.output.lower():
            score += 1
        
        # Add points for implementation details
        if "css" in result.output.lower() or "implementation" in result.output.lower():
            score += 1
        
        # Add points for accessibility considerations
        if "accessibility" in result.output.lower():
            score += 1
        
        # Add points for responsive design
        if "responsive" in result.output.lower() or "mobile" in result.output.lower():
            score += 1
        
        return min(score, 10)  # Cap at 10
    
    async def _design_interface(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Design a complete interface."""
        await asyncio.sleep(0.5)  # Simulate design time
        
        output = f"""# Interface Design Specification

## Design Context
Task: {task}
Context: {context.get('description', 'Interface design')}

## Layout Structure

### Header Section
- **Height:** 64px (4rem)
- **Background:** White with subtle shadow
- **Content:** Logo, navigation, user menu
- **Responsive:** Collapses to hamburger on mobile

### Main Content Area
- **Layout:** Grid system with 12 columns
- **Gutters:** 24px (1.5rem) between columns
- **Breakpoints:** Mobile (320px), Tablet (768px), Desktop (1024px+)

### Footer Section
- **Height:** Auto with minimum 200px
- **Background:** Neutral color with brand accents
- **Content:** Links, social media, legal info

## Component Layout

### Hero Section
```html
<section class="hero">
  <div class="container">
    <h1 class="hero-title">Main Headline</h1>
    <p class="hero-subtitle">Supporting text</p>
    <div class="hero-actions">
      <button class="btn btn-primary">Primary Action</button>
      <button class="btn btn-secondary">Secondary Action</button>
    </div>
  </div>
</section>
```

### Feature Cards
```html
<div class="features-grid">
  <div class="feature-card">
    <div class="feature-icon">📱</div>
    <h3 class="feature-title">Feature Title</h3>
    <p class="feature-description">Description text</p>
  </div>
</div>
```

## Responsive Behavior

### Mobile (< 768px)
- Single column layout
- Stacked navigation
- Larger touch targets (44px minimum)
- Reduced font sizes for readability

### Tablet (768px - 1024px)
- Two-column grid where appropriate
- Side navigation option
- Medium-sized components

### Desktop (> 1024px)
- Multi-column layouts
- Hover effects and interactions
- Full navigation menu
- Larger content areas

## Interaction Patterns

### Navigation
- **Primary:** Horizontal menu with dropdowns
- **Mobile:** Hamburger menu with slide-out panel
- **Breadcrumbs:** For deep page hierarchies
- **Pagination:** For long content lists

### Forms
- **Layout:** Single column on mobile, multi-column on desktop
- **Validation:** Real-time feedback with clear error messages
- **Progress:** Step indicators for multi-step forms
- **Submission:** Clear success/error states

### Modals & Overlays
- **Trigger:** Button clicks or automatic display
- **Backdrop:** Semi-transparent overlay
- **Close:** X button, escape key, or backdrop click
- **Focus:** Trap focus within modal content

## Visual Hierarchy

### Typography Scale
- **H1:** 36px/40px - Page titles
- **H2:** 24px/32px - Section headers
- **H3:** 20px/28px - Card titles
- **Body:** 16px/24px - Main content
- **Small:** 14px/20px - Secondary text

### Color Usage
- **Primary:** Main actions and branding
- **Secondary:** Supporting elements
- **Neutral:** Text and backgrounds
- **Semantic:** Success, warning, error states

## Accessibility Features
- **Keyboard Navigation:** All interactive elements
- **Screen Reader:** Proper ARIA labels
- **Color Contrast:** 4.5:1 minimum ratio
- **Focus Indicators:** Clear visual feedback
- **Alternative Text:** For all images and icons

## Performance Considerations
- **Lazy Loading:** For images and heavy content
- **Critical CSS:** Inline above-the-fold styles
- **Optimized Assets:** Compressed images and fonts
- **Caching:** Static assets with long cache times
"""
        
        return AgentResult(
            success=True,
            output=output,
            error_message=None,
            metadata={
                "layout_sections": 3,
                "components_defined": 4,
                "responsive_breakpoints": 3,
                "design_method": "interface_design"
            }
        )
    
    async def _create_color_system(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Create a comprehensive color system."""
        await asyncio.sleep(0.5)  # Simulate design time
        
        output = f"""# Color System Specification

## Design Context
Task: {task}
Context: {context.get('description', 'Color system creation')}

## Color Palette

### Primary Colors
```css
/* Blue Primary */
--primary-50: #eff6ff;
--primary-100: #dbeafe;
--primary-200: #bfdbfe;
--primary-300: #93c5fd;
--primary-400: #60a5fa;
--primary-500: #3b82f6;
--primary-600: #2563eb;
--primary-700: #1d4ed8;
--primary-800: #1e40af;
--primary-900: #1e3a8a;
```

### Neutral Colors
```css
/* Gray Neutral */
--neutral-50: #f9fafb;
--neutral-100: #f3f4f6;
--neutral-200: #e5e7eb;
--neutral-300: #d1d5db;
--neutral-400: #9ca3af;
--neutral-500: #6b7280;
--neutral-600: #4b5563;
--neutral-700: #374151;
--neutral-800: #1f2937;
--neutral-900: #111827;
```

### Semantic Colors
```css
/* Success Colors */
--success-50: #f0fdf4;
--success-500: #10b981;
--success-600: #059669;
--success-700: #047857;

/* Warning Colors */
--warning-50: #fffbeb;
--warning-500: #f59e0b;
--warning-600: #d97706;
--warning-700: #b45309;

/* Error Colors */
--error-50: #fef2f2;
--error-500: #ef4444;
--error-600: #dc2626;
--error-700: #b91c1c;
```

## Color Usage Guidelines

### Primary Color Usage
- **Brand Identity:** Logo, primary buttons, links
- **Navigation:** Active states, selected items
- **Accents:** Highlights, important information
- **Interactive Elements:** Hover states, focus indicators

### Neutral Color Usage
- **Text:** Primary text (neutral-900), secondary text (neutral-600)
- **Backgrounds:** Page backgrounds (neutral-50), card backgrounds (white)
- **Borders:** Subtle borders (neutral-200), dividers (neutral-300)
- **Shadows:** Drop shadows with neutral colors

### Semantic Color Usage
- **Success:** Completed actions, positive feedback
- **Warning:** Caution states, pending actions
- **Error:** Error messages, destructive actions
- **Info:** Informational messages, tips

## Accessibility Compliance

### Contrast Ratios
- **Normal Text:** 4.5:1 minimum (WCAG AA)
- **Large Text:** 3:1 minimum (WCAG AA)
- **UI Components:** 3:1 minimum (WCAG AA)

### Color Blindness Considerations
- **Red-Green:** Avoid relying solely on color for information
- **Blue-Yellow:** Use sufficient contrast for blue text
- **Monochrome:** Ensure information is distinguishable without color

## Implementation Examples

### Button Color Classes
```css
.btn-primary {
  background-color: var(--primary-600);
  color: white;
  border: 1px solid var(--primary-600);
}

.btn-primary:hover {
  background-color: var(--primary-700);
  border-color: var(--primary-700);
}

.btn-secondary {
  background-color: transparent;
  color: var(--primary-600);
  border: 1px solid var(--primary-600);
}

.btn-success {
  background-color: var(--success-600);
  color: white;
  border: 1px solid var(--success-600);
}

.btn-error {
  background-color: var(--error-600);
  color: white;
  border: 1px solid var(--error-600);
}
```

### Text Color Classes
```css
.text-primary {
  color: var(--primary-600);
}

.text-secondary {
  color: var(--neutral-600);
}

.text-success {
  color: var(--success-600);
}

.text-warning {
  color: var(--warning-600);
}

.text-error {
  color: var(--error-600);
}
```

### Background Color Classes
```css
.bg-primary {
  background-color: var(--primary-50);
}

.bg-neutral {
  background-color: var(--neutral-50);
}

.bg-success {
  background-color: var(--success-50);
}

.bg-warning {
  background-color: var(--warning-50);
}

.bg-error {
  background-color: var(--error-50);
}
```

## Dark Mode Support

### Dark Mode Color Mapping
```css
/* Light Mode (Default) */
:root {
  --bg-primary: var(--neutral-50);
  --bg-secondary: white;
  --text-primary: var(--neutral-900);
  --text-secondary: var(--neutral-600);
  --border-color: var(--neutral-200);
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
  :root {
    --bg-primary: var(--neutral-900);
    --bg-secondary: var(--neutral-800);
    --text-primary: var(--neutral-100);
    --text-secondary: var(--neutral-400);
    --border-color: var(--neutral-700);
  }
}
```

## Color Testing Checklist
- [ ] All color combinations meet WCAG AA standards
- [ ] Colors work in grayscale/monochrome
- [ ] Colors are distinguishable for color-blind users
- [ ] Dark mode colors provide sufficient contrast
- [ ] Colors maintain brand consistency across platforms
- [ ] Color usage is consistent throughout the interface
"""
        
        return AgentResult(
            success=True,
            output=output,
            error_message=None,
            metadata={
                "color_palettes": 4,
                "accessibility_guidelines": 3,
                "implementation_examples": 3,
                "design_method": "color_system"
            }
        )
    
    async def _design_typography(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Design typography system."""
        await asyncio.sleep(0.5)  # Simulate design time
        
        output = f"""# Typography System Specification

## Design Context
Task: {task}
Context: {context.get('description', 'Typography design')}

## Font Stack

### Primary Font Family
```css
/* System Font Stack */
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, 
             "Helvetica Neue", Arial, "Noto Sans", sans-serif, 
             "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", 
             "Noto Color Emoji";
```

### Monospace Font Family
```css
/* Code Font Stack */
font-family: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", 
             Consolas, "Courier New", monospace;
```

## Type Scale

### Display Text
```css
.display-1 {{
  font-size: 3.75rem;    /* 60px */
  line-height: 1.2;
  font-weight: 700;
  letter-spacing: -0.025em;
}}

.display-2 {{
  font-size: 3rem;       /* 48px */
  line-height: 1.25;
  font-weight: 700;
  letter-spacing: -0.025em;
}}
```

### Heading Text
```css
.h1 {{
  font-size: 2.25rem;    /* 36px */
  line-height: 1.3;
  font-weight: 700;
  letter-spacing: -0.025em;
}}

.h2 {{
  font-size: 1.875rem;   /* 30px */
  line-height: 1.35;
  font-weight: 600;
  letter-spacing: -0.025em;
}}

.h3 {{
  font-size: 1.5rem;     /* 24px */
  line-height: 1.4;
  font-weight: 600;
  letter-spacing: -0.025em;
}}

.h4 {{
  font-size: 1.25rem;    /* 20px */
  line-height: 1.45;
  font-weight: 600;
  letter-spacing: -0.025em;
}}

.h5 {{
  font-size: 1.125rem;   /* 18px */
  line-height: 1.5;
  font-weight: 600;
  letter-spacing: -0.025em;
}}

.h6 {{
  font-size: 1rem;       /* 16px */
  line-height: 1.5;
  font-weight: 600;
  letter-spacing: -0.025em;
}}
```

### Body Text
```css
.body-large {{
  font-size: 1.125rem;   /* 18px */
  line-height: 1.6;
  font-weight: 400;
}}

.body {{
  font-size: 1rem;       /* 16px */
  line-height: 1.6;
  font-weight: 400;
}}

.body-small {{
  font-size: 0.875rem;   /* 14px */
  line-height: 1.6;
  font-weight: 400;
}}
```

### Caption Text
```css
.caption {{
  font-size: 0.75rem;    /* 12px */
  line-height: 1.5;
  font-weight: 400;
}}

.caption-small {{
  font-size: 0.625rem;   /* 10px */
  line-height: 1.5;
  font-weight: 400;
}}
```

## Font Weights

### Weight Scale
```css
.font-thin {{
  font-weight: 100;
}}

.font-light {{
  font-weight: 300;
}}

.font-normal {{
  font-weight: 400;
}}

.font-medium {{
  font-weight: 500;
}}

.font-semibold {{
  font-weight: 600;
}}

.font-bold {{
  font-weight: 700;
}}

.font-extrabold {{
  font-weight: 800;
}}

.font-black {{
  font-weight: 900;
}}
```

## Text Utilities

### Text Alignment
```css
.text-left {{
  text-align: left;
}}

.text-center {{
  text-align: center;
}}

.text-right {{
  text-align: right;
}}

.text-justify {{
  text-align: justify;
}}
```

### Text Transform
```css
.text-uppercase {{
  text-transform: uppercase;
}}

.text-lowercase {{
  text-transform: lowercase;
}}

.text-capitalize {{
  text-transform: capitalize;
}}
```

### Text Decoration
```css
.text-underline {{
  text-decoration: underline;
}}

.text-line-through {{
  text-decoration: line-through;
}}

.text-no-decoration {{
  text-decoration: none;
}}
```

## Responsive Typography

### Mobile-First Approach
```css
/* Base styles (mobile) */
h1 {{ font-size: 1.875rem; }}  /* 30px */
h2 {{ font-size: 1.5rem; }}    /* 24px */
h3 {{ font-size: 1.25rem; }}   /* 20px */

/* Tablet (768px+) */
@media (min-width: 768px) {{
  h1 {{ font-size: 2.25rem; }}  /* 36px */
  h2 {{ font-size: 1.875rem; }} /* 30px */
  h3 {{ font-size: 1.5rem; }}   /* 24px */
}}

/* Desktop (1024px+) */
@media (min-width: 1024px) {{
  h1 {{ font-size: 3rem; }}     /* 48px */
  h2 {{ font-size: 2.25rem; }}  /* 36px */
  h3 {{ font-size: 1.875rem; }} /* 30px */
}}
```

## Accessibility Guidelines

### Minimum Font Sizes
- **Body Text:** 16px minimum for readability
- **Captions:** 12px minimum for legibility
- **UI Elements:** 14px minimum for touch targets

### Line Height Guidelines
- **Headings:** 1.2-1.4 for tight, professional look
- **Body Text:** 1.5-1.7 for comfortable reading
- **Long Text:** 1.6-1.8 for extended content

### Contrast Requirements
- **Normal Text:** 4.5:1 contrast ratio minimum
- **Large Text:** 3:1 contrast ratio minimum
- **UI Elements:** 3:1 contrast ratio minimum

## Implementation Best Practices

### CSS Custom Properties
```css
:root {{
  /* Font Sizes */
  --font-size-xs: 0.75rem;
  --font-size-sm: 0.875rem;
  --font-size-base: 1rem;
  --font-size-lg: 1.125rem;
  --font-size-xl: 1.25rem;
  
  /* Line Heights */
  --line-height-tight: 1.25;
  --line-height-normal: 1.5;
  --line-height-relaxed: 1.75;
  
  /* Font Weights */
  --font-weight-normal: 400;
  --font-weight-medium: 500;
  --font-weight-semibold: 600;
  --font-weight-bold: 700;
}}
```

### Utility Classes
```css
/* Size Utilities */
.text-xs {{ font-size: var(--font-size-xs); }}
.text-sm {{ font-size: var(--font-size-sm); }}
.text-base {{ font-size: var(--font-size-base); }}
.text-lg {{ font-size: var(--font-size-lg); }}
.text-xl {{ font-size: var(--font-size-xl); }}

/* Weight Utilities */
.font-normal {{ font-weight: var(--font-weight-normal); }}
.font-medium {{ font-weight: var(--font-weight-medium); }}
.font-semibold {{ font-weight: var(--font-weight-semibold); }}
.font-bold {{ font-weight: var(--font-weight-bold); }}
```

## Testing Checklist
- [ ] All font sizes meet accessibility standards
- [ ] Line heights provide comfortable reading
- [ ] Font weights are distinguishable
- [ ] Responsive typography works across devices
- [ ] Text remains readable at all sizes
- [ ] Font loading doesn't cause layout shifts
- [ ] Fallback fonts provide good experience
"""
        
        return AgentResult(
            success=True,
            output=output,
            error_message=None,
            metadata={
                "font_scales": 4,
                "utility_classes": 12,
                "accessibility_guidelines": 3,
                "design_method": "typography_design"
            }
        )
    
    async def _comprehensive_ui_design(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Conduct comprehensive UI design analysis."""
        await asyncio.sleep(0.5)  # Simulate design time
        
        output = f"""# Comprehensive UI Design Analysis

## Design Context
Task: {task}
Context: {context.get('description', 'Comprehensive UI design')}

## Design Strategy

### Visual Hierarchy
1. **Primary Elements:** Headlines, main CTAs, key information
2. **Secondary Elements:** Supporting text, secondary actions
3. **Tertiary Elements:** Metadata, fine print, decorative elements

### Layout Principles
- **Grid System:** 12-column responsive grid
- **Spacing:** Consistent 8px base unit system
- **Alignment:** Left-aligned text, centered CTAs
- **Proximity:** Related elements grouped together

### Color Strategy
- **Primary:** Brand color for main actions
- **Secondary:** Supporting color for accents
- **Neutral:** Grays for text and backgrounds
- **Semantic:** Success, warning, error states

## Component Design

### Button System
```css
/* Primary Button */
.btn-primary {
  background: var(--primary-600);
  color: white;
  padding: 0.75rem 1.5rem;
  border-radius: 0.375rem;
  font-weight: 600;
  transition: all 0.2s;
}

/* Secondary Button */
.btn-secondary {
  background: transparent;
  color: var(--primary-600);
  border: 2px solid var(--primary-600);
  padding: 0.75rem 1.5rem;
  border-radius: 0.375rem;
  font-weight: 600;
  transition: all 0.2s;
}
```

### Card System
```css
.card {
  background: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
  border: 1px solid var(--neutral-200);
}

.card-hover {
  transition: all 0.2s;
}

.card-hover:hover {
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}
```

### Form System
```css
.form-group {
  margin-bottom: 1rem;
}

.form-label {
  display: block;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: var(--neutral-700);
}

.form-input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid var(--neutral-300);
  border-radius: 0.375rem;
  font-size: 1rem;
  transition: all 0.2s;
}

.form-input:focus {
  outline: none;
  border-color: var(--primary-500);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
```

## Responsive Design

### Breakpoint Strategy
- **Mobile:** 320px - 767px
- **Tablet:** 768px - 1023px
- **Desktop:** 1024px+

### Mobile-First Approach
1. **Start with mobile layout**
2. **Add tablet styles with min-width media queries**
3. **Add desktop styles with min-width media queries**

### Touch-Friendly Design
- **Minimum touch target:** 44px x 44px
- **Spacing between touch targets:** 8px minimum
- **Thumb-friendly navigation:** Bottom navigation on mobile

## Accessibility Features

### Keyboard Navigation
- **Tab order:** Logical flow through interactive elements
- **Focus indicators:** Clear visual feedback
- **Skip links:** Jump to main content

### Screen Reader Support
- **Semantic HTML:** Proper heading hierarchy
- **ARIA labels:** Descriptive labels for complex elements
- **Alt text:** Meaningful descriptions for images

### Color and Contrast
- **Contrast ratio:** 4.5:1 minimum for normal text
- **Color independence:** Information not conveyed by color alone
- **High contrast mode:** Support for system preferences

## Performance Optimization

### Font Loading
```css
/* Preload critical fonts */
<link rel="preload" href="/fonts/inter-var.woff2" as="font" type="font/woff2" crossorigin>

/* Font display swap for better performance */
@font-face {
  font-family: 'Inter';
  font-display: swap;
  src: url('/fonts/inter-var.woff2') format('woff2');
}
```

### Image Optimization
- **WebP format:** Modern browsers
- **Fallback formats:** JPEG/PNG for older browsers
- **Responsive images:** Different sizes for different screens
- **Lazy loading:** Images load as needed

### CSS Optimization
- **Critical CSS:** Inline above-the-fold styles
- **Minification:** Remove unnecessary whitespace
- **Purge unused CSS:** Remove unused styles in production

## Implementation Guidelines

### CSS Architecture
```css
/* Base styles */
@import 'base/reset.css';
@import 'base/typography.css';
@import 'base/colors.css';

/* Components */
@import 'components/buttons.css';
@import 'components/cards.css';
@import 'components/forms.css';

/* Layout */
@import 'layout/header.css';
@import 'layout/footer.css';
@import 'layout/grid.css';

/* Utilities */
@import 'utilities/spacing.css';
@import 'utilities/text.css';
@import 'utilities/colors.css';
```

### Component Structure
```html
<!-- Button Component -->
<button class="btn btn-primary" type="button">
  <span class="btn-text">Button Text</span>
  <svg class="btn-icon" aria-hidden="true">
    <!-- Icon SVG -->
  </svg>
</button>

<!-- Card Component -->
<div class="card">
  <div class="card-header">
    <h3 class="card-title">Card Title</h3>
  </div>
  <div class="card-body">
    <p class="card-text">Card content</p>
  </div>
  <div class="card-footer">
    <button class="btn btn-primary">Action</button>
  </div>
</div>
```

## Quality Assurance

### Design Review Checklist
- [ ] Visual hierarchy is clear and logical
- [ ] Color usage is consistent and accessible
- [ ] Typography is readable at all sizes
- [ ] Spacing is consistent throughout
- [ ] Interactive elements have clear states
- [ ] Responsive behavior works correctly
- [ ] Accessibility requirements are met
- [ ] Performance optimizations are implemented

### Testing Strategy
1. **Cross-browser testing:** Chrome, Firefox, Safari, Edge
2. **Device testing:** Mobile, tablet, desktop
3. **Accessibility testing:** Screen readers, keyboard navigation
4. **Performance testing:** Page load times, Core Web Vitals
5. **User testing:** Real user feedback and behavior

## Success Metrics
- **Design Consistency:** 95% component reuse
- **Accessibility:** WCAG 2.1 AA compliance
- **Performance:** 90+ Lighthouse score
- **User Satisfaction:** 4.5+ rating on design quality
- **Development Speed:** 50% faster component implementation
"""
        
        return AgentResult(
            success=True,
            output=output,
            error_message=None,
            metadata={
                "design_principles": 4,
                "component_systems": 3,
                "implementation_guidelines": 6,
                "quality_metrics": 5,
                "design_method": "comprehensive_ui_design"
            }
        )


# Export the main class
__all__ = ['UIDesigner', 'DesignComponent', 'ColorPalette', 'DesignStyle'] 
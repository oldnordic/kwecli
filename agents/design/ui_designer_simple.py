#!/usr/bin/env python3
"""
UI Designer Agent - Interface Design Specialist (Simplified)

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
- Primary Colors: Blue (#3b82f6, #2563eb, #1d4ed8)
- Neutral Colors: Gray scale (#f9fafb to #111827)
- Semantic Colors: Success (#10b981), Warning (#f59e0b), Error (#ef4444)

### Typography Scale
- Display: 60px/48px for hero headlines
- H1: 36px for page titles
- H2: 30px for section headers
- H3: 24px for card titles
- Body: 16px for main content
- Small: 14px for secondary text
- Caption: 12px for fine print

### Spacing System
- Base unit: 4px
- Scale: 4px, 8px, 12px, 16px, 24px, 32px, 48px, 64px

## Core Components

### Button Component
- **Primary Button:** Blue background, white text, rounded corners
- **Secondary Button:** Transparent background, blue border and text
- **States:** Default, hover, active, disabled, loading

### Card Component
- **Background:** White with subtle shadow
- **Border Radius:** 8px
- **Padding:** 24px
- **States:** Default, hover (elevated shadow)

### Input Component
- **Border:** Light gray with focus state
- **Padding:** 12px
- **Border Radius:** 6px
- **States:** Default, focus, error, success, disabled

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
- **Structure:** Full-width section with centered content
- **Typography:** Large headline with supporting text
- **Actions:** Primary and secondary buttons
- **Background:** Optional gradient or image overlay

### Feature Cards
- **Layout:** Grid of cards with icons
- **Content:** Title, description, optional action
- **Spacing:** Consistent padding and margins
- **Responsive:** Stack on mobile, grid on desktop

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
- **Blue Primary:** #eff6ff, #dbeafe, #bfdbfe, #93c5fd, #60a5fa, #3b82f6, #2563eb, #1d4ed8, #1e40af, #1e3a8a

### Neutral Colors
- **Gray Neutral:** #f9fafb, #f3f4f6, #e5e7eb, #d1d5db, #9ca3af, #6b7280, #4b5563, #374151, #1f2937, #111827

### Semantic Colors
- **Success Colors:** #f0fdf4, #10b981, #059669, #047857
- **Warning Colors:** #fffbeb, #f59e0b, #d97706, #b45309
- **Error Colors:** #fef2f2, #ef4444, #dc2626, #b91c1c

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
- **Primary Button:** Blue background, white text, blue border
- **Secondary Button:** Transparent background, blue text and border
- **Success Button:** Green background, white text, green border
- **Error Button:** Red background, white text, red border

### Text Color Classes
- **Primary Text:** Blue color for brand elements
- **Secondary Text:** Gray color for supporting text
- **Success Text:** Green color for positive feedback
- **Warning Text:** Amber color for caution states
- **Error Text:** Red color for error messages

### Background Color Classes
- **Primary Background:** Light blue for brand sections
- **Neutral Background:** Light gray for content areas
- **Success Background:** Light green for success states
- **Warning Background:** Light amber for warning states
- **Error Background:** Light red for error states

## Dark Mode Support

### Dark Mode Color Mapping
- **Light Mode (Default):** Light backgrounds, dark text
- **Dark Mode:** Dark backgrounds, light text
- **Contrast:** Maintain accessibility in both modes

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
- **System Font Stack:** -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif

### Monospace Font Family
- **Code Font Stack:** "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, "Courier New", monospace

## Type Scale

### Display Text
- **Display 1:** 60px, line-height 1.2, font-weight 700
- **Display 2:** 48px, line-height 1.25, font-weight 700

### Heading Text
- **H1:** 36px, line-height 1.3, font-weight 700
- **H2:** 30px, line-height 1.35, font-weight 600
- **H3:** 24px, line-height 1.4, font-weight 600
- **H4:** 20px, line-height 1.45, font-weight 600
- **H5:** 18px, line-height 1.5, font-weight 600
- **H6:** 16px, line-height 1.5, font-weight 600

### Body Text
- **Body Large:** 18px, line-height 1.6, font-weight 400
- **Body:** 16px, line-height 1.6, font-weight 400
- **Body Small:** 14px, line-height 1.6, font-weight 400

### Caption Text
- **Caption:** 12px, line-height 1.5, font-weight 400
- **Caption Small:** 10px, line-height 1.5, font-weight 400

## Font Weights

### Weight Scale
- **Thin:** 100
- **Light:** 300
- **Normal:** 400
- **Medium:** 500
- **Semibold:** 600
- **Bold:** 700
- **Extrabold:** 800
- **Black:** 900

## Text Utilities

### Text Alignment
- **Left:** text-align left
- **Center:** text-align center
- **Right:** text-align right
- **Justify:** text-align justify

### Text Transform
- **Uppercase:** text-transform uppercase
- **Lowercase:** text-transform lowercase
- **Capitalize:** text-transform capitalize

### Text Decoration
- **Underline:** text-decoration underline
- **Line Through:** text-decoration line-through
- **No Decoration:** text-decoration none

## Responsive Typography

### Mobile-First Approach
- **Base styles (mobile):** H1 30px, H2 24px, H3 20px
- **Tablet (768px+):** H1 36px, H2 30px, H3 24px
- **Desktop (1024px+):** H1 48px, H2 36px, H3 30px

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
- **Font Sizes:** --font-size-xs, --font-size-sm, --font-size-base, --font-size-lg, --font-size-xl
- **Line Heights:** --line-height-tight, --line-height-normal, --line-height-relaxed
- **Font Weights:** --font-weight-normal, --font-weight-medium, --font-weight-semibold, --font-weight-bold

### Utility Classes
- **Size Utilities:** .text-xs, .text-sm, .text-base, .text-lg, .text-xl
- **Weight Utilities:** .font-normal, .font-medium, .font-semibold, .font-bold

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
- **Primary Button:** Brand color background, white text, rounded corners
- **Secondary Button:** Transparent background, brand color border and text
- **States:** Default, hover, active, disabled, loading

### Card System
- **Background:** White with subtle shadow
- **Border Radius:** 8px
- **Padding:** 24px
- **Hover Effect:** Elevated shadow with smooth transition

### Form System
- **Layout:** Single column on mobile, multi-column on desktop
- **Validation:** Real-time feedback with clear error messages
- **States:** Default, focus, error, success, disabled
- **Accessibility:** Proper labels and ARIA attributes

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
- **Preload critical fonts:** Optimize loading performance
- **Font display swap:** Prevent layout shifts
- **Fallback fonts:** Ensure text displays immediately

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
- **Base styles:** Reset, typography, colors
- **Components:** Buttons, cards, forms
- **Layout:** Header, footer, grid
- **Utilities:** Spacing, text, colors

### Component Structure
- **Button Component:** Semantic HTML with proper states
- **Card Component:** Flexible container with header, body, footer
- **Form Component:** Accessible inputs with validation

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


# Export the main class
__all__ = ['UIDesigner', 'DesignComponent', 'ColorPalette', 'DesignStyle'] 
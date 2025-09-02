#!/usr/bin/env python3
"""
Frontend Developer Sub-Agent Implementation.

This agent specializes in frontend development, UI/UX implementation,
responsive design, and interactive features for web and mobile applications.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from agents.base_agent import (
    SubAgent, AgentResult, AgentStatus, AgentExpertise
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrontendRequest:
    """Request for frontend development implementation."""
    frontend_type: str  # "react", "vue", "mobile", "responsive", "interactive"
    requirements: List[str]
    context: Optional[str] = None
    design_constraints: Optional[Dict[str, Any]] = None
    technology_preferences: Optional[List[str]] = None


class FrontendDeveloper(SubAgent):
    """
    Frontend Developer sub-agent specializing in user interface development.
    
    This agent handles:
    - React and Vue.js development
    - Mobile app development (React Native, Flutter)
    - Responsive web design
    - Interactive features and animations
    - UI/UX implementation
    - Frontend performance optimization
    """

    def __init__(self):
        """Initialize the Frontend Developer agent."""
        super().__init__(
            name="Frontend Developer",
            expertise=[
                AgentExpertise.FRONTEND_DEVELOPMENT,
                AgentExpertise.UI_DESIGN,
                AgentExpertise.MOBILE_DEVELOPMENT
            ],
            tools=[
                "React", "Vue", "Angular", "TypeScript", "JavaScript",
                "React Native", "Flutter", "Ionic", "Cordova",
                "CSS3", "Sass", "Less", "Tailwind CSS", "Bootstrap",
                "Webpack", "Vite", "Babel", "ESLint", "Prettier",
                "Jest", "Cypress", "Storybook", "Framer Motion",
                "Redux", "Vuex", "Context API", "Hooks"
            ],
            description="Frontend development specialist focusing on user interface implementation"
        )

    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """
        Execute frontend development task.
        
        Args:
            task: The frontend task to execute
            context: Additional context for the task
            
        Returns:
            AgentResult with the frontend implementation results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Frontend Developer starting task: {task}")
            
            # Parse the task to determine the type of frontend work needed
            frontend_type = await self._determine_frontend_type(task)
            
            # Create implementation plan
            implementation_plan = await self._create_implementation_plan(
                task, frontend_type, context
            )
            
            # Generate the frontend implementation
            implementation = await self._generate_implementation(
                task, frontend_type, implementation_plan, context
            )
            
            # Validate the implementation
            validation_result = await self._validate_implementation(
                implementation, frontend_type, context
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(
                implementation, frontend_type
            )
            
            # Create result
            result = AgentResult(
                success=validation_result.success,
                output=implementation,
                metadata={
                    "frontend_type": frontend_type,
                    "implementation_plan": implementation_plan,
                    "validation_result": validation_result.metadata,
                    "performance_metrics": performance_metrics,
                    "agent": self.name
                },
                error_message=validation_result.error_message
            )
            
            # Record the work
            self._record_work(task, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Frontend Developer task failed: {e}")
            execution_time = asyncio.get_event_loop().time() - start_time
            
            error_result = AgentResult(
                success=False,
                output="",
                metadata={"error": str(e), "agent": self.name},
                error_message=str(e)
            )
            
            # Record the failed work
            self._record_work(task, error_result)
            
            return error_result

    def can_handle(self, task: str) -> bool:
        """
        Check if this agent can handle the given task.
        
        Args:
            task: The task to check
            
        Returns:
            True if the agent can handle the task, False otherwise
        """
        task_lower = task.lower()
        
        # Frontend development keywords - more specific to avoid false positives
        frontend_keywords = [
            "react", "vue", "angular", "component", "frontend",
            "ui", "ux", "interface", "user interface", "web app",
            "mobile app", "mobile", "responsive", "layout",
            "css", "javascript", "typescript", "html",
            "animation", "interactive", "user experience",
            "react native", "flutter", "ionic", "cordova",
            "bootstrap", "tailwind", "sass", "less",
            "webpack", "vite", "babel", "eslint", "prettier",
            "landing", "page", "dashboard"
        ]
        
        # Exclude keywords that might cause false positives
        exclude_keywords = [
            "database", "schema", "backend", "api", "server",
            "machine learning", "ml", "ai", "model", "algorithm",
            "infrastructure", "devops", "pipeline", "deployment"
        ]
        
        # Check if task contains any excluded keywords (but allow "design" in frontend context)
        for exclude_keyword in exclude_keywords:
            if exclude_keyword in task_lower:
                # Special case: allow "design" if it's clearly frontend-related
                if (exclude_keyword == "design" and 
                    any(frontend_keyword in task_lower 
                        for frontend_keyword in ["ui", "ux", "interface", "frontend", 
                                               "mobile", "web", "landing", "page", 
                                               "component"])):
                    continue
                return False
        
        return any(keyword in task_lower for keyword in frontend_keywords)

    def get_expertise(self) -> List[AgentExpertise]:
        """
        Get the agent's areas of expertise.
        
        Returns:
            List of expertise areas
        """
        return self.expertise

    async def _determine_frontend_type(self, task: str) -> str:
        """Determine the type of frontend development needed."""
        task_lower = task.lower()
        
        # Check for specific technology mentions first
        if any(keyword in task_lower for keyword in ["react", "jsx"]):
            return "react"
        elif any(keyword in task_lower for keyword in ["vue", "template", "script"]):
            return "vue"
        elif any(keyword in task_lower for keyword in ["mobile", "app", "native", "flutter"]):
            return "mobile"
        elif any(keyword in task_lower for keyword in ["animation", "interactive", "transition"]):
            return "interactive"
        elif any(keyword in task_lower for keyword in ["responsive", "layout", "css", "design"]):
            return "responsive"
        else:
            # Default to React for general UI tasks
            return "react"

    async def _create_implementation_plan(self, task: str, frontend_type: str, 
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed implementation plan for the frontend feature."""
        
        plans = {
            "react": {
                "components": [
                    "Component Design",
                    "State Management",
                    "Props and Events",
                    "Styling and CSS",
                    "Testing and Validation"
                ],
                "tools": ["React", "TypeScript", "CSS", "Jest"],
                "considerations": [
                    "Component reusability",
                    "Performance optimization",
                    "Accessibility standards",
                    "Cross-browser compatibility"
                ]
            },
            "vue": {
                "components": [
                    "Component Design",
                    "Vue Router",
                    "Vuex State Management",
                    "Template and Script",
                    "Styling and Scoped CSS"
                ],
                "tools": ["Vue", "TypeScript", "CSS", "Vue Test Utils"],
                "considerations": [
                    "Vue ecosystem integration",
                    "Component composition",
                    "Reactive data flow",
                    "Build optimization"
                ]
            },
            "mobile": {
                "components": [
                    "Mobile Design",
                    "Navigation Structure",
                    "Platform-specific Features",
                    "Performance Optimization",
                    "Testing on Devices"
                ],
                "tools": ["React Native", "Flutter", "Ionic", "Cordova"],
                "considerations": [
                    "Platform differences",
                    "Performance on mobile",
                    "Touch interactions",
                    "App store requirements"
                ]
            },
            "responsive": {
                "components": [
                    "Responsive Layout",
                    "CSS Grid and Flexbox",
                    "Media Queries",
                    "Mobile-first Design",
                    "Cross-device Testing"
                ],
                "tools": ["CSS3", "Sass", "Tailwind CSS", "Bootstrap"],
                "considerations": [
                    "Breakpoint strategy",
                    "Performance across devices",
                    "Touch-friendly interfaces",
                    "Loading optimization"
                ]
            },
            "interactive": {
                "components": [
                    "Animation Design",
                    "User Interactions",
                    "State Transitions",
                    "Performance Monitoring",
                    "Accessibility Features"
                ],
                "tools": ["Framer Motion", "CSS Animations", "JavaScript", "React Spring"],
                "considerations": [
                    "Animation performance",
                    "User experience flow",
                    "Accessibility compliance",
                    "Cross-browser support"
                ]
            }
        }
        
        # Handle general case with a default plan
        if frontend_type == "general":
            return {
                "components": [
                    "Frontend Analysis",
                    "Component Design",
                    "Implementation",
                    "Testing and Validation"
                ],
                "tools": ["JavaScript", "CSS", "HTML", "Testing"],
                "considerations": [
                    "User experience",
                    "Performance optimization",
                    "Cross-browser compatibility",
                    "Maintainability"
                ]
            }
        
        return plans.get(frontend_type, plans["react"])  # Default to React

    async def _generate_implementation(self, task: str, frontend_type: str,
                                    plan: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate the actual frontend implementation code."""
        
        implementation_templates = {
            "react": self._generate_react_implementation(task, plan, context),
            "vue": self._generate_vue_implementation(task, plan, context),
            "mobile": self._generate_mobile_implementation(task, plan, context),
            "responsive": self._generate_responsive_implementation(task, plan, context),
            "interactive": self._generate_interactive_implementation(task, plan, context)
        }
        
        return implementation_templates.get(frontend_type, self._generate_general_implementation(task, plan, context))

    def _generate_react_implementation(self, task: str, plan: Dict[str, Any], 
                                     context: Dict[str, Any]) -> str:
        """Generate React implementation."""
        
        return f'''
// Frontend Developer Implementation: React Component
// Task: {task}

import React, {{ useState, useEffect }} from 'react';
import './styles.css';

interface Props {{
    // Add your props here
}}

const Component: React.FC<Props> = ({{ }}) => {{
    const [state, setState] = useState<string>('');
    
    useEffect(() => {{
        // Component initialization
        console.log('Component mounted');
    }}, []);
    
    const handleClick = () => {{
        setState('Updated');
    }};
    
    return (
        <div className="component">
            <h2>React Component</h2>
            <p>State: {{state}}</p>
            <button onClick={{handleClick}}>
                Update State
            </button>
        </div>
    );
}};

export default Component;
'''

    def _generate_vue_implementation(self, task: str, plan: Dict[str, Any], 
                                   context: Dict[str, Any]) -> str:
        """Generate Vue implementation."""
        
        return f'''
<!-- Frontend Developer Implementation: Vue Component -->
<!-- Task: {task} -->

<template>
  <div class="component">
    <h2>Vue Component</h2>
    <p>Message: {{ message }}</p>
    <button @click="updateMessage">
      Update Message
    </button>
  </div>
</template>

<script>
export default {{
  name: 'Component',
  data() {{
    return {{
      message: 'Hello Vue!'
    }}
  }},
  methods: {{
    updateMessage() {{
      this.message = 'Message updated!'
    }}
  }},
  mounted() {{
    console.log('Component mounted')
  }}
}}
</script>

<style scoped>
.component {{
  padding: 20px;
  border: 1px solid #ccc;
  border-radius: 8px;
}}
</style>
'''

    def _generate_mobile_implementation(self, task: str, plan: Dict[str, Any], 
                                      context: Dict[str, Any]) -> str:
        """Generate mobile implementation."""
        
        return f'''
// Frontend Developer Implementation: React Native Component
// Task: {task}

import React, {{ useState }} from 'react';
import {{ View, Text, TouchableOpacity, StyleSheet }} from 'react-native';

interface Props {{
    // Add your props here
}}

const MobileComponent: React.FC<Props> = ({{ }}) => {{
    const [count, setCount] = useState(0);
    
    const handlePress = () => {{
        setCount(count + 1);
    }};
    
    return (
        <View style={{styles.container}}>
            <Text style={{styles.title}}>Mobile Component</Text>
            <Text style={{styles.counter}}>Count: {{count}}</Text>
            <TouchableOpacity style={{styles.button}} onPress={{handlePress}}>
                <Text style={{styles.buttonText}}>Increment</Text>
            </TouchableOpacity>
        </View>
    );
}};

const styles = StyleSheet.create({{
    container: {{
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    }},
    title: {{
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 20,
    }},
    counter: {{
        fontSize: 18,
        marginBottom: 20,
    }},
    button: {{
        backgroundColor: '#007AFF',
        padding: 15,
        borderRadius: 8,
    }},
    buttonText: {{
        color: 'white',
        fontSize: 16,
        fontWeight: 'bold',
    }},
}});

export default MobileComponent;
'''

    def _generate_responsive_implementation(self, task: str, plan: Dict[str, Any], 
                                         context: Dict[str, Any]) -> str:
        """Generate responsive implementation."""
        
        return f'''
/* Frontend Developer Implementation: Responsive CSS */
/* Task: {task} */

.responsive-container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}}

.responsive-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
}}

.responsive-card {{
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
}}

.responsive-card:hover {{
    transform: translateY(-2px);
}}

/* Mobile-first responsive design */
@media (max-width: 768px) {{
    .responsive-container {{
        padding: 10px;
    }}
    
    .responsive-grid {{
        grid-template-columns: 1fr;
    }}
    
    .responsive-card {{
        padding: 15px;
    }}
}}

@media (max-width: 480px) {{
    .responsive-container {{
        padding: 5px;
    }}
    
    .responsive-card {{
        padding: 10px;
    }}
}}

/* Tablet styles */
@media (min-width: 769px) and (max-width: 1024px) {{
    .responsive-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
}}
'''

    def _generate_interactive_implementation(self, task: str, plan: Dict[str, Any], 
                                          context: Dict[str, Any]) -> str:
        """Generate interactive implementation."""
        
        return f'''
// Frontend Developer Implementation: Interactive Features
// Task: {task}

import React, {{ useState, useEffect }} from 'react';
import './interactive.css';

const InteractiveComponent = () => {{
    const [isVisible, setIsVisible] = useState(false);
    const [isAnimating, setIsAnimating] = useState(false);
    
    useEffect(() => {{
        // Trigger animation on mount
        const timer = setTimeout(() => {{
            setIsVisible(true);
        }}, 100);
        
        return () => clearTimeout(timer);
    }}, []);
    
    const handleAnimation = () => {{
        setIsAnimating(true);
        setTimeout(() => setIsAnimating(false), 1000);
    }};
    
    return (
        <div className="interactive-container">
            <div className={{`animated-element ${{isVisible ? 'visible' : ''}}`}}>
                <h2>Interactive Component</h2>
                <p>This component demonstrates smooth animations and interactions.</p>
            </div>
            
            <button 
                className={{`interactive-button ${{isAnimating ? 'animating' : ''}}`}}
                onClick={{handleAnimation}}
            >
                Trigger Animation
            </button>
            
            <div className="hover-effect">
                <p>Hover over this element for effects</p>
            </div>
        </div>
    );
}};

export default InteractiveComponent;

/* CSS for interactive features */
.interactive-container {{
    padding: 20px;
    text-align: center;
}}

.animated-element {{
    opacity: 0;
    transform: translateY(20px);
    transition: all 0.6s ease;
}}

.animated-element.visible {{
    opacity: 1;
    transform: translateY(0);
}}

.interactive-button {{
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    border: none;
    padding: 12px 24px;
    border-radius: 25px;
    color: white;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
}}

.interactive-button:hover {{
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}}

.interactive-button.animating {{
    animation: pulse 1s ease-in-out;
}}

.hover-effect {{
    margin-top: 20px;
    padding: 15px;
    border-radius: 8px;
    transition: all 0.3s ease;
}}

.hover-effect:hover {{
    background: #f0f0f0;
    transform: scale(1.02);
}}

@keyframes pulse {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.1); }}
    100% {{ transform: scale(1); }}
}}
'''

    def _generate_general_implementation(self, task: str, plan: Dict[str, Any], 
                                       context: Dict[str, Any]) -> str:
        """Generate general frontend implementation."""
        
        return f'''
// Frontend Developer Implementation: General Frontend
// Task: {task}

// HTML Structure
const htmlStructure = `
<div class="frontend-component">
    <header class="component-header">
        <h1>Frontend Component</h1>
        <p>This is a general frontend implementation</p>
    </header>
    
    <main class="component-main">
        <section class="content-section">
            <h2>Content Area</h2>
            <p>This component demonstrates general frontend development practices.</p>
        </section>
        
        <aside class="sidebar">
            <h3>Sidebar</h3>
            <ul>
                <li>Feature 1</li>
                <li>Feature 2</li>
                <li>Feature 3</li>
            </ul>
        </aside>
    </main>
    
    <footer class="component-footer">
        <p>&copy; 2024 Frontend Component</p>
    </footer>
</div>
`;

// CSS Styles
const cssStyles = `
.frontend-component {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    font-family: Arial, sans-serif;
}}

.component-header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
}}

.component-main {{
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 20px;
}}

.content-section {{
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}}

.sidebar {{
    background: #f8f9fa;
    padding: 20px;
    border-radius: 8px;
}}

.component-footer {{
    margin-top: 20px;
    text-align: center;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 8px;
}}

/* Responsive design */
@media (max-width: 768px) {{
    .component-main {{
        grid-template-columns: 1fr;
    }}
}}
`;

// JavaScript functionality
const initializeComponent = () => {{
    console.log('Frontend component initialized');
    
    // Add event listeners
    document.addEventListener('DOMContentLoaded', () => {{
        console.log('DOM loaded');
    }});
    
    // Add interactive features
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {{
        button.addEventListener('click', (e) => {{
            console.log('Button clicked:', e.target.textContent);
        }});
    }});
}};

// Initialize the component
initializeComponent();
'''

    async def _validate_implementation(self, implementation: str, frontend_type: str,
                                    context: Dict[str, Any]) -> AgentResult:
        """Validate the generated frontend implementation."""
        
        try:
            # Basic validation checks
            validation_checks = {
                "react": ["react", "import", "export", "component"],
                "vue": ["vue", "template", "script", "component"],
                "mobile": ["react native", "flutter", "mobile", "component", "react"],
                "responsive": ["css", "media", "responsive", "grid"],
                "interactive": ["animation", "transition", "interactive", "hover"]
            }
            
            required_keywords = validation_checks.get(frontend_type, [])
            implementation_lower = implementation.lower()
            
            # For mobile type, we need at least one of the mobile keywords
            if frontend_type == "mobile":
                mobile_keywords = ["react native", "flutter", "mobile", "component", "react"]
                if not any(keyword in implementation_lower for keyword in mobile_keywords):
                    return AgentResult(
                        success=False,
                        output=implementation,
                        metadata={"validation_errors": ["No mobile keywords found"]},
                        error_message="Missing required mobile keywords"
                    )
                return AgentResult(
                    success=True,
                    output=implementation,
                    metadata={"validation_passed": True, "frontend_type": frontend_type}
                )
            
            # Check if implementation contains required keywords
            missing_keywords = [kw for kw in required_keywords if kw not in implementation_lower]
            
            if missing_keywords:
                return AgentResult(
                    success=False,
                    output=implementation,
                    metadata={"validation_errors": missing_keywords},
                    error_message=f"Missing required keywords: {missing_keywords}"
                )
            
            # Check for basic syntax (for React/JavaScript)
            if frontend_type in ["react", "general"]:
                try:
                    # Basic JavaScript syntax check
                    if "function" in implementation_lower or "const" in implementation_lower:
                        # Simple syntax validation
                        if "{" in implementation and "}" in implementation:
                            pass  # Basic structure check
                        else:
                            return AgentResult(
                                success=False,
                                output=implementation,
                                metadata={"syntax_error": "Missing braces"},
                                error_message="Syntax error: Missing braces"
                            )
                except Exception as e:
                    return AgentResult(
                        success=False,
                        output=implementation,
                        metadata={"syntax_error": str(e)},
                        error_message=f"Syntax error: {e}"
                    )
            
            return AgentResult(
                success=True,
                output=implementation,
                metadata={"validation_passed": True, "frontend_type": frontend_type}
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                output=implementation,
                metadata={"validation_error": str(e)},
                error_message=str(e)
            )

    async def _calculate_performance_metrics(self, implementation: str, 
                                           frontend_type: str) -> Dict[str, Any]:
        """Calculate performance metrics for the implementation."""
        
        # Simple metrics based on implementation characteristics
        lines_of_code = len(implementation.split('\n'))
        complexity_score = len([line for line in implementation.split('\n') 
                              if any(keyword in line.lower() 
                                    for keyword in ['function', 'const', 'let', 'var', 'class', 'component'])])
        
        return {
            "lines_of_code": lines_of_code,
            "complexity_score": complexity_score,
            "frontend_type": frontend_type,
            "estimated_performance": "high" if lines_of_code < 100 else "medium",
            "maintainability": "high" if complexity_score < 10 else "medium"
        } 
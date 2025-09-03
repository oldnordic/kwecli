#!/usr/bin/env python3
"""
KWECLI Command Handlers - Modular Architecture
===============================================

Command handlers for KWECLI modes: command, chat, plan, research.
Max 300 lines, using tool-equipped Ollama approach.

File: kwecli/handlers.py
Purpose: CLI command handlers with Ollama+tools integration
"""

import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import availability flags and functions
try:
    from agents.ollama_interface import process_development_request, get_available_tools
    OLLAMA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Ollama interface not available: {e}")
    OLLAMA_AVAILABLE = False

try:
    from bridge.ltmc_local import save_thought
    LTMC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Local LTMC not available: {e}")
    LTMC_AVAILABLE = False


def log_usage(mode: str, query: str, project_path: str):
    """Log usage to LTMC for learning."""
    if not LTMC_AVAILABLE:
        return
    
    try:
        save_thought(
            kind="usage",
            content=f"Mode: {mode}\nQuery: {query}\nProject: {project_path}",
            metadata={
                "mode": mode,
                "project_path": project_path,
                "timestamp": datetime.now().isoformat(),
                "source": "kwecli_handlers"
            }
        )
    except Exception as e:
        logger.warning(f"Failed to log usage: {e}")


def handle_command(command: str, project_path: str, language: str, verbose: bool) -> int:
    """Handle natural language development command."""
    print(f"🚀 Processing command: {command}")
    log_usage("command", command, project_path)
    
    if not OLLAMA_AVAILABLE:
        print("❌ Ollama interface not available")
        return 1
    
    try:
        # Build context for development request
        context = {
            "project_path": project_path,
            "language": language,
            "mode": "command"
        }
        
        print("🤖 AI Development Assistant processing request...")
        
        # Process with Ollama and tools
        result = asyncio.run(process_development_request(command, context))
        
        if result["success"]:
            print("\n📋 AI Analysis:")
            print(result["response"])
            
            if result.get("tools_executed"):
                print(f"\n🔧 Tools used: {len(result['tools_executed'])}")
                for tool in result["tools_executed"]:
                    if tool["success"]:
                        print(f"  ✅ {tool['tool_call']}")
                    else:
                        print(f"  ❌ {tool['tool_call']}: {tool['result'].get('error', 'Failed')}")
            
            print(f"\n🎯 Using model: {result['model']}")
            return 0
        else:
            print(f"❌ Command processing failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Command processing error: {e}")
        if verbose:
            raise
        return 1


def handle_chat(query: str, project_path: str, verbose: bool) -> int:
    """Handle chat mode interaction."""
    print(f"💬 Chat mode: {query}")
    log_usage("chat", query, project_path)
    
    if not OLLAMA_AVAILABLE:
        print("❌ Ollama interface not available")
        return 1
    
    try:
        # Build context for chat request
        context = {
            "project_path": project_path,
            "mode": "chat"
        }
        
        print("💭 AI Assistant thinking...")
        
        # Process with Ollama and tools
        result = asyncio.run(process_development_request(query, context))
        
        if result["success"]:
            print("\n🤖 Assistant:")
            print(result["response"])
            
            if result.get("tools_executed"):
                print(f"\n🔧 Actions taken: {len(result['tools_executed'])}")
                for tool in result["tools_executed"]:
                    if tool["success"]:
                        print(f"  ✅ {tool['tool_call']}")
                    else:
                        print(f"  ❌ {tool['tool_call']}: {tool['result'].get('error', 'Failed')}")
            
            return 0
        else:
            print(f"❌ Chat processing failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        if verbose:
            raise
        return 1


def handle_plan(goal: str, project_path: str, language: str, verbose: bool) -> int:
    """Handle planning mode - create development plan."""
    print(f"📋 Planning: {goal}")
    log_usage("plan", goal, project_path)
    
    if not OLLAMA_AVAILABLE:
        print("❌ Ollama interface not available")
        return 1
    
    try:
        # Build enhanced context for planning
        context = {
            "project_path": project_path,
            "language": language,
            "mode": "plan"
        }
        
        # Enhance planning prompt
        planning_prompt = f"""Create a detailed development plan for: {goal}

Please analyze the project and create:
1. Project structure and architecture
2. Step-by-step implementation plan
3. Required files and components
4. Dependencies and tools needed
5. Testing strategy

Use tools to explore the project if needed."""
        
        print("🧠 AI Planner analyzing requirements...")
        
        # Process with Ollama and tools
        result = asyncio.run(process_development_request(planning_prompt, context))
        
        if result["success"]:
            print("\n📋 Development Plan:")
            print(result["response"])
            
            if result.get("tools_executed"):
                print(f"\n🔍 Analysis tools used: {len(result['tools_executed'])}")
                for tool in result["tools_executed"]:
                    if tool["success"]:
                        print(f"  ✅ {tool['tool_call']}")
                    else:
                        print(f"  ❌ {tool['tool_call']}: {tool['result'].get('error', 'Failed')}")
            
            return 0
        else:
            print(f"❌ Planning failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Planning error: {e}")
        if verbose:
            raise
        return 1


def handle_research(query: str, project_path: str, verbose: bool) -> int:
    """Handle research mode - gather information."""
    print(f"🔍 Research: {query}")
    log_usage("research", query, project_path)
    
    if not OLLAMA_AVAILABLE:
        print("❌ Ollama interface not available")
        return 1
    
    try:
        # Build context for research request
        context = {
            "project_path": project_path,
            "mode": "research"
        }
        
        # Enhance research prompt
        research_prompt = f"""Research and gather information about: {query}

Please:
1. Search through project files for relevant information
2. Analyze existing code and documentation
3. Identify patterns, best practices, and examples
4. Provide comprehensive findings with citations
5. Use appropriate tools to explore the codebase

Focus on actionable insights and practical information."""
        
        print("🔬 AI Researcher analyzing...")
        
        # Process with Ollama and tools
        result = asyncio.run(process_development_request(research_prompt, context))
        
        if result["success"]:
            print("\n📚 Research Findings:")
            print(result["response"])
            
            if result.get("tools_executed"):
                print(f"\n🔍 Research tools used: {len(result['tools_executed'])}")
                for tool in result["tools_executed"]:
                    if tool["success"]:
                        print(f"  ✅ {tool['tool_call']}")
                    else:
                        print(f"  ❌ {tool['tool_call']}: {tool['result'].get('error', 'Failed')}")
            
            return 0
        else:
            print(f"❌ Research failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Research error: {e}")
        if verbose:
            raise
        return 1


def get_tool_status() -> Dict[str, Any]:
    """Get status of available tools for health check."""
    if not OLLAMA_AVAILABLE:
        return {"ollama": False, "error": "Ollama interface not available"}
    
    try:
        tools = get_available_tools()
        return tools
    except Exception as e:
        return {"ollama": False, "error": str(e)}
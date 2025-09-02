"""
Enhanced Ollama Planner with Sophisticated Chat System
======================================================

Integrates kwecli's sophisticated chat system with behavioral enforcement
and LTMC integration while maintaining backward compatibility.
"""

import asyncio
import logging
from typing import Optional

# Import sophisticated chat system
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from chat.sophisticated_ollama_chat import SophisticatedOllamaChat, ChatConfig, plan_task as sophisticated_plan_task
    from behavioral.kwecli_behavioral_system import check_quality_standards
    SOPHISTICATED_CHAT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Sophisticated chat system not available: {e}")
    # Fallback to simple implementation
    import ollama
    import yaml
    SOPHISTICATED_CHAT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global sophisticated chat instance for session continuity  
_chat_instance = None  # Will be SophisticatedOllamaChat instance when available

def get_chat_instance():
    """Get or create sophisticated chat instance for session continuity."""
    global _chat_instance
    if _chat_instance is None:
        config = ChatConfig(
            ollama_model="llama3.2:latest",  # Default model
            enable_hooks=True,
            enable_ltmc=True,
            behavioral_enforcement=True
        )
        _chat_instance = SophisticatedOllamaChat(config)
    return _chat_instance

def plan_task(prompt: str) -> str:
    """
    Enhanced task planning with sophisticated chat capabilities.
    
    This now provides Claude Code-level experience with:
    - Behavioral enforcement (quality over speed)
    - Conversation continuity via hooks
    - LTMC integration for memory
    - Real functionality requirements
    """
    
    if SOPHISTICATED_CHAT_AVAILABLE:
        logger.info("🚀 Using sophisticated kwecli chat system")
        
        try:
            # Check quality standards first
            should_proceed, quality_message = check_quality_standards(prompt)
            
            if not should_proceed:
                logger.warning("❌ Request rejected due to quality standards violation")
                return f"""❌ **Quality Standards Violation**

{quality_message}

Please rephrase your request following kwecli quality standards:
- Focus on quality over speed
- Request real implementations, not shortcuts
- Avoid asking for mocks, stubs, or placeholder code

I'm here to help you build sophisticated, production-ready solutions! 🚀"""
            
            # Use sophisticated chat system - fix asyncio issue
            chat_instance = get_chat_instance()
            
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, use await in a proper way
                import nest_asyncio
                nest_asyncio.apply()
                result = asyncio.run(chat_instance.chat(prompt))
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                result = asyncio.run(chat_instance.chat(prompt))
            except ImportError:
                # nest_asyncio not available, try different approach
                import threading
                import concurrent.futures
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(chat_instance.chat(prompt))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    result = future.result(timeout=30)
            
            # Add quality warning if needed
            if quality_message:
                result = f"{quality_message}\n\n{result}"
            
            return result
            
        except Exception as e:
            logger.error(f"Sophisticated chat system failed: {e}")
            logger.info("Falling back to simple ollama chat")
            # Fall through to simple implementation
    
    # Simple fallback implementation (original behavior)
    logger.info("📝 Using simple ollama fallback")
    try:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        
        response = ollama.chat(
            model=cfg["ollama_model"],
            messages=[
                {"role": "system", "content": """You are kwecli, a sophisticated AI assistant.

BEHAVIORAL REQUIREMENTS:
- Quality over speed always
- Real implementations only - no shortcuts, stubs, or mocks
- Test everything before completion
- Provide working, production-ready solutions

You're a lead AI architect focused on excellence."""},
                {"role": "user", "content": prompt}
            ],
        )
        return response["message"]["content"]
        
    except Exception as e:
        logger.error(f"Ollama chat failed: {e}")
        return f"❌ I encountered an error processing your request: {e}. Please ensure ollama is running and try again."

# Async version for advanced usage
async def plan_task_async(prompt: str) -> str:
    """Async version of plan_task for advanced usage."""
    if SOPHISTICATED_CHAT_AVAILABLE:
        try:
            # Check quality standards
            should_proceed, quality_message = check_quality_standards(prompt)
            
            if not should_proceed:
                return f"❌ **Quality Standards Violation**\n\n{quality_message}"
            
            # Use sophisticated async chat
            chat_instance = get_chat_instance()
            result = await chat_instance.chat(prompt)
            
            if quality_message:
                result = f"{quality_message}\n\n{result}"
                
            return result
            
        except Exception as e:
            logger.error(f"Sophisticated async chat failed: {e}")
    
    # Fallback to sync version
    return plan_task(prompt)

# Conversation management functions
def get_conversation_stats():
    """Get current conversation statistics."""
    if SOPHISTICATED_CHAT_AVAILABLE and _chat_instance:
        stats = _chat_instance.get_conversation_stats()
        # Ensure 'mode' key exists for compatibility
        if 'mode' not in stats:
            stats['mode'] = 'sophisticated'
        return stats
    return {"mode": "simple", "sophisticated_chat": False}

def restore_conversation(conversation_id: str) -> bool:
    """Restore conversation from previous session."""
    if SOPHISTICATED_CHAT_AVAILABLE:
        try:
            chat_instance = get_chat_instance()
            return asyncio.run(chat_instance.restore_conversation(conversation_id))
        except Exception as e:
            logger.error(f"Conversation restoration failed: {e}")
    return False

def search_conversation_history(query: str, limit: int = 5):
    """Search conversation history."""
    if SOPHISTICATED_CHAT_AVAILABLE and _chat_instance:
        try:
            return asyncio.run(_chat_instance.search_conversation_history(query, limit))
        except Exception as e:
            logger.error(f"History search failed: {e}")
    return []

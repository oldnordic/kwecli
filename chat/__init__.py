"""
kwecli Chat System
==================

Sophisticated chat interface with Claude Code-level capabilities using local
ollama models with LTMC integration and conversation continuity.
"""

from .sophisticated_ollama_chat import (
    SophisticatedOllamaChat, 
    ChatConfig,
    plan_task,
    chat_with_kwecli
)

__all__ = ['SophisticatedOllamaChat', 'ChatConfig', 'plan_task', 'chat_with_kwecli']
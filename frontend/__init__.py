#!/usr/bin/env python3
"""
KWE CLI Frontend Module

Modular frontend architecture for the Knowledge Workflow Engine CLI.
This module provides a clean separation of concerns across 4 main components:

1. CLI Interface - Main navigation and UI framework
2. Interactive Modes - AI-powered interaction features (chat, code gen, analysis)
3. Context Management - Context storage and retrieval operations
4. Backend Communication - HTTP API communication layer

This modular design ensures maintainability and compliance with the 300-line rule.
"""

from .cli_interface import CLIInterface
from .interactive_modes import InteractiveModes
from .context_management import ContextManagement
from .backend_communication import BackendCommunication
from .coordination_interface import CoordinationInterface

__all__ = [
    "CLIInterface",
    "InteractiveModes", 
    "ContextManagement",
    "BackendCommunication",
    "CoordinationInterface"
]
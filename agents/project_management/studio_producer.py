#!/usr/bin/env python3
"""
Studio Producer Agent

This module implements the Studio Producer agent which specializes in cross-team
coordination, resource management, and process optimization within the 6-day
development cycle.

This is the main entry point that imports the modularized components.
"""

# Import the core StudioProducer class from the modularized version
from .studio_producer_core import StudioProducer

# Expose the main class at module level for backward compatibility
__all__ = ['StudioProducer']
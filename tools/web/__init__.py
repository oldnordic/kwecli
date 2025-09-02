"""
Web operations module for KWE CLI.

This module provides web-focused tools including:
- Web content fetching with security validation
- HTML to markdown conversion
- Web search capabilities
- Content processing and sanitization
"""

from .web_fetch_tool import WebFetchTool
from .web_search_tool import WebSearchTool

__all__ = [
    "WebFetchTool",
    "WebSearchTool"
]
"""
Filesystem tools package for KWE CLI.

Provides file operations, search operations, and directory operations
for generic filesystem manipulation.
"""

# Import all filesystem tools
from .file_operations import ReadTool, WriteTool, EditTool, MultiEditTool
from .search_operations import GlobTool, GrepTool
from .directory_operations import DirectoryListerTool
from .security import SecurityValidator, PathTraversalError

# Aliases for compatibility with registry expectations
FileReaderTool = ReadTool
FileWriterTool = WriteTool
FileEditorTool = EditTool
MultiEditorTool = MultiEditTool
FileSearcherTool = GrepTool  # Primary search tool

__all__ = [
    "ReadTool",
    "WriteTool",
    "EditTool", 
    "MultiEditTool",
    "GlobTool",
    "GrepTool",
    "DirectoryListerTool",
    "SecurityValidator",
    "PathTraversalError",
    # Compatibility aliases
    "FileReaderTool",
    "FileWriterTool",
    "FileEditorTool",
    "MultiEditorTool", 
    "FileSearcherTool"
]
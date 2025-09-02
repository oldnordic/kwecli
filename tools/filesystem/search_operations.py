"""
Search operation tools for KWE CLI.

Provides pattern-based file finding (glob) and content searching (grep).
These tools match the functionality of Claude Code's search tools.
"""

import os
import re
import glob
import fnmatch
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, Pattern
import asyncio
import logging
import chardet

from ..core.tool_interface import BaseTool, ToolValidationMixin
from .security import SecurityValidator, PathTraversalError


class GlobTool(BaseTool, ToolValidationMixin):
    """
    Generic pattern-based file finding tool.
    
    Finds files using glob patterns with filtering and sorting.
    Matches functionality of Claude Code's Glob tool.
    """
    
    def __init__(self):
        super().__init__()
        self.security = SecurityValidator()
    
    @property
    def name(self) -> str:
        return "glob"
    
    @property
    def category(self) -> str:
        return "filesystem"
    
    @property
    def capabilities(self) -> List[str]:
        return ["search", "files", "patterns", "filtering"]
    
    @property
    def description(self) -> str:
        return "Find files using glob patterns with filtering and sorting"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute glob search operation."""
        try:
            pattern = parameters["pattern"]
            base_path = parameters.get("path", ".")
            ignore_patterns = parameters.get("ignore_patterns", [])
            file_types = parameters.get("file_types", [])
            min_size = parameters.get("min_size")
            max_size = parameters.get("max_size")
            sort_by = parameters.get("sort_by", "name")
            
            # Security validation for base path
            try:
                self.security.validate_path(base_path)
            except PathTraversalError as e:
                return {
                    "success": False,
                    "error": f"Path validation failed: {e}"
                }
            
            base_path_obj = Path(base_path).resolve()
            
            if not base_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Path does not exist: {base_path}"
                }
            
            # Find matching files
            matches = []
            
            if pattern.startswith("**/"):
                # Recursive pattern
                search_pattern = os.path.join(base_path, pattern)
                raw_matches = glob.glob(search_pattern, recursive=True)
            else:
                # Non-recursive pattern
                search_pattern = os.path.join(base_path, pattern)
                raw_matches = glob.glob(search_pattern)
            
            # Process and filter matches
            for match_path in raw_matches:
                path_obj = Path(match_path)
                
                # Skip if not accessible
                try:
                    if not self.security.check_permissions(str(path_obj), "read"):
                        continue
                except:
                    continue
                
                # Apply ignore patterns
                if self._should_ignore(path_obj, ignore_patterns):
                    continue
                
                # Apply file type filtering
                if file_types and not self._matches_file_types(path_obj, file_types):
                    continue
                
                # Apply size filtering
                if not self._matches_size_filter(path_obj, min_size, max_size):
                    continue
                
                # Collect file info
                file_info = self._get_file_info(path_obj, base_path_obj)
                matches.append(file_info)
            
            # Sort results
            matches = self._sort_matches(matches, sort_by)
            
            return {
                "success": True,
                "matches": [match["path"] for match in matches],
                "detailed_matches": matches,
                "pattern": pattern,
                "base_path": str(base_path_obj),
                "total_found": len(matches)
            }
            
        except Exception as e:
            self.logger.error(f"Error in glob search: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _should_ignore(self, path: Path, ignore_patterns: List[str]) -> bool:
        """Check if path matches ignore patterns."""
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern):
                return True
        return False
    
    def _matches_file_types(self, path: Path, file_types: List[str]) -> bool:
        """Check if path matches file type filters."""
        if path.is_dir():
            return False
        
        file_ext = path.suffix.lstrip('.').lower()
        return file_ext in [ft.lower() for ft in file_types]
    
    def _matches_size_filter(self, path: Path, min_size: Optional[int], max_size: Optional[int]) -> bool:
        """Check if file matches size filters."""
        try:
            if path.is_dir():
                return True  # Directories pass size filter
            
            file_size = path.stat().st_size
            
            if min_size is not None and file_size < min_size:
                return False
            
            if max_size is not None and file_size > max_size:
                return False
            
            return True
            
        except (OSError, PermissionError):
            return False
    
    def _get_file_info(self, path: Path, base_path: Path) -> Dict[str, Any]:
        """Get detailed file information."""
        try:
            stat_info = path.stat()
            relative_path = path.relative_to(base_path) if path.is_relative_to(base_path) else path
            
            return {
                "path": str(path),
                "relative_path": str(relative_path),
                "name": path.name,
                "type": "directory" if path.is_dir() else "file",
                "size": stat_info.st_size if path.is_file() else 0,
                "modified_time": stat_info.st_mtime,
                "extension": path.suffix.lstrip('.') if path.is_file() else None
            }
            
        except (OSError, PermissionError):
            return {
                "path": str(path),
                "relative_path": str(path),
                "name": path.name,
                "type": "unknown",
                "size": 0,
                "modified_time": 0,
                "extension": None
            }
    
    def _sort_matches(self, matches: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        """Sort matches by specified criteria."""
        if sort_by == "name":
            return sorted(matches, key=lambda x: x["name"].lower())
        elif sort_by == "size":
            return sorted(matches, key=lambda x: x["size"])
        elif sort_by == "mtime":
            return sorted(matches, key=lambda x: x["modified_time"], reverse=True)
        elif sort_by == "type":
            return sorted(matches, key=lambda x: (x["type"], x["name"].lower()))
        else:
            return matches
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        required = ["pattern"]
        if not self.validate_required_params(parameters, required):
            return False
        
        type_specs = {
            "pattern": str,
            "path": str,
            "ignore_patterns": list,
            "file_types": list,
            "min_size": int,
            "max_size": int,
            "sort_by": str
        }
        if not self.validate_param_types(parameters, type_specs):
            return False
        
        # Validate sort_by value
        if "sort_by" in parameters:
            valid_sorts = ["name", "size", "mtime", "type"]
            if parameters["sort_by"] not in valid_sorts:
                return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to search for (e.g., '*.py', '**/*.txt')",
                    "required": True
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (default: current directory)",
                    "required": False,
                    "default": "."
                },
                "ignore_patterns": {
                    "type": "array",
                    "description": "Patterns to ignore",
                    "required": False,
                    "items": {"type": "string"}
                },
                "file_types": {
                    "type": "array",
                    "description": "File extensions to include (without dots)",
                    "required": False,
                    "items": {"type": "string"}
                },
                "min_size": {
                    "type": "integer",
                    "description": "Minimum file size in bytes",
                    "required": False,
                    "minimum": 0
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum file size in bytes",
                    "required": False,
                    "minimum": 0
                },
                "sort_by": {
                    "type": "string",
                    "description": "Sort results by: name, size, mtime, type",
                    "required": False,
                    "enum": ["name", "size", "mtime", "type"],
                    "default": "name"
                }
            }
        }


class GrepTool(BaseTool, ToolValidationMixin):
    """
    Generic content searching tool.
    
    Searches file contents using regex patterns with context and filtering.
    Matches functionality of Claude Code's Grep tool.
    """
    
    def __init__(self):
        super().__init__()
        self.security = SecurityValidator()
    
    @property
    def name(self) -> str:
        return "grep"
    
    @property
    def category(self) -> str:
        return "filesystem"
    
    @property
    def capabilities(self) -> List[str]:
        return ["search", "content", "regex", "context"]
    
    @property
    def description(self) -> str:
        return "Search file contents using regex patterns with context and filtering"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute grep search operation."""
        try:
            pattern = parameters["pattern"]
            search_path = parameters.get("path", ".")
            glob_pattern = parameters.get("glob", "**/*")
            use_regex = parameters.get("use_regex", False)
            ignore_case = parameters.get("ignore_case", False)
            context_before = parameters.get("context_before", 0)
            context_after = parameters.get("context_after", 0)
            show_line_numbers = parameters.get("show_line_numbers", False)
            file_types = parameters.get("file_types", [])
            exclude_patterns = parameters.get("exclude_patterns", [])
            count_only = parameters.get("count_only", False)
            multiline = parameters.get("multiline", False)
            max_results = parameters.get("max_results")
            
            # Security validation
            try:
                if hasattr(self.security, 'base_path') and self.security.base_path:
                    self.security.validate_path(search_path)
            except PathTraversalError as e:
                return {
                    "success": False,
                    "error": f"Path validation failed: {e}"
                }
            
            search_path_obj = Path(search_path).resolve()
            
            if not search_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Path does not exist: {search_path}"
                }
            
            # Compile search pattern
            try:
                regex_flags = re.IGNORECASE if ignore_case else 0
                if multiline:
                    regex_flags |= re.MULTILINE | re.DOTALL
                
                if use_regex:
                    search_regex = re.compile(pattern, regex_flags)
                else:
                    # Escape special regex characters for literal search
                    escaped_pattern = re.escape(pattern)
                    search_regex = re.compile(escaped_pattern, regex_flags)
            except re.error as e:
                return {
                    "success": False,
                    "error": f"Invalid regex pattern: {e}"
                }
            
            # Find files to search
            files_to_search = self._find_files_to_search(
                search_path_obj, glob_pattern, file_types, exclude_patterns
            )
            
            # Search files
            matches = []
            total_matches = 0
            
            for file_path in files_to_search:
                if max_results and len(matches) >= max_results:
                    break
                
                file_matches = self._search_file(
                    file_path, search_regex, context_before, context_after,
                    show_line_numbers, multiline, count_only
                )
                
                if file_matches:
                    matches.extend(file_matches)
                    total_matches += len(file_matches)
            
            # Limit results if specified
            if max_results and len(matches) > max_results:
                matches = matches[:max_results]
            
            if count_only:
                return {
                    "success": True,
                    "total_matches": total_matches,
                    "files_searched": len(files_to_search),
                    "pattern": pattern
                }
            else:
                return {
                    "success": True,
                    "matches": matches,
                    "total_matches": total_matches,
                    "files_searched": len(files_to_search),
                    "pattern": pattern
                }
            
        except Exception as e:
            self.logger.error(f"Error in grep search: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _find_files_to_search(
        self, 
        search_path: Path, 
        glob_pattern: str,
        file_types: List[str],
        exclude_patterns: List[str]
    ) -> List[Path]:
        """Find files that match search criteria."""
        files = []
        
        if search_path.is_file():
            # Single file
            if self._should_search_file(search_path, file_types, exclude_patterns):
                files.append(search_path)
        else:
            # Directory search
            if glob_pattern.startswith("**/"):
                # Recursive search
                for file_path in search_path.rglob(glob_pattern[3:]):
                    if file_path.is_file() and self._should_search_file(file_path, file_types, exclude_patterns):
                        files.append(file_path)
            else:
                # Non-recursive search
                for file_path in search_path.glob(glob_pattern):
                    if file_path.is_file() and self._should_search_file(file_path, file_types, exclude_patterns):
                        files.append(file_path)
        
        return files
    
    def _should_search_file(self, file_path: Path, file_types: List[str], exclude_patterns: List[str]) -> bool:
        """Check if file should be searched."""
        # Check permissions
        try:
            if not self.security.check_permissions(str(file_path), "read"):
                return False
        except:
            return False
        
        # Check file types
        if file_types:
            file_ext = file_path.suffix.lstrip('.').lower()
            if file_ext not in [ft.lower() for ft in file_types]:
                return False
        
        # Check exclude patterns
        for pattern in exclude_patterns:
            # Try multiple matching approaches
            file_str = str(file_path)
            file_name = file_path.name
            relative_path = str(file_path).replace(str(file_path.anchor), '') if file_path.is_absolute() else file_str
            
            # Direct match on full path, filename, or relative path
            if (fnmatch.fnmatch(file_str, pattern) or 
                fnmatch.fnmatch(file_name, pattern) or
                fnmatch.fnmatch(relative_path, pattern)):
                return False
                
            # Check if file is in excluded directory
            if pattern.endswith('/*'):
                excluded_dir = pattern[:-2]  # Remove /*
                if excluded_dir in file_path.parts or file_str.find(f'/{excluded_dir}/') != -1 or file_str.find(f'{excluded_dir}/') != -1:
                    return False
        
        # Check if binary file
        if self._is_binary_file(file_path):
            return False
        
        return True
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
            
            # Check for null bytes (common in binary files)
            if b'\x00' in chunk:
                return True
            
            # Try to decode as text
            try:
                chunk.decode('utf-8')
                return False
            except UnicodeDecodeError:
                # Try chardet detection
                result = chardet.detect(chunk)
                return result.get('confidence', 0) < 0.7
                
        except (OSError, PermissionError):
            return True
    
    def _search_file(
        self,
        file_path: Path,
        search_regex: Pattern,
        context_before: int,
        context_after: int,
        show_line_numbers: bool,
        multiline: bool,
        count_only: bool
    ) -> List[Dict[str, Any]]:
        """Search individual file for pattern matches."""
        matches = []
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                if multiline:
                    content = f.read()
                    lines = content.splitlines()
                else:
                    lines = f.readlines()
            
            if multiline:
                # Multiline search
                for match in search_regex.finditer(content):
                    start_line = content[:match.start()].count('\n')
                    end_line = content[:match.end()].count('\n')
                    
                    match_info = {
                        "file": str(file_path),
                        "match": match.group(),
                        "start_line": start_line + 1,
                        "end_line": end_line + 1
                    }
                    
                    if not count_only:
                        match_info.update(self._get_context(lines, start_line, context_before, context_after))
                    
                    matches.append(match_info)
            else:
                # Line-by-line search
                for line_num, line in enumerate(lines):
                    line_content = line.rstrip('\n\r')
                    
                    if search_regex.search(line_content):
                        match_info = {
                            "file": str(file_path),
                            "line": line_content,
                            "line_number": line_num + 1 if show_line_numbers else None
                        }
                        
                        if not count_only:
                            match_info.update(self._get_context(lines, line_num, context_before, context_after))
                        
                        matches.append(match_info)
            
        except (OSError, UnicodeDecodeError, PermissionError) as e:
            # Skip files that can't be read
            self.logger.debug(f"Skipped file {file_path}: {e}")
        
        return matches
    
    def _get_context(self, lines: List[str], match_line: int, before: int, after: int) -> Dict[str, Any]:
        """Get context lines around match."""
        context = {}
        
        if before > 0:
            start_idx = max(0, match_line - before)
            context_lines = [line.rstrip('\n\r') for line in lines[start_idx:match_line]]
            context["context_before"] = context_lines
        
        if after > 0:
            end_idx = min(len(lines), match_line + after + 1)
            context_lines = [line.rstrip('\n\r') for line in lines[match_line + 1:end_idx]]
            context["context_after"] = context_lines
        
        return context
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        required = ["pattern"]
        if not self.validate_required_params(parameters, required):
            return False
        
        type_specs = {
            "pattern": str,
            "path": str,
            "glob": str,
            "use_regex": bool,
            "ignore_case": bool,
            "context_before": int,
            "context_after": int,
            "show_line_numbers": bool,
            "file_types": list,
            "exclude_patterns": list,
            "count_only": bool,
            "multiline": bool,
            "max_results": int
        }
        if not self.validate_param_types(parameters, type_specs):
            return False
        
        # Validate context values are non-negative
        for param in ["context_before", "context_after", "max_results"]:
            if param in parameters and parameters[param] < 0:
                return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (literal or regex)",
                    "required": True
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search (default: current directory)",
                    "required": False,
                    "default": "."
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern for file selection (default: *)",
                    "required": False,
                    "default": "*"
                },
                "use_regex": {
                    "type": "boolean",
                    "description": "Treat pattern as regex (default: false)",
                    "required": False,
                    "default": False
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case insensitive search (default: false)",
                    "required": False,
                    "default": False
                },
                "context_before": {
                    "type": "integer",
                    "description": "Lines of context before match",
                    "required": False,
                    "minimum": 0,
                    "default": 0
                },
                "context_after": {
                    "type": "integer",
                    "description": "Lines of context after match",
                    "required": False,
                    "minimum": 0,
                    "default": 0
                },
                "show_line_numbers": {
                    "type": "boolean",
                    "description": "Include line numbers in results",
                    "required": False,
                    "default": False
                },
                "file_types": {
                    "type": "array",
                    "description": "File extensions to search (without dots)",
                    "required": False,
                    "items": {"type": "string"}
                },
                "exclude_patterns": {
                    "type": "array",
                    "description": "Patterns to exclude from search",
                    "required": False,
                    "items": {"type": "string"}
                },
                "count_only": {
                    "type": "boolean",
                    "description": "Return only match counts, not content",
                    "required": False,
                    "default": False
                },
                "multiline": {
                    "type": "boolean",
                    "description": "Enable multiline pattern matching",
                    "required": False,
                    "default": False
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "required": False,
                    "minimum": 1
                }
            }
        }
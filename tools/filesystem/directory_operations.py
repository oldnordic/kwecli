"""
Directory operation tools for KWE CLI.

Provides directory listing with metadata, filtering, and sorting.
Matches functionality of Claude Code's LS tool.
"""

import os
import stat
import pwd
import grp
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import fnmatch
import asyncio
import logging

from ..core.tool_interface import BaseTool, ToolValidationMixin
from .security import SecurityValidator, PathTraversalError


class DirectoryListerTool(BaseTool, ToolValidationMixin):
    """
    Generic directory listing tool.
    
    Lists directory contents with metadata, filtering, and sorting.
    Matches functionality of Claude Code's LS tool.
    """
    
    def __init__(self):
        super().__init__()
        self.security = SecurityValidator()
    
    @property
    def name(self) -> str:
        return "ls"
    
    @property
    def category(self) -> str:
        return "filesystem"
    
    @property
    def capabilities(self) -> List[str]:
        return ["list", "directories", "metadata", "filtering"]
    
    @property
    def description(self) -> str:
        return "List directory contents with metadata, filtering, and sorting"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute directory listing operation."""
        try:
            target_path = parameters["path"]
            detailed = parameters.get("detailed", False)
            recursive = parameters.get("recursive", False)
            show_hidden = parameters.get("show_hidden", False)
            file_types = parameters.get("file_types", [])
            ignore_patterns = parameters.get("ignore_patterns", [])
            min_size = parameters.get("min_size")
            max_size = parameters.get("max_size")
            sort_by = parameters.get("sort_by", "name")
            summary = parameters.get("summary", False)
            limit = parameters.get("limit")
            
            # Security validation
            try:
                self.security.validate_path(target_path)
            except PathTraversalError as e:
                return {
                    "success": False,
                    "error": f"Path validation failed: {e}"
                }
            
            path_obj = Path(target_path).resolve()
            
            if not path_obj.exists():
                return {
                    "success": False,
                    "error": f"Path does not exist: {target_path}"
                }
            
            if not path_obj.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {target_path}"
                }
            
            # Check read permissions
            if not self.security.check_permissions(str(path_obj), "read"):
                return {
                    "success": False,
                    "error": f"Permission denied: cannot read directory {target_path}"
                }
            
            # Collect entries
            entries = []
            summary_stats = {
                "total_entries": 0,
                "directories": 0,
                "files": 0,
                "symlinks": 0,
                "total_size": 0
            }
            
            if recursive:
                for entry in self._walk_directory(path_obj, show_hidden, ignore_patterns):
                    if self._should_include_entry(entry, file_types, min_size, max_size):
                        entry_info = self._get_entry_info(entry, path_obj, detailed)
                        entries.append(entry_info)
                        self._update_summary(entry_info, summary_stats)
            else:
                try:
                    for entry in path_obj.iterdir():
                        if not show_hidden and entry.name.startswith('.'):
                            continue
                        
                        if self._should_ignore_entry(entry, ignore_patterns):
                            continue
                        
                        if self._should_include_entry(entry, file_types, min_size, max_size):
                            entry_info = self._get_entry_info(entry, path_obj, detailed)
                            entries.append(entry_info)
                            self._update_summary(entry_info, summary_stats)
                            
                except (OSError, PermissionError) as e:
                    return {
                        "success": False,
                        "error": f"Permission denied reading directory: {e}"
                    }
            
            # Sort entries
            entries = self._sort_entries(entries, sort_by)
            
            # Apply limit if specified
            truncated = False
            if limit and len(entries) > limit:
                entries = entries[:limit]
                truncated = True
            
            result = {
                "success": True,
                "entries": entries,
                "path": str(path_obj),
                "total_entries": len(entries)
            }
            
            if truncated:
                result["truncated"] = True
                result["limit"] = limit
            
            if summary:
                result["summary"] = summary_stats
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error listing directory {parameters.get('path', 'unknown')}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _walk_directory(self, path: Path, show_hidden: bool, ignore_patterns: List[str]):
        """Recursively walk directory tree."""
        try:
            for entry in path.iterdir():
                if not show_hidden and entry.name.startswith('.'):
                    continue
                
                if self._should_ignore_entry(entry, ignore_patterns):
                    continue
                
                yield entry
                
                if entry.is_dir():
                    try:
                        yield from self._walk_directory(entry, show_hidden, ignore_patterns)
                    except (OSError, PermissionError):
                        # Skip directories we can't read
                        continue
                        
        except (OSError, PermissionError):
            # Skip directories we can't read
            return
    
    def _should_ignore_entry(self, entry: Path, ignore_patterns: List[str]) -> bool:
        """Check if entry should be ignored based on patterns."""
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(entry.name, pattern) or fnmatch.fnmatch(str(entry), pattern):
                return True
        return False
    
    def _should_include_entry(
        self, 
        entry: Path, 
        file_types: List[str], 
        min_size: Optional[int], 
        max_size: Optional[int]
    ) -> bool:
        """Check if entry should be included based on filters."""
        # File type filtering
        if file_types:
            if entry.is_dir():
                return False  # Exclude directories when filtering by file types
            
            file_ext = entry.suffix.lstrip('.').lower()
            if file_ext not in [ft.lower() for ft in file_types]:
                return False
        
        # Size filtering
        if (min_size is not None or max_size is not None) and entry.is_file():
            try:
                file_size = entry.stat().st_size
                
                if min_size is not None and file_size < min_size:
                    return False
                
                if max_size is not None and file_size > max_size:
                    return False
                    
            except (OSError, PermissionError):
                return False
        
        return True
    
    def _get_entry_info(self, entry: Path, base_path: Path, detailed: bool) -> Dict[str, Any]:
        """Get information about directory entry."""
        try:
            stat_info = entry.stat()
            relative_path = entry.relative_to(base_path) if entry.is_relative_to(base_path) else entry
            
            entry_info = {
                "name": entry.name,
                "path": str(relative_path),
                "absolute_path": str(entry),
                "type": self._get_entry_type(entry),
                "size": stat_info.st_size if entry.is_file() else 0
            }
            
            if detailed:
                entry_info.update({
                    "permissions": self._get_permissions_info(stat_info),
                    "owner": self._get_owner_info(stat_info),
                    "modified_time": stat_info.st_mtime,
                    "created_time": stat_info.st_ctime,
                    "accessed_time": stat_info.st_atime,
                    "mode": oct(stat_info.st_mode),
                    "inode": stat_info.st_ino,
                    "device": stat_info.st_dev,
                    "links": stat_info.st_nlink
                })
                
                # Additional info for symlinks
                if entry.is_symlink():
                    try:
                        entry_info["target"] = str(entry.readlink())
                        entry_info["broken_link"] = not entry.exists()
                    except (OSError, PermissionError):
                        entry_info["target"] = "<unreadable>"
                        entry_info["broken_link"] = True
            
            return entry_info
            
        except (OSError, PermissionError):
            # Return basic info if we can't stat the file
            return {
                "name": entry.name,
                "path": str(entry),
                "absolute_path": str(entry),
                "type": "unknown",
                "size": 0,
                "error": "Permission denied"
            }
    
    def _get_entry_type(self, entry: Path) -> str:
        """Determine entry type."""
        if entry.is_symlink():
            return "symlink"
        elif entry.is_dir():
            return "directory"
        elif entry.is_file():
            return "file"
        elif entry.is_fifo():
            return "fifo"
        elif entry.is_socket():
            return "socket"
        elif entry.is_block_device():
            return "block_device"
        elif entry.is_char_device():
            return "char_device"
        else:
            return "unknown"
    
    def _get_permissions_info(self, stat_info: os.stat_result) -> Dict[str, Any]:
        """Get detailed permissions information."""
        mode = stat_info.st_mode
        
        return {
            "readable": bool(mode & stat.S_IRUSR),
            "writable": bool(mode & stat.S_IWUSR),
            "executable": bool(mode & stat.S_IXUSR),
            "group_readable": bool(mode & stat.S_IRGRP),
            "group_writable": bool(mode & stat.S_IWGRP),
            "group_executable": bool(mode & stat.S_IXGRP),
            "other_readable": bool(mode & stat.S_IROTH),
            "other_writable": bool(mode & stat.S_IWOTH),
            "other_executable": bool(mode & stat.S_IXOTH),
            "setuid": bool(mode & stat.S_ISUID),
            "setgid": bool(mode & stat.S_ISGID),
            "sticky": bool(mode & stat.S_ISVTX),
            "mode_string": stat.filemode(mode)
        }
    
    def _get_owner_info(self, stat_info: os.stat_result) -> Dict[str, Any]:
        """Get owner information."""
        try:
            # Try to get username and group name
            try:
                username = pwd.getpwuid(stat_info.st_uid).pw_name
            except KeyError:
                username = str(stat_info.st_uid)
            
            try:
                groupname = grp.getgrgid(stat_info.st_gid).gr_name
            except KeyError:
                groupname = str(stat_info.st_gid)
            
            return {
                "uid": stat_info.st_uid,
                "gid": stat_info.st_gid,
                "username": username,
                "groupname": groupname
            }
        except (ImportError, AttributeError):
            # pwd/grp modules not available on all systems
            return {
                "uid": stat_info.st_uid,
                "gid": stat_info.st_gid,
                "username": str(stat_info.st_uid),
                "groupname": str(stat_info.st_gid)
            }
    
    def _update_summary(self, entry_info: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """Update summary statistics."""
        summary["total_entries"] += 1
        summary["total_size"] += entry_info.get("size", 0)
        
        entry_type = entry_info.get("type", "unknown")
        if entry_type == "directory":
            summary["directories"] += 1
        elif entry_type == "file":
            summary["files"] += 1
        elif entry_type == "symlink":
            summary["symlinks"] += 1
    
    def _sort_entries(self, entries: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        """Sort entries by specified criteria."""
        if sort_by == "name":
            return sorted(entries, key=lambda x: x["name"].lower())
        elif sort_by == "size":
            return sorted(entries, key=lambda x: x.get("size", 0))
        elif sort_by == "mtime" and entries and "modified_time" in entries[0]:
            return sorted(entries, key=lambda x: x.get("modified_time", 0), reverse=True)
        elif sort_by == "type":
            return sorted(entries, key=lambda x: (x.get("type", ""), x["name"].lower()))
        else:
            return entries
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        required = ["path"]
        if not self.validate_required_params(parameters, required):
            return False
        
        type_specs = {
            "path": str,
            "detailed": bool,
            "recursive": bool,
            "show_hidden": bool,
            "file_types": list,
            "ignore_patterns": list,
            "min_size": int,
            "max_size": int,
            "sort_by": str,
            "summary": bool,
            "limit": int
        }
        if not self.validate_param_types(parameters, type_specs):
            return False
        
        # Validate sort_by value
        if "sort_by" in parameters:
            valid_sorts = ["name", "size", "mtime", "type"]
            if parameters["sort_by"] not in valid_sorts:
                return False
        
        # Validate size values are non-negative
        for param in ["min_size", "max_size", "limit"]:
            if param in parameters and parameters[param] < 0:
                return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to directory to list",
                    "required": True
                },
                "detailed": {
                    "type": "boolean",
                    "description": "Include detailed metadata for each entry",
                    "required": False,
                    "default": False
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List directory contents recursively",
                    "required": False,
                    "default": False
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files and directories",
                    "required": False,
                    "default": False
                },
                "file_types": {
                    "type": "array",
                    "description": "File extensions to include (without dots)",
                    "required": False,
                    "items": {"type": "string"}
                },
                "ignore_patterns": {
                    "type": "array",
                    "description": "Patterns to ignore",
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
                },
                "summary": {
                    "type": "boolean",
                    "description": "Include summary statistics",
                    "required": False,
                    "default": False
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries to return",
                    "required": False,
                    "minimum": 1
                }
            }
        }



"""
File operation tools for KWE CLI.

Provides generic file operations: read, write, edit, and multi-edit.
These tools match the functionality of Claude Code's file tools.
"""

import os
import shutil
import tempfile
import chardet
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import asyncio
import logging

from ..core.tool_interface import BaseTool, ToolValidationMixin
from .security import SecurityValidator, PathTraversalError
from config.unified_config import get_config


class ReadTool(BaseTool, ToolValidationMixin):
    """
    Generic file reading tool.
    
    Reads files with encoding detection, line limits, and offset support.
    Matches functionality of Claude Code's Read tool.
    """
    
    def __init__(self):
        super().__init__()
        self.security = SecurityValidator()
    
    @property
    def name(self) -> str:
        return "read"
    
    @property
    def category(self) -> str:
        return "filesystem"
    
    @property
    def capabilities(self) -> List[str]:
        return ["read", "files", "encoding_detection", "line_limits"]
    
    @property
    def description(self) -> str:
        return "Read files with encoding detection, line limits, and offset support"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file read operation."""
        try:
            file_path = parameters["file_path"]
            limit = parameters.get("limit")
            offset = parameters.get("offset", 0)
            
            path_obj = Path(file_path)
            
            # Check if file exists first
            if not path_obj.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            # Security validation (only if file exists)
            validation = self.security.comprehensive_validate(file_path, "read")
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Security validation failed: {', '.join(validation['errors'])}"
                }
            
            if not path_obj.is_file():
                return {
                    "success": False,
                    "error": f"Path is not a file: {file_path}"
                }
            
            # Detect encoding
            encoding = self._detect_encoding(path_obj)
            
            if encoding == "binary":
                return {
                    "success": True,
                    "content": f"<Binary file: {path_obj.stat().st_size} bytes>",
                    "encoding": "binary",
                    "lines": 0,
                    "file_path": file_path
                }
            
            # Read file content
            try:
                with open(path_obj, 'r', encoding=encoding, errors='replace') as f:
                    lines = f.readlines()
                
                # Apply offset and limit
                if offset > 0:
                    lines = lines[offset:]
                
                if limit is not None:
                    lines = lines[:limit]
                
                content = ''.join(lines)
                
                return {
                    "success": True,
                    "content": content,
                    "lines": len(lines),
                    "encoding": encoding,
                    "file_path": file_path,
                    "total_lines": len(open(path_obj, 'r', encoding=encoding, errors='replace').readlines()),
                    "offset": offset,
                    "limit": limit
                }
                
            except UnicodeDecodeError:
                return {
                    "success": False,
                    "error": f"Could not decode file with encoding {encoding}"
                }
            
        except Exception as e:
            self.logger.error(f"Error reading file {parameters.get('file_path', 'unknown')}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _detect_encoding(self, path: Path) -> str:
        """Detect file encoding."""
        try:
            # Read a sample to detect encoding
            with open(path, 'rb') as f:
                raw_data = f.read(8192)  # Read first 8KB
            
            # Check for binary content
            if b'\x00' in raw_data:
                return "binary"
            
            # Use chardet for encoding detection
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            
            # Prefer utf-8 for text files, especially for test content
            if not encoding or result.get('confidence', 0) < 0.8:
                encoding = 'utf-8'
            
            # Map common encodings to utf-8 for consistency
            if encoding.lower() in ['ascii', 'us-ascii']:
                encoding = 'utf-8'
            
            return encoding
            
        except Exception:
            return 'utf-8'
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        required = ["file_path"]
        if not self.validate_required_params(parameters, required):
            return False
        
        type_specs = {
            "file_path": str,
            "limit": int,
            "offset": int
        }
        if not self.validate_param_types(parameters, type_specs):
            return False
        
        # Validate limit and offset are non-negative
        if "limit" in parameters and parameters["limit"] < 0:
            return False
        
        if "offset" in parameters and parameters["offset"] < 0:
            return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to file to read",
                    "required": True
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                    "required": False,
                    "minimum": 0
                },
                "offset": {
                    "type": "integer", 
                    "description": "Line number to start reading from (0-based)",
                    "required": False,
                    "minimum": 0
                }
            }
        }


class WriteTool(BaseTool, ToolValidationMixin):
    """
    Generic file writing tool.
    
    Writes files with atomic operations and backup creation.
    Matches functionality of Claude Code's Write tool.
    """
    
    def __init__(self):
        super().__init__()
        self.security = SecurityValidator()
    
    @property
    def name(self) -> str:
        return "write"
    
    @property
    def category(self) -> str:
        return "filesystem"
    
    @property
    def capabilities(self) -> List[str]:
        return ["write", "files", "atomic", "backup"]
    
    @property
    def description(self) -> str:
        return "Write files with atomic operations and backup creation"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file write operation."""
        try:
            file_path = parameters["file_path"]
            content = parameters["content"]
            create_dirs = parameters.get("create_dirs", False)
            
            # Security validation
            content_size = len(content.encode('utf-8'))
            validation = self.security.comprehensive_validate(
                file_path, "write", content_size=content_size
            )
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Security validation failed: {', '.join(validation['errors'])}"
                }
            
            path_obj = Path(file_path)
            
            # Create parent directories if requested
            if create_dirs:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
            elif not path_obj.parent.exists():
                return {
                    "success": False,
                    "error": f"Parent directory does not exist: {path_obj.parent}"
                }
            
            # Check parent directory is writable
            if not self.security.check_permissions(str(path_obj.parent), "write"):
                return {
                    "success": False,
                    "error": f"Permission denied: cannot write to directory {path_obj.parent}"
                }
            
            # Enforce quality policy (line limits and placeholder bans)
            policy_ok, policy_error = self._validate_policy(file_path, content)
            if not policy_ok:
                return {"success": False, "error": policy_error}

            # Perform atomic write
            backup_path = None
            try:
                # Create backup if file exists
                if path_obj.exists():
                    backup_path = self._create_backup(path_obj)
                
                # Write to temporary file first
                self._write_file_content(file_path, content)
                
                # Clean up backup on success
                if backup_path and backup_path.exists():
                    backup_path.unlink()
                
                return {
                    "success": True,
                    "file_path": file_path,
                    "bytes_written": content_size,
                    "created": not backup_path
                }
                
            except Exception as e:
                # Restore from backup on failure
                if backup_path and backup_path.exists():
                    shutil.move(str(backup_path), file_path)
                
                return {
                    "success": False,
                    "error": f"Write failed: {e}"
                }
            
        except Exception as e:
            self.logger.error(f"Error writing file {parameters.get('file_path', 'unknown')}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_backup(self, path: Path) -> Path:
        """Create backup of existing file."""
        backup_path = path.with_suffix(path.suffix + '.backup')
        shutil.copy2(path, backup_path)
        return backup_path
    
    def _write_file_content(self, file_path: str, content: str) -> None:
        """Write content to file atomically."""
        path_obj = Path(file_path)
        
        # Write to temporary file in same directory
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=path_obj.parent,
            delete=False,
            encoding='utf-8'
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Atomic move
        shutil.move(temp_path, file_path)

    def _validate_policy(self, file_path: str, content: str) -> (bool, str):
        cfg = get_config()
        # Line limit
        lines = content.splitlines()
        if cfg.modularization_required and cfg.max_file_lines and len(lines) > cfg.max_file_lines:
            return False, (
                f"File exceeds {cfg.max_file_lines} lines ({len(lines)}). "
                f"Please modularize into smaller components."
            )
        # Placeholder/mocks/stubs/pass bans for code files
        if cfg.disallow_placeholders and Path(file_path).suffix in {'.py', '.rs', '.js', '.ts', '.go', '.java', '.cpp', '.c'}:
            banned_tokens = ("TODO", "FIXME", "MOCK", "STUB", "PLACEHOLDER")
            for tok in banned_tokens:
                if tok in content:
                    return False, f"Disallowed placeholder token detected: {tok}. Provide fully implemented code."
            # Disallow bare 'pass' statements used as stubs
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped == 'pass' or stripped == 'pass;':
                    return False, f"Disallowed 'pass' stub at line {i}. Implement functionality or remove."
        return True, ""
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        required = ["file_path", "content"]
        if not self.validate_required_params(parameters, required):
            return False
        
        type_specs = {
            "file_path": str,
            "content": str,
            "create_dirs": bool
        }
        if not self.validate_param_types(parameters, type_specs):
            return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to file to write",
                    "required": True
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to file",
                    "required": True
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Create parent directories if they don't exist",
                    "required": False,
                    "default": False
                }
            }
        }


class EditTool(BaseTool, ToolValidationMixin):
    """
    Generic file editing tool.
    
    Performs exact string replacements in files.
    Matches functionality of Claude Code's Edit tool.
    """
    
    def __init__(self):
        super().__init__()
        self.security = SecurityValidator()
    
    @property
    def name(self) -> str:
        return "edit"
    
    @property
    def category(self) -> str:
        return "filesystem"
    
    @property
    def capabilities(self) -> List[str]:
        return ["edit", "files", "string_replacement"]
    
    @property
    def description(self) -> str:
        return "Perform exact string replacements in files"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file edit operation."""
        try:
            file_path = parameters["file_path"]
            old_string = parameters["old_string"]
            new_string = parameters["new_string"]
            replace_all = parameters.get("replace_all", False)
            
            # Validate strings are different
            if old_string == new_string:
                return {
                    "success": False,
                    "error": "old_string and new_string cannot be identical"
                }
            
            # Security validation
            validation = self.security.comprehensive_validate(file_path, "write")
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Security validation failed: {', '.join(validation['errors'])}"
                }
            
            path_obj = Path(file_path)
            
            if not path_obj.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            # Read current content
            try:
                with open(path_obj, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except UnicodeDecodeError:
                return {
                    "success": False,
                    "error": "Could not decode file as UTF-8"
                }
            
            # Check if old_string exists in content
            if old_string not in content:
                return {
                    "success": False,
                    "error": f"String not found in file: '{old_string}'"
                }
            
            # Perform replacement
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacements = content.count(old_string)
            else:
                new_content = content.replace(old_string, new_string, 1)
                replacements = 1
            
            # Write modified content
            write_tool = WriteTool()
            write_result = await write_tool.execute({
                "file_path": file_path,
                "content": new_content
            })
            
            if not write_result["success"]:
                return write_result
            
            return {
                "success": True,
                "file_path": file_path,
                "replacements_made": replacements,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all
            }
            
        except Exception as e:
            self.logger.error(f"Error editing file {parameters.get('file_path', 'unknown')}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        required = ["file_path", "old_string", "new_string"]
        if not self.validate_required_params(parameters, required):
            return False
        
        type_specs = {
            "file_path": str,
            "old_string": str,
            "new_string": str,
            "replace_all": bool
        }
        if not self.validate_param_types(parameters, type_specs):
            return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to file to edit",
                    "required": True
                },
                "old_string": {
                    "type": "string",
                    "description": "String to replace",
                    "required": True
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement string",
                    "required": True
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)",
                    "required": False,
                    "default": False
                }
            }
        }


class MultiEditTool(BaseTool, ToolValidationMixin):
    """
    Generic multi-edit tool.
    
    Performs batch editing operations on files with atomic rollback.
    Matches functionality of Claude Code's MultiEdit tool.
    """
    
    def __init__(self):
        super().__init__()
        self.security = SecurityValidator()
    
    @property
    def name(self) -> str:
        return "multi_edit"
    
    @property
    def category(self) -> str:
        return "filesystem"
    
    @property
    def capabilities(self) -> List[str]:
        return ["edit", "files", "batch", "atomic"]
    
    @property
    def description(self) -> str:
        return "Perform batch editing operations with atomic rollback"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-edit operation."""
        try:
            file_path = parameters["file_path"]
            edits = parameters["edits"]
            
            # Security validation
            validation = self.security.comprehensive_validate(file_path, "write")
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Security validation failed: {', '.join(validation['errors'])}"
                }
            
            path_obj = Path(file_path)
            
            if not path_obj.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            # Read original content for rollback
            try:
                with open(path_obj, 'r', encoding='utf-8', errors='replace') as f:
                    original_content = f.read()
            except UnicodeDecodeError:
                return {
                    "success": False,
                    "error": "Could not decode file as UTF-8"
                }
            
            # Apply edits sequentially
            current_content = original_content
            total_replacements = 0
            
            for i, edit in enumerate(edits):
                old_string = edit["old_string"]
                new_string = edit["new_string"]
                replace_all = edit.get("replace_all", False)
                
                # Validate edit
                if old_string == new_string:
                    return self._rollback_error(
                        file_path, original_content,
                        f"Edit {i+1}: old_string and new_string cannot be identical"
                    )
                
                if old_string not in current_content:
                    return self._rollback_error(
                        file_path, original_content,
                        f"Edit {i+1}: String not found in file: '{old_string}'"
                    )
                
                # Apply edit
                if replace_all:
                    replacements = current_content.count(old_string)
                    current_content = current_content.replace(old_string, new_string)
                else:
                    replacements = 1 if old_string in current_content else 0
                    current_content = current_content.replace(old_string, new_string, 1)
                
                total_replacements += replacements
            
            # Write final content
            write_tool = WriteTool()
            write_result = await write_tool.execute({
                "file_path": file_path,
                "content": current_content
            })
            
            if not write_result["success"]:
                return self._rollback_error(
                    file_path, original_content,
                    f"Failed to write final content: {write_result['error']}"
                )
            
            return {
                "success": True,
                "file_path": file_path,
                "total_replacements": total_replacements,
                "edits_applied": len(edits)
            }
            
        except Exception as e:
            self.logger.error(f"Error in multi-edit {parameters.get('file_path', 'unknown')}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _rollback_error(self, file_path: str, original_content: str, error_msg: str) -> Dict[str, Any]:
        """Rollback file to original content and return error."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
        except Exception as rollback_error:
            self.logger.error(f"Failed to rollback {file_path}: {rollback_error}")
        
        return {
            "success": False,
            "error": f"{error_msg} (rollback performed)"
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        required = ["file_path", "edits"]
        if not self.validate_required_params(parameters, required):
            return False
        
        if not isinstance(parameters["edits"], list) or len(parameters["edits"]) == 0:
            return False
        
        # Validate each edit
        for edit in parameters["edits"]:
            if not isinstance(edit, dict):
                return False
            
            edit_required = ["old_string", "new_string"]
            if not all(key in edit for key in edit_required):
                return False
            
            if not isinstance(edit["old_string"], str) or not isinstance(edit["new_string"], str):
                return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to file to edit",
                    "required": True
                },
                "edits": {
                    "type": "array",
                    "description": "Array of edit operations to perform",
                    "required": True,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string", "description": "String to replace"},
                            "new_string": {"type": "string", "description": "Replacement string"},
                            "replace_all": {"type": "boolean", "description": "Replace all occurrences", "default": False}
                        },
                        "required": ["old_string", "new_string"]
                    }
                }
            }
        }

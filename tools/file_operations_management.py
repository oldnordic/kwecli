#!/usr/bin/env python3
"""
File Operations Management
==========================

File management utilities for KWE CLI tools with backup and template support.
Extracted from file_operations.py for smart modularization.

File: tools/file_operations_management.py
Purpose: File management operations with templates and validation (≤300 lines)
"""

import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import KWECLI's sequential thinking and artifact logging
try:
    from tools.utils import save_thought, log_artifact
    HAS_KWECLI_UTILS = True
    logger.info("KWECLI utilities imported successfully")
except ImportError:
    HAS_KWECLI_UTILS = False
    logger.warning("KWECLI utilities not available")
    def save_thought(category: str, content: str, context: Dict[str, Any] = None):
        """Fallback save_thought function."""
        pass
    def log_artifact(path: str, operation: str, identifier: str):
        """Fallback log_artifact function."""
        pass


class FileManagementOperations:
    """
    Advanced file management operations with templates, validation, and backup support.
    
    Provides comprehensive file management capabilities:
    - File creation with template support
    - File editing with line-based operations
    - File copying with metadata preservation
    - File deletion with backup protection
    - Python syntax validation
    - Template-based file creation
    """
    
    def __init__(self, backup_dir: str = None):
        """
        Initialize file management operations.
        
        Args:
            backup_dir: Directory for storing backups (creates if needed)
        """
        self.backup_dir = Path(backup_dir or "backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # File templates for different types
        self.templates = {
            '.py': '#!/usr/bin/env python3\n"""\nModule description.\n"""\n\n',
            '.js': '// JavaScript module\n\n',
            '.rs': '// Rust module\n\n',
            '.md': '# Document Title\n\n',
            '.sh': '#!/bin/bash\n\n',
            '.html': '<!DOCTYPE html>\n<html>\n<head>\n    <title>Document</title>\n</head>\n<body>\n\n</body>\n</html>\n',
            '.json': '{\n    "name": "value"\n}\n'
        }
        
        logger.info(f"FileManagementOperations initialized with backup_dir: {self.backup_dir}")
    
    def create_file(self, file_path: str, content: str = "", template: str = None, 
                   create_dirs: bool = True) -> Dict[str, Any]:
        """
        Create new file with optional template and content.
        
        Args:
            file_path: Path where to create the file
            content: Initial content (overrides template if provided)
            template: Template name or file extension for template
            create_dirs: Whether to create parent directories
            
        Returns:
            Dictionary with creation result and file information
        """
        try:
            path = Path(file_path)
            
            if path.exists():
                return {"success": False, "error": f"File already exists: {file_path}"}
            
            # Create directories if needed
            if create_dirs and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directories for: {path.parent}")
            
            # Determine content to write
            write_content = content
            if not content and template:
                template_content = self._get_template_content(template, path.suffix)
                write_content = template_content
            
            # Write the file
            path.write_text(write_content, encoding='utf-8')
            
            # Get file statistics
            file_stat = path.stat()
            
            # Log the operation
            if HAS_KWECLI_UTILS:
                save_thought(
                    "file_creation",
                    f"Created file: {file_path} ({len(write_content.splitlines())} lines)",
                    {
                        "operation": "create",
                        "file_path": str(path),
                        "template_used": template,
                        "size": file_stat.st_size
                    }
                )
                log_artifact(str(path), "file_create", f"create_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            return {
                "success": True,
                "file_path": str(path),
                "size": file_stat.st_size,
                "lines": len(write_content.splitlines()),
                "template_used": template,
                "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def edit_file(self, file_path: str, line_start: int, line_end: int, 
                 new_content: str, validate: bool = True) -> Dict[str, Any]:
        """
        Edit file by replacing specific line range with new content.
        
        Args:
            file_path: Path to the file to edit
            line_start: Starting line number (1-based)
            line_end: Ending line number (1-based, inclusive)
            new_content: New content to replace the line range
            validate: Whether to validate syntax for supported file types
            
        Returns:
            Dictionary with edit result and backup information
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            # Read current content
            current_content = path.read_text(encoding='utf-8')
            lines = current_content.splitlines()
            
            # Validate line numbers
            if line_start < 1 or line_end > len(lines) or line_start > line_end:
                return {
                    "success": False, 
                    "error": f"Invalid line range: {line_start}-{line_end} (file has {len(lines)} lines)"
                }
            
            # Create backup
            backup_path = self._create_backup(path)
            
            # Replace lines (convert to 0-based indexing)
            new_lines = new_content.splitlines() if new_content else []
            updated_lines = lines[:line_start-1] + new_lines + lines[line_end:]
            updated_content = '\n'.join(updated_lines)
            
            # Validate syntax if requested
            if validate and path.suffix == '.py':
                validation_result = self._validate_python_syntax(updated_content)
                if not validation_result["valid"]:
                    return {
                        "success": False, 
                        "error": f"Python syntax error: {validation_result['error']}"
                    }
            
            # Write updated content
            path.write_text(updated_content, encoding='utf-8')
            
            # Log the operation
            if HAS_KWECLI_UTILS:
                save_thought(
                    "file_edit",
                    f"Edited file: {file_path} (lines {line_start}-{line_end})",
                    {
                        "operation": "edit", 
                        "file_path": str(path),
                        "line_range": f"{line_start}-{line_end}",
                        "lines_added": len(new_lines),
                        "lines_removed": line_end - line_start + 1
                    }
                )
                log_artifact(str(path), "file_edit", f"edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            return {
                "success": True,
                "file_path": str(path),
                "lines_before": len(lines),
                "lines_after": len(updated_lines),
                "backup_path": str(backup_path),
                "range_edited": f"{line_start}-{line_end}",
                "syntax_validated": validate and path.suffix == '.py'
            }
            
        except Exception as e:
            logger.error(f"Failed to edit file {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_file(self, file_path: str, create_backup: bool = True) -> Dict[str, Any]:
        """
        Delete file with optional backup creation.
        
        Args:
            file_path: Path to the file to delete
            create_backup: Whether to create backup before deletion
            
        Returns:
            Dictionary with deletion result and backup information
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            if not path.is_file():
                return {"success": False, "error": f"Path is not a file: {file_path}"}
            
            backup_path = None
            if create_backup:
                backup_path = self._create_backup(path)
            
            # Get file info before deletion
            file_size = path.stat().st_size
            
            # Delete the file
            path.unlink()
            
            # Log the operation
            if HAS_KWECLI_UTILS:
                save_thought(
                    "file_deletion",
                    f"Deleted file: {file_path} ({file_size} bytes)",
                    {
                        "operation": "delete",
                        "file_path": str(path),
                        "backup_created": backup_path is not None,
                        "file_size": file_size
                    }
                )
            
            return {
                "success": True,
                "file_path": str(path),
                "backup_path": str(backup_path) if backup_path else None,
                "size_deleted": file_size
            }
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def copy_file(self, src_path: str, dst_path: str, preserve_metadata: bool = True,
                 overwrite: bool = False) -> Dict[str, Any]:
        """
        Copy file to destination with metadata preservation option.
        
        Args:
            src_path: Source file path
            dst_path: Destination file path  
            preserve_metadata: Whether to preserve file metadata
            overwrite: Whether to overwrite existing destination
            
        Returns:
            Dictionary with copy result and file information
        """
        try:
            src = Path(src_path)
            dst = Path(dst_path)
            
            if not src.exists():
                return {"success": False, "error": f"Source file not found: {src_path}"}
            
            if not src.is_file():
                return {"success": False, "error": f"Source is not a file: {src_path}"}
            
            if dst.exists() and not overwrite:
                return {"success": False, "error": f"Destination exists and overwrite=False: {dst_path}"}
            
            # Create destination directory if needed
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Perform copy
            if preserve_metadata:
                shutil.copy2(src, dst)  # Preserves metadata
            else:
                shutil.copy(src, dst)   # Content only
            
            # Get result statistics
            dst_stat = dst.stat()
            
            # Log the operation
            if HAS_KWECLI_UTILS:
                save_thought(
                    "file_copy",
                    f"Copied file: {src_path} → {dst_path}",
                    {
                        "operation": "copy",
                        "src_path": str(src),
                        "dst_path": str(dst),
                        "preserve_metadata": preserve_metadata,
                        "size": dst_stat.st_size
                    }
                )
                log_artifact(str(dst), "file_copy", f"copy_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            return {
                "success": True,
                "src_path": str(src),
                "dst_path": str(dst),
                "size": dst_stat.st_size,
                "metadata_preserved": preserve_metadata,
                "copied": datetime.fromtimestamp(dst_stat.st_ctime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to copy file {src_path} to {dst_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_backup(self, path: Path) -> Path:
        """
        Create timestamped backup of file.
        
        Args:
            path: Path to file to backup
            
        Returns:
            Path to created backup file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{path.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def _validate_python_syntax(self, content: str) -> Dict[str, Any]:
        """
        Validate Python syntax using compile.
        
        Args:
            content: Python source code to validate
            
        Returns:
            Dictionary with validation result
        """
        try:
            compile(content, '<string>', 'exec')
            return {"valid": True}
        except SyntaxError as e:
            return {
                "valid": False, 
                "error": f"Line {e.lineno}: {e.msg}",
                "line": e.lineno,
                "message": e.msg
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Compilation error: {e}"
            }
    
    def _get_template_content(self, template: str, file_extension: str) -> str:
        """
        Get template content for file creation.
        
        Args:
            template: Template name or file extension
            file_extension: File extension (as fallback)
            
        Returns:
            Template content string
        """
        # Direct template lookup
        if template in self.templates:
            return self.templates[template]
        
        # Extension-based lookup
        if file_extension in self.templates:
            return self.templates[file_extension]
        
        # Default empty template
        return ""
    
    def get_available_templates(self) -> Dict[str, str]:
        """Get all available file templates."""
        return self.templates.copy()


# Test functionality if run directly
if __name__ == "__main__":
    manager = FileManagementOperations()
    print(f"✅ Management operations: {len(manager.get_available_templates())} templates")
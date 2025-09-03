#!/usr/bin/env python3
"""
File Operations Core
====================

Core file I/O operations for KWE CLI tools with advanced encoding support.
Extracted from file_operations.py for smart modularization.

File: tools/file_operations_core.py
Purpose: Core file I/O with encoding fallbacks and validation (≤300 lines)
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import KWECLI's sequential thinking for file operations
try:
    from tools.utils import save_thought
    HAS_SEQUENTIAL_THINKING = True
    logger.info("Sequential thinking imported successfully")
except ImportError:
    HAS_SEQUENTIAL_THINKING = False
    logger.warning("Sequential thinking not available")
    def save_thought(category: str, content: str, context: Dict[str, Any] = None):
        """Fallback save_thought function when sequential thinking unavailable."""
        pass


class FileOperationsCore:
    """
    Core file operations with advanced encoding support and validation.
    
    Provides fundamental file I/O operations with:
    - Multi-encoding support with fallback chains
    - Comprehensive error handling and validation
    - File metadata and statistics tracking
    - Sequential thinking integration for operation logging
    """
    
    def __init__(self, default_encoding: str = "utf-8"):
        """
        Initialize core file operations.
        
        Args:
            default_encoding: Default encoding for file operations
        """
        self.default_encoding = default_encoding
        self.encoding_fallbacks = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        logger.info(f"FileOperationsCore initialized with encoding: {default_encoding}")
    
    def read_file_from_disk(self, file_path: str, encoding: str = None, 
                           max_lines: Optional[int] = None) -> Dict[str, Any]:
        """
        Read file from disk with encoding fallback and line limiting.
        
        Args:
            file_path: Path to the file to read
            encoding: Primary encoding to try (uses default if None)
            max_lines: Maximum lines to return (None for all)
            
        Returns:
            Dictionary with file content and metadata
        """
        path = Path(file_path)
        
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        if not path.is_file():
            return {"success": False, "error": f"Path is not a file: {file_path}"}
        
        # Use provided encoding or default
        primary_encoding = encoding or self.default_encoding
        
        # Try reading with primary encoding first
        content = None
        used_encoding = None
        
        for attempt_encoding in [primary_encoding] + self.encoding_fallbacks:
            if attempt_encoding == primary_encoding:
                continue  # Already tried
            try:
                content = path.read_text(encoding=attempt_encoding)
                used_encoding = attempt_encoding
                break
            except UnicodeDecodeError:
                logger.debug(f"Failed to decode {file_path} with {attempt_encoding}")
                continue
        
        if content is None:
            return {
                "success": False, 
                "error": f"Could not decode file with any encoding: {self.encoding_fallbacks}"
            }
        
        # Process content and apply line limits
        lines = content.splitlines()
        truncated = False
        
        if max_lines and len(lines) > max_lines:
            content = '\n'.join(lines[:max_lines])
            truncated = True
        
        # Get file statistics
        try:
            file_stat = path.stat()
            modified_time = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            file_size = file_stat.st_size
        except OSError as e:
            logger.warning(f"Could not get file stats for {file_path}: {e}")
            modified_time = datetime.now().isoformat()
            file_size = len(content)
        
        # Log the operation using sequential thinking
        if HAS_SEQUENTIAL_THINKING:
            save_thought(
                "file_access",
                f"Read file: {file_path} ({len(lines)} lines, {used_encoding} encoding)",
                {
                    "operation": "read",
                    "file_path": str(path),
                    "encoding": used_encoding,
                    "size": file_size,
                    "truncated": truncated
                }
            )
        
        return {
            "success": True,
            "content": content,
            "encoding": used_encoding,
            "size": file_size,
            "lines": len(lines),
            "truncated": truncated,
            "modified": modified_time,
            "source": "disk"
        }
    
    def write_file_to_disk(self, file_path: str, content: str, encoding: str = None,
                          create_backup: bool = True, create_dirs: bool = True) -> Dict[str, Any]:
        """
        Write content to file with backup creation and directory handling.
        
        Args:
            file_path: Path to write to
            content: Content to write
            encoding: Encoding to use (uses default if None)
            create_backup: Whether to create backup of existing file
            create_dirs: Whether to create parent directories
            
        Returns:
            Dictionary with write result and backup information
        """
        try:
            path = Path(file_path)
            write_encoding = encoding or self.default_encoding
            backup_path = None
            
            # Create directories if needed
            if create_dirs and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directories for: {path.parent}")
            
            # Create backup if file exists and backup is requested
            if create_backup and path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = path.with_suffix(f"{path.suffix}.bak_{timestamp}")
                
                try:
                    # Copy existing file to backup
                    import shutil
                    shutil.copy2(path, backup_path)
                    logger.info(f"Created backup: {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to create backup: {e}")
                    backup_path = None
            
            # Write the new content
            path.write_text(content, encoding=write_encoding)
            
            # Get file statistics after write
            file_stat = path.stat()
            
            # Log the operation
            if HAS_SEQUENTIAL_THINKING:
                save_thought(
                    "file_modification",
                    f"Wrote file: {file_path} ({len(content.splitlines())} lines, {write_encoding} encoding)",
                    {
                        "operation": "write",
                        "file_path": str(path),
                        "encoding": write_encoding,
                        "size": file_stat.st_size,
                        "backup_created": backup_path is not None,
                        "backup_path": str(backup_path) if backup_path else None
                    }
                )
            
            return {
                "success": True,
                "file_path": str(path),
                "size": file_stat.st_size,
                "encoding": write_encoding,
                "lines": len(content.splitlines()),
                "backup_path": str(backup_path) if backup_path else None,
                "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_file_path(self, file_path: str, must_exist: bool = False, 
                          must_be_file: bool = False) -> Dict[str, Any]:
        """
        Validate file path and return detailed information.
        
        Args:
            file_path: Path to validate
            must_exist: Whether path must exist
            must_be_file: Whether path must be a file (not directory)
            
        Returns:
            Validation result with path information
        """
        try:
            path = Path(file_path)
            
            result = {
                "success": True,
                "path": str(path.resolve()),
                "exists": path.exists(),
                "is_file": path.is_file() if path.exists() else None,
                "is_dir": path.is_dir() if path.exists() else None,
                "parent_exists": path.parent.exists(),
                "readable": None,
                "writable": None
            }
            
            # Check existence requirements
            if must_exist and not path.exists():
                result["success"] = False
                result["error"] = f"Path does not exist: {file_path}"
                return result
            
            if must_be_file and path.exists() and not path.is_file():
                result["success"] = False
                result["error"] = f"Path is not a file: {file_path}"
                return result
            
            # Check permissions if file exists
            if path.exists():
                try:
                    result["readable"] = path.stat().st_mode & 0o444 != 0
                    result["writable"] = path.stat().st_mode & 0o222 != 0
                except OSError:
                    result["readable"] = False
                    result["writable"] = False
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Path validation failed: {e}",
                "path": file_path
            }
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive file information and metadata.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            Dictionary with complete file information
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            stat = path.stat()
            
            return {
                "success": True,
                "path": str(path.resolve()),
                "name": path.name,
                "stem": path.stem,
                "suffix": path.suffix,
                "parent": str(path.parent),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "is_symlink": path.is_symlink(),
                "permissions": oct(stat.st_mode)[-3:],
                "owner_readable": bool(stat.st_mode & 0o400),
                "owner_writable": bool(stat.st_mode & 0o200),
                "owner_executable": bool(stat.st_mode & 0o100)
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {"success": False, "error": str(e)}


# Test functionality if run directly
if __name__ == "__main__":
    print("🧪 Testing File Operations Core...")
    
    # Test core functionality
    core = FileOperationsCore()
    print(f"✅ Core initialized with encoding: {core.default_encoding}")
    
    # Test path validation
    validation = core.validate_file_path(__file__, must_exist=True, must_be_file=True)
    print(f"✅ Path validation: {validation['success']}")
    
    # Test file info
    info = core.get_file_info(__file__)
    print(f"✅ File info: {info['success']}, size: {info.get('size', 'unknown')} bytes")
    
    # Test file reading
    read_result = core.read_file_from_disk(__file__, max_lines=10)
    print(f"✅ File read: {read_result['success']}, lines: {read_result.get('lines', 'unknown')}")
    
    print("✅ File Operations Core test complete")
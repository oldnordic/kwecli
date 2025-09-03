#!/usr/bin/env python3
"""
File Operations - Modular Entry Point
=====================================

Comprehensive file operations system using smart modular architecture.
Rebuilt from file_operations.py following CLAUDE.md ≤300 lines rule.

File: tools/file_operations_modular.py
Purpose: Main file operations interface with modular imports (≤300 lines)
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List

# Import modular components
from .file_operations_memory import FileMemoryManager
from .file_operations_core import FileOperationsCore
from .file_operations_management import FileManagementOperations

# Import KWECLI utilities
try:
    from bridge.ltmc_native import save_thought, log_artifact
    HAS_KWECLI_BRIDGE = True
    logger = logging.getLogger(__name__)
    logger.info("KWECLI bridge imported successfully")
except ImportError:
    HAS_KWECLI_BRIDGE = False
    logger = logging.getLogger(__name__)
    logger.warning("KWECLI bridge not available")
    def save_thought(category: str, content: str, context: Dict[str, Any] = None):
        pass
    def log_artifact(path: str, operation: str, identifier: str):
        pass


class FileOperations:
    """
    Comprehensive file operations system with LTMC memory integration.
    
    Modular Architecture:
    - FileMemoryManager: LTMC atomic memory integration and caching
    - FileOperationsCore: Core I/O operations with encoding support
    - FileManagementOperations: Advanced file management with templates
    
    This class provides a unified interface while leveraging the power
    of specialized modules for different operational aspects.
    """
    
    def __init__(self, enable_memory_storage: bool = True, backup_dir: str = None,
                 default_encoding: str = "utf-8"):
        """
        Initialize comprehensive file operations system.
        
        Args:
            enable_memory_storage: Enable LTMC atomic memory integration
            backup_dir: Directory for backups (creates if needed)
            default_encoding: Default text encoding for operations
        """
        # Initialize modular components
        self.memory_manager = FileMemoryManager(enable_memory_storage)
        self.core_ops = FileOperationsCore(default_encoding)
        self.management_ops = FileManagementOperations(backup_dir)
        
        # Session statistics
        self.session_stats = {
            "total_reads": 0,
            "disk_reads": 0,
            "memory_hits": 0,
            "estimated_tokens_saved": 0,
            "writes": 0,
            "edits": 0,
            "creates": 0,
            "deletes": 0,
            "copies": 0
        }
        
        self.enable_memory_storage = enable_memory_storage
        self.default_encoding = default_encoding
        
        logger.info(f"FileOperations initialized - Memory: {enable_memory_storage}, Encoding: {default_encoding}")
    
    def read_file(self, file_path: str, encoding: str = "utf-8", max_lines: int = None,
                 force_disk: bool = False) -> Dict[str, Any]:
        """
        Read file with memory-first strategy and LTMC integration.
        
        Memory Strategy:
        1. Try LTMC atomic memory retrieval first (sub-20ms)
        2. Validate cache freshness if found
        3. Fall back to disk if needed
        4. Store in LTMC for future access
        
        Args:
            file_path: Path to the file to read
            encoding: Text encoding to use
            max_lines: Maximum lines to return (truncation)
            force_disk: Force reading from disk (bypass cache)
            
        Returns:
            File content with metadata and performance metrics
        """
        self.session_stats["total_reads"] += 1
        
        try:
            # Try LTMC atomic memory-first strategy
            if self.enable_memory_storage and not force_disk:
                cached_content = self.memory_manager.retrieve_from_ltmc(file_path)
                if cached_content:
                    # Validate cache freshness
                    if self.memory_manager.check_cache_freshness(file_path, cached_content):
                        # Process cached content
                        lines = cached_content.splitlines()
                        if max_lines and len(lines) > max_lines:
                            cached_content = '\n'.join(lines[:max_lines])
                            truncated = True
                        else:
                            truncated = False
                        
                        # Update stats
                        self.session_stats["memory_hits"] += 1
                        tokens_saved = self.memory_manager.calculate_token_savings(len(cached_content))
                        self.session_stats["estimated_tokens_saved"] += tokens_saved
                        
                        logger.info(f"💾 KWECLI hit: {file_path} (~{tokens_saved} tokens saved)")
                        
                        return {
                            "success": True,
                            "content": cached_content,
                            "encoding": encoding,
                            "size": len(cached_content),
                            "lines": len(lines),
                            "truncated": truncated,
                            "source": "kwecli_cache"
                        }
            
            # Fallback to disk reading
            self.session_stats["disk_reads"] += 1
            result = self.core_ops.read_file_from_disk(file_path, encoding, max_lines)
            
            # Store in LTMC atomic memory for future access
            if self.enable_memory_storage and result.get("success"):
                self.memory_manager.store_in_ltmc(file_path, result["content"])
            
            logger.info(f"📁 Disk read: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def write_file(self, file_path: str, content: str, encoding: str = "utf-8",
                  create_backup: bool = True, create_dirs: bool = True) -> Dict[str, Any]:
        """
        Write file with backup, logging, and memory cache updates.
        
        Memory Integration:
        1. Write to disk with backup
        2. Invalidate existing cache
        3. Update LTMC atomic memory with new content
        
        Args:
            file_path: Path to write to
            content: Content to write
            encoding: Text encoding
            create_backup: Create backup of existing file
            create_dirs: Create parent directories if needed
        """
        self.session_stats["writes"] += 1
        
        try:
            # Write to disk using core operations
            result = self.core_ops.write_file_to_disk(
                file_path, content, encoding, create_backup, create_dirs
            )
            
            # Update LTMC atomic memory if successful
            if result.get("success") and self.enable_memory_storage:
                self.memory_manager.store_in_ltmc(file_path, content)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def edit_file(self, file_path: str, line_start: int, line_end: int,
                 new_content: str, validate: bool = True) -> Dict[str, Any]:
        """Edit file using management operations with validation."""
        self.session_stats["edits"] += 1
        
        result = self.management_ops.edit_file(file_path, line_start, line_end, new_content, validate)
        
        # Update memory cache if successful
        if result.get("success") and self.enable_memory_storage:
            # Read updated content and store in memory
            updated_result = self.core_ops.read_file_from_disk(file_path)
            if updated_result.get("success"):
                self.memory_manager.store_in_ltmc(file_path, updated_result["content"])
        
        return result
    
    def create_file(self, file_path: str, content: str = "", template: str = None,
                   create_dirs: bool = True) -> Dict[str, Any]:
        """Create file using management operations with templates."""
        self.session_stats["creates"] += 1
        
        result = self.management_ops.create_file(file_path, content, template, create_dirs)
        
        # Store in memory if successful
        if result.get("success") and self.enable_memory_storage:
            final_content = content or self.management_ops._get_template_content(template, Path(file_path).suffix)
            self.memory_manager.store_in_ltmc(file_path, final_content)
        
        return result
    
    def delete_file(self, file_path: str, create_backup: bool = True) -> Dict[str, Any]:
        """Delete file using management operations."""
        self.session_stats["deletes"] += 1
        return self.management_ops.delete_file(file_path, create_backup)
    
    def copy_file(self, src_path: str, dst_path: str, preserve_metadata: bool = True,
                 overwrite: bool = False) -> Dict[str, Any]:
        """Copy file using management operations."""
        self.session_stats["copies"] += 1
        
        result = self.management_ops.copy_file(src_path, dst_path, preserve_metadata, overwrite)
        
        # Store destination in memory if successful
        if result.get("success") and self.enable_memory_storage:
            src_result = self.core_ops.read_file_from_disk(src_path)
            if src_result.get("success"):
                self.memory_manager.store_in_ltmc(dst_path, src_result["content"])
        
        return result
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information."""
        return self.core_ops.get_file_info(file_path)
    
    def validate_file_path(self, file_path: str, must_exist: bool = False,
                          must_be_file: bool = False) -> Dict[str, Any]:
        """Validate file path."""
        return self.core_ops.validate_file_path(file_path, must_exist, must_be_file)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        stats = self.session_stats.copy()
        stats.update(self.memory_manager.get_memory_stats())
        return stats
    
    def get_available_templates(self) -> Dict[str, str]:
        """Get available file templates."""
        return self.management_ops.get_available_templates()
    
    def is_memory_enabled(self) -> bool:
        """Check if memory storage is enabled."""
        return self.memory_manager.is_memory_enabled()


# Convenience functions for backward compatibility
def read_file(file_path: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for reading files."""
    ops = FileOperations()
    return ops.read_file(file_path, **kwargs)

def write_file(file_path: str, content: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for writing files."""
    ops = FileOperations()
    return ops.write_file(file_path, content, **kwargs)

def edit_file(file_path: str, line_start: int, line_end: int, new_content: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for editing files."""
    ops = FileOperations()
    return ops.edit_file(file_path, line_start, line_end, new_content, **kwargs)

def create_file(file_path: str, content: str = "", **kwargs) -> Dict[str, Any]:
    """Convenience function for creating files."""
    ops = FileOperations()
    return ops.create_file(file_path, content, **kwargs)


# Test functionality if run directly
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing Modular File Operations...")
    
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        ops = FileOperations(backup_dir=f"{tmp_dir}/backups")
        print(f"✅ File operations initialized")
        
        # Test file creation
        test_file = f"{tmp_dir}/test.py"
        create_result = ops.create_file(test_file, template=".py")
        print(f"✅ File creation: {create_result['success']}")
        
        # Test file reading
        read_result = ops.read_file(test_file)
        print(f"✅ File read: {read_result['success']}, source: {read_result.get('source')}")
        
        # Test memory hit on second read
        read_result2 = ops.read_file(test_file)
        print(f"✅ Second read: {read_result2['success']}, source: {read_result2.get('source')}")
        
        # Test session stats
        stats = ops.get_session_stats()
        print(f"✅ Session stats: {stats['total_reads']} reads, {stats['memory_hits']} memory hits")
        
        print("✅ Modular File Operations test complete")
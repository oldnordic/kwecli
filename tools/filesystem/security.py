"""
Security validation module for filesystem operations.

Provides comprehensive security controls including path traversal protection,
permission validation, and operation-specific security checks.
"""

import os
import stat
import pathlib
import re
import threading
from typing import Dict, Any, List, Optional, Set
from pathlib import Path


class PathTraversalError(Exception):
    """Raised when path traversal attempt is detected."""
    pass


class PermissionError(Exception):
    """Raised when insufficient permissions for operation."""
    pass


class SecurityValidator:
    """
    Comprehensive security validator for filesystem operations.
    
    Provides path validation, permission checking, and security controls
    to prevent malicious filesystem access attempts.
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        max_path_length: int = 4096,
        blocked_extensions: Optional[List[str]] = None,
        context: str = "user",
        allowed_operations: Optional[List[str]] = None
    ):
        self.base_path = Path(base_path).resolve() if base_path else None
        self.max_file_size = max_file_size
        self.max_path_length = max_path_length
        self.blocked_extensions = blocked_extensions or [
            ".exe", ".bat", ".cmd", ".com", ".scr", ".pif"
        ]
        self.context = context
        self.allowed_operations = allowed_operations or ["read", "write", "execute"]
        
        # Thread-safe lock tracking
        self._locks: Dict[str, Dict[str, str]] = {}
        self._lock = threading.RLock()
        
        # Windows reserved names
        self.reserved_names = {
            "con", "prn", "aux", "nul",
            "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
            "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"
        }
    
    def validate_path(
        self, 
        path: str, 
        working_dir: Optional[str] = None,
        follow_symlinks: bool = False
    ) -> bool:
        """
        Validate path for security issues.
        
        Args:
            path: File/directory path to validate
            working_dir: Working directory for relative path resolution
            follow_symlinks: Whether to follow symbolic links
            
        Returns:
            True if path is valid
            
        Raises:
            PathTraversalError: If path traversal attempt detected
        """
        try:
            # Convert to Path object for normalization
            if working_dir:
                full_path = (Path(working_dir) / path).resolve()
            else:
                full_path = Path(path).resolve()
            
            # Check if path escapes base directory
            if self.base_path:
                try:
                    full_path.relative_to(self.base_path)
                except ValueError:
                    raise PathTraversalError(f"Path traversal attempt detected: '{path}' escapes base directory")
            
            # Check for symlink traversal if not following symlinks
            if not follow_symlinks and full_path.is_symlink():
                link_target = full_path.readlink()
                if link_target.is_absolute():
                    # Check if symlink target escapes base directory
                    if self.base_path:
                        try:
                            link_target.relative_to(self.base_path)
                        except ValueError:
                            raise PathTraversalError(f"Path traversal via symlink: '{path}' points outside base directory")
            
            return True
            
        except (OSError, ValueError) as e:
            raise PathTraversalError(f"Path validation failed: '{path}': {e}")
    
    def check_permissions(self, path: str, operation: str) -> bool:
        """
        Check if path has required permissions for operation.
        
        Args:
            path: File/directory path
            operation: Operation type ('read', 'write', 'execute')
            
        Returns:
            True if permissions are sufficient
        """
        try:
            path_obj = Path(path)
            
            if not path_obj.exists():
                return operation == "write"  # Can write new files
            
            file_stat = path_obj.stat()
            mode = file_stat.st_mode
            
            if operation == "read":
                return bool(mode & stat.S_IRUSR)
            elif operation == "write":
                if path_obj.is_dir():
                    return bool(mode & stat.S_IWUSR)
                else:
                    return bool(mode & stat.S_IWUSR)
            elif operation == "execute":
                return bool(mode & stat.S_IXUSR)
            else:
                return False
                
        except (OSError, PermissionError):
            return False
    
    def validate_parent_writable(self, file_path: str) -> bool:
        """
        Validate that parent directory is writable for file creation.
        
        Args:
            file_path: Path to file that would be created
            
        Returns:
            True if parent directory is writable
        """
        parent_dir = Path(file_path).parent
        return self.check_permissions(str(parent_dir), "write")
    
    def validate_file_size(self, path: str) -> bool:
        """
        Validate file size against limits.
        
        Args:
            path: File path to check
            
        Returns:
            True if file size is within limits
        """
        try:
            file_size = Path(path).stat().st_size
            return file_size <= self.max_file_size
        except (OSError, FileNotFoundError):
            return True  # File doesn't exist yet
    
    def validate_file_extension(self, filename: str) -> bool:
        """
        Validate file extension against blocked list.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if extension is allowed
        """
        ext = Path(filename).suffix.lower()
        return ext not in [e.lower() for e in self.blocked_extensions]
    
    def validate_path_length(self, path: str) -> bool:
        """
        Validate path length against limits.
        
        Args:
            path: Path to check
            
        Returns:
            True if path length is within limits
        """
        return len(path) <= self.max_path_length
    
    def validate_filename(self, filename: str) -> bool:
        """
        Validate filename for dangerous characters and patterns.
        
        Args:
            filename: Filename to validate
            
        Returns:
            True if filename is valid
        """
        if not filename or filename in [".", ".."]:
            return False
        
        # Check for control characters
        if any(ord(c) < 32 for c in filename):
            return False
        
        # Check for Windows reserved names
        name_without_ext = Path(filename).stem.lower()
        if name_without_ext in self.reserved_names:
            return False
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        if any(char in filename for char in dangerous_chars):
            return False
        
        return True
    
    def comprehensive_validate(
        self, 
        path: str, 
        operation: str,
        content_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation of path for operation.
        
        Args:
            path: Path to validate
            operation: Operation type
            content_size: Size of content for write operations
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        
        try:
            # Path traversal validation
            self.validate_path(path)
        except PathTraversalError as e:
            errors.append(str(e))
        
        # Path length validation
        if not self.validate_path_length(path):
            errors.append(f"Path length exceeds maximum of {self.max_path_length}")
        
        # Filename validation
        filename = Path(path).name
        if filename and not self.validate_filename(filename):
            errors.append(f"Invalid filename: {filename}")
        
        # Extension validation
        if not self.validate_file_extension(filename):
            errors.append(f"Blocked file extension: {Path(filename).suffix}")
        
        # Permission validation
        if not self.validate_operation(path, operation):
            errors.append(f"Insufficient permissions for {operation} operation")
        
        # File size validation for existing files
        if Path(path).exists() and not self.validate_file_size(path):
            errors.append(f"File size exceeds maximum of {self.max_file_size} bytes")
        
        # Content size validation for write operations
        if content_size and content_size > self.max_file_size:
            errors.append(f"Content size {content_size} exceeds maximum of {self.max_file_size} bytes")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "path": path,
            "operation": operation
        }
    
    def validate_operation(self, path: str, operation: str) -> bool:
        """
        Validate specific operation against context and permissions.
        
        Args:
            path: Path for operation
            operation: Operation type
            
        Returns:
            True if operation is allowed
        """
        # Check if operation is allowed in current context
        if operation not in self.allowed_operations:
            return False
        
        # Check file system permissions
        if operation == "write":
            # For write operations, check parent directory if file doesn't exist
            if not Path(path).exists():
                return self.validate_parent_writable(path)
        
        return self.check_permissions(path, operation)
    
    def is_temporary_file(self, path: str) -> bool:
        """
        Check if path appears to be a temporary file.
        
        Args:
            path: Path to check
            
        Returns:
            True if appears to be temporary file
        """
        filename = Path(path).name.lower()
        temp_patterns = [
            r'.*\.tmp$',
            r'.*\.temp$',
            r'.*~$',
            r'^\._.*',
            r'^temp_.*',
            r'^tmp_.*'
        ]
        
        return any(re.match(pattern, filename) for pattern in temp_patterns)
    
    def validate_temporary_file(self, path: str) -> bool:
        """
        Validate temporary file access.
        
        Args:
            path: Temporary file path
            
        Returns:
            True if temporary file access is valid
        """
        # Basic validation for temp files
        if not self.is_temporary_file(path):
            return False
        
        # Ensure temp file is in appropriate directory
        temp_dirs = ["/tmp", "/var/tmp", os.path.expanduser("~/tmp")]
        path_obj = Path(path)
        
        return any(
            str(path_obj).startswith(temp_dir) 
            for temp_dir in temp_dirs
            if Path(temp_dir).exists()
        )
    
    def validate_cleanup_operation(self, path: str) -> bool:
        """
        Validate file cleanup/deletion operation.
        
        Args:
            path: Path to cleanup
            
        Returns:
            True if cleanup is allowed
        """
        return self.validate_operation(path, "write") and Path(path).exists()
    
    def can_acquire_lock(self, path: str, lock_type: str) -> bool:
        """
        Check if file lock can be acquired.
        
        Args:
            path: File path
            lock_type: Type of lock ('read', 'write')
            
        Returns:
            True if lock can be acquired
        """
        with self._lock:
            abs_path = str(Path(path).resolve())
            
            if abs_path not in self._locks:
                return True
            
            existing_locks = self._locks[abs_path]
            
            # Write locks are exclusive
            if lock_type == "write" or any(t == "write" for t in existing_locks.values()):
                return False
            
            # Multiple read locks are allowed
            return lock_type == "read"
    
    def acquire_lock(self, path: str, lock_type: str, process_id: str) -> bool:
        """
        Acquire file lock.
        
        Args:
            path: File path
            lock_type: Type of lock
            process_id: Process identifier
            
        Returns:
            True if lock acquired successfully
        """
        with self._lock:
            if not self.can_acquire_lock(path, lock_type):
                return False
            
            abs_path = str(Path(path).resolve())
            if abs_path not in self._locks:
                self._locks[abs_path] = {}
            
            self._locks[abs_path][process_id] = lock_type
            return True
    
    def release_lock(self, path: str, process_id: str) -> bool:
        """
        Release file lock.
        
        Args:
            path: File path
            process_id: Process identifier
            
        Returns:
            True if lock released successfully
        """
        with self._lock:
            abs_path = str(Path(path).resolve())
            
            if abs_path in self._locks and process_id in self._locks[abs_path]:
                del self._locks[abs_path][process_id]
                
                # Clean up empty entries
                if not self._locks[abs_path]:
                    del self._locks[abs_path]
                
                return True
            
            return False
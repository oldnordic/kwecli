"""
Package Management Tool for KWE CLI.

Provides secure package management operations across multiple ecosystems including
Python (pip, poetry, pipenv), Node.js (npm, yarn, pnpm), Rust (cargo), and Go.

This tool integrates with the BashTool security framework to provide enterprise-grade
security validation while enabling comprehensive package management workflows.
"""

import os
import re
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import sys
sys.path.append('/home/feanor/Projects/kwecli')

from tools.core.tool_interface import BaseTool, ToolValidationMixin
from tools.system.bash_tool import BashTool


class PackageManagerTool(BaseTool, ToolValidationMixin):
    """
    Multi-language package management with comprehensive security validation.
    
    Provides secure package operations across Python, Node.js, Rust, Go, and other
    ecosystems while integrating with the existing BashTool security framework.
    """
    
    # Supported operations
    SUPPORTED_OPERATIONS = ["install", "update", "build", "test", "audit", "init"]
    
    # Ecosystem detection patterns
    ECOSYSTEM_PATTERNS = {
        "python": [
            "requirements.txt", "setup.py", "pyproject.toml", 
            "Pipfile", "environment.yml", "*.py"
        ],
        "nodejs": [
            "package.json", "package-lock.json", "yarn.lock", 
            "pnpm-lock.yaml", "*.js", "*.ts"
        ],
        "rust": [
            "Cargo.toml", "Cargo.lock", "*.rs"
        ],
        "go": [
            "go.mod", "go.sum", "*.go"
        ]
    }
    
    def __init__(
        self,
        bash_tool: BashTool,
        max_install_timeout: int = 600,
        enable_auto_audit: bool = True
    ):
        super().__init__()
        self.bash_tool = bash_tool
        self.max_install_timeout = max_install_timeout
        self.enable_auto_audit = enable_auto_audit
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @property
    def name(self) -> str:
        return "package_manager"
    
    @property
    def category(self) -> str:
        return "development"
    
    @property
    def capabilities(self) -> List[str]:
        return [
            "package_installation", "dependency_management", "vulnerability_scanning",
            "environment_setup", "build_automation", "security_auditing",
            "multi_ecosystem", "project_detection"
        ]
    
    @property
    def description(self) -> str:
        return "Multi-ecosystem package management with comprehensive security validation"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute package management operation with comprehensive security validation."""
        try:
            # Validate required parameters
            if not self.validate_parameters(parameters):
                missing_params = []
                if "operation" not in parameters:
                    missing_params.append("operation")
                
                if missing_params:
                    return {
                        "success": False,
                        "error": f"Missing required parameters: {missing_params}"
                    }
            
            operation = parameters["operation"]
            
            # Validate operation type
            if operation not in self.SUPPORTED_OPERATIONS:
                return {
                    "success": False,
                    "error": f"Invalid operation: {operation}. Supported: {self.SUPPORTED_OPERATIONS}"
                }
            
            # Detect or validate ecosystem
            ecosystem = parameters.get("ecosystem")
            working_directory = parameters.get("working_directory", ".")
            
            if not ecosystem:
                ecosystem = await self.detect_ecosystem(working_directory)
                if ecosystem == "unknown":
                    return {
                        "success": False,
                        "error": f"Could not detect project ecosystem in {working_directory}"
                    }
            
            # Validate ecosystem is supported
            if ecosystem not in self.ECOSYSTEM_PATTERNS:
                return {
                    "success": False,
                    "error": f"Unsupported ecosystem: {ecosystem}",
                    "supported_ecosystems": list(self.ECOSYSTEM_PATTERNS.keys())
                }
            
            # Security validation for packages
            packages = parameters.get("packages", [])
            if packages:
                security_result = await self._validate_package_security(packages, ecosystem)
                if not security_result["valid"]:
                    return {
                        "success": False,
                        "error": f"Security validation failed: {', '.join(security_result['errors'])}",
                        "security_errors": security_result["errors"]
                    }
            
            # Execute operation based on ecosystem
            if ecosystem == "python":
                result = await self._execute_python_operation(operation, parameters)
            elif ecosystem == "nodejs":
                result = await self._execute_nodejs_operation(operation, parameters)
            elif ecosystem == "rust":
                result = await self._execute_rust_operation(operation, parameters)
            elif ecosystem == "go":
                result = await self._execute_go_operation(operation, parameters)
            else:
                return {
                    "success": False,
                    "error": f"Operation handler not implemented for ecosystem: {ecosystem}"
                }
            
            return result
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Operation timeout exceeded"
            }
        except Exception as e:
            self.logger.error(f"Package operation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": parameters.get("operation", "unknown")
            }
    
    async def detect_ecosystem(self, directory: str) -> str:
        """Detect project ecosystem based on project files."""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            return "unknown"
        
        # Check for ecosystem-specific files
        for ecosystem, patterns in self.ECOSYSTEM_PATTERNS.items():
            for pattern in patterns:
                if pattern.startswith("*."):
                    # Glob pattern
                    extension = pattern[1:]
                    if any(f.suffix == extension for f in directory_path.rglob("*") if f.is_file()):
                        return ecosystem
                else:
                    # Exact file name
                    if (directory_path / pattern).exists():
                        return ecosystem
        
        return "unknown"
    
    async def _validate_package_security(self, packages: List[str], ecosystem: str) -> Dict[str, Any]:
        """Validate package names for security issues."""
        errors = []
        warnings = []
        
        # Known malicious patterns
        malicious_patterns = [
            r".*;.*",  # Command injection attempts
            r".*\|.*", # Pipe injection attempts
            r".*&.*",  # Background execution attempts
        ]
        
        # Popular package typosquatting protection
        popular_packages = {
            "python": ["requests", "urllib3", "numpy", "pandas", "django", "flask"],
            "nodejs": ["express", "react", "vue", "lodash", "moment", "axios"],
            "rust": ["serde", "tokio", "reqwest", "clap", "anyhow"],
            "go": ["gin", "gorilla", "logrus", "viper"]
        }
        
        for package in packages:
            # Check for command injection patterns
            for pattern in malicious_patterns:
                if re.match(pattern, package):
                    errors.append(f"Package name contains dangerous characters: {package}")
                    break
            
            # Check for typosquatting
            if ecosystem in popular_packages:
                package_name = package.split("==")[0].split(">=")[0].split("<=")[0]
                for popular in popular_packages[ecosystem]:
                    if self._is_typosquatting_attempt(package_name, popular):
                        errors.append(f"Potential typosquatting detected: '{package_name}' similar to '{popular}'")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _is_typosquatting_attempt(self, package_name: str, popular_name: str) -> bool:
        """Check if package name is a typosquatting attempt."""
        # Simple edit distance check
        if len(package_name) != len(popular_name):
            return False
        
        # Count character differences
        differences = sum(c1 != c2 for c1, c2 in zip(package_name.lower(), popular_name.lower()))
        
        # Flag as typosquatting if 1-2 character differences
        return 1 <= differences <= 2 and package_name.lower() != popular_name.lower()
    
    async def _detect_python_package_manager(self, directory: str) -> str:
        """Detect which Python package manager to use."""
        directory_path = Path(directory)
        
        # Check for Poetry project
        if (directory_path / "pyproject.toml").exists():
            pyproject_content = (directory_path / "pyproject.toml").read_text()
            if "[tool.poetry]" in pyproject_content:
                return "poetry"
        
        # Check for Pipenv project
        if (directory_path / "Pipfile").exists():
            return "pipenv"
        
        # Check for requirements.txt or fallback to pip
        if (directory_path / "requirements.txt").exists():
            return "pip"
        
        # Default to pip
        return "pip"

    async def _execute_python_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python ecosystem operation."""
        working_directory = parameters.get("working_directory", ".")
        packages = parameters.get("packages", [])
        
        if operation == "install":
            if not packages:
                return {
                    "success": False,
                    "error": "No packages specified for installation"
                }
            
            # Detect package manager
            package_manager = await self._detect_python_package_manager(working_directory)
            
            # Build appropriate command based on package manager
            if package_manager == "poetry":
                package_list = " ".join(packages)
                command = f"poetry add {package_list}"
            elif package_manager == "pipenv":
                package_list = " ".join(packages)
                command = f"pipenv install {package_list}"
            else:  # pip
                package_list = " ".join(packages)
                command = f"pip install {package_list}"
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": self.max_install_timeout
            })
            
            if result["success"]:
                return {
                    "success": True,
                    "ecosystem": "python",
                    "package_manager": package_manager,
                    "operation": "install",
                    "packages": packages,
                    "stdout": result["stdout"],
                    "execution_time": result["execution_time"]
                }
            else:
                return {
                    "success": False,
                    "error": result.get("stderr") or result.get("error", "Installation failed"),
                    "ecosystem": "python",
                    "package_manager": package_manager,
                    "operation": "install",
                    "packages": packages
                }
        
        elif operation == "update":
            if not packages:
                return {
                    "success": False,
                    "error": "No packages specified for update"
                }
            
            # Detect package manager
            package_manager = await self._detect_python_package_manager(working_directory)
            
            # Build appropriate update command based on package manager
            if package_manager == "poetry":
                package_list = " ".join(packages)
                command = f"poetry update {package_list}"
            elif package_manager == "pipenv":
                package_list = " ".join(packages)
                command = f"pipenv update {package_list}"
            else:  # pip
                package_list = " ".join(packages)
                command = f"pip install --upgrade {package_list}"
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": self.max_install_timeout
            })
            
            if result["success"]:
                return {
                    "success": True,
                    "ecosystem": "python",
                    "package_manager": package_manager,
                    "operation": "update",
                    "packages": packages,
                    "stdout": result["stdout"],
                    "execution_time": result["execution_time"]
                }
            else:
                return {
                    "success": False,
                    "error": result.get("stderr") or result.get("error", "Update failed"),
                    "ecosystem": "python",
                    "package_manager": package_manager,
                    "operation": "update",
                    "packages": packages
                }
        
        elif operation == "test":
            # Detect package manager for appropriate test command
            package_manager = await self._detect_python_package_manager(working_directory)
            
            # Build appropriate test command based on package manager
            if package_manager == "poetry":
                command = "poetry run pytest"
            elif package_manager == "pipenv":
                command = "pipenv run pytest"
            else:  # pip or general Python
                # Try pytest first, fall back to unittest
                command = "pytest"
                # Check if pytest is available, otherwise use python -m pytest
                try:
                    import subprocess
                    subprocess.run(["which", "pytest"], capture_output=True, check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    command = "python -m pytest"
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": 300  # Tests may take longer
            })
            
            return {
                "success": result["success"],
                "ecosystem": "python",
                "package_manager": package_manager,
                "operation": "test",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": result.get("execution_time", 0),
                **({"error": result.get("error", "Test execution failed")} if not result["success"] else {})
            }
        
        elif operation == "audit":
            # Detect package manager for appropriate audit command
            package_manager = await self._detect_python_package_manager(working_directory)
            
            # Build appropriate audit command based on package manager
            if package_manager == "poetry":
                command = "poetry audit"
            elif package_manager == "pipenv":
                command = "pipenv check"
            else:  # pip
                # Use pip-audit if available, otherwise safety
                command = "pip-audit"
                # Fall back to safety if pip-audit not available
                try:
                    import subprocess
                    subprocess.run(["which", "pip-audit"], capture_output=True, check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    command = "safety check"
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": 120
            })
            
            return {
                "success": result["success"],
                "ecosystem": "python",
                "package_manager": package_manager,
                "operation": "audit",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": result.get("execution_time", 0),
                **({"error": result.get("error", "Audit failed")} if not result["success"] else {})
            }
        
        else:
            return {
                "success": False,
                "error": f"Python operation '{operation}' not yet implemented",
                "supported_operations": ["install", "update", "test", "audit"]
            }
    
    async def _detect_nodejs_package_manager(self, directory: str) -> str:
        """Detect which Node.js package manager to use based on enterprise best practices."""
        directory_path = Path(directory)
        
        # Priority-based detection following enterprise patterns
        # 1. Check for explicit packageManager field in package.json (highest priority)
        package_json_path = directory_path / "package.json"
        if package_json_path.exists():
            try:
                import json
                package_json_content = json.loads(package_json_path.read_text())
                if "packageManager" in package_json_content:
                    package_manager_spec = package_json_content["packageManager"]
                    # Extract manager name from spec like "pnpm@8.10.0"
                    if package_manager_spec.startswith("pnpm"):
                        return "pnpm"
                    elif package_manager_spec.startswith("yarn"):
                        return "yarn"
                    elif package_manager_spec.startswith("npm"):
                        return "npm"
            except (json.JSONDecodeError, OSError):
                pass
        
        # 2. Check for lock files (enterprise critical for deterministic installs)
        if (directory_path / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (directory_path / "yarn.lock").exists():
            return "yarn"
        if (directory_path / "package-lock.json").exists():
            return "npm"
        if (directory_path / "npm-shrinkwrap.json").exists():
            return "npm"
        
        # 3. Check for workspace configuration patterns
        if (directory_path / "pnpm-workspace.yaml").exists():
            return "pnpm"
        if (directory_path / ".yarnrc.yml").exists():
            return "yarn"
        
        # 4. Default to npm (most universal)
        return "npm"

    async def _execute_nodejs_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Node.js ecosystem operation with enterprise security patterns."""
        working_directory = parameters.get("working_directory", ".")
        packages = parameters.get("packages", [])
        
        if operation == "install":
            if not packages:
                return {
                    "success": False,
                    "error": "No packages specified for installation"
                }
            
            # Detect package manager using enterprise patterns
            package_manager = await self._detect_nodejs_package_manager(working_directory)
            
            # Build appropriate command based on package manager
            if package_manager == "npm":
                package_list = " ".join(packages)
                command = f"npm install {package_list}"
            elif package_manager == "yarn":
                package_list = " ".join(packages)
                command = f"yarn add {package_list}"
            elif package_manager == "pnpm":
                package_list = " ".join(packages)
                command = f"pnpm add {package_list}"
            else:
                return {
                    "success": False,
                    "error": f"Unsupported Node.js package manager: {package_manager}"
                }
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": self.max_install_timeout
            })
            
            if result["success"]:
                return {
                    "success": True,
                    "ecosystem": "nodejs",
                    "package_manager": package_manager,
                    "operation": "install",
                    "packages": packages,
                    "stdout": result["stdout"],
                    "execution_time": result["execution_time"]
                }
            else:
                return {
                    "success": False,
                    "error": result.get("stderr") or result.get("error", "Installation failed"),
                    "ecosystem": "nodejs",
                    "package_manager": package_manager,
                    "operation": "install",
                    "packages": packages
                }
        
        elif operation == "update":
            if not packages:
                return {
                    "success": False,
                    "error": "No packages specified for update"
                }
            
            # Detect package manager
            package_manager = await self._detect_nodejs_package_manager(working_directory)
            
            # Build appropriate update command
            if package_manager == "npm":
                package_list = " ".join(packages)
                command = f"npm update {package_list}"
            elif package_manager == "yarn":
                # Yarn has different syntax for v1 vs v2+, use universal approach
                package_list = " ".join(packages)
                command = f"yarn up {package_list}"  # Works for both v1 and v2+
            elif package_manager == "pnpm":
                package_list = " ".join(packages)
                command = f"pnpm update {package_list}"
            else:
                return {
                    "success": False,
                    "error": f"Unsupported Node.js package manager: {package_manager}"
                }
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": self.max_install_timeout
            })
            
            if result["success"]:
                return {
                    "success": True,
                    "ecosystem": "nodejs",
                    "package_manager": package_manager,
                    "operation": "update",
                    "packages": packages,
                    "stdout": result["stdout"],
                    "execution_time": result["execution_time"]
                }
            else:
                return {
                    "success": False,
                    "error": result.get("stderr") or result.get("error", "Update failed"),
                    "ecosystem": "nodejs",
                    "package_manager": package_manager,
                    "operation": "update",
                    "packages": packages
                }
        
        elif operation == "audit":
            # Detect package manager for appropriate audit command
            package_manager = await self._detect_nodejs_package_manager(working_directory)
            
            # Build appropriate audit command (enterprise security focus)
            if package_manager == "npm":
                command = "npm audit"
            elif package_manager == "yarn":
                # Check if it's modern yarn (v2+) or classic (v1)
                command = "yarn audit"  # Works for both versions with different output
            elif package_manager == "pnpm":
                command = "pnpm audit"
            else:
                return {
                    "success": False,
                    "error": f"Unsupported Node.js package manager: {package_manager}"
                }
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": 120
            })
            
            return {
                "success": result["success"],
                "ecosystem": "nodejs",
                "package_manager": package_manager,
                "operation": "audit",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": result.get("execution_time", 0),
                **({"error": result.get("error", "Audit failed")} if not result["success"] else {})
            }
        
        elif operation == "test":
            # Detect package manager for appropriate test command
            package_manager = await self._detect_nodejs_package_manager(working_directory)
            
            # Build appropriate test command
            if package_manager == "npm":
                command = "npm test"
            elif package_manager == "yarn":
                command = "yarn test"
            elif package_manager == "pnpm":
                command = "pnpm test"
            else:
                return {
                    "success": False,
                    "error": f"Unsupported Node.js package manager: {package_manager}"
                }
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": 300  # Tests may take longer
            })
            
            return {
                "success": result["success"],
                "ecosystem": "nodejs",
                "package_manager": package_manager,
                "operation": "test",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": result.get("execution_time", 0),
                **({"error": result.get("error", "Test execution failed")} if not result["success"] else {})
            }
        
        else:
            return {
                "success": False,
                "error": f"Node.js operation '{operation}' not yet implemented",
                "supported_operations": ["install", "update", "audit", "test"]
            }
    
    async def _detect_rust_package_manager(self, directory: str) -> str:
        """Detect Rust package manager - always cargo for Rust ecosystem."""
        # Rust ecosystem only has cargo as package manager
        # This method exists for API consistency with other ecosystems
        return "cargo"

    async def _execute_rust_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Rust ecosystem operation with enterprise security patterns."""
        working_directory = parameters.get("working_directory", ".")
        packages = parameters.get("packages", [])
        
        if operation == "install":
            if not packages:
                return {
                    "success": False,
                    "error": "No packages specified for installation"
                }
            
            # Rust only uses cargo
            package_manager = await self._detect_rust_package_manager(working_directory)
            
            # Build cargo add command (modern cargo supports cargo add)
            package_list = " ".join(packages)
            command = f"cargo add {package_list}"
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": self.max_install_timeout
            })
            
            if result["success"]:
                return {
                    "success": True,
                    "ecosystem": "rust",
                    "package_manager": package_manager,
                    "operation": "install",
                    "packages": packages,
                    "stdout": result["stdout"],
                    "execution_time": result["execution_time"]
                }
            else:
                return {
                    "success": False,
                    "error": result.get("stderr") or result.get("error", "Installation failed"),
                    "ecosystem": "rust",
                    "package_manager": package_manager,
                    "operation": "install",
                    "packages": packages
                }
        
        elif operation == "update":
            # Detect package manager (always cargo)
            package_manager = await self._detect_rust_package_manager(working_directory)
            
            # Build cargo update command
            if packages:
                # Update specific packages
                package_list = " ".join(packages)
                command = f"cargo update -p {package_list.replace(' ', ' -p ')}"
            else:
                # Update all packages
                command = "cargo update"
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": self.max_install_timeout
            })
            
            if result["success"]:
                return {
                    "success": True,
                    "ecosystem": "rust",
                    "package_manager": package_manager,
                    "operation": "update",
                    "packages": packages,
                    "stdout": result["stdout"],
                    "execution_time": result["execution_time"]
                }
            else:
                return {
                    "success": False,
                    "error": result.get("stderr") or result.get("error", "Update failed"),
                    "ecosystem": "rust",
                    "package_manager": package_manager,
                    "operation": "update",
                    "packages": packages
                }
        
        elif operation == "build":
            # Detect package manager
            package_manager = await self._detect_rust_package_manager(working_directory)
            
            # Build cargo build command (default to dev profile)
            command = "cargo build"
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": 300  # Build may take longer
            })
            
            return {
                "success": result["success"],
                "ecosystem": "rust",
                "package_manager": package_manager,
                "operation": "build",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": result.get("execution_time", 0),
                **({"error": result.get("error", "Build failed")} if not result["success"] else {})
            }
        
        elif operation == "test":
            # Detect package manager
            package_manager = await self._detect_rust_package_manager(working_directory)
            
            # Build cargo test command
            command = "cargo test"
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": 300  # Tests may take longer
            })
            
            return {
                "success": result["success"],
                "ecosystem": "rust",
                "package_manager": package_manager,
                "operation": "test",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": result.get("execution_time", 0),
                **({"error": result.get("error", "Test execution failed")} if not result["success"] else {})
            }
        
        elif operation == "audit":
            # Detect package manager
            package_manager = await self._detect_rust_package_manager(working_directory)
            
            # Build cargo audit command (requires cargo-audit to be installed)
            command = "cargo audit"
            
            result = await self.bash_tool.execute({
                "command": command,
                "working_directory": working_directory,
                "timeout": 120
            })
            
            return {
                "success": result["success"],
                "ecosystem": "rust",
                "package_manager": package_manager,
                "operation": "audit",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": result.get("execution_time", 0),
                **({"error": result.get("error", "Audit failed")} if not result["success"] else {})
            }
        
        else:
            return {
                "success": False,
                "error": f"Rust operation '{operation}' not yet implemented",
                "supported_operations": ["install", "update", "build", "test", "audit"]
            }
    
    async def _execute_go_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Go ecosystem operation."""
        return {
            "success": False,
            "error": "Go operations not yet implemented"
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for package management operation."""
        required = ["operation"]
        if not self.validate_required_params(parameters, required):
            return False
        
        type_specs = {
            "operation": str,
            "ecosystem": str,
            "packages": list,
            "working_directory": str,
            "timeout": int
        }
        if not self.validate_param_types(parameters, type_specs):
            return False
        
        # Validate working directory if provided
        wd = parameters.get("working_directory")
        if wd is not None:
            from pathlib import Path
            p = Path(wd)
            if not p.exists() or not p.is_dir():
                return False
        
        # Validate packages list contents if provided
        pkgs = parameters.get("packages")
        if pkgs is not None:
            if not isinstance(pkgs, list):
                return False
            forbidden = set(";|&`$><")
            for item in pkgs:
                if not isinstance(item, str) or not item.strip():
                    return False
                if any(ch in item for ch in forbidden):
                    return False
                if len(item) > 200:
                    return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema for package management tool."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "operation": {
                    "type": "string",
                    "description": "Operation to perform",
                    "required": True,
                    "enum": self.SUPPORTED_OPERATIONS
                },
                "ecosystem": {
                    "type": "string", 
                    "description": "Package ecosystem (auto-detected if not specified)",
                    "required": False,
                    "enum": list(self.ECOSYSTEM_PATTERNS.keys())
                },
                "packages": {
                    "type": "array",
                    "description": "List of packages to operate on",
                    "required": False,
                    "items": {"type": "string"}
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory for the operation",
                    "required": False,
                    "default": "."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Operation timeout in seconds",
                    "required": False,
                    "default": self.max_install_timeout,
                    "minimum": 1,
                    "maximum": 3600
                }
            }
        }

"""
CodeQualityTool – runs formatters and linters via BashTool.

Supports Python (black, isort, flake8, mypy) and can be extended.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from tools.core.tool_interface import BaseTool
from tools.system.bash_tool import BashTool


@dataclass
class CodeQualityConfig:
    python_paths: List[str] = None
    check_only: bool = True  # formatters run in check-mode by default


class CodeQualityTool(BaseTool):
    """Runs black/isort/flake8/mypy through a secure BashTool wrapper."""

    def __init__(self, bash_tool: Optional[BashTool] = None, config: Optional[CodeQualityConfig] = None):
        super().__init__()
        self._name = "code_quality"
        self._category = "development"
        self._capabilities = ["format", "lint", "type_check"]
        self._description = "Run Python code quality tools (black, isort, flake8, mypy)."
        self.bash_tool = bash_tool or BashTool()
        self.config = config or CodeQualityConfig(python_paths=["."])

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def capabilities(self) -> List[str]:
        return self._capabilities

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_parameters(parameters):
            return {"success": False, "error": "Invalid parameters"}

        paths = parameters.get("paths") or self.config.python_paths or ["."]
        check_only = parameters.get("check_only", self.config.check_only)

        cmds = []
        # formatters
        black_cmd = f"python -m black {'--check ' if check_only else ''}{' '.join(paths)}"
        isort_cmd = f"python -m isort {'--check-only ' if check_only else ''}{' '.join(paths)}"
        # linters
        flake8_cmd = f"python -m flake8 {' '.join(paths)}"
        mypy_cmd = f"python -m mypy {' '.join(paths)}"
        cmds.extend([black_cmd, isort_cmd, flake8_cmd, mypy_cmd])

        results: List[Dict[str, Any]] = []
        overall_success = True
        for cmd in cmds:
            res = await self.bash_tool.execute({"command": cmd})
            results.append({"command": cmd, **res})
            # flake8/mypy non-zero exit should mark as failure but not crash
            if not res.get("success", False) or res.get("exit_code", 0) not in (0,):
                overall_success = False

        return {
            "success": overall_success,
            "results": results,
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        if not isinstance(parameters, dict):
            return False
        if "paths" in parameters and not isinstance(parameters["paths"], list):
            return False
        return True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "capabilities": self.capabilities,
            "parameters": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to check/format",
                    "default": self.config.python_paths or ["."]
                },
                "check_only": {
                    "type": "boolean",
                    "description": "Run formatters in check-only mode",
                    "default": self.config.check_only,
                },
            },
            "description": self.description,
        }


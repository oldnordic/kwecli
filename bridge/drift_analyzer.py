#!/usr/bin/env python3
"""
KWECLI Drift Analysis Engine
============================

Core drift analysis logic extracted from drift.py for smart modularization.
Handles specific analysis types: code, documentation, semantic, and relationship drift.

File: bridge/drift_analyzer.py
Purpose: Drift analysis engine (≤300 lines)
"""

import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DriftAnalyzer:
    """Core drift analysis engine for different drift types."""
    
    def __init__(self, project_path: Path):
        """Initialize analyzer with project path."""
        self.project_path = project_path
    
    def analyze_code_drift(self, python_files: List[Path]) -> List[Dict[str, Any]]:
        """Detect code changes using git integration."""
        code_changes = []
        
        try:
            for file_path in python_files[:10]:  # Limit for performance
                try:
                    # Check if file has recent changes
                    result = subprocess.run(
                        ['git', 'diff', '--stat', 'HEAD~1', str(file_path)],
                        cwd=self.project_path,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.stdout.strip():
                        code_changes.append({
                            "file": str(file_path.relative_to(self.project_path)),
                            "change_type": "modified",
                            "git_diff_stat": result.stdout.strip(),
                            "detected_at": datetime.now().isoformat()
                        })
                        
                except subprocess.TimeoutExpired:
                    continue
                except Exception as e:
                    logger.debug(f"Git diff failed for {file_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"Git integration not available: {e}")
        
        return code_changes
    
    def analyze_documentation_drift(self, python_files: List[Path], doc_files: List[Path]) -> List[Dict[str, Any]]:
        """Detect documentation inconsistencies."""
        doc_issues = []
        
        # Check for functions without docstrings
        for py_file in python_files[:5]:  # Limit for performance
            try:
                content = py_file.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                # Simple check for functions without docstrings
                in_function = False
                function_line = 0
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    if stripped.startswith('def '):
                        in_function = True
                        function_line = i
                        function_name = stripped.split('(')[0].replace('def ', '')
                        
                    elif in_function and stripped:
                        if not stripped.startswith('"""') and not stripped.startswith("'''"):
                            if not stripped.startswith('#'):
                                doc_issues.append({
                                    "file": str(py_file.relative_to(self.project_path)),
                                    "issue_type": "missing_docstring",
                                    "function": function_name,
                                    "line": function_line + 1,
                                    "detected_at": datetime.now().isoformat()
                                })
                        in_function = False
                        
            except Exception as e:
                logger.debug(f"Doc analysis failed for {py_file}: {e}")
                continue
        
        return doc_issues
    
    def analyze_semantic_drift(self, python_files: List[Path]) -> List[Dict[str, Any]]:
        """Detect semantic changes in code structure."""
        semantic_changes = []
        
        for py_file in python_files[:3]:  # Limit for performance
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Simple complexity analysis
                function_count = content.count('def ')
                class_count = content.count('class ')
                import_count = content.count('import ')
                
                # Heuristic: high complexity might indicate drift
                if function_count > 20 or import_count > 15:
                    semantic_changes.append({
                        "file": str(py_file.relative_to(self.project_path)),
                        "issue_type": "high_complexity",
                        "function_count": function_count,
                        "class_count": class_count,
                        "import_count": import_count,
                        "detected_at": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.debug(f"Semantic analysis failed for {py_file}: {e}")
                continue
        
        return semantic_changes
    
    def analyze_relationship_drift(self) -> List[Dict[str, Any]]:
        """Detect changes in code relationships and dependencies."""
        relationship_changes = []
        
        try:
            requirements_file = self.project_path / "requirements.txt"
            setup_py = self.project_path / "setup.py"
            pyproject_toml = self.project_path / "pyproject.toml"
            
            dependency_files = [f for f in [requirements_file, setup_py, pyproject_toml] if f.exists()]
            
            if len(dependency_files) > 1:
                relationship_changes.append({
                    "issue_type": "multiple_dependency_files",
                    "files": [str(f.relative_to(self.project_path)) for f in dependency_files],
                    "recommendation": "Consolidate dependency management",
                    "detected_at": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.debug(f"Relationship analysis failed: {e}")
        
        return relationship_changes
    
    def get_project_files(self, extensions: List[str]) -> List[Path]:
        """Get project files with specific extensions."""
        files = []
        for ext in extensions:
            pattern = f"**/*{ext}"
            files.extend(self.project_path.glob(pattern))
        
        # Filter out common non-source directories
        excluded_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'venv', '.venv', 'htmlcov'}
        return [f for f in files if not any(part in excluded_dirs for part in f.parts)]


# Test functionality if run directly
if __name__ == "__main__":
    print("🧪 Testing KWECLI Drift Analyzer...")
    
    # Test analyzer with current project
    analyzer = DriftAnalyzer(Path("."))
    
    # Test file discovery
    python_files = analyzer.get_project_files([".py"])
    doc_files = analyzer.get_project_files([".md", ".rst"])
    
    print(f"📄 Found {len(python_files)} Python files, {len(doc_files)} doc files")
    
    # Test analysis functions
    code_drift = analyzer.analyze_code_drift(python_files)
    print(f"🔍 Code drift analysis: {len(code_drift)} changes found")
    
    doc_drift = analyzer.analyze_documentation_drift(python_files, doc_files)
    print(f"📚 Documentation drift analysis: {len(doc_drift)} issues found")
    
    semantic_drift = analyzer.analyze_semantic_drift(python_files)
    print(f"🧠 Semantic drift analysis: {len(semantic_drift)} issues found")
    
    relationship_drift = analyzer.analyze_relationship_drift()
    print(f"🔗 Relationship drift analysis: {len(relationship_drift)} issues found")
    
    print("✅ KWECLI Drift Analyzer test complete")
"""
Prompt Analysis Module for Tool Integration.

Analyzes user prompts to detect when filesystem tools should be used
and determines which tools are needed with their parameters.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ToolIntention:
    """Represents an intention to use a specific tool with parameters."""
    tool_name: str
    parameters: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    priority: int      # Lower number = higher priority


@dataclass
class PromptAnalysis:
    """Result of analyzing a user prompt for tool requirements."""
    requires_tools: bool
    tool_intentions: List[ToolIntention]
    original_prompt: str
    analysis_confidence: float
    metadata: Dict[str, Any]


class PromptAnalyzer:
    """Analyzes prompts to determine if and which filesystem tools should be used."""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for intent detection."""
        # Project analysis patterns
        self.project_patterns = [
            (re.compile(r'analyze\s+(?:the\s+)?project\s+(?:at\s+)?["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.9),
            (re.compile(r'examine\s+(?:the\s+)?project\s+(?:structure\s+)?(?:in\s+)?["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.85),
            (re.compile(r'verify\s+(?:the\s+)?["\']?([^"\'?\s]+)["\']?\s+(?:looks\s+for|for)', re.IGNORECASE), 0.9),
            (re.compile(r'(?:project\s+)?analysis\s+(?:of\s+)?["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.8),
            (re.compile(r'(?:can\s+you\s+)?(?:please\s+)?analyze\s+(?:this\s+)?(?:project[\s:]+)["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.85),
            (re.compile(r'check\s+(?:the\s+)?(?:project\s+)?(?:directory\s+)?["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.8),
            (re.compile(r'inspect\s+(?:the\s+)?(?:project\s+)?["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.8),
        ]
        
        # File reading patterns
        self.file_read_patterns = [
            (re.compile(r'read\s+(?:the\s+)?file\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.9),
            (re.compile(r'show\s+(?:me\s+)?(?:the\s+)?contents?\s+of\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.85),
            (re.compile(r'what[\'\s]*s\s+in\s+(?:the\s+)?["\']?([^"\'?\s]+)["\']?\s+file', re.IGNORECASE), 0.8),
            (re.compile(r'examine\s+(?:the\s+)?(?:code\s+in\s+)?["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.75),
            (re.compile(r'(?:open|view)\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.7),
        ]
        
        # Directory listing patterns
        self.directory_patterns = [
            (re.compile(r'list\s+(?:the\s+)?(?:contents?\s+of\s+)?(?:directory|folder)\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.9),
            (re.compile(r'(?:what[\'\s]*s\s+in\s+)?(?:directory|folder)\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.85),
            (re.compile(r'ls\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.8),
            (re.compile(r'dir\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.75),
        ]
        
        # Search patterns
        self.search_patterns = [
            (re.compile(r'search\s+for\s+["\']([^"\']+)["\']\s+in\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.9),
            (re.compile(r'find\s+(?:all\s+)?["\']([^"\']+)["\']\s+(?:comments?\s+)?(?:in\s+)?(?:the\s+)?(?:project|codebase|files?)', re.IGNORECASE), 0.85),
            (re.compile(r'grep\s+(?:for\s+)?["\']([^"\']+)["\']\s*(?:in\s+["\']?([^"\'?\s]*)["\']?)?', re.IGNORECASE), 0.8),
            (re.compile(r'look\s+for\s+["\']([^"\']+)["\']\s+(?:in\s+)?(?:the\s+)?(?:codebase|project|files?)', re.IGNORECASE), 0.75),
        ]
        
        # Edit/modify patterns
        self.edit_patterns = [
            (re.compile(r'edit\s+(?:the\s+)?file\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.9),
            (re.compile(r'modify\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.85),
            (re.compile(r'change\s+(?:the\s+)?(?:contents?\s+of\s+)?["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.8),
            (re.compile(r'update\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.75),
        ]
        
        # Write/create patterns
        self.write_patterns = [
            (re.compile(r'(?:create|write)\s+(?:a\s+)?(?:new\s+)?file\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.9),
            (re.compile(r'write\s+(?:to\s+)?["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.8),
            (re.compile(r'create\s+["\']?([^"\'?\s]+)["\']?', re.IGNORECASE), 0.75),
        ]

        # Development (format/lint/type-check) patterns
        self.dev_patterns = [
            # format on <path>
            (re.compile(r'(?:run\s+)?(?:format|formatter|black|isort)(?:\s+on\s+([\w./-]+))?', re.IGNORECASE), 0.9),
            # lint on <path>
            (re.compile(r'(?:run\s+)?(?:lint|flake8|mypy)(?:\s+on\s+([\w./-]+))?', re.IGNORECASE), 0.9),
            # lint <path> with <tool>
            (re.compile(r'(?:run\s+)?lint\s+([\w./-]+)\s+(?:with\s+)?(?:flake8|mypy)', re.IGNORECASE), 0.95),
            # type-check <path>
            (re.compile(r'(?:type[- ]?check|static\s+typing|mypy)(?:\s+([\w./-]+))?', re.IGNORECASE), 0.85),
        ]
    
    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """
        Analyze a prompt to determine tool requirements.
        
        Args:
            prompt: User prompt to analyze
            
        Returns:
            PromptAnalysis with tool intentions and confidence scores
        """
        try:
            tool_intentions = []
            overall_confidence = 0.0
            
            # Analyze for different types of operations
            intentions = []
            intentions.extend(self._analyze_project_operations(prompt))
            intentions.extend(self._analyze_file_operations(prompt))
            intentions.extend(self._analyze_directory_operations(prompt))
            intentions.extend(self._analyze_search_operations(prompt))
            intentions.extend(self._analyze_edit_operations(prompt))
            intentions.extend(self._analyze_write_operations(prompt))
            intentions.extend(self._analyze_development_operations(prompt))
            
            # Filter and sort intentions
            valid_intentions = [intent for intent in intentions if intent.confidence >= self.confidence_threshold]
            valid_intentions.sort(key=lambda x: (x.priority, -x.confidence))
            
            # Calculate overall confidence
            if valid_intentions:
                overall_confidence = sum(intent.confidence for intent in valid_intentions) / len(valid_intentions)
            
            requires_tools = len(valid_intentions) > 0
            
            return PromptAnalysis(
                requires_tools=requires_tools,
                tool_intentions=valid_intentions,
                original_prompt=prompt,
                analysis_confidence=overall_confidence,
                metadata={
                    "total_patterns_matched": len(intentions),
                    "valid_patterns_matched": len(valid_intentions),
                    "analysis_timestamp": self._get_timestamp()
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {e}")
            return PromptAnalysis(
                requires_tools=False,
                tool_intentions=[],
                original_prompt=prompt,
                analysis_confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _analyze_project_operations(self, prompt: str) -> List[ToolIntention]:
        """Analyze for project-level operations."""
        intentions = []
        
        for pattern, confidence in self.project_patterns:
            match = pattern.search(prompt)
            if match:
                project_path = match.group(1)
                
                # Validate path exists or is reasonable
                if self._is_valid_path(project_path):
                    intentions.append(ToolIntention(
                        tool_name="directory_lister",
                        parameters={"path": project_path},
                        confidence=confidence,
                        priority=1
                    ))
                    
                    # For project analysis, also add file reading for key files
                    intentions.append(ToolIntention(
                        tool_name="read",
                        parameters={
                            "file_path": f"{project_path}/README.md",
                            "limit": 50
                        },
                        confidence=confidence * 0.8,
                        priority=2
                    ))
                    
                    # Add search for common patterns
                    intentions.append(ToolIntention(
                        tool_name="grep",
                        parameters={
                            "pattern": "TODO|FIXME|BUG|HACK",
                            "path": project_path,
                            "output_mode": "count"
                        },
                        confidence=confidence * 0.7,
                        priority=3
                    ))
        
        return intentions

    def _analyze_development_operations(self, prompt: str) -> List[ToolIntention]:
        """Analyze for development tooling operations (format/lint/type-check)."""
        intentions: List[ToolIntention] = []
        for pattern, confidence in self.dev_patterns:
            match = pattern.search(prompt)
            if match:
                path = None
                if match and match.lastindex:
                    try:
                        path = match.group(1)
                    except IndexError:
                        path = None
                params: Dict[str, Any] = {"paths": [path] if path else ["."], "check_only": True}
                intentions.append(ToolIntention(
                    tool_name="code_quality",
                    parameters=params,
                    confidence=confidence,
                    priority=2
                ))
        return intentions
    
    def _analyze_file_operations(self, prompt: str) -> List[ToolIntention]:
        """Analyze for file reading operations."""
        intentions = []
        
        for pattern, confidence in self.file_read_patterns:
            match = pattern.search(prompt)
            if match:
                file_path = match.group(1)
                
                if self._is_valid_path(file_path):
                    intentions.append(ToolIntention(
                        tool_name="read",
                        parameters={"file_path": file_path},
                        confidence=confidence,
                        priority=1
                    ))
        
        return intentions
    
    def _analyze_directory_operations(self, prompt: str) -> List[ToolIntention]:
        """Analyze for directory listing operations."""
        intentions = []
        
        for pattern, confidence in self.directory_patterns:
            match = pattern.search(prompt)
            if match:
                dir_path = match.group(1)
                
                if self._is_valid_path(dir_path):
                    intentions.append(ToolIntention(
                        tool_name="directory_lister",
                        parameters={"path": dir_path},
                        confidence=confidence,
                        priority=1
                    ))
        
        return intentions
    
    def _analyze_search_operations(self, prompt: str) -> List[ToolIntention]:
        """Analyze for search operations."""
        intentions = []
        
        for pattern, confidence in self.search_patterns:
            match = pattern.search(prompt)
            if match:
                search_term = match.group(1)
                search_path = match.group(2) if match.lastindex >= 2 else "."
                
                # Handle common search terms
                if search_term.lower() in ['todo', 'fixme', 'bug', 'hack']:
                    pattern_str = f"\\b{search_term.upper()}\\b"
                else:
                    pattern_str = search_term
                
                intentions.append(ToolIntention(
                    tool_name="grep",
                    parameters={
                        "pattern": pattern_str,
                        "path": search_path or ".",
                        "output_mode": "content",
                        "-C": 2
                    },
                    confidence=confidence,
                    priority=1
                ))
        
        return intentions
    
    def _analyze_edit_operations(self, prompt: str) -> List[ToolIntention]:
        """Analyze for edit operations."""
        intentions = []
        
        for pattern, confidence in self.edit_patterns:
            match = pattern.search(prompt)
            if match:
                file_path = match.group(1)
                
                if self._is_valid_path(file_path):
                    # First read the file to understand current content
                    intentions.append(ToolIntention(
                        tool_name="read",
                        parameters={"file_path": file_path},
                        confidence=confidence,
                        priority=1
                    ))
        
        return intentions
    
    def _analyze_write_operations(self, prompt: str) -> List[ToolIntention]:
        """Analyze for write/create operations."""
        intentions = []
        
        for pattern, confidence in self.write_patterns:
            match = pattern.search(prompt)
            if match:
                file_path = match.group(1)
                
                # For write operations, we might need to check if directory exists
                dir_path = str(Path(file_path).parent)
                if dir_path != file_path:  # Not root path
                    intentions.append(ToolIntention(
                        tool_name="directory_lister",
                        parameters={"path": dir_path},
                        confidence=confidence * 0.8,
                        priority=1
                    ))
        
        return intentions
    
    def _is_valid_path(self, path: str) -> bool:
        """
        Check if a path is valid and reasonable.
        
        Args:
            path: Path to validate
            
        Returns:
            True if path seems valid
        """
        if not path or len(path) < 1:
            return False
        
        # Basic sanity checks
        if path.startswith('http'):  # URLs are not filesystem paths
            return False
        
        if '..' in path and len(path.split('..')) > 3:  # Too many parent references
            return False
        
        if len(path) > 500:  # Unreasonably long path
            return False
        
        # Check for obvious non-path content
        if any(char in path for char in ['<', '>', '|', '^', '&']):
            return False
        
        return True
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        return self.confidence_threshold
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set confidence threshold for tool selection.
        
        Args:
            threshold: Minimum confidence required (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

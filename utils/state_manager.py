#!/usr/bin/env python3
"""
State Management System

This module provides persistent state management, conversation history,
and context management for the KWE CLI agent.
"""

import json
import os
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """A message in the conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentState:
    """Complete agent state."""
    project_name: str
    working_directory: str
    tasks: List[str]
    completed_tasks: List[str]
    current_task: str
    conversation_history: List[ConversationMessage]
    code_generated: str
    code_review: str
    execution_result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    context_window: List[str]
    last_updated: float


class StateManager:
    """Manages persistent state and conversation history."""
    
    def __init__(self, state_dir: str = "~/.kwe"):
        self.state_dir = Path(state_dir).expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_size = 1000
        self.max_context_window = 50
        
    def _get_state_file(self, project_name: str) -> Path:
        """Get the state file path for a project."""
        return self.state_dir / f"{project_name}_state.json"
    
    def _get_conversation_file(self, project_name: str) -> Path:
        """Get the conversation file path for a project."""
        return self.state_dir / f"{project_name}_conversation.json"
    
    def save_state(self, state: AgentState) -> bool:
        """Save agent state to disk."""
        try:
            state_file = self._get_state_file(state.project_name)
            
            # Convert to dict for JSON serialization
            state_dict = asdict(state)
            state_dict["last_updated"] = time.time()
            
            with open(state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
            
            logger.info(f"State saved for project: {state.project_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, project_name: str) -> Optional[AgentState]:
        """Load agent state from disk."""
        try:
            state_file = self._get_state_file(project_name)
            
            if not state_file.exists():
                return None
            
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
            
            # Convert conversation history back to objects
            if "conversation_history" in state_dict:
                conv_history = []
                for msg_dict in state_dict["conversation_history"]:
                    conv_history.append(ConversationMessage(**msg_dict))
                state_dict["conversation_history"] = conv_history
            
            state = AgentState(**state_dict)
            logger.info(f"State loaded for project: {project_name}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def create_initial_state(self, project_name: str, working_directory: str, 
                           initial_task: str) -> AgentState:
        """Create initial state for a new project."""
        return AgentState(
            project_name=project_name,
            working_directory=working_directory,
            tasks=[initial_task],
            completed_tasks=[],
            current_task=initial_task,
            conversation_history=[],
            code_generated="",
            code_review="",
            execution_result=None,
            error_message=None,
            context_window=[],
            last_updated=time.time()
        )
    
    def add_message(self, state: AgentState, role: str, content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> AgentState:
        """Add a message to the conversation history."""
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata
        )
        
        state.conversation_history.append(message)
        
        # Limit history size
        if len(state.conversation_history) > self.max_history_size:
            state.conversation_history = state.conversation_history[-self.max_history_size:]
        
        # Update context window
        self._update_context_window(state)
        
        state.last_updated = time.time()
        return state
    
    def _update_context_window(self, state: AgentState):
        """Update the context window with recent messages."""
        recent_messages = state.conversation_history[-self.max_context_window:]
        state.context_window = [msg.content for msg in recent_messages]
    
    def get_context_summary(self, state: AgentState) -> str:
        """Get a summary of the current context."""
        if not state.conversation_history:
            return "No conversation history."
        
        summary_parts = [f"Project: {state.project_name}"]
        summary_parts.append(f"Current Task: {state.current_task}")
        summary_parts.append(f"Working Directory: {state.working_directory}")
        summary_parts.append(f"Messages: {len(state.conversation_history)}")
        
        if state.code_generated:
            summary_parts.append(f"Code Generated: {len(state.code_generated)} chars")
        
        if state.execution_result:
            summary_parts.append(f"Last Execution: {state.execution_result.get('status', 'unknown')}")
        
        return "\n".join(summary_parts)
    
    def search_history(self, state: AgentState, query: str) -> List[ConversationMessage]:
        """Search conversation history for messages containing the query."""
        results = []
        query_lower = query.lower()
        
        for message in state.conversation_history:
            if query_lower in message.content.lower():
                results.append(message)
        
        return results
    
    def export_conversation(self, state: AgentState, format: str = "json") -> str:
        """Export conversation history."""
        if format == "json":
            return json.dumps([asdict(msg) for msg in state.conversation_history], indent=2)
        elif format == "text":
            lines = []
            for msg in state.conversation_history:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", 
                                       time.localtime(msg.timestamp))
                lines.append(f"[{timestamp}] {msg.role.upper()}: {msg.content}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def cleanup_old_states(self, max_age_days: int = 30) -> int:
        """Clean up old state files."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0
        
        for state_file in self.state_dir.glob("*_state.json"):
            try:
                if state_file.stat().st_mtime < cutoff_time:
                    state_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup {state_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} old state files")
        return cleaned_count
    
    def list_projects(self) -> List[str]:
        """List all projects with saved state."""
        projects = []
        for state_file in self.state_dir.glob("*_state.json"):
            project_name = state_file.stem.replace("_state", "")
            projects.append(project_name)
        return sorted(projects)
    
    def delete_project_state(self, project_name: str) -> bool:
        """Delete state for a specific project."""
        try:
            state_file = self._get_state_file(project_name)
            conv_file = self._get_conversation_file(project_name)
            
            if state_file.exists():
                state_file.unlink()
            
            if conv_file.exists():
                conv_file.unlink()
            
            logger.info(f"Deleted state for project: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete state for {project_name}: {e}")
            return False


# Global instance for easy use
state_manager = StateManager()


def save_state(state: AgentState) -> bool:
    """Save agent state."""
    return state_manager.save_state(state)


def load_state(project_name: str) -> Optional[AgentState]:
    """Load agent state."""
    return state_manager.load_state(project_name)


def create_initial_state(project_name: str, working_directory: str, 
                       initial_task: str) -> AgentState:
    """Create initial state for a new project."""
    return state_manager.create_initial_state(project_name, working_directory, initial_task)


def add_message(state: AgentState, role: str, content: str, 
               metadata: Optional[Dict[str, Any]] = None) -> AgentState:
    """Add a message to the conversation history."""
    return state_manager.add_message(state, role, content, metadata) 
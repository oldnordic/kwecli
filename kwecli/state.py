#!/usr/bin/env python3
"""
KWECLI State Management Module
==============================

Main state management coordinator using focused modules.
Manages user sessions and LTMC integration.

Features:
- Session coordination and management
- LTMC integration for advanced state tracking
- Modular architecture with focused components

File: kwecli/state.py
Purpose: State management coordinator
"""

import sys
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bridge.ltmc_native import get_ltmc_native
from .config import load_config
from .session_data import KWECLISession
from .session_storage import SessionStorage

logger = logging.getLogger(__name__)


class StateManager:
    """State manager coordinating sessions and LTMC integration."""
    
    def __init__(self, project_path: str = "."):
        """Initialize state manager."""
        self.project_path = Path(project_path).resolve()
        self.config = load_config(str(self.project_path))
        self.ltmc = None
        
        # Initialize components
        self.storage = SessionStorage(self.project_path)
        self.current_session: Optional[KWECLISession] = None
        
        # Initialize LTMC
        self._init_ltmc()
    
    def _init_ltmc(self):
        """Initialize LTMC connection for state management."""
        try:
            self.ltmc = get_ltmc_native()
            logger.info("LTMC connected for state management")
        except Exception as e:
            logger.warning(f"LTMC not available for state: {e}")
            self.ltmc = None
    
    def create_session(self, interface_mode: str = "tui") -> KWECLISession:
        """Create new session."""
        try:
            # Create session with current config
            session = KWECLISession(
                project_path=str(self.project_path),
                project_name=self.project_path.name,
                interface_mode=interface_mode,
                config_snapshot=self.config.__dict__.copy(),
                expires_at=datetime.now() + timedelta(seconds=self.config.session_timeout)
            )
            
            # Save to database
            self.storage.save_session(session)
            
            # Save to LTMC
            if self.ltmc:
                self._save_session_to_ltmc(session)
            
            # Set as current session
            self.current_session = session
            
            logger.info(f"Created new session: {session.session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            # Return basic session on error
            return KWECLISession(project_path=str(self.project_path))
    
    def load_session(self, session_id: str) -> Optional[KWECLISession]:
        """Load existing session."""
        session = self.storage.load_session(session_id)
        if session and session.is_active():
            self.current_session = session
            logger.info(f"Loaded active session: {session_id}")
            return session
        elif session:
            logger.info(f"Session expired or inactive: {session_id}")
        return None
    
    def get_current_session(self) -> Optional[KWECLISession]:
        """Get current active session."""
        if self.current_session and self.current_session.is_active():
            return self.current_session
        
        # Try to load most recent active session
        recent_session = self.storage.load_recent_session(str(self.project_path))
        if recent_session and recent_session.is_active():
            self.current_session = recent_session
        
        return self.current_session
    
    def save_session(self, session: Optional[KWECLISession] = None) -> bool:
        """Save session to database."""
        session = session or self.current_session
        if not session:
            return False
        
        return self.storage.save_session(session)
    
    def end_session(self, session: Optional[KWECLISession] = None) -> bool:
        """End session gracefully."""
        session = session or self.current_session
        if not session:
            return True
        
        try:
            # Mark session as inactive
            session.active = False
            session.update_activity()
            
            # Save final state
            self.storage.save_session(session)
            
            # Save end session to LTMC
            if self.ltmc:
                self.ltmc.save_thought(
                    kind="session_end",
                    content=f"Session ended: {session.session_id}",
                    metadata=session.get_session_summary()
                )
            
            # Clear current session if it's this one
            if self.current_session and self.current_session.session_id == session.session_id:
                self.current_session = None
            
            logger.info(f"Session ended: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return False
    
    def _save_session_to_ltmc(self, session: KWECLISession):
        """Save session creation to LTMC."""
        try:
            self.ltmc.save_thought(
                kind="session_start",
                content=f"KWECLI session started: {session.interface_mode} mode",
                metadata={
                    "session_id": session.session_id,
                    "project_path": session.project_path,
                    "project_name": session.project_name,
                    "interface_mode": session.interface_mode,
                    "created_at": session.created_at.isoformat()
                }
            )
        except Exception as e:
            logger.debug(f"Failed to save session to LTMC: {e}")
    
    def cleanup_old_sessions(self, days: int = 7) -> int:
        """Clean up old/expired sessions."""
        return self.storage.cleanup_old_sessions(days)


# Global state manager instance
_state_manager = None

def get_state_manager(project_path: str = ".") -> StateManager:
    """Get or create global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager(project_path)
    return _state_manager

def get_current_session(project_path: str = ".") -> Optional[KWECLISession]:
    """Get current active session."""
    manager = get_state_manager(project_path)
    return manager.get_current_session()

def create_session(project_path: str = ".", interface_mode: str = "tui") -> KWECLISession:
    """Create new KWECLI session."""
    manager = get_state_manager(project_path)
    return manager.create_session(interface_mode)


# Test functionality if run directly
if __name__ == "__main__":
    print("🧪 Testing KWECLI State Manager...")
    
    # Test state manager
    manager = get_state_manager(".")
    
    # Test session creation
    session = manager.create_session("tui")
    print(f"✅ Session created: {session.session_id}")
    print(f"  Project: {session.project_name} at {session.project_path}")
    print(f"  Mode: {session.interface_mode}")
    print(f"  Active: {session.is_active()}")
    
    # Test session operations
    session.add_command("analyze project")
    session.set_context("current_view", "main")
    
    # Test save/load
    save_ok = manager.save_session(session)
    print(f"✅ Session save: {'success' if save_ok else 'failed'}")
    
    # Test session loading
    loaded_session = manager.load_session(session.session_id)
    if loaded_session:
        print(f"✅ Session loaded: {loaded_session.session_id}")
        print(f"  Commands: {len(loaded_session.command_history)}")
        print(f"  Context: {loaded_session.context_data}")
    
    # Test cleanup
    cleanup_count = manager.cleanup_old_sessions(0)  # Clean everything for test
    print(f"✅ Cleanup: {cleanup_count} sessions cleaned")
    
    print("✅ KWECLI State Manager test complete")
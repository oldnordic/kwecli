import subprocess
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TmuxWindow:
    session_name: str
    window_index: int
    window_name: str
    active: bool
    
@dataclass
class TmuxSession:
    name: str
    windows: List[TmuxWindow]
    attached: bool

class TmuxManager:
    def __init__(self):
        self.safety_mode = True
        self.max_lines_capture = 1000
        
    def _run_tmux_command(self, cmd: List[str]) -> Tuple[str, str, int]:
        """Helper to run tmux commands and return stdout, stderr, and exit code."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except Exception as e:
            return "", str(e), 1

    def get_tmux_sessions(self) -> List[TmuxSession]:
        """Get all tmux sessions and their windows"""
        stdout, stderr, exit_code = self._run_tmux_command(["tmux", "list-sessions", "-F", "#{session_name}:#{session_attached}"])
        if exit_code != 0:
            print(f"Error getting tmux sessions: {stderr}")
            return []
            
        sessions = []
        for line in stdout.split('\n'):
            if not line:
                continue
            session_name, attached = line.split(':')
            
            stdout_windows, stderr_windows, exit_code_windows = self._run_tmux_command(["tmux", "list-windows", "-t", session_name, "-F", "#{window_index}:#{window_name}:#{window_active}"])
            if exit_code_windows != 0:
                print(f"Error getting windows for session {session_name}: {stderr_windows}")
                continue

            windows = []
            for window_line in stdout_windows.split('\n'):
                if not window_line:
                    continue
                window_index, window_name, window_active = window_line.split(':')
                windows.append(TmuxWindow(
                    session_name=session_name,
                    window_index=int(window_index),
                    window_name=window_name,
                    active=window_active == '1'
                ))
            
            sessions.append(TmuxSession(
                name=session_name,
                windows=windows,
                attached=attached == '1'
            ))
        
        return sessions
    
    def capture_window_content(self, session_name: str, window_index: int, num_lines: int = 50) -> str:
        """Safely capture the last N lines from a tmux window"""
        if num_lines > self.max_lines_capture:
            num_lines = self.max_lines_capture
            
        stdout, stderr, exit_code = self._run_tmux_command(["tmux", "capture-pane", "-t", f"{session_name}:{window_index}", "-p", "-S", f"-{num_lines}"])
        if exit_code != 0:
            return f"Error capturing window content: {stderr}"
        return stdout
    
    def get_window_info(self, session_name: str, window_index: int) -> Dict:
        """Get detailed information about a specific window"""
        stdout, stderr, exit_code = self._run_tmux_command(["tmux", "display-message", "-t", f"{session_name}:{window_index}", "-p", 
               "#{window_name}:#{window_active}:#{window_panes}:#{window_layout}"])
        if exit_code != 0:
            return {"error": f"Could not get window info: {stderr}"}
            
        if stdout:
            parts = stdout.split(':')
            return {
                "name": parts[0],
                "active": parts[1] == '1',
                "panes": int(parts[2]),
                "layout": parts[3],
                "content": self.capture_window_content(session_name, window_index)
            }
        return {"error": "No window info found"}
    
    def send_keys_to_window(self, session_name: str, window_index: int, keys: str, confirm: bool = False) -> bool:
        """Safely send keys to a tmux window with confirmation"""
        if self.safety_mode and confirm:
            print(f"SAFETY CHECK: About to send '{keys}' to {session_name}:{window_index}")
            response = input("Confirm? (yes/no): ")
            if response.lower() != 'yes':
                print("Operation cancelled")
                return False
        
        stdout, stderr, exit_code = self._run_tmux_command(["tmux", "send-keys", "-t", f"{session_name}:{window_index}", keys])
        if exit_code != 0:
            print(f"Error sending keys: {stderr}")
            return False
        return True
    
    def send_command_to_window(self, session_name: str, window_index: int, command: str, confirm: bool = False) -> bool:
        """Send a command to a window (adds Enter automatically)"""
        # First send the command text
        if not self.send_keys_to_window(session_name, window_index, command, confirm):
            return False
        # Then send the actual Enter key (C-m)
        stdout, stderr, exit_code = self._run_tmux_command(["tmux", "send-keys", "-t", f"{session_name}:{window_index}", "C-m"])
        if exit_code != 0:
            print(f"Error sending Enter key: {stderr}")
            return False
        return True
    
    def create_new_session(self, session_name: str, path: str) -> bool:
        """Creates a new tmux session."""
        stdout, stderr, exit_code = self._run_tmux_command(["tmux", "new-session", "-d", "-s", session_name, "-c", path])
        if exit_code != 0:
            print(f"Error creating new session: {stderr}")
            return False
        return True

    def create_new_window(self, session_name: str, window_name: str, path: str) -> bool:
        """Creates a new tmux window within a session."""
        stdout, stderr, exit_code = self._run_tmux_command(["tmux", "new-window", "-t", session_name, "-n", window_name, "-c", path])
        if exit_code != 0:
            print(f"Error creating new window: {stderr}")
            return False
        return True

    def rename_window(self, session_name: str, window_index: int, new_name: str) -> bool:
        """Renames a tmux window."""
        stdout, stderr, exit_code = self._run_tmux_command(["tmux", "rename-window", "-t", f"{session_name}:{window_index}", new_name])
        if exit_code != 0:
            print(f"Error renaming window: {stderr}")
            return False
        return True

    def create_monitoring_snapshot(self) -> str:
        """Create a comprehensive snapshot for Claude analysis"""
        status = self.get_all_windows_status()
        
        # Format for Claude consumption
        snapshot = f"Tmux Monitoring Snapshot - {status['timestamp']}\n"
        snapshot += "=" * 50 + "\n\n"
        
        for session in status['sessions']:
            snapshot += f"Session: {session['name']} ({'ATTACHED' if session['attached'] else 'DETACHED'})\n"
            snapshot += "-" * 30 + "\n"
            
            for window in session['windows']:
                snapshot += f"  Window {window['index']}: {window['name']}"
                if window['active']:
                    snapshot += " (ACTIVE)"
                snapshot += "\n"
                
                if 'content' in window['info']:
                    # Get last 10 lines for overview
                    content_lines = window['info']['content'].split('\n')
                    recent_lines = content_lines[-10:] if len(content_lines) > 10 else content_lines
                    snapshot += "    Recent output:\n"
                    for line in recent_lines:
                        if line.strip():
                            snapshot += f"    | {line}\n"
                snapshot += "\n"
        
        return snapshot

if __name__ == "__main__":
    manager = TmuxManager()
    status = manager.get_tmux_sessions()
    print(json.dumps(status, indent=2))

    print("\n\n--- Monitoring Snapshot ---")
    snapshot = manager.create_monitoring_snapshot()
    print(snapshot)

    print("\n\n--- Creating a new session and window ---")
    session_name = "test_session"
    window_name = "test_window"
    path = "/tmp"
    if manager.create_new_session(session_name, path):
        print(f"Session '{session_name}' created.")
        if manager.create_new_window(session_name, window_name, path):
            print(f"Window '{window_name}' created in '{session_name}'.")
            manager.send_command_to_window(session_name, 0, "echo 'Hello from tmux!'")
            time.sleep(1)
            content = manager.capture_window_content(session_name, 0)
            print(f"Captured content:\n{content}")
            manager.rename_window(session_name, 0, "renamed_window")
            print(f"Window renamed.")
        else:
            print("Failed to create window.")
    else:
        print("Failed to create session.")

    # Clean up
    # subprocess.run(["tmux", "kill-session", "-t", session_name])

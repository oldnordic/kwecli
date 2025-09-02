#!/usr/bin/env python3
"""
KWE CLI Interface Module

This module handles the main CLI interface, navigation, and UI framework.
Provides welcome screen, main menu, settings, and chat history functionality.
"""

import asyncio
from typing import Dict, Any, List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.markdown import Markdown
from rich import box
import json
import sys
import termios
import tty
from typing import List, Tuple

# Initialize Rich console
console = Console()


class CLIInterface:
    """Handles CLI interface navigation and main UI components."""
    
    def __init__(self, backend_comm, interactive_modes, context_mgmt, coordination_interface):
        """
        Initialize CLI interface with module dependencies.
        
        Args:
            backend_comm: Backend communication module instance
            interactive_modes: Interactive modes module instance  
            context_mgmt: Context management module instance
            coordination_interface: Coordination interface module instance
        """
        self.backend_comm = backend_comm
        self.interactive_modes = interactive_modes
        self.context_mgmt = context_mgmt
        self.coordination_interface = coordination_interface
        self.chat_history: List[Dict[str, Any]] = []
        self.is_running = True
        
    async def display_welcome(self) -> bool:
        """Display welcome screen and check backend health."""
        welcome_text = """
# 🚀 KWE CLI - Knowledge Workflow Engine

Welcome to the **Knowledge Workflow Engine Command Line Interface**!

This powerful CLI provides:
- 🤖 **AI-Powered Development** - Get help with coding tasks
- 💻 **Code Generation** - Generate code from descriptions
- 🔍 **Code Analysis** - Analyze and improve your code
- 📝 **Context Management** - Store and retrieve project context
- 🎯 **Smart Assistance** - AI-driven development guidance

## Quick Start
1. **Chat Mode** - Interactive AI conversations
2. **Code Generation** - Create code from descriptions
3. **Code Analysis** - Analyze existing code
4. **Context Management** - Manage project knowledge

Ready to boost your development productivity?
        """
        
        console.clear()
        console.print(Panel(
            Markdown(welcome_text),
            title="🎯 KWE CLI - Knowledge Workflow Engine",
            border_style="blue",
            padding=(1, 2)
        ))
        
        # Check backend status
        if await self.backend_comm.check_health():
            console.print("✅ [green]Backend server is running[/green]")
            return True
        else:
            console.print("❌ [red]Backend server is not running[/red]")
            console.print("💡 Start the backend with: [yellow]python cli_backend_server.py[/yellow]")
            return False
    
    async def main_menu(self):
        """Display main menu with arrow-key navigation."""
        # Define menu options as (key, description)
        menu_options: List[Tuple[str, str]] = [
            ("1", "💬 Chat Mode - Interactive AI conversations"),
            ("2", "💻 Code Generation - Generate code from descriptions"),
            ("3", "🔍 Code Analysis - Analyze and improve code"),
            ("4", "📝 Context Management - Store and retrieve context"),
            ("5", "🎯 Coordination & Task Delegation - AI agent coordination"),
            ("8", "🛠️  Development Actions - Format/Lint/Tests"),
            ("A", "🧭 Planning Mode - Plan & Docs"),
            ("B", "🧪 Guided Code + Tests"),
            ("9", "🧩 Diff & Apply Patch"),
            ("6", "📊 View Chat History"),
            ("C", "🧾 View Run History"),
            ("D", "💾 Save Session to Markdown"),
            ("7", "⚙️  Settings & Configuration"),
            ("G", "📚 RAG Search"),
            ("H", "💡 RAG Answer"),
            ("0", "🚪 Exit")
        ]
        # Loop until exit selected
        while self.is_running:
            choice = self._arrow_menu(menu_options)
            await self._handle_menu_choice(choice)
    
    async def _handle_menu_choice(self, choice: str):
        """Handle menu choice selection."""
        if choice == "0":
            await self.exit_cli()
        elif choice == "1":
            await self.interactive_modes.chat_mode(self.chat_history)
        elif choice == "2":
            await self.interactive_modes.code_generation_mode()
        elif choice == "3":
            await self.interactive_modes.code_analysis_mode()
        elif choice == "4":
            await self.context_mgmt.context_management_mode()
        elif choice == "5":
            await self.coordination_interface.coordination_mode()
        elif choice == "8":
            await self.interactive_modes.dev_actions_mode()
        elif choice == "9":
            await self.interactive_modes.diff_apply_mode()
        elif choice == "A":
            await self.interactive_modes.planning_mode()
        elif choice == "B":
            await self._guided_code_tests()
        elif choice == "6":
            await self.view_chat_history()
        elif choice == "7":
            await self.settings_mode()
        elif choice == "G":
            await self.interactive_modes.rag_search_mode()
        elif choice == "H":
            await self.interactive_modes.rag_answer_mode()
        elif choice == "C":
            await self.view_run_history()
        elif choice == "D":
            await self.save_session_markdown()
    
    async def view_chat_history(self):
        """View chat history."""
        console.clear()
        console.print(Panel(
            "📊 Chat History",
            style="bold cyan",
            border_style="cyan"
        ))
        
        if not self.chat_history:
            console.print("📭 No chat history yet")
            Prompt.ask("\nPress Enter to continue")
            return
        
        console.print(f"💬 Total conversations: {len(self.chat_history)}")
        console.print()
        
        for i, chat in enumerate(self.chat_history, 1):
            console.print(Panel(
                f"👤 You: {chat.get('user', 'N/A')}\n"
                f"🤖 AI: {chat.get('ai', 'N/A')}\n"
                f"🕒 {chat.get('timestamp', 'N/A')}",
                title=f"Conversation {i}",
                border_style="blue"
            ))
            console.print()
        
        Prompt.ask("\nPress Enter to continue")

    async def view_run_history(self):
        """View persisted run history from the backend session store."""
        console.clear()
        console.print(Panel("🧾 Run History", style="bold cyan", border_style="cyan"))
        # Interactive loop with filters, paging, and detail view
        current_limit = 200  # fetch a larger batch, page locally
        type_filter = ""
        search = ""
        meta_filter = ""  # e.g., key or key=value substring
        page_size = 20
        page_index = 0
        while True:
            try:
                data = await self.backend_comm.session_history(limit=current_limit)
            except Exception as e:
                data = {"error": str(e)}
            if data.get("error"):
                console.print(f"❌ [red]Error: {data['error']}[/red]")
                Prompt.ask("\nPress Enter to return")
                return
            items = data.get("history", [])
            # Apply client-side filters
            filtered = []
            for ev in items:
                if type_filter and ev.get("type") != type_filter:
                    continue
                if search and search.lower() not in (ev.get("message", "") or "").lower():
                    continue
                if meta_filter:
                    md = ev.get("metadata", {}) or {}
                    text = str(meta_filter)
                    if "=" in text:
                        k, v = text.split("=", 1)
                        if k and (k not in md or v.lower() not in str(md.get(k, "")).lower()):
                            continue
                    else:
                        # presence of key or in any value
                        has_key = text in md
                        has_val = any(text.lower() in str(val).lower() for val in md.values())
                        if not (has_key or has_val):
                            continue
                filtered.append(ev)
            total = len(filtered)
            total_pages = max(1, (total + page_size - 1) // page_size)
            if page_index >= total_pages:
                page_index = max(0, total_pages - 1)
            start = page_index * page_size
            end = start + page_size
            page_slice = filtered[start:end]

            if not filtered:
                console.print("📭 No matching history events")
            subtitle = (
                f"filters: type='{type_filter or '*'}' search='{search or '*'}' meta='{meta_filter or '*'}' | "
                f"page {page_index+1}/{total_pages} (size {page_size})"
            )
            table = Table(title=f"Events (showing {len(page_slice)}/{total} of {len(items)}; limit={current_limit})\n{subtitle}", box=box.SIMPLE_HEAVY)
            table.add_column("#", style="magenta", no_wrap=True)
            table.add_column("Time", style="cyan")
            table.add_column("Type", style="white")
            table.add_column("Message", style="white")
            for i, ev in enumerate(page_slice, start=1):
                absolute_index = start + i
                table.add_row(str(absolute_index), ev.get("timestamp", ""), ev.get("type", "event"), ev.get("message", ""))
            console.print(table)
            console.print("Options: [L]imit  [T]ype  [S]earch  [M]eta  [P]ageSize  [N]ext  [P]rev  [G]oto  [V]iew <#>  [R]efresh  [C]lear  [B]ack")
            choice = Prompt.ask("Action", default="R").strip()
            if not choice:
                continue
            c = choice.lower()
            if c == "b":
                break
            if c == "l":
                try:
                    current_limit = int(Prompt.ask("New limit", default=str(current_limit)))
                except Exception:
                    current_limit = 100
            elif c == "t":
                # Gather types
                types = sorted({ev.get("type", "event") for ev in items})
                console.print(f"Available types: {', '.join(types) if types else '(none)'}")
                val = Prompt.ask("Type filter (empty for all)", default=type_filter)
                type_filter = val.strip()
                page_index = 0
            elif c == "s":
                val = Prompt.ask("Search text (in message)", default=search)
                search = val.strip()
                page_index = 0
            elif c == "m":
                console.print("Enter metadata filter. Examples: 'key', 'key=value'")
                val = Prompt.ask("Meta filter", default=meta_filter)
                meta_filter = val.strip()
                page_index = 0
            elif c == "p":
                try:
                    page_size = int(Prompt.ask("Page size", default=str(page_size)))
                    if page_size <= 0:
                        page_size = 20
                except Exception:
                    page_size = 20
                page_index = 0
            elif c == "n":
                if page_index + 1 < total_pages:
                    page_index += 1
            elif c == "g":
                try:
                    target = int(Prompt.ask("Go to page #", default=str(page_index+1))) - 1
                    if 0 <= target < total_pages:
                        page_index = target
                except Exception:
                    pass
            elif c == "prev" or c == "pr" or c == "p":
                # note: 'p' also used for PageSize; disambiguate by exact match above
                if choice.lower() in ("prev", "pr") and page_index > 0:
                    page_index -= 1
            elif c.startswith("v"):
                # allow 'v' or 'v 3'
                parts = choice.split()
                idx_str = parts[1] if len(parts) > 1 else Prompt.ask("Row # to view details (absolute # as shown)", default=str(start+1))
                try:
                    idx = int(idx_str)
                    if 1 <= idx <= len(filtered):
                        ev = filtered[idx - 1]
                        meta = ev.get("metadata", {})
                        pretty = json.dumps(meta, indent=2, ensure_ascii=False) if isinstance(meta, dict) else str(meta)
                        console.print(Panel(pretty or "{}", title=f"Metadata for #{idx}", border_style="green"))
                        Prompt.ask("Press Enter to return")
                except Exception:
                    pass
            elif c == "c":
                type_filter = ""; search = ""; meta_filter = ""; page_index = 0
            # Clear for next loop
            console.clear()
            console.print(Panel("🧾 Run History", style="bold cyan", border_style="cyan"))
    
    async def settings_mode(self):
        """Interactive configuration: change model, toggle ollama/rag, manage sessions."""
        console.clear()
        console.print(Panel("⚙️  Settings & Configuration", style="bold yellow", border_style="yellow"))
        try:
            cfg = await self.backend_comm.get_config()
        except Exception as e:
            cfg = {"error": str(e)}
        if cfg.get("error"):
            console.print(f"❌ [red]Config error: {cfg['error']}[/red]")
            Prompt.ask("Press Enter to continue")
            return

        while True:
            console.clear()
            table = Table(title="Current Configuration")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="white")
            for k in ["use_ollama", "ollama_model", "rag_enabled", "docs_cache_path", "backend_host", "backend_port"]:
                table.add_row(k, str(cfg.get(k)))
            console.print(table)
            choice = Prompt.ask(
                "Select action",
                choices=[
                    "change-model", "toggle-ollama", "toggle-rag", "change-docs-path",
                    "test-ollama", "sessions", "back"
                ],
                default="back"
            )
            if choice == "back":
                break
            if choice == "change-model":
                # Show available models to help selection
                models_info = await self.backend_comm.get_models()
                if models_info.get("available_models"):
                    console.print(Panel(
                        "\n".join(models_info.get("available_models", [])) or "(no models)",
                        title="Available Models (Ollama)", border_style="yellow"
                    ))
                new_model = Prompt.ask("New Ollama model (e.g., qwen2.5-coder:7b)", default=str(cfg.get("ollama_model", "qwen2.5-coder:7b")))
                res = await self.backend_comm.update_config(ollama_model=new_model)
                if res.get("success"):
                    cfg = res.get("config", cfg)
                else:
                    console.print(f"❌ [red]Update failed: {res.get('error','unknown')}[/red]")
            elif choice == "toggle-ollama":
                val = not bool(cfg.get("use_ollama", True))
                res = await self.backend_comm.update_config(use_ollama=val)
                cfg = res.get("config", cfg)
            elif choice == "toggle-rag":
                val = not bool(cfg.get("rag_enabled", False))
                res = await self.backend_comm.update_config(rag_enabled=val)
                cfg = res.get("config", cfg)
            elif choice == "change-docs-path":
                current = str(cfg.get("docs_cache_path") or "")
                new_path = Prompt.ask("Docs cache path (absolute or relative)", default=current)
                res = await self.backend_comm.update_config(docs_cache_path=new_path)
                cfg = res.get("config", cfg)
            elif choice == "test-ollama":
                info = await self.backend_comm.get_models()
                ok = bool(info.get("ollama_available"))
                if ok:
                    default_model = info.get("default_model")
                    configured_model = info.get("configured_model")
                    available = info.get("available_models", [])
                    console.print(Panel(
                        f"✅ Ollama available\nConfigured: {configured_model}\nDefault: {default_model}\nModels: {', '.join(available) if available else '(none)'}",
                        title="Ollama Status", border_style="green"
                    ))
                else:
                    console.print(Panel("❌ Ollama models not available", title="Ollama Status", border_style="red"))
            elif choice == "sessions":
                await self._sessions_menu()
        Prompt.ask("Press Enter to continue")

    async def _sessions_menu(self):
        console.clear()
        console.print(Panel("🗂️ Sessions", style="bold blue", border_style="blue"))
        data = await self.backend_comm.session_list()
        sessions = data.get("sessions", [])
        current = data.get("current")
        table = Table(title="Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Created", style="white")
        table.add_column("Current", style="green")
        for s in sessions:
            table.add_row(s.get("id",""), s.get("name",""), s.get("created_at",""), "*" if current and current.get("id") == s.get("id") else "")
        console.print(table)
        action = Prompt.ask("Action", choices=["select", "create", "back"], default="back")
        if action == "create":
            name = Prompt.ask("Session name", default="")
            await self.backend_comm.session_create(name=name or None)
        elif action == "select":
            sid = Prompt.ask("Session ID")
            await self.backend_comm.session_select(session_id=sid)
    
    async def exit_cli(self):
        """Exit the CLI."""
        console.print("\n👋 Thank you for using KWE CLI!")
        console.print("🚀 Keep building amazing things!")
        self.is_running = False
    
    def add_chat_to_history(self, user_input: str, ai_response: str, timestamp: str):
        """Add a chat interaction to history."""
        self.chat_history.append({
            "timestamp": timestamp,
            "user": user_input,
            "ai": ai_response
        })
    
    def get_console(self):
        """Get the Rich console instance."""
        return console

    async def save_session_markdown(self):
        """Save current session to docs/chats via backend endpoint."""
        console.clear()
        console.print(Panel("💾 Save Session", style="bold green", border_style="green"))
        try:
            res = await self.backend_comm.session_save_markdown()
        except Exception as e:
            res = {"error": str(e)}
        if res.get("success"):
            console.print(f"✅ Saved: {res.get('path')}")
        else:
            console.print(f"❌ [red]Failed:[/red] {res.get('error','unknown')}")
        Prompt.ask("\nPress Enter to continue")
    
    def _arrow_menu(self, options: List[Tuple[str, str]]) -> str:
        """Render a menu navigable with arrow keys and return the selected key."""
        idx = 0
        # Save original terminal settings
        fd = sys.stdin.fileno()
        while True:
            console.clear()
            console.print(Panel(
                "🎯 KWE CLI - Knowledge Workflow Engine",
                style="bold blue",
                border_style="blue"
            ))
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Description", style="white")
            for i, (key, desc) in enumerate(options):
                style = "reverse" if i == idx else ""
                table.add_row(f"{key}", desc, style=style)
            console.print(table)
            console.print("\nUse ↑/↓ arrows and Enter to select.")
            # Read a single keypress
            ch = self._getch()
            if ch == "\x1b[A":  # Up arrow
                idx = (idx - 1) % len(options)
            elif ch == "\x1b[B":  # Down arrow
                idx = (idx + 1) % len(options)
            elif ch in ("\r", "\n"):  # Enter
                return options[idx][0]
            # ignore other keys

    def _getch(self) -> str:
        """Read single character or escape sequence from stdin."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # read two more chars
                ch += sys.stdin.read(2)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


    async def _guided_code_tests(self):
        console.clear()
        console.print(Panel("🧪 Guided Code + Tests", style="bold green", border_style="green"))
        desc = Prompt.ask("Describe the feature to implement")
        lang = Prompt.ask("Language", default="python")
        code_path = Prompt.ask("Code file path", default="generated_code.py")
        test_path = Prompt.ask("Test file path", default="tests/test_generated_code.py")
        with console.status("[bold green]Generating code and tests...", spinner="dots"):
            res = await self.backend_comm.generate_code_with_tests(desc, lang, code_path, test_path)
        ok = res.get("success", False)
        table = Table(title="Guided Code + Tests Result")
        table.add_column("Key", style="cyan"); table.add_column("Value", style="white")
        table.add_row("success", str(ok))
        files = res.get("files", {})
        table.add_row("code", str(files.get("code")))
        table.add_row("tests", str(files.get("tests")))
        test_res = res.get("test_result", {})
        table.add_row("pytest_exit", str(test_res.get("exit_code")))
        console.print(table)
        console.print(Panel((test_res.get("stdout", "") or "")[:2000], title="pytest STDOUT", border_style="yellow"))
        if test_res.get("stderr"):
            console.print(Panel((test_res.get("stderr", "") or "")[:1000], title="pytest STDERR", border_style="red"))
        Prompt.ask("\nPress Enter to continue")

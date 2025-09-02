#!/usr/bin/env python3
"""
KWE CLI Interactive Modes Module

This module handles the core AI interaction features:
- Chat mode for interactive conversations
- Code generation mode for creating code from descriptions
- Code analysis mode for analyzing and improving existing code
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from rich.syntax import Syntax


class InteractiveModes:
    """Handles all interactive AI-powered modes."""
    
    def __init__(self, backend_comm, console: Console):
        """
        Initialize interactive modes with backend communication.
        
        Args:
            backend_comm: Backend communication module instance
            console: Rich console instance for output
        """
        self.backend_comm = backend_comm
        self.console = console
    
    async def chat_mode(self, chat_history: List[Dict[str, Any]]):
        """Interactive chat mode with AI."""
        self.console.clear()
        self.console.print(Panel(
            "💬 Chat Mode - Interactive AI Conversations",
            style="bold green",
            border_style="green"
        ))
        
        self.console.print("🤖 Start chatting with the AI! Type 'exit' to return to main menu.")
        self.console.print("💡 You can ask about coding, get help, or discuss development topics.")
        self.console.print()
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("👤 You")
                
                if user_input.lower() in ['exit', 'quit', 'back']:
                    break
                
                if not user_input.strip():
                    continue
                
                # Show typing indicator
                with self.console.status("[bold green]AI is thinking...", spinner="dots"):
                    # Send to backend
                    response = await self.backend_comm.send_chat_request(user_input)
                
                # Display AI response
                self.console.print()
                self.console.print(Panel(
                    response,
                    title="🤖 AI Assistant",
                    border_style="green",
                    padding=(1, 2)
                ))
                self.console.print()
                
                # Store in history (handled by CLI interface)
                timestamp = datetime.now().isoformat()
                chat_history.append({
                    "timestamp": timestamp,
                    "user": user_input,
                    "ai": response
                })
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"❌ [red]Error: {e}[/red]")
    
    async def code_generation_mode(self):
        """Code generation mode with live streaming via SSE."""
        self.console.clear()
        self.console.print(Panel(
            "💻 Code Generation (Streaming)",
            style="bold blue",
            border_style="blue"
        ))

        try:
            prompt = Prompt.ask("📝 Describe what to build (single line)")
            if not prompt.strip():
                return

            layout = Layout()
            layout.split_column(
                Layout(name="plan", size=6),
                Layout(name="code", ratio=2),
                Layout(name="review", size=7),
                Layout(name="exec", size=5),
                Layout(name="status", size=3),
            )

            plan_text = Text("", style="bold cyan")
            code_text = Text("", style="white")
            review_text = Text("", style="yellow")
            exec_text = Text("", style="magenta")
            status_text = Text("Press Ctrl+C to cancel; Enter to return", style="dim")

            layout["plan"].update(Panel(plan_text, title="Plan"))
            layout["code"].update(Panel(code_text, title="Code (streaming)"))
            layout["review"].update(Panel(review_text, title="Review"))
            layout["exec"].update(Panel(exec_text, title="Execute"))
            layout["status"].update(Panel(status_text, title="Status"))

            async def run_stream():
                try:
                    async for evt in self.backend_comm.stream_generate(prompt):
                        typ = evt.get("event"); data = evt.get("data", "")
                        if typ == "plan":
                            plan_text.plain = data.replace("\\n", "\n")
                            layout["plan"].update(Panel(plan_text, title="Plan"))
                        elif typ == "code":
                            code_text.append(data.replace("\\n", "\n"))
                            layout["code"].update(Panel(code_text, title="Code (streaming)"))
                        elif typ == "review":
                            review_text.plain = data.replace("\\n", "\n")
                            layout["review"].update(Panel(review_text, title="Review"))
                        elif typ == "execute":
                            exec_text.plain = data.replace("\\n", "\n")
                            layout["exec"].update(Panel(exec_text, title="Execute"))
                        elif typ == "done":
                            status_text.plain = "Done. Press Enter to return."
                            layout["status"].update(Panel(status_text, title="Status"))
                            break
                        elif typ == "error":
                            status_text.plain = f"Error: {data}"
                            layout["status"].update(Panel(status_text, title="Status"))
                            break
                        await asyncio.sleep(0)
                except KeyboardInterrupt:
                    status_text.plain = "Cancelled by user"
                    layout["status"].update(Panel(status_text, title="Status"))

            with Live(layout, refresh_per_second=20, screen=True):
                await run_stream()

        except Exception as e:
            self.console.print(f"❌ [red]Error: {e}[/red]")

        Prompt.ask("\nPress Enter to continue")
    
    async def _display_code_generation_result(self, response: Dict[str, Any], language: str):
        """Display code generation results."""
        # Display generated code
        self.console.print()
        self.console.print(Panel(
            "✅ Code Generated Successfully!",
            style="bold green",
            border_style="green"
        ))
        self.console.print()
        
        # Show code with syntax highlighting
        if response.get("code"):
            syntax = Syntax(response["code"], language, theme="monokai")
            self.console.print(Panel(
                syntax,
                title=f"Generated {language.title()} Code",
                border_style="blue"
            ))
            
            # Show explanation
            if response.get("explanation"):
                self.console.print()
                self.console.print(Panel(
                    response["explanation"],
                    title="💡 Code Explanation",
                    border_style="cyan"
                ))
        
        # Ask if user wants to save
        if Confirm.ask("💾 Save this code to a file?"):
            filename = Prompt.ask(
                "📁 Filename", 
                default=f"generated_code.{self._get_file_extension(language)}"
            )
            await self._save_code_to_file(filename, response.get("code", ""))
    
    async def code_analysis_mode(self):
        """Code analysis mode."""
        self.console.clear()
        self.console.print(Panel(
            "🔍 Code Analysis Mode",
            style="bold yellow",
            border_style="yellow"
        ))
        
        try:
            # Get file path
            file_path = Prompt.ask("📁 Path to file to analyze")
            if not file_path or not Path(file_path).exists():
                self.console.print("❌ [red]File not found![/red]")
                Prompt.ask("\nPress Enter to continue")
                return
            
            # Get language (optional)
            language = Prompt.ask("🔤 Programming language (optional, will auto-detect)")
            if not language.strip():
                language = None
            
            # Show analysis progress
            with self.console.status("[bold yellow]Analyzing code...", spinner="dots"):
                response = await self.backend_comm.send_code_analysis_request(file_path, language)
            
            await self._display_code_analysis_result(response)
        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.console.print(f"❌ [red]Error: {e}[/red]")
        
        Prompt.ask("\nPress Enter to continue")
    
    async def _display_code_analysis_result(self, response: Dict[str, Any]):
        """Display code analysis results."""
        # Display analysis results
        self.console.print()
        self.console.print(Panel(
            "✅ Code Analysis Complete!",
            style="bold green",
            border_style="green"
        ))
        self.console.print()
        
        # Create analysis table
        analysis_table = Table(title="Code Analysis Results")
        analysis_table.add_column("Metric", style="cyan")
        analysis_table.add_column("Value", style="white")
        
        analysis_table.add_row("File Path", response.get("file_path", "N/A"))
        analysis_table.add_row("Language", response.get("language", "N/A"))
        analysis_table.add_row("Complexity Score", f"{response.get('complexity_score', 0.0):.2f}/10")
        analysis_table.add_row("Lines of Code", str(response.get("lines_of_code", 0)))
        
        self.console.print(analysis_table)
        self.console.print()
        
        # Show issues
        issues = response.get("issues", [])
        if issues:
            self.console.print(Panel(
                f"⚠️  Found {len(issues)} potential issues",
                style="bold yellow",
                border_style="yellow"
            ))
            
            for issue in issues:
                self.console.print(f"• [yellow]{issue.get('message', 'Unknown issue')}[/yellow]")
                if issue.get('suggestion'):
                    self.console.print(f"  💡 {issue.get('suggestion')}")
                self.console.print()
        
        # Show suggestions
        suggestions = response.get("suggestions", [])
        if suggestions:
            self.console.print(Panel(
                "💡 Improvement Suggestions",
                style="bold cyan",
                border_style="cyan"
            ))
            
            for suggestion in suggestions:
                self.console.print(f"• {suggestion}")
            self.console.print()
        
        # Show patterns
        patterns = response.get("patterns", [])
        if patterns:
            self.console.print(Panel(
                "🔍 Code Patterns Detected",
                style="bold blue",
                border_style="blue"
            ))
            
            for pattern in patterns:
                pattern_desc = pattern.get('description', 'Unknown pattern')
                pattern_count = pattern.get('count', 0)
                self.console.print(f"• {pattern_desc}: {pattern_count} occurrences")
            self.console.print()
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for a programming language."""
        extensions = {
            "python": "py",
            "rust": "rs",
            "javascript": "js",
            "typescript": "ts",
            "go": "go",
            "java": "java",
            "cpp": "cpp",
            "c": "c"
        }
        return extensions.get(language.lower(), "txt")
    
    async def _save_code_to_file(self, filename: str, code: str):
        """Save generated code to a file."""
        try:
            with open(filename, 'w') as f:
                f.write(code)
            self.console.print(f"✅ [green]Code saved to {filename}[/green]")
        except Exception as e:
            self.console.print(f"❌ [red]Error saving file: {e}[/red]")

    async def dev_actions_mode(self):
        """Run development actions: format/lint/type-check/tests."""
        self.console.clear()
        self.console.print(Panel(
            "🛠️  Development Actions",
            style="bold cyan",
            border_style="cyan"
        ))

        action = Prompt.ask(
            "Select action",
            choices=["quality-check", "tests", "back"],
            default="quality-check"
        )
        if action == "back":
            return
        if action == "quality-check":
            paths = Prompt.ask("Paths (comma-separated)", default=".")
            with self.console.status("[bold cyan]Running code quality...", spinner="dots"):
                result = await self.backend_comm.run_code_quality(
                    paths=[p.strip() for p in paths.split(",") if p.strip()], check_only=True
                )
            self._print_tool_result("Code Quality", result)
        elif action == "tests":
            marker = Prompt.ask("pytest marker (optional)", default="")
            path = Prompt.ask("path (default: tests)", default="tests")
            with self.console.status("[bold cyan]Running tests...", spinner="dots"):
                result = await self.backend_comm.run_tests(marker=marker or None, path=path)
            self._print_tool_result("Tests", result)
        Prompt.ask("\nPress Enter to continue")

    def _print_tool_result(self, title: str, result: Dict[str, Any]):
        ok = result.get("success", True)
        color = "green" if ok else "red"
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", result.get("error", ""))
        self.console.print(Panel(f"Status: [{'OK' if ok else 'FAIL'}]\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}", title=title, border_style=color))

    async def diff_apply_mode(self):
        """Interactive diff and patch apply."""
        self.console.clear()
        self.console.print(Panel("🧩 Diff & Apply Patch", style="bold magenta", border_style="magenta"))
        file_path = Prompt.ask("File to patch (absolute or relative)")
        new_content_path = Prompt.ask("Path to file containing NEW content")
        try:
            with open(new_content_path, 'r') as f:
                new_content = f.read()
        except Exception as e:
            self.console.print(f"❌ [red]Error reading new content: {e}[/red]")
            Prompt.ask("\nPress Enter to continue")
            return

        with self.console.status("[bold magenta]Computing diff...", spinner="dots"):
            diff = await self.backend_comm.diff_patch(file_path, new_content)
        if not diff.get("success"):
            self.console.print(f"❌ [red]Diff failed: {diff.get('error','unknown')}[/red]")
            Prompt.ask("\nPress Enter to continue")
            return

        diff_text = diff.get("diff", "(no changes)")
        if not diff_text.strip():
            diff_text = "(no changes)"
        self.console.print(Panel(Syntax(diff_text, "diff"), title="Proposed Diff", border_style="magenta"))
        if Confirm.ask("Apply this patch?", default=False):
            with self.console.status("[bold magenta]Applying patch...", spinner="dots"):
                res = await self.backend_comm.apply_patch(file_path, new_content, create_dirs=True)
            ok = res.get("success", False)
            if ok:
                self.console.print("✅ [green]Patch applied successfully[/green]")
            else:
                self.console.print(f"❌ [red]Apply failed: {res.get('error','unknown')}[/red]")
        Prompt.ask("\nPress Enter to continue")

    async def planning_mode(self):
        """Interactive planning flow: generate plan/TODO/Tech Stack, save docs, inject into session."""
        self.console.clear()
        self.console.print(Panel("🧭 Planning Mode", style="bold green", border_style="green"))
        goal = Prompt.ask("Project goal (one-liner)")
        constraints = Prompt.ask("Constraints (optional)", default="")
        with self.console.status("[bold green]Generating plan...", spinner="dots"):
            res = await self.backend_comm.plan_generate(goal, constraints or None)
        if not res.get("success"):
            self.console.print(f"❌ [red]Plan generation failed: {res.get('error','unknown')}[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        md = res.get("markdown", "")
        parsed = res.get("parsed", {})
        self.console.print(Panel(md, title="Proposed Plan", border_style="green"))
        if Confirm.ask("Save documentation (PROJECT_PLAN.md, TODO.md, TECH_STACK.md) and inject into session?", default=True):
            with self.console.status("[bold green]Saving docs...", spinner="dots"):
                save = await self.backend_comm.plan_save(md, parsed)
            if not save.get("success"):
                self.console.print(f"❌ [red]Save failed: {save.get('error','unknown')}[/red]")
            else:
                self.console.print("✅ [green]Documentation saved and session updated[/green]")
        Prompt.ask("\nPress Enter to continue")

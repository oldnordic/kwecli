#!/usr/bin/env python3
"""
KWE CLI Context Management Module

This module handles all context-related operations:
- Adding new context entries
- Searching through stored context
- Listing all context entries
- Deleting context entries (when backend supports it)
"""

from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table


class ContextManagement:
    """Handles all context management operations."""
    
    def __init__(self, backend_comm, console: Console):
        """
        Initialize context management with backend communication.
        
        Args:
            backend_comm: Backend communication module instance
            console: Rich console instance for output
        """
        self.backend_comm = backend_comm
        self.console = console
    
    async def context_management_mode(self):
        """Context management mode main menu."""
        self.console.clear()
        self.console.print(Panel(
            "📝 Context Management Mode",
            style="bold magenta",
            border_style="magenta"
        ))
        
        while True:
            self.console.print()
            context_menu = Table(show_header=False, box=None, padding=(0, 2))
            context_menu.add_column("Option", style="cyan", no_wrap=True)
            context_menu.add_column("Description", style="white")
            
            context_options = [
                ("1", "➕ Add new context"),
                ("2", "🔍 Search context"),
                ("3", "📋 List all context"),
                ("4", "🗑️  Delete context"),
                ("0", "🔙 Back to main menu")
            ]
            
            for option, description in context_options:
                context_menu.add_row(option, description)
            
            self.console.print(context_menu)
            self.console.print()
            
            choice = Prompt.ask(
                "Select an option",
                choices=["0", "1", "2", "3", "4"],
                default="1"
            )
            
            if choice == "0":
                break
            elif choice == "1":
                await self.add_context()
            elif choice == "2":
                await self.search_context()
            elif choice == "3":
                await self.list_context()
            elif choice == "4":
                await self.delete_context()
    
    async def add_context(self):
        """Add new context entry."""
        self.console.print("\n➕ Adding New Context")
        self.console.print("=" * 30)
        
        try:
            content = Prompt.ask("📝 Context content")
            if not content.strip():
                return
            
            tags_input = Prompt.ask("🏷️  Tags (comma-separated)")
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
            
            priority = Prompt.ask(
                "⚡ Priority",
                choices=["low", "medium", "high"],
                default="medium"
            )
            
            # Send to backend
            with self.console.status("[bold magenta]Saving context...", spinner="dots"):
                response = await self.backend_comm.send_context_request(content, tags, priority)
            
            if response:
                self.console.print("✅ [green]Context saved successfully![/green]")
                self.console.print(f"🆔 Context ID: {response.get('id', 'N/A')}")
            
        except Exception as e:
            self.console.print(f"❌ [red]Error: {e}[/red]")
        
        Prompt.ask("\nPress Enter to continue")
    
    async def search_context(self):
        """Search through stored context entries."""
        self.console.print("\n🔍 Search Context")
        self.console.print("=" * 20)
        
        try:
            query = Prompt.ask("🔎 Search query")
            if not query.strip():
                return
            
            limit = Prompt.ask("📊 Maximum results", default="10")
            
            # Send to backend
            with self.console.status("[bold magenta]Searching...", spinner="dots"):
                response = await self.backend_comm.search_context_request(query, int(limit))
            
            if response and response.get("results"):
                results = response["results"]
                self.console.print(f"✅ Found {len(results)} results")
                self.console.print()
                
                await self._display_context_results(results, "Search Results")
            else:
                self.console.print("🔍 No results found")
        
        except Exception as e:
            self.console.print(f"❌ [red]Error: {e}[/red]")
        
        Prompt.ask("\nPress Enter to continue")
    
    async def list_context(self):
        """List all stored context entries."""
        self.console.print("\n📋 All Context Entries")
        self.console.print("=" * 25)
        
        try:
            # Send to backend
            with self.console.status("[bold magenta]Loading context...", spinner="dots"):
                response = await self.backend_comm.list_context_request()
            
            if response and response.get("contexts"):
                contexts = response["contexts"]
                self.console.print(f"📊 Total contexts: {len(contexts)}")
                self.console.print()
                
                await self._display_context_results(contexts, "All Context Entries", "green")
            else:
                self.console.print("📭 No context entries found")
        
        except Exception as e:
            self.console.print(f"❌ [red]Error: {e}[/red]")
        
        Prompt.ask("\nPress Enter to continue")
    
    async def delete_context(self):
        """Delete a context entry."""
        self.console.print("\n🗑️  Delete Context")
        self.console.print("=" * 20)
        
        try:
            # First list all contexts to choose from
            response = await self.backend_comm.list_context_request()
            if not response or not response.get("contexts"):
                self.console.print("📭 No contexts to delete")
                Prompt.ask("\nPress Enter to continue")
                return
            
            contexts = response["contexts"]
            self.console.print("📋 Available contexts:")
            
            # Display contexts with numbers for selection
            for i, context in enumerate(contexts):
                content_preview = context.get('content', 'No content')
                if len(content_preview) > 50:
                    content_preview = content_preview[:50] + "..."
                self.console.print(f"{i+1}. {content_preview}")
            
            self.console.print()
            choice = Prompt.ask(
                "Select context to delete",
                choices=[str(i+1) for i in range(len(contexts))],
                default="1"
            )
            
            if Confirm.ask(f"🗑️  Delete context {choice}?"):
                selected_context = contexts[int(choice) - 1]
                context_id = selected_context.get('id')
                
                if not context_id:
                    self.console.print("❌ [red]Error: Context ID not found[/red]")
                    Prompt.ask("\nPress Enter to continue")
                    return
                
                # Delete the context
                with self.console.status("[bold magenta]Deleting context...", spinner="dots"):
                    response = await self.backend_comm.delete_context_request(context_id)
                
                if response and response.get("success"):
                    self.console.print("✅ [green]Context deleted successfully![/green]")
                    self.console.print(f"🗑️  Deleted: {selected_context.get('content', 'Unknown')[:50]}...")
                    self.console.print(f"📊 Remaining contexts: {response.get('remaining_contexts', 'Unknown')}")
                elif response and response.get("error"):
                    self.console.print(f"❌ [red]Error: {response['error']}[/red]")
                else:
                    self.console.print("❌ [red]Unknown error occurred during deletion[/red]")
        
        except Exception as e:
            self.console.print(f"❌ [red]Error: {e}[/red]")
        
        Prompt.ask("\nPress Enter to continue")
    
    async def _display_context_results(self, results: List, title: str, border_style: str = "blue"):
        """Display context results in a formatted way."""
        for i, result in enumerate(results, 1):
            content = result.get('content', 'No content')
            tags = result.get('tags', [])
            priority = result.get('priority', 'N/A')
            timestamp = result.get('timestamp', 'N/A')
            
            # Format tags display
            tags_display = ', '.join(tags) if tags else 'No tags'
            
            self.console.print(Panel(
                f"📝 {content}\n"
                f"🏷️  Tags: {tags_display}\n"
                f"⚡ Priority: {priority}\n"
                f"🕒 {timestamp}",
                title=f"{title} - Entry {i}",
                border_style=border_style
            ))
            self.console.print()
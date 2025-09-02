#!/usr/bin/env python3
"""
KWE CLI Coordination Interface Module

This module handles coordination functionality for the Python frontend,
providing a complete interface for task delegation, monitoring, and management.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax

# Initialize Rich console
console = Console()
from agents.agent_registry import AgentRegistry
registry = AgentRegistry()


class CoordinationInterface:
    """Handles coordination and task delegation interface for the Python frontend."""
    
    def __init__(self, backend_comm):
        """
        Initialize coordination interface.
        
        Args:
            backend_comm: Backend communication module instance
        """
        self.backend_comm = backend_comm
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
    
    async def coordination_mode(self):
        """Main coordination mode interface."""
        while True:
            console.clear()
            console.print(Panel(
                "🎯 KWE CLI - Coordination & Task Delegation",
                style="bold blue",
                border_style="blue"
            ))
            
            # Display coordination menu
            coord_menu = Table(show_header=False, box=None, padding=(0, 2))
            coord_menu.add_column("Option", style="cyan", no_wrap=True)
            coord_menu.add_column("Description", style="white")
            
            menu_options = [
                ("1", "📋 View Coordination Capabilities"),
                ("2", "🎯 Delegate Single Task"),
                ("3", "🔄 Parallel Task Execution"),
                ("4", "📊 Monitor Active Tasks"),
                ("5", "📈 View Coordination Metrics"),
                ("6", "🔍 Task Status & Results"),
                ("7", "🚫 Cancel Tasks"),
                ("0", "🔙 Back to Main Menu")
            ]
            
            for option, description in menu_options:
                coord_menu.add_row(option, description)
            
            console.print(coord_menu)
            console.print()
            
            choice = Prompt.ask(
                "Select coordination option",
                choices=["0", "1", "2", "3", "4", "5", "6", "7"],
                default="1"
            )
            
            if choice == "0":
                break
            elif choice == "1":
                await self.view_capabilities()
            elif choice == "2":
                await self.delegate_single_task()
            elif choice == "3":
                await self.parallel_task_execution()
            elif choice == "4":
                await self.monitor_active_tasks()
            elif choice == "5":
                await self.view_metrics()
            elif choice == "6":
                await self.task_status_and_results()
            elif choice == "7":
                await self.cancel_tasks()
    
    async def view_capabilities(self):
        """Display coordination capabilities and available agents."""
        console.clear()
        console.print(Panel(
            "📋 Coordination Capabilities",
            style="bold cyan",
            border_style="cyan"
        ))
        
        console.clear()
        console.print(Panel(
            "📋 Coordination Capabilities",
            style="bold cyan",
            border_style="cyan"
        ))
        # Fetch registered agents from backend
        try:
            response = await self.backend_comm.send_request("/api/agents/list", method="GET")
            agents = response.get("agents", [])
        except Exception as e:
            console.print(f"❌ [red]Error fetching agents list: {e}[/red]")
            Prompt.ask("Press Enter to continue")
            return
        if not agents:
            console.print("📭 No agents registered")
        else:
            console.print("🤖 [bold]Registered Agents:[/bold]")
            table = Table(show_header=True)
            table.add_column("Agent Name", style="cyan")
            table.add_column("Expertise", style="green")
            table.add_column("Description", style="white")
            for a in agents:
                table.add_row(a.get("name", ""), ", ".join(a.get("expertise", [])), a.get("description", ""))
            console.print(table)
        console.print()
        Prompt.ask("Press Enter to continue")
        
        console.print()
        Prompt.ask("Press Enter to continue")
    
    async def delegate_single_task(self):
        """Interface for delegating a single task."""
        console.clear()
        console.print(Panel(
            "🎯 Delegate Single Task",
            style="bold cyan",
            border_style="cyan"
        ))
        
        # Get task type
        task_type = Prompt.ask("Enter task type", default="bash")
        
        # Get command for bash tasks
        if task_type.lower() == "bash":
            command = Prompt.ask("Enter bash command")
            parameters = {"command": command}
        else:
            # For other task types, get generic parameters
            param_input = Prompt.ask("Enter parameters (JSON format)", default="{}")
            try:
                parameters = json.loads(param_input)
            except json.JSONDecodeError:
                console.print("❌ [red]Invalid JSON format[/red]")
                Prompt.ask("Press Enter to continue")
                return
        
        # Get additional options
        timeout = Prompt.ask("Enter timeout (seconds)", default="60")
        requester_id = Prompt.ask("Enter requester ID", default="python-frontend")
        
        try:
            timeout_int = int(timeout)
        except ValueError:
            timeout_int = 60
        
        # Prepare task request
        task_request = {
            "task_type": task_type,
            "parameters": parameters,
            "requester_id": requester_id,
            "timeout": timeout_int
        }
        
        console.print("\n🚀 [bold]Delegating task...[/bold]")
        
        try:
            # Send delegation request
            response = await self.backend_comm.send_request(
                "/api/agents/delegate",
                data=task_request,
                method="POST"
            )
            task_id = response.get("task_id")
            
            if task_id:
                console.print(f"✅ [green]Task delegated successfully! Task ID:[/green] [yellow]{task_id}[/yellow]")
                
                # Store task info for monitoring
                self.active_tasks[task_id] = {
                    "task_type": task_type,
                    "parameters": parameters,
                    "delegated_at": time.time()
                }
                
                # Ask if user wants to monitor
                if Confirm.ask("\nMonitor this task now?"):
                    await self.monitor_specific_task(task_id)
            else:
                console.print("❌ [red]Task delegation failed[/red]")
                if response and response.get("error"):
                    console.print(f"Error: {response['error']}")
        
        except Exception as e:
            console.print(f"❌ [red]Error delegating task: {e}[/red]")
        
        console.print()
        Prompt.ask("Press Enter to continue")
    
    async def parallel_task_execution(self):
        """Interface for parallel task execution."""
        console.clear()
        console.print(Panel(
            "🔄 Parallel Task Execution",
            style="bold cyan",
            border_style="cyan"
        ))
        
        tasks = []
        
        while True:
            console.print(f"\n📝 [bold]Task {len(tasks) + 1}:[/bold]")
            
            task_type = Prompt.ask("Enter task type", default="bash")
            
            if task_type.lower() == "bash":
                command = Prompt.ask("Enter bash command")
                parameters = {"command": command}
            else:
                param_input = Prompt.ask("Enter parameters (JSON format)", default="{}")
                try:
                    parameters = json.loads(param_input)
                except json.JSONDecodeError:
                    console.print("❌ [red]Invalid JSON format, skipping task[/red]")
                    continue
            
            tasks.append({
                "task_type": task_type,
                "parameters": parameters
            })
            
            if not Confirm.ask("Add another task?"):
                break
        
        if not tasks:
            console.print("❌ [red]No tasks to execute[/red]")
            Prompt.ask("Press Enter to continue")
            return
        
        # Get coordination options
        timeout = Prompt.ask("Enter coordination timeout (seconds)", default="120")
        requester_id = Prompt.ask("Enter requester ID", default="python-frontend")
        
        try:
            timeout_int = int(timeout)
        except ValueError:
            timeout_int = 120
        
        # Prepare parallel request
        parallel_request = {
            "tasks": tasks,
            "requester_id": requester_id,
            "coordination_timeout": timeout_int
        }
        
        console.print(f"\n🚀 [bold]Executing {len(tasks)} tasks in parallel...[/bold]")
        
        # Delegate each task individually and poll for results
        task_ids = []
        for idx, t in enumerate(tasks, start=1):
            console.print(f"Delegating task {idx}/{len(tasks)}: [cyan]{t['task_type']}[/cyan]")
            resp = await self.backend_comm.send_request(
                "/api/agents/delegate",
                data={"task_type": t["task_type"], "context": t.get("parameters", {})},
                method="POST"
            )
            tid = resp.get("task_id")
            task_ids.append(tid)
            console.print(f"→ Task ID: [yellow]{tid}[/yellow]")
        
        console.print("\n⏳ All tasks delegated, polling for completion...")
        results = {}
        # Poll until all tasks complete
        pending = set(task_ids)
        while pending:
            await asyncio.sleep(2)
            for tid in list(pending):
                resp = await self.backend_comm.send_request(f"/api/agents/tasks/{tid}", method="GET")
                status = resp.get("status")
                if status in ("completed", "failed", "cancelled"):
                    results[tid] = resp
                    pending.remove(tid)
                    console.print(f"Task [yellow]{tid}[/yellow] status: [cyan]{status}[/cyan]")
        
        # Display summary
        console.print("\n📊 [bold]Parallel Execution Summary:[/bold]")
        table = Table()
        table.add_column("Task ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Result", style="white")
        for tid, data in results.items():
            res = data.get("result")
            table.add_row(tid, data.get("status"), str(res))
        console.print(table)
        
        Prompt.ask("\nPress Enter to continue")
    
    async def monitor_active_tasks(self):
        """Monitor all active tasks."""
        console.clear()
        console.print(Panel(
            "📊 Monitor Active Tasks",
            style="bold cyan",
            border_style="cyan"
        ))
        
        if not self.active_tasks:
            console.print("📭 [yellow]No active tasks to monitor[/yellow]")
            Prompt.ask("Press Enter to continue")
            return
        
        # Show active tasks
        for task_id, task_info in self.active_tasks.items():
            await self.show_task_status(task_id, task_info)
            console.print("─" * 60)
        
        console.print()
        Prompt.ask("Press Enter to continue")
    
    async def monitor_specific_task(self, task_id: str):
        """Monitor a specific task until completion."""
        console.print(f"\n👀 [bold]Monitoring task: {task_id}[/bold]")
        console.print("Press Ctrl+C to stop monitoring...")
        
        try:
            while True:
                # Get task status
                response = await self.backend_comm.send_request(
                    f"/api/agents/tasks/{task_id}",
                    method="GET"
                )
                
                if response and response.get("success"):
                    status = response.get("status", "unknown")
                    progress = response.get("progress_percent", 0)
                    elapsed = response.get("elapsed_seconds", 0)
                    
                    console.print(f"\r[{time.strftime('%H:%M:%S')}] Status: {status} ({progress:.1f}%) - Elapsed: {elapsed:.1f}s", end="")
                    
                    if response.get("completed"):
                        console.print(f"\n✅ [green]Task completed with status: {status}[/green]")
                        
                        # Get final result
                        result_response = await self.backend_comm.send_request(
                            f"/api/agents/tasks/{task_id}",
                            method="GET"
                        )
                        
                        if result_response and result_response.get("success"):
                            console.print("\n📄 [bold]Final Result:[/bold]")
                            result_data = result_response.get("result")
                            if result_data:
                                console.print(Syntax(
                                    json.dumps(result_data, indent=2),
                                    "json",
                                    theme="monokai"
                                ))
                        break
                else:
                    console.print(f"\n❌ [red]Failed to get task status[/red]")
                    break
                
                await asyncio.sleep(2)  # Check every 2 seconds
        
        except KeyboardInterrupt:
            console.print(f"\n⚠️ [yellow]Monitoring stopped by user[/yellow]")
        except Exception as e:
            console.print(f"\n❌ [red]Monitoring error: {e}[/red]")
    
    async def show_task_status(self, task_id: str, task_info: Dict[str, Any]):
        """Display status for a single task."""
        try:
            response = await self.backend_comm.send_request(
                f"/api/agents/tasks/{task_id}",
                method="GET"
            )
            
            console.print(f"🎯 [bold]Task: {task_id[:8]}...[/bold]")
            console.print(f"   Type: {task_info.get('task_type', 'unknown')}")
            
            if response and response.get("success"):
                status = response.get("status", "unknown")
                progress = response.get("progress_percent", 0)
                elapsed = response.get("elapsed_seconds", 0)
                
                status_color = "green" if status == "completed" else "yellow" if status == "in_progress" else "red"
                console.print(f"   Status: [{status_color}]{status}[/{status_color}]")
                console.print(f"   Progress: {progress:.1f}%")
                console.print(f"   Elapsed: {elapsed:.1f}s")
                
                if response.get("completed"):
                    # Remove from active tasks
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
            else:
                console.print("   Status: [red]Error getting status[/red]")
        
        except Exception as e:
            console.print(f"   Status: [red]Error: {e}[/red]")
    
    async def view_metrics(self):
        """Display coordination metrics."""
        console.clear()
        console.print(Panel(
            "📈 Coordination Metrics",
            style="bold cyan",
            border_style="cyan"
        ))
        
        try:
            response = await self.backend_comm.send_request(
                "/api/agents/metrics",
                method="GET"
            )
            
            if response and response.get("success"):
                console.print("📊 [bold]Performance Metrics:[/bold]")
                console.print(Syntax(
                    json.dumps(response, indent=2),
                    "json",
                    theme="monokai"
                ))
            else:
                console.print("❌ [red]Failed to retrieve metrics[/red]")
        
        except Exception as e:
            console.print(f"❌ [red]Error retrieving metrics: {e}[/red]")
        
        console.print()
        Prompt.ask("Press Enter to continue")
    
    async def task_status_and_results(self):
        """Interface for checking task status and results."""
        console.clear()
        console.print(Panel(
            "🔍 Task Status & Results",
            style="bold cyan",
            border_style="cyan"
        ))
        
        task_id = Prompt.ask("Enter task ID")
        
        try:
            # Get status
            status_response = await self.backend_comm.send_request(
                f"/api/acp/task/{task_id}/status",
                method="GET"
            )
            
            if status_response and status_response.get("success"):
                console.print("📊 [bold]Task Status:[/bold]")
                console.print(Syntax(
                    json.dumps(status_response, indent=2),
                    "json",
                    theme="monokai"
                ))
                
                # If completed, offer to show result
                if status_response.get("completed"):
                    if Confirm.ask("\nTask is completed. View result?"):
                        result_response = await self.backend_comm.send_request(
                            f"/api/acp/task/{task_id}/result",
                            method="GET"
                        )
                        
                        if result_response and result_response.get("success"):
                            console.print("\n📄 [bold]Task Result:[/bold]")
                            console.print(Syntax(
                                json.dumps(result_response, indent=2),
                                "json",
                                theme="monokai"
                            ))
                        else:
                            console.print("❌ [red]Failed to retrieve result[/red]")
            else:
                console.print("❌ [red]Failed to retrieve task status[/red]")
                if status_response and status_response.get("error"):
                    console.print(f"Error: {status_response['error']}")
        
        except Exception as e:
            console.print(f"❌ [red]Error: {e}[/red]")
        
        console.print()
        Prompt.ask("Press Enter to continue")
    
    async def cancel_tasks(self):
        """Interface for canceling tasks."""
        console.clear()
        console.print(Panel(
            "🚫 Cancel Tasks",
            style="bold cyan",
            border_style="cyan"
        ))
        
        task_id = Prompt.ask("Enter task ID to cancel")
        
        if Confirm.ask(f"Are you sure you want to cancel task {task_id}?"):
            try:
                response = await self.backend_comm.send_request(
                    f"/api/agents/tasks/{task_id}/cancel",
                    method="POST"
                )
                
                if response and response.get("success"):
                    console.print("✅ [green]Task cancelled successfully[/green]")
                    if response.get("message"):
                        console.print(f"Message: {response['message']}")
                    
                    # Remove from active tasks
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                else:
                    console.print("❌ [red]Failed to cancel task[/red]")
                    if response and response.get("error"):
                        console.print(f"Error: {response['error']}")
            
            except Exception as e:
                console.print(f"❌ [red]Error canceling task: {e}[/red]")
        else:
            console.print("🔔 [yellow]Cancellation aborted[/yellow]")
        
        console.print()
        Prompt.ask("Press Enter to continue")

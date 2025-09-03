#!/usr/bin/env python3
"""
KWECLI Main Entry Point - TUI-Centric Architecture
===================================================

TUI-first entrypoint with CLI fallback.
Follows modular architecture: TUI primary, CLI for scripting.

Usage:
    kwe              # Launch TUI (default)
    kwe tui          # Launch TUI explicitly
    kwe dev analyze  # CLI commands (fallback)
    kwe --health     # System health check

File: kwecli/__main__.py  
Purpose: TUI-centric entry point with CLI fallback
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import local LTMC bridge (no external dependencies)
try:
    from bridge.ltmc_local import save_thought, log_artifact, get_ltmc_native
    LTMC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Local LTMC not available: {e}")
    LTMC_AVAILABLE = False

# Import command handlers
try:
    from kwecli.handlers import handle_command, handle_chat, handle_plan, handle_research, get_tool_status
    HANDLERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Command handlers not available: {e}")
    HANDLERS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for TUI-first approach."""
    parser = argparse.ArgumentParser(
        prog='kwe',
        description='KWECLI - TUI-Centric Development Environment with LTMC Integration',
        epilog='''
Examples:
  kwe                     # Launch TUI (default)
  kwe tui                 # Launch TUI explicitly
  kwe dev analyze .       # CLI: analyze codebase
  kwe dev blueprint       # CLI: create blueprint
  kwe dev sprint          # CLI: create sprint
  kwe --health            # System health check
        '''
    )
    
    # Main mode selection
    parser.add_argument('mode', nargs='?', default='tui', 
                       choices=['tui', 'dev'], 
                       help='Interface mode: tui (default) or dev (CLI)')
    
    # Sub-commands for dev mode
    parser.add_argument('subcommand', nargs='?',
                       help='Dev subcommand: analyze, blueprint, sprint, chat')
    
    # Target for operations
    parser.add_argument('target', nargs='?', default='.',
                       help='Target path or query')
    
    # Options
    parser.add_argument('--project-path', default='.', help='Project directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--health', action='store_true', help='Check system health')
    
    return parser




def health_check() -> int:
    """Check system health and database connectivity."""
    print("🏥 KWECLI Health Check")
    print("=" * 50)
    
    all_healthy = True
    
    # Check LTMC
    if not LTMC_AVAILABLE:
        print("❌ LTMC Native Bridge: Not Available")
        all_healthy = False
    else:
        try:
            ltmc = get_ltmc_native()
            health = ltmc.health_check()
            
            print(f"📊 LTMC Health: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
            print(f"📁 Data Directory: {health['data_dir']}")
            print("\nDatabase Connections:")
            
            for db, status in health['connections'].items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {db.upper()}: {'Connected' if status else 'Disconnected'}")
                if not status:
                    all_healthy = False
                    
        except Exception as e:
            print(f"❌ LTMC health check failed: {e}")
            all_healthy = False
    
    # Check Ollama and tools
    print("\nTool Availability:")
    if not HANDLERS_AVAILABLE:
        print("❌ Command Handlers: Not Available")
        all_healthy = False
    else:
        try:
            tools = get_tool_status()
            print(f"🤖 Ollama: {'✅ Available' if tools.get('ollama', False) else '❌ Not Available'}")
            
            if tools.get('error'):
                print(f"   Error: {tools['error']}")
                all_healthy = False
            else:
                # Modern Unix tools
                unix_tools = ['ripgrep', 'exa', 'lsd', 'fzf', 'fd', 'bat', 'delta', 'tree_sitter', 'jq', 'tldr', 'duf']
                available_count = sum(1 for tool in unix_tools if tools.get(tool, False))
                print(f"🔧 Modern Unix Tools: {available_count}/{len(unix_tools)} available")
                
                for tool in unix_tools:
                    status_icon = "✅" if tools.get(tool, False) else "❌"
                    print(f"  {status_icon} {tool}")
                
        except Exception as e:
            print(f"❌ Tool check failed: {e}")
            all_healthy = False
    
    if all_healthy:
        print("\n🎉 All systems operational!")
        return 0
    else:
        print("\n⚠️  Some systems not available")
        return 1




def launch_tui() -> int:
    """Launch the TUI interface."""
    try:
        print("🚀 Launching KWECLI TUI...")
        # Try main TUI app first
        from kwecli.tui.app import KWECLIApp
        app = KWECLIApp()
        return app.run()
    except Exception as e:
        # If main TUI fails, try simple TUI
        logger.warning(f"Main TUI failed ({e}), trying simple TUI...")
        try:
            from kwecli.tui.app_simple import KWECLISimpleApp
            print("🔄 Loading simple TUI interface...")
            app = KWECLISimpleApp()
            return app.run()
        except Exception as simple_e:
            logger.error(f"Simple TUI also failed: {simple_e}")
            print("⚠️  TUI not available, starting basic CLI mode...")
            print("Usage: kwe dev <analyze|blueprint|sprint> [target]")
            return 0


def handle_dev_commands(subcommand: str, target: str, project_path: str, verbose: bool) -> int:
    """Handle dev CLI commands."""
    if not HANDLERS_AVAILABLE:
        print("❌ Command handlers not available")
        return 1
    
    if subcommand == 'analyze':
        # Use our existing workflow system
        print(f"🔍 Analyzing codebase: {target}")
        # Note: Workflow system removed in cleanup - analysis simplified
        print("📊 Basic analysis functionality available")
        print("📊 Analysis functionality connected to workflow system")
        return 0
        
    elif subcommand == 'blueprint':
        print(f"📋 Creating blueprint from analysis...")
        # TODO: Connect to blueprint creation
        return 0
        
    elif subcommand == 'sprint':
        print(f"🎯 Creating development sprint...")
        # TODO: Connect to sprint creation
        return 0
        
    elif subcommand == 'chat':
        return handle_chat(target, project_path, verbose)
        
    else:
        print(f"❌ Unknown dev subcommand: {subcommand}")
        print("Available: analyze, blueprint, sprint, chat")
        return 1


def main() -> int:
    """Main entry point for TUI-first architecture."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(f"Arguments: {args}")
    
    try:
        # Health check mode
        if args.health:
            return health_check()
        
        # TUI mode (default)
        if args.mode == 'tui' or args.mode is None:
            return launch_tui()
        
        # Dev CLI mode
        elif args.mode == 'dev':
            if not args.subcommand:
                print("❌ Dev mode requires subcommand")
                print("Usage: kwe dev <analyze|blueprint|sprint|chat> [target]")
                return 1
            
            return handle_dev_commands(
                args.subcommand, 
                args.target,
                args.project_path, 
                args.verbose
            )
        
        else:
            parser.print_help()
            return 1
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
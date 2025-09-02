#!/usr/bin/env python3
"""
Real File Agent Implementation

This agent provides real file operations using actual filesystem operations.
NO mocks, stubs, or placeholders are used - all file operations are real.

Features:
- Real file reading with error handling
- Real directory listing with content analysis
- Real markdown file analysis and summarization
- Real large file handling with size limits
- Async execution support for non-blocking operations
"""

import os
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from agents.base_agent import SubAgent, AgentResult, AgentStatus, AgentExpertise


class RealFileAgent(SubAgent):
    """Real file operations agent using actual filesystem operations."""
    
    def __init__(self):
        super().__init__(
            name="real_file_agent",
            description="Real file operations agent using actual filesystem operations",
            expertise=[AgentExpertise.FILE_OPERATIONS, AgentExpertise.ANALYTICS],
            tools=["filesystem", "file_analysis", "markdown_parsing"]
        )
        self.max_file_size = 10 * 1024 * 1024  # 10MB default limit
    
    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle file operation tasks."""
        task_lower = task.lower()
        file_keywords = [
            "read file", "file", "document", "analyze contents",
            "list files", "directory", "markdown", "text",
            "analyze markdown", "read document"
        ]
        
        # Should handle file-related tasks
        if any(keyword in task_lower for keyword in file_keywords):
            return True
        
        # Should not handle unrelated tasks
        unrelated_keywords = [
            "compile", "build", "email", "network", "database",
            "api", "server", "deploy", "install"
        ]
        
        if any(keyword in task_lower for keyword in unrelated_keywords):
            return False
        
        return False
    
    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute file operation task with real filesystem operations."""
        start_time = time.time()
        
        try:
            self.update_status(AgentStatus.BUSY)
            
            task_lower = task.lower()
            
            # Check most specific patterns first
            if "analyze markdown" in task_lower or ("markdown" in task_lower and "analyze" in task_lower):
                result = await self._analyze_markdown_files_task(context)
            elif "list" in task_lower and "directory" in task_lower:
                result = await self._list_directory_task(context)
            elif "read file" in task_lower or ("file" in task_lower and "read" in task_lower):
                result = await self._read_file_task(context)
            else:
                # Default routing based on context
                if "file_paths" in context:
                    # Multiple files - use markdown analysis
                    result = await self._analyze_markdown_files_task(context)
                elif "file_path" in context:
                    # Single file - use file reading
                    result = await self._read_file_task(context)
                elif "directory_path" in context:
                    # Directory - use directory listing
                    result = await self._list_directory_task(context)
                else:
                    result = AgentResult(
                        success=False,
                        output="",
                        error_message=f"Unsupported task: {task}. Supported: read file, list directory, analyze markdown. Provide file_path, file_paths, or directory_path in context."
                    )
            
            # Add execution time
            execution_time = time.time() - start_time
            if result.metadata is None:
                result.metadata = {}
            result.metadata["execution_time"] = execution_time
            result.execution_time = execution_time
            
            # Add to work history
            self.add_work_history(task, result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = AgentResult(
                success=False,
                output="",
                error_message=f"Task execution failed: {str(e)}",
                metadata={"execution_time": execution_time},
                execution_time=execution_time
            )
            self.add_work_history(task, error_result)
            return error_result
            
        finally:
            self.update_status(AgentStatus.IDLE)
    
    async def execute_task_async(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Async wrapper for execute_task to support async testing."""
        return await self.execute_task(task, context)
    
    async def _read_file_task(self, context: Dict[str, Any]) -> AgentResult:
        """Read a real file from the filesystem."""
        file_path = context.get("file_path")
        if not file_path:
            return AgentResult(
                success=False,
                output="",
                error_message="file_path required in context"
            )
        
        # Check if file exists
        if not os.path.exists(file_path):
            return AgentResult(
                success=False,
                output="",
                error_message=f"File not found: {file_path}"
            )
        
        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            return AgentResult(
                success=False,
                output="",
                error_message=f"Path is not a file: {file_path}"
            )
        
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            max_size = context.get("max_size", self.max_file_size)
            
            # Check size limit
            if file_size > max_size:
                return AgentResult(
                    success=False,
                    output="",
                    error_message=f"File too large: {file_size} bytes (max: {max_size})"
                )
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return AgentResult(
                success=True,
                output=content,
                metadata={
                    "file_path": file_path,
                    "file_size": file_size,
                    "encoding": "utf-8",
                    "lines_count": len(content.splitlines())
                }
            )
            
        except UnicodeDecodeError:
            # Try reading as binary if UTF-8 fails
            try:
                with open(file_path, 'rb') as f:
                    binary_content = f.read()
                
                return AgentResult(
                    success=True,
                    output=f"Binary file content (first 1000 bytes): {binary_content[:1000]}",
                    metadata={
                        "file_path": file_path,
                        "file_size": os.path.getsize(file_path),
                        "encoding": "binary",
                        "is_binary": True
                    }
                )
            except Exception as e:
                return AgentResult(
                    success=False,
                    output="",
                    error_message=f"Failed to read file as binary: {str(e)}"
                )
                
        except PermissionError:
            return AgentResult(
                success=False,
                output="",
                error_message=f"Permission denied: {file_path}"
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                error_message=f"Failed to read file: {str(e)}"
            )
    
    async def _list_directory_task(self, context: Dict[str, Any]) -> AgentResult:
        """List real directory contents."""
        directory_path = context.get("directory_path")
        if not directory_path:
            return AgentResult(
                success=False,
                output="",
                error_message="directory_path required in context"
            )
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            return AgentResult(
                success=False,
                output="",
                error_message=f"Directory not found: {directory_path}"
            )
        
        # Check if it's a directory
        if not os.path.isdir(directory_path):
            return AgentResult(
                success=False,
                output="",
                error_message=f"Path is not a directory: {directory_path}"
            )
        
        try:
            # List directory contents
            entries = []
            file_count = 0
            dir_count = 0
            total_size = 0
            
            for entry in os.listdir(directory_path):
                entry_path = os.path.join(directory_path, entry)
                
                if os.path.isfile(entry_path):
                    size = os.path.getsize(entry_path)
                    entries.append(f"📄 {entry} ({size} bytes)")
                    file_count += 1
                    total_size += size
                elif os.path.isdir(entry_path):
                    entries.append(f"📁 {entry}/")
                    dir_count += 1
                else:
                    entries.append(f"❓ {entry}")
            
            # Sort entries
            entries.sort()
            
            output = f"Directory: {directory_path}\n"
            output += f"Files: {file_count}, Directories: {dir_count}\n"
            output += f"Total size: {total_size} bytes\n\n"
            output += "\n".join(entries)
            
            return AgentResult(
                success=True,
                output=output,
                metadata={
                    "directory_path": directory_path,
                    "file_count": file_count,
                    "directory_count": dir_count,
                    "total_size": total_size,
                    "entries": len(entries)
                }
            )
            
        except PermissionError:
            return AgentResult(
                success=False,
                output="",
                error_message=f"Permission denied: {directory_path}"
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                error_message=f"Failed to list directory: {str(e)}"
            )
    
    async def _analyze_markdown_files_task(self, context: Dict[str, Any]) -> AgentResult:
        """Analyze real markdown files and provide summary."""
        file_paths = context.get("file_paths", [])
        if not file_paths:
            return AgentResult(
                success=False,
                output="",
                error_message="file_paths list required in context"
            )
        
        analysis_results = []
        successful_files = 0
        failed_files = 0
        total_size = 0
        
        for file_path in file_paths:
            try:
                # Check if file exists and is markdown
                if not os.path.exists(file_path):
                    analysis_results.append(f"❌ {file_path}: File not found")
                    failed_files += 1
                    continue
                
                if not file_path.lower().endswith('.md'):
                    analysis_results.append(f"⚠️  {file_path}: Not a markdown file")
                    continue
                
                # Read and analyze the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_size = len(content)
                total_size += file_size
                
                # Extract markdown structure
                lines = content.splitlines()
                headers = []
                sections = 0
                
                for line in lines:
                    if line.startswith('#'):
                        header_level = len(line) - len(line.lstrip('#'))
                        header_text = line.lstrip('#').strip()
                        headers.append(f"{'  ' * (header_level - 1)}• {header_text}")
                        sections += 1
                
                # Create file summary
                filename = os.path.basename(file_path)
                summary = f"📄 {filename}:\n"
                summary += f"   Size: {file_size} bytes, Lines: {len(lines)}\n"
                summary += f"   Sections: {sections}\n"
                
                if headers:
                    summary += f"   Structure:\n"
                    for header in headers[:10]:  # Limit to first 10 headers
                        summary += f"     {header}\n"
                    if len(headers) > 10:
                        summary += f"     ... and {len(headers) - 10} more sections\n"
                
                # Extract first few lines as preview
                preview_lines = [line for line in lines[:5] if line.strip()]
                if preview_lines:
                    summary += f"   Preview: {preview_lines[0][:100]}...\n"
                
                analysis_results.append(summary)
                successful_files += 1
                
            except UnicodeDecodeError:
                analysis_results.append(f"❌ {file_path}: Not a text file (binary content)")
                failed_files += 1
            except PermissionError:
                analysis_results.append(f"❌ {file_path}: Permission denied")
                failed_files += 1
            except Exception as e:
                analysis_results.append(f"❌ {file_path}: Error - {str(e)}")
                failed_files += 1
        
        # Generate final report
        output = f"Markdown Files Analysis Report\n"
        output += f"{'=' * 35}\n\n"
        output += f"Files processed: {len(file_paths)}\n"
        output += f"Successful: {successful_files}\n"
        output += f"Failed: {failed_files}\n"
        output += f"Total content size: {total_size} bytes\n\n"
        output += "File Details:\n"
        output += "\n".join(analysis_results)
        
        return AgentResult(
            success=successful_files > 0,
            output=output,
            metadata={
                "files_analyzed": successful_files,
                "files_failed": failed_files,
                "total_files": len(file_paths),
                "total_content_size": total_size,
                "analysis_type": context.get("analysis_type", "markdown_summary")
            }
        )
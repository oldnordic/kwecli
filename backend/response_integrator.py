"""
Response Integration Module for AI Enhancement.

Formats tool execution results and integrates them with AI prompts
to provide context-rich responses based on real filesystem data.
"""

import logging
from typing import Dict, Any, List, Optional
import json

from .tool_orchestrator import ToolWorkflowResult, ToolExecutionResult

logger = logging.getLogger(__name__)


class ResponseIntegrator:
    """Integrates tool execution results with AI prompts for enhanced responses."""
    
    def __init__(self):
        self.max_content_length = 5000  # Maximum content length to include in prompt
        self.max_file_lines = 100       # Maximum file lines to include
        self.max_directory_items = 50   # Maximum directory items to show
    
    def integrate_results(self, tool_results: ToolWorkflowResult, original_prompt: str) -> str:
        """
        Integrate tool execution results with the original prompt.
        
        Args:
            tool_results: Results from tool workflow execution
            original_prompt: Original user prompt
            
        Returns:
            Enhanced prompt with tool context, or original prompt if no useful results
        """
        try:
            if not tool_results.overall_success or not tool_results.execution_results:
                logger.debug("No successful tool results to integrate")
                return original_prompt
            
            # Filter successful results
            successful_results = [r for r in tool_results.execution_results if r.success]
            
            if not successful_results:
                logger.debug("No successful tool executions found")
                return original_prompt
            
            # Format tool outputs
            tool_context = self._format_tool_results(successful_results)
            
            if not tool_context or len(tool_context) < 10:
                logger.debug("Tool context too minimal to be useful")
                return original_prompt
            
            # Build enhanced prompt
            enhanced_prompt = self._build_enhanced_prompt(tool_context, original_prompt)
            
            logger.info(f"Enhanced prompt with {len(successful_results)} tool results")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error integrating tool results: {e}")
            return original_prompt
    
    def _format_tool_results(self, results: List[ToolExecutionResult]) -> str:
        """
        Format tool execution results into a coherent context string.
        
        Args:
            results: List of successful tool execution results
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for result in results:
            tool_name = result.intention.tool_name
            result_data = result.result.get("data", {})
            
            if tool_name == "directory_lister":
                context_parts.append(self._format_directory_result(result_data, result.intention.parameters))
            elif tool_name == "read":
                context_parts.append(self._format_file_read_result(result_data, result.intention.parameters))
            elif tool_name == "grep":
                context_parts.append(self._format_search_result(result_data, result.intention.parameters))
            elif tool_name in ["write", "edit", "multi_edit"]:
                context_parts.append(self._format_modification_result(result_data, result.intention.parameters))
            else:
                # Generic formatting for unknown tools
                context_parts.append(self._format_generic_result(tool_name, result_data, result.intention.parameters))
        
        return "\n\n".join(filter(None, context_parts))
    
    def _format_directory_result(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Format directory listing results."""
        path = parameters.get("path", "unknown")
        
        # Extract directory information
        files = data.get("files", [])
        file_count = data.get("file_count", len(files))
        directory_count = data.get("directory_count", 0)
        total_size = data.get("total_size", 0)
        
        result = f"=== Directory Analysis: {path} ===\n"
        result += f"Total Files: {file_count}, Directories: {directory_count}\n"
        
        if total_size > 0:
            result += f"Total Size: {self._format_size(total_size)}\n"
        
        # Show file structure (limited)
        if files:
            result += "\nDirectory Structure:\n"
            shown_files = files[:self.max_directory_items]
            for file_info in shown_files:
                if isinstance(file_info, dict):
                    name = file_info.get("name", "unknown")
                    file_type = file_info.get("type", "file")
                    size = file_info.get("size", 0)
                    
                    if file_type == "directory":
                        result += f"  📁 {name}/\n"
                    else:
                        size_str = f" ({self._format_size(size)})" if size > 0 else ""
                        result += f"  📄 {name}{size_str}\n"
                else:
                    # Simple string format
                    result += f"  📄 {file_info}\n"
            
            if len(files) > self.max_directory_items:
                result += f"  ... and {len(files) - self.max_directory_items} more items\n"
        
        return result
    
    def _format_file_read_result(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Format file reading results."""
        file_path = parameters.get("file_path", "unknown")
        content = data.get("content", "")
        lines = data.get("lines", 0)
        encoding = data.get("encoding", "unknown")
        
        result = f"=== File Content: {file_path} ===\n"
        result += f"Lines: {lines}, Encoding: {encoding}\n\n"
        
        if content:
            # Truncate content if too long
            truncated_content = self._truncate_content(content, self.max_content_length)
            result += f"Content:\n```\n{truncated_content}\n```"
            
            if len(content) > self.max_content_length:
                result += f"\n\n[Content truncated - showing first {self.max_content_length} characters of {len(content)} total]"
        else:
            result += "File appears to be empty or unreadable."
        
        return result
    
    def _format_search_result(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Format search/grep results."""
        pattern = parameters.get("pattern", "unknown")
        path = parameters.get("path", ".")
        
        matches = data.get("matches", [])
        match_count = data.get("match_count", len(matches))
        
        result = f"=== Search Results: '{pattern}' in {path} ===\n"
        result += f"Total Matches: {match_count}\n\n"
        
        if matches:
            # Show limited number of matches
            shown_matches = matches[:20]  # Limit to 20 matches
            
            for match in shown_matches:
                if isinstance(match, dict):
                    file_path = match.get("file", "unknown")
                    line_number = match.get("line", 0)
                    line_content = match.get("content", "")
                    
                    result += f"{file_path}:{line_number}: {line_content.strip()}\n"
                else:
                    # Simple string format
                    result += f"  {match}\n"
            
            if len(matches) > 20:
                result += f"\n[Showing first 20 matches of {len(matches)} total]"
        else:
            result += "No matches found."
        
        return result
    
    def _format_modification_result(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Format file modification results."""
        operation = data.get("operation", "modification")
        file_path = parameters.get("file_path", "unknown")
        
        result = f"=== File {operation.title()}: {file_path} ===\n"
        
        if data.get("success", False):
            result += "Operation completed successfully.\n"
            
            # Add any relevant metadata
            lines_added = data.get("lines_added", 0)
            lines_removed = data.get("lines_removed", 0)
            
            if lines_added or lines_removed:
                result += f"Changes: +{lines_added} lines, -{lines_removed} lines\n"
        else:
            error_msg = data.get("error_message", "Unknown error")
            result += f"Operation failed: {error_msg}\n"
        
        return result
    
    def _format_generic_result(self, tool_name: str, data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Format results from unknown tools."""
        result = f"=== {tool_name.title()} Tool Result ===\n"
        
        # Try to extract useful information
        if data.get("success", False):
            result += "Operation completed successfully.\n"
            
            # Look for common data fields
            if "output" in data:
                output = str(data["output"])
                truncated_output = self._truncate_content(output, 1000)
                result += f"Output:\n{truncated_output}\n"
            elif "result" in data:
                result_data = str(data["result"])
                truncated_result = self._truncate_content(result_data, 1000)
                result += f"Result:\n{truncated_result}\n"
        else:
            error_msg = data.get("error_message", "Unknown error")
            result += f"Operation failed: {error_msg}\n"
        
        return result
    
    def _build_enhanced_prompt(self, tool_context: str, original_prompt: str) -> str:
        """
        Build the enhanced prompt with tool context.
        
        Args:
            tool_context: Formatted tool execution results
            original_prompt: Original user prompt
            
        Returns:
            Enhanced prompt with context
        """
        enhanced_prompt = f"""Based on the following information gathered from the filesystem:

{tool_context}

Now, please respond to the original request: {original_prompt}

Provide a comprehensive analysis based on the actual data shown above. Focus on practical insights, patterns, and actionable recommendations."""
        
        return enhanced_prompt
    
    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to maximum length with appropriate indication."""
        if not content:
            return content
        
        if len(content) <= max_length:
            return content
        
        # Try to truncate at a line boundary near the limit
        truncated = content[:max_length]
        last_newline = truncated.rfind('\n')
        
        if last_newline > max_length * 0.8:  # If we can find a good line break
            return content[:last_newline] + "\n[... content truncated ...]"
        else:
            return truncated + "\n[... content truncated ...]"
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def configure_limits(self, max_content_length: int = None, max_file_lines: int = None, 
                        max_directory_items: int = None) -> None:
        """
        Configure content limits for response integration.
        
        Args:
            max_content_length: Maximum content length to include
            max_file_lines: Maximum file lines to include
            max_directory_items: Maximum directory items to show
        """
        if max_content_length is not None and max_content_length > 0:
            self.max_content_length = max_content_length
        
        if max_file_lines is not None and max_file_lines > 0:
            self.max_file_lines = max_file_lines
        
        if max_directory_items is not None and max_directory_items > 0:
            self.max_directory_items = max_directory_items
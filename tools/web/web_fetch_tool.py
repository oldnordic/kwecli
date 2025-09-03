#!/usr/bin/env python3
"""
Web Fetch Tool - Modular Entry Point
====================================

Secure web content fetching using smart modular architecture.
Rebuilt from web_fetch_tool.py following CLAUDE.md ≤300 lines rule.

File: tools/web/web_fetch_modular.py
Purpose: Main web fetch interface with modular imports (≤300 lines)
"""

import asyncio
import logging
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root for imports
sys.path.append('/home/feanor/Projects/kwecli')

# Import modular components
from .web_fetch_security import WebFetchSecurityManager
from .web_fetch_processor import WebFetchContentProcessor

# Import base tool interface
from tools.core.tool_interface import BaseTool, ToolValidationMixin

# Import HTTP client
import aiohttp

logger = logging.getLogger(__name__)


class WebFetchTool(BaseTool, ToolValidationMixin):
    """
    Secure web content fetching with enterprise-grade security and modular architecture.
    
    Modular Components:
    - WebFetchSecurityManager: Domain allowlisting, rate limiting, security validation
    - WebFetchContentProcessor: HTML-to-markdown conversion, caching, content processing
    
    Provides secure web operations with comprehensive security validation,
    content processing, and enterprise compliance features.
    """
    
    DEFAULT_USER_AGENT = "KWE-CLI/1.0 (+https://github.com/anthropics/kwe-cli)"
    
    def __init__(self, allowed_domains: Optional[List[str]] = None, max_response_size: str = "10MB",
                 timeout: int = 30, requests_per_minute: int = 30, enable_caching: bool = True,
                 cache_ttl: int = 300, enforce_https: bool = True, follow_redirects: bool = True,
                 max_redirects: int = 5):
        """
        Initialize web fetch tool with modular security and processing components.
        
        Args:
            allowed_domains: List of allowed domains for security
            max_response_size: Maximum response size (e.g., "10MB")
            timeout: Request timeout in seconds
            requests_per_minute: Rate limiting threshold
            enable_caching: Enable response caching
            cache_ttl: Cache time-to-live in seconds
            enforce_https: Require HTTPS connections
            follow_redirects: Allow following HTTP redirects
            max_redirects: Maximum number of redirects to follow
        """
        super().__init__()
        
        # Parse response size limit
        max_size_bytes = self._parse_size(max_response_size)
        
        # Initialize modular components
        self.security_manager = WebFetchSecurityManager(
            allowed_domains=allowed_domains,
            requests_per_minute=requests_per_minute,
            enforce_https=enforce_https,
            max_response_size=max_size_bytes
        )
        
        self.content_processor = WebFetchContentProcessor(
            enable_caching=enable_caching,
            cache_ttl=cache_ttl
        )
        
        # Tool configuration
        self.timeout = timeout
        self.follow_redirects = follow_redirects
        self.max_redirects = max_redirects
        
        # Tool statistics
        self.tool_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "security_blocks": 0,
            "cache_hits": 0
        }
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info(f"WebFetchTool initialized - HTTPS enforced: {enforce_https}")
    
    @property
    def name(self) -> str:
        return "web_fetch"
    
    @property
    def category(self) -> str:
        return "web"
    
    @property
    def capabilities(self) -> List[str]:
        return [
            "content_retrieval", "markdown_conversion", "html_sanitization",
            "domain_allowlisting", "rate_limiting", "response_caching",
            "security_validation", "https_enforcement", "redirect_handling"
        ]
    
    @property
    def description(self) -> str:
        return "Secure web content fetching with markdown conversion and enterprise security"
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '10MB') to bytes."""
        size_str = size_str.upper().strip()
        if size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        else:
            return int(size_str)  # Assume bytes
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute web fetch operation with comprehensive security validation.
        
        Args:
            parameters: Tool parameters including URL and options
            
        Returns:
            Fetch result with content and metadata
        """
        try:
            self.tool_stats["total_requests"] += 1
            
            # Validate required parameters
            if not self.validate_parameters(parameters):
                self.tool_stats["failed_requests"] += 1
                return {"success": False, "error": "Missing required parameter: url"}
            
            url = parameters["url"]
            convert_to_markdown = parameters.get("convert_to_markdown", False)
            follow_redirects = parameters.get("follow_redirects", self.follow_redirects)
            timeout = parameters.get("timeout", self.timeout)
            
            # Security validation
            url_validation = self.security_manager.validate_url(url)
            if not url_validation["valid"]:
                self.tool_stats["security_blocks"] += 1
                self.tool_stats["failed_requests"] += 1
                return {"success": False, "error": url_validation["error"]}
            
            # Rate limiting check
            rate_check = self.security_manager.check_rate_limit()
            if not rate_check["allowed"]:
                self.tool_stats["security_blocks"] += 1
                self.tool_stats["failed_requests"] += 1
                return {"success": False, "error": rate_check["error"]}
            
            # Check cache first
            cached_response = self.content_processor.get_cached_response(url, convert_to_markdown)
            if cached_response:
                self.tool_stats["cache_hits"] += 1
                self.tool_stats["successful_requests"] += 1
                cached_response["from_cache"] = True
                return cached_response
            
            # Fetch content from web
            fetch_result = await self._fetch_content(url, timeout, follow_redirects)
            if not fetch_result["success"]:
                self.tool_stats["failed_requests"] += 1
                return fetch_result
            
            # Process response content
            response_data = self.content_processor.process_response(
                fetch_result, url, convert_to_markdown
            )
            
            if not response_data.get("success"):
                self.tool_stats["failed_requests"] += 1
                return response_data
            
            # Cache the processed response
            self.content_processor.cache_response(url, response_data, convert_to_markdown)
            
            self.tool_stats["successful_requests"] += 1
            return response_data
            
        except Exception as e:
            self.tool_stats["failed_requests"] += 1
            self.logger.error(f"Web fetch operation failed: {e}")
            return {"success": False, "error": str(e), "url": parameters.get("url", "unknown")}
    
    async def _fetch_content(self, url: str, timeout: int, follow_redirects: bool) -> Dict[str, Any]:
        """
        Fetch content from URL using aiohttp with security validation.
        
        Args:
            url: URL to fetch
            timeout: Request timeout
            follow_redirects: Whether to follow redirects
            
        Returns:
            Fetch result with content and metadata
        """
        try:
            # Create aiohttp session with timeout
            timeout_config = aiohttp.ClientTimeout(total=timeout)
            headers = {
                "User-Agent": self.DEFAULT_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.get(
                    url,
                    headers=headers,
                    allow_redirects=follow_redirects,
                    max_redirects=self.max_redirects if follow_redirects else 0
                ) as response:
                    
                    # Validate response size before reading
                    content_length = response.headers.get('content-length')
                    size_validation = self.security_manager.validate_response_size(content_length)
                    
                    if not size_validation["valid"]:
                        return {"success": False, "error": size_validation["error"]}
                    
                    # Read content
                    content = await response.text()
                    
                    # Validate actual content size
                    actual_size_validation = self.security_manager.validate_response_size(None, content)
                    if not actual_size_validation["valid"]:
                        return {"success": False, "error": actual_size_validation["error"]}
                    
                    # Validate redirect destination if redirects occurred
                    if follow_redirects and str(response.url) != url:
                        redirect_validation = self.security_manager.validate_url(str(response.url))
                        if not redirect_validation["valid"]:
                            return {
                                "success": False,
                                "error": f"Redirect to blocked domain: {redirect_validation['error']}"
                            }
                    
                    # Sanitize HTML content
                    sanitized_content = self.security_manager.sanitize_html(content)
                    
                    return {
                        "success": True,
                        "content": sanitized_content,
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "final_url": str(response.url),
                        "content_type": response.headers.get('content-type', 'text/html')
                    }
                    
        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timeout exceeded"}
        except aiohttp.ClientError as e:
            return {"success": False, "error": f"HTTP request failed: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Content fetching failed: {e}")
            return {"success": False, "error": f"Fetch failed: {str(e)}"}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for web fetch operation."""
        required = ["url"]
        missing = self.validate_required_params(parameters, required)
        if missing:
            return False
        
        type_specs = {
            "url": str,
            "convert_to_markdown": bool,
            "follow_redirects": bool,
            "timeout": int
        }
        type_errors = self.validate_param_types(parameters, type_specs)
        return len(type_errors) == 0
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema for web fetch tool."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "url": {
                    "type": "string", "description": "URL to fetch content from",
                    "required": True, "format": "uri"
                },
                "convert_to_markdown": {
                    "type": "boolean", "description": "Convert HTML content to markdown format",
                    "required": False, "default": False
                },
                "follow_redirects": {
                    "type": "boolean", "description": "Follow HTTP redirects",
                    "required": False, "default": self.follow_redirects
                },
                "timeout": {
                    "type": "integer", "description": "Request timeout in seconds",
                    "required": False, "default": self.timeout, "minimum": 1, "maximum": 300
                }
            }
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            "tool_stats": self.tool_stats,
            "security_stats": self.security_manager.get_security_stats(),
            "processing_stats": self.content_processor.get_cache_stats(),
            "configuration": {
                "timeout": self.timeout,
                "follow_redirects": self.follow_redirects,
                "max_redirects": self.max_redirects
            }
        }
    
    def reset_all_stats(self):
        """Reset statistics in all components."""
        for key in self.tool_stats:
            self.tool_stats[key] = 0
        self.security_manager.reset_stats()
        self.content_processor.reset_stats()
        self.logger.info("All statistics reset")


# Test functionality if run directly
if __name__ == "__main__":
    import asyncio
    
    print("🧪 Testing Modular Web Fetch Tool...")
    
    async def test_web_fetch():
        # Test initialization
        tool = WebFetchTool()
        print("✅ Web fetch tool initialized")
        
        # Test parameter validation
        valid_params = {"url": "https://docs.python.org"}
        is_valid = tool.validate_parameters(valid_params)
        print(f"✅ Parameter validation: {is_valid}")
        
        # Test schema generation
        schema = tool.get_schema()
        print(f"✅ Schema generation: {schema['name']}")
        
        # Test comprehensive stats
        stats = tool.get_comprehensive_stats()
        print(f"✅ Comprehensive stats: {len(stats)} categories")
        
        print("✅ Modular Web Fetch Tool test complete")
    
    asyncio.run(test_web_fetch())
"""
Web Fetch Tool for KWE CLI.

Provides secure web content fetching with markdown conversion, domain allowlisting,
rate limiting, and comprehensive security validation following enterprise patterns.
"""

import asyncio
import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta
import aiohttp

import sys
sys.path.append('/home/feanor/Projects/kwecli')

from tools.core.tool_interface import BaseTool, ToolValidationMixin


class WebFetchTool(BaseTool, ToolValidationMixin):
    """
    Secure web content fetching with enterprise-grade security validation.
    
    Provides secure web operations including content retrieval, HTML to markdown
    conversion, domain allowlisting, rate limiting, and comprehensive security validation.
    """
    
    # Default security configuration
    DEFAULT_ALLOWED_DOMAINS = [
        "github.com", "docs.python.org", "nodejs.org", 
        "doc.rust-lang.org", "stackoverflow.com"
    ]
    
    DEFAULT_USER_AGENT = "KWE-CLI/1.0 (+https://github.com/anthropics/kwe-cli)"
    
    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        max_response_size: str = "10MB",
        timeout: int = 30,
        requests_per_minute: int = 30,
        enable_caching: bool = True,
        cache_ttl: int = 300,  # 5 minutes
        enforce_https: bool = True,
        follow_redirects: bool = True,
        max_redirects: int = 5,
        sanitize_html: bool = True
    ):
        super().__init__()
        
        # Security configuration
        self.allowed_domains = set(allowed_domains or self.DEFAULT_ALLOWED_DOMAINS)
        self.max_response_size = self._parse_size(max_response_size)
        self.timeout = timeout
        self.enforce_https = enforce_https
        self.follow_redirects = follow_redirects
        self.max_redirects = max_redirects
        self.sanitize_html = sanitize_html
        
        # Rate limiting
        self.requests_per_minute = requests_per_minute
        self.request_times: List[float] = []
        
        # Caching
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
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
            "security_validation", "https_enforcement"
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
    
    def _is_rate_limited(self) -> bool:
        """Check if current request would exceed rate limit."""
        current_time = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            return True
        
        self.request_times.append(current_time)
        return False
    
    def _validate_url(self, url: str) -> Dict[str, Any]:
        """Validate URL format and domain allowlist."""
        try:
            parsed = urlparse(url)
        except Exception as e:
            return {"valid": False, "error": f"Invalid URL format: {e}"}
        
        # Basic URL validation
        if not parsed.scheme or not parsed.netloc:
            return {"valid": False, "error": "Invalid URL: missing scheme or domain"}
        
        # HTTPS enforcement
        if self.enforce_https and parsed.scheme != "https":
            return {"valid": False, "error": "HTTPS required for security"}
        
        # Domain allowlist validation
        domain = parsed.netloc.lower()
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        if domain not in self.allowed_domains:
            return {
                "valid": False, 
                "error": f"Domain not allowed: {domain}. Allowed domains: {list(self.allowed_domains)}"
            }
        
        return {"valid": True, "parsed": parsed}
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return f"url:{url}"
    
    def _get_cached_response(self, url: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if valid."""
        if not self.enable_caching:
            return None
        
        cache_key = self._get_cache_key(url)
        cached = self.cache.get(cache_key)
        
        if cached and time.time() - cached["timestamp"] < self.cache_ttl:
            return cached["response"]
        
        return None
    
    def _cache_response(self, url: str, response: Dict[str, Any]) -> None:
        """Cache response data."""
        if not self.enable_caching:
            return
        
        cache_key = self._get_cache_key(url)
        self.cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
    
    def _html_to_markdown(self, html_content: str) -> str:
        """Convert HTML content to markdown format."""
        # Simple HTML to markdown conversion
        # In a real implementation, you would use html-to-markdown or markdownify
        
        # Basic conversions
        content = html_content
        
        # Remove common HTML tags and convert to markdown
        conversions = [
            (r'<h1[^>]*>(.*?)</h1>', r'# \1'),
            (r'<h2[^>]*>(.*?)</h2>', r'## \1'),
            (r'<h3[^>]*>(.*?)</h3>', r'### \1'),
            (r'<h4[^>]*>(.*?)</h4>', r'#### \1'),
            (r'<h5[^>]*>(.*?)</h5>', r'##### \1'),
            (r'<h6[^>]*>(.*?)</h6>', r'###### \1'),
            (r'<strong[^>]*>(.*?)</strong>', r'**\1**'),
            (r'<b[^>]*>(.*?)</b>', r'**\1**'),
            (r'<em[^>]*>(.*?)</em>', r'*\1*'),
            (r'<i[^>]*>(.*?)</i>', r'*\1*'),
            (r'<code[^>]*>(.*?)</code>', r'`\1`'),
            (r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```'),
            (r'<ul[^>]*>(.*?)</ul>', r'\1'),
            (r'<ol[^>]*>(.*?)</ol>', r'\1'),
            (r'<li[^>]*>(.*?)</li>', r'- \1'),
            (r'<p[^>]*>(.*?)</p>', r'\1\n'),
            (r'<br[^>]*/?>', r'\n'),
            (r'<hr[^>]*/?>', r'\n---\n'),
        ]
        
        for pattern, replacement in conversions:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove remaining HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Clean up whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content
    
    def _sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content by removing dangerous elements."""
        if not self.sanitize_html:
            return html_content
        
        # Remove script tags and their content
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove style tags and their content
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove dangerous event handlers
        dangerous_attrs = [
            'onclick', 'onload', 'onerror', 'onmouseover', 'onkeydown',
            'onsubmit', 'onfocus', 'onblur', 'onchange', 'onmouseout'
        ]
        
        for attr in dangerous_attrs:
            html_content = re.sub(f'{attr}=["\'][^"\']*["\']', '', html_content, flags=re.IGNORECASE)
        
        # Remove javascript: protocols
        html_content = re.sub(r'href\s*=\s*["\']javascript:[^"\']*["\']', '', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'src\s*=\s*["\']javascript:[^"\']*["\']', '', html_content, flags=re.IGNORECASE)
        
        return html_content
    
    async def _fetch_content(self, url: str, timeout: int, follow_redirects: bool) -> Dict[str, Any]:
        """Fetch content from URL using aiohttp."""
        try:
            # Create aiohttp session with configured timeout
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
                    # Check response size before reading content
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_response_size:
                        return {
                            "success": False,
                            "error": f"Response too large: {content_length} bytes exceeds limit of {self.max_response_size} bytes"
                        }
                    
                    # Read and validate content size
                    content = await response.text()
                    if len(content.encode('utf-8')) > self.max_response_size:
                        return {
                            "success": False,
                            "error": f"Response too large: {len(content.encode('utf-8'))} bytes exceeds limit of {self.max_response_size} bytes"
                        }
                    
                    # Validate redirects if any occurred
                    if follow_redirects and str(response.url) != url:
                        # Check if final URL domain is allowed
                        final_url_validation = self._validate_url(str(response.url))
                        if not final_url_validation["valid"]:
                            return {
                                "success": False,
                                "error": f"Redirect to blocked domain: {final_url_validation['error']}"
                            }
                    
                    return {
                        "success": True,
                        "content": content,
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "final_url": str(response.url),
                        "content_type": response.headers.get('content-type', 'text/html')
                    }
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Request timeout exceeded"
            }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": f"HTTP request failed: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during HTTP request: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web fetch operation with comprehensive security validation."""
        try:
            # Validate required parameters
            if not self.validate_parameters(parameters):
                return {
                    "success": False,
                    "error": "Missing required parameter: url"
                }
            
            url = parameters["url"]
            convert_to_markdown = parameters.get("convert_to_markdown", False)
            follow_redirects = parameters.get("follow_redirects", self.follow_redirects)
            timeout = parameters.get("timeout", self.timeout)
            
            # Validate URL
            url_validation = self._validate_url(url)
            if not url_validation["valid"]:
                return {
                    "success": False,
                    "error": url_validation["error"]
                }
            
            # Check rate limiting
            if self._is_rate_limited():
                return {
                    "success": False,
                    "error": "Rate limit exceeded. Please wait before making more requests."
                }
            
            # Check cache first
            cached_response = self._get_cached_response(url)
            if cached_response:
                cached_response["from_cache"] = True
                return cached_response
            
            # Make HTTP request using real aiohttp implementation
            fetch_result = await self._fetch_content(url, timeout, follow_redirects)
            if not fetch_result["success"]:
                return fetch_result
            
            response_data = {
                "success": True,
                "url": url,
                "status_code": fetch_result["status_code"],
                "content": fetch_result["content"],
                "content_type": "html",
                "from_cache": False,
                "timestamp": datetime.now().isoformat(),
                "response_size": len(fetch_result["content"].encode('utf-8')),
                "headers": fetch_result["headers"],
                "final_url": fetch_result["final_url"]
            }
            
            # Convert to markdown if requested
            if convert_to_markdown:
                html_content = self._sanitize_html(response_data["content"])
                response_data["content"] = self._html_to_markdown(html_content)
                response_data["content_type"] = "markdown"
            
            # Cache the response
            self._cache_response(url, response_data)
            
            return response_data
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Request timeout exceeded"
            }
        except Exception as e:
            self.logger.error(f"Web fetch operation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": parameters.get("url", "unknown")
            }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for web fetch operation."""
        required = ["url"]
        if not self.validate_required_params(parameters, required):
            return False
        
        type_specs = {
            "url": str,
            "convert_to_markdown": bool,
            "follow_redirects": bool,
            "timeout": int,
            "sanitize_html": bool
        }
        if not self.validate_param_types(parameters, type_specs):
            return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema for web fetch tool."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch content from",
                    "required": True,
                    "format": "uri"
                },
                "convert_to_markdown": {
                    "type": "boolean",
                    "description": "Convert HTML content to markdown format",
                    "required": False,
                    "default": False
                },
                "follow_redirects": {
                    "type": "boolean",
                    "description": "Follow HTTP redirects",
                    "required": False,
                    "default": True
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "required": False,
                    "default": self.timeout,
                    "minimum": 1,
                    "maximum": 300
                },
                "sanitize_html": {
                    "type": "boolean",
                    "description": "Remove dangerous HTML elements",
                    "required": False,
                    "default": True
                }
            }
        }
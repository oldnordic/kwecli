"""
DuckDuckGo Search Result Parser for w3m Output.

This module provides robust parsing of DuckDuckGo Lite search results from w3m 
dump output, including proper URL decoding and multi-line description handling.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, unquote
from dataclasses import dataclass


@dataclass
class ParsedResult:
    """Represents a parsed search result."""
    title: str
    url: str
    snippet: str
    domain: str
    relevance_score: float = 0.8
    result_type: str = "web"
    link_number: Optional[str] = None
    result_number: Optional[str] = None


class DuckDuckGoURLDecoder:
    """Handles DuckDuckGo redirect URL decoding."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def decode_redirect_url(self, redirect_url: str) -> Optional[str]:
        """
        Decode DuckDuckGo redirect URL to extract the real target URL.
        
        DuckDuckGo uses redirect URLs like:
        https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.example.com&rut=...
        
        The real URL is URL-encoded in the 'uddg' parameter.
        """
        try:
            if not redirect_url or "duckduckgo.com/l/?" not in redirect_url:
                return redirect_url  # Not a DuckDuckGo redirect, return as-is
            
            # Parse query parameters from redirect URL
            parsed = urlparse(redirect_url)
            query_params = parse_qs(parsed.query)
            
            # Extract and decode the 'uddg' parameter
            if 'uddg' in query_params and query_params['uddg']:
                encoded_url = query_params['uddg'][0]
                real_url = unquote(encoded_url)
                
                # Validate the decoded URL
                if self._is_valid_url(real_url):
                    return real_url
                else:
                    self.logger.warning(f"Decoded URL appears invalid: {real_url}")
                    return redirect_url
            
            # No 'uddg' parameter found, return original URL
            return redirect_url
            
        except Exception as e:
            self.logger.error(f"Failed to decode DuckDuckGo URL {redirect_url}: {e}")
            return redirect_url  # Fallback to original URL
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate that a URL looks reasonable."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL with fallback handling."""
        try:
            if not url or not url.strip():
                return "unknown"
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower() if parsed.netloc else "unknown"
            
            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]
            
            return domain if domain else "unknown"
            
        except Exception:
            return "unknown"


class DuckDuckGoResultParser:
    """Main parser for DuckDuckGo w3m output."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.url_decoder = DuckDuckGoURLDecoder()
        
        # Compiled regex patterns for efficiency
        self.result_pattern = re.compile(r'^\s*(\d+)\.\s*\[(\d+)\](.+?)$')
        self.reference_pattern = re.compile(r'^\s*\[(\d+)\]\s+(https?://\S+)$')
        self.navigation_pattern = re.compile(r'^\s*\[[^\]]*\].*$')
    
    def parse(self, w3m_output: str) -> List[ParsedResult]:
        """
        Parse w3m DuckDuckGo output into structured results.
        
        Returns a list of ParsedResult objects with properly decoded URLs
        and extracted descriptions.
        """
        try:
            if not w3m_output or not w3m_output.strip():
                return []
            
            lines = w3m_output.split('\n')
            
            # Phase 1: Build link number to URL mapping
            url_map = self._build_url_mapping(lines)
            
            # Phase 2: Parse numbered search results
            results = self._parse_result_entries(lines, url_map)
            
            self.logger.info(f"Successfully parsed {len(results)} DuckDuckGo results")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to parse DuckDuckGo results: {e}")
            return []
    
    def _build_url_mapping(self, lines: List[str]) -> Dict[str, str]:
        """Build mapping from link numbers to decoded URLs."""
        url_map = {}
        
        for line in lines:
            ref_match = self.reference_pattern.match(line)
            if ref_match:
                link_num, redirect_url = ref_match.groups()
                
                # Decode DuckDuckGo redirect URL to get real target URL
                real_url = self.url_decoder.decode_redirect_url(redirect_url)
                
                if real_url:
                    url_map[link_num] = real_url
                    self.logger.debug(f"Mapped link [{link_num}] to {real_url}")
                else:
                    self.logger.warning(f"Failed to decode URL for link [{link_num}]: {redirect_url}")
        
        self.logger.info(f"Built URL mapping for {len(url_map)} links")
        return url_map
    
    def _parse_result_entries(self, lines: List[str], url_map: Dict[str, str]) -> List[ParsedResult]:
        """Parse numbered search result entries."""
        results = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            result_match = self.result_pattern.match(line)
            
            if result_match:
                result_num, link_num, title_text = result_match.groups()
                title = title_text.strip()
                
                # Collect description from following lines
                description_lines, next_i = self._collect_description(lines, i + 1)
                
                # Build result if we have a URL for this link number
                if link_num in url_map:
                    url = url_map[link_num]
                    domain = self.url_decoder.extract_domain(url)
                    snippet = " ".join(description_lines).strip()
                    
                    # Calculate relevance score (higher for top results)
                    relevance_score = max(0.1, 1.0 - (int(result_num) - 1) * 0.05)
                    
                    result = ParsedResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        domain=domain,
                        relevance_score=relevance_score,
                        result_type="web",
                        link_number=link_num,
                        result_number=result_num
                    )
                    
                    results.append(result)
                    self.logger.debug(f"Parsed result {result_num}: {title}")
                else:
                    self.logger.warning(f"No URL found for link number [{link_num}] in result {result_num}")
                
                i = next_i  # Skip to next unprocessed line
            else:
                i += 1
        
        return results
    
    def _collect_description(self, lines: List[str], start_index: int) -> Tuple[List[str], int]:
        """
        Collect description lines following a search result.
        
        Returns tuple of (description_lines, next_index_to_process)
        """
        description_lines = []
        i = start_index
        max_description_lines = 4  # Reasonable limit for descriptions
        
        while i < len(lines) and len(description_lines) < max_description_lines:
            line = lines[i].strip()
            
            if not line:
                # Empty line - skip but continue looking
                i += 1
                continue
            elif self.result_pattern.match(line):
                # Found next result - stop collecting
                break
            elif self.reference_pattern.match(line):
                # Found reference section - stop collecting
                break
            elif self.navigation_pattern.match(line):
                # Found navigation element - skip
                i += 1
                continue
            elif line.startswith('http') and ('duckduckgo.com' in line or 'uddg=' in line):
                # Found URL reference - skip  
                i += 1
                continue
            else:
                # This looks like description text
                description_lines.append(line)
                i += 1
        
        return description_lines, i
    
    def _is_navigation_element(self, line: str) -> bool:
        """Check if line is a navigation element (Search button, filters, etc.)."""
        navigation_indicators = [
            '[Search]', '[Next Page', '[Previous Page', 
            '[All Regions', '[Any Time', '[Safe Search'
        ]
        return any(indicator in line for indicator in navigation_indicators)


def parse_duckduckgo_w3m_output(w3m_output: str) -> List[Dict[str, Any]]:
    """
    Main entry point for parsing DuckDuckGo w3m output.
    
    Returns a list of dictionaries compatible with WebSearchTool format.
    """
    parser = DuckDuckGoResultParser()
    parsed_results = parser.parse(w3m_output)
    
    # Convert ParsedResult objects to dictionaries
    results = []
    for parsed_result in parsed_results:
        result_dict = {
            "title": parsed_result.title,
            "url": parsed_result.url,
            "snippet": parsed_result.snippet,
            "domain": parsed_result.domain,
            "relevance_score": parsed_result.relevance_score,
            "result_type": parsed_result.result_type
        }
        
        # Add optional metadata if available
        if parsed_result.link_number:
            result_dict["link_number"] = parsed_result.link_number
        if parsed_result.result_number:
            result_dict["result_number"] = parsed_result.result_number
            
        results.append(result_dict)
    
    return results


# For backward compatibility and testing
def decode_duckduckgo_url(redirect_url: str) -> Optional[str]:
    """Standalone function to decode DuckDuckGo redirect URLs."""
    decoder = DuckDuckGoURLDecoder()
    return decoder.decode_redirect_url(redirect_url)
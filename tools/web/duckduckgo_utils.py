"""
Additional utilities for DuckDuckGo search result processing.

This module provides advanced features like result ranking, snippet enhancement,
date extraction, and content quality scoring.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse


class DuckDuckGoResultEnhancer:
    """Enhances DuckDuckGo search results with additional features."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Patterns for extracting dates, ratings, and other metadata
        self.date_patterns = [
            re.compile(r'(\d{1,2}/\d{1,2}/\d{4})', re.IGNORECASE),  # MM/DD/YYYY
            re.compile(r'(\d{4}-\d{1,2}-\d{1,2})', re.IGNORECASE),  # YYYY-MM-DD
            re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}', re.IGNORECASE),
            re.compile(r'(\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})', re.IGNORECASE)
        ]
        
        self.rating_patterns = [
            re.compile(r'(\d+\.?\d*)\s*stars?', re.IGNORECASE),
            re.compile(r'(\d+\.?\d*)/5', re.IGNORECASE),
            re.compile(r'rating:?\s*(\d+\.?\d*)', re.IGNORECASE)
        ]
        
        self.price_patterns = [
            re.compile(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', re.IGNORECASE),
            re.compile(r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*USD', re.IGNORECASE),
            re.compile(r'price:?\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', re.IGNORECASE)
        ]
        
        # Domain quality rankings (higher is better)
        self.domain_quality_scores = {
            'github.com': 0.9,
            'stackoverflow.com': 0.95,
            'docs.python.org': 1.0,
            'wikipedia.org': 0.85,
            'medium.com': 0.7,
            'dev.to': 0.75,
            'arxiv.org': 0.95,
            'papers.nips.cc': 0.9,
            'openai.com': 0.85,
            'w3schools.com': 0.8,
            'geeksforgeeks.org': 0.75,
            'tutorialspoint.com': 0.7,
            'realpython.com': 0.85
        }
    
    def enhance_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance search results with additional metadata and improved scoring.
        
        Args:
            results: List of parsed search result dictionaries
            
        Returns:
            Enhanced results with additional metadata
        """
        enhanced_results = []
        
        for i, result in enumerate(results):
            try:
                enhanced = self._enhance_single_result(result, i)
                enhanced_results.append(enhanced)
            except Exception as e:
                self.logger.warning(f"Failed to enhance result {i}: {e}")
                enhanced_results.append(result)  # Keep original on error
        
        # Re-rank results based on enhanced scores
        enhanced_results = self._rerank_results(enhanced_results)
        
        return enhanced_results
    
    def _enhance_single_result(self, result: Dict[str, Any], position: int) -> Dict[str, Any]:
        """Enhance a single search result."""
        enhanced = result.copy()
        
        # Extract dates from snippet
        dates = self._extract_dates(result.get('snippet', ''))
        if dates:
            enhanced['extracted_dates'] = dates
            enhanced['most_recent_date'] = max(dates, key=lambda d: d['parsed_date'])
        
        # Extract ratings
        ratings = self._extract_ratings(result.get('snippet', ''))
        if ratings:
            enhanced['extracted_ratings'] = ratings
            enhanced['average_rating'] = sum(r['value'] for r in ratings) / len(ratings)
        
        # Extract prices
        prices = self._extract_prices(result.get('snippet', ''))
        if prices:
            enhanced['extracted_prices'] = prices
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(result, position)
        enhanced['quality_score'] = quality_score
        
        # Enhanced relevance score combining multiple factors
        enhanced_relevance = self._calculate_enhanced_relevance(result, quality_score, position)
        enhanced['enhanced_relevance_score'] = enhanced_relevance
        
        # Content type detection
        content_type = self._detect_content_type(result)
        enhanced['detected_content_type'] = content_type
        
        # Snippet quality improvements
        enhanced['snippet'] = self._improve_snippet(result.get('snippet', ''))
        
        return enhanced
    
    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract and parse dates from text."""
        dates = []
        
        for pattern in self.date_patterns:
            matches = pattern.findall(text)
            for match in matches:
                date_str = match if isinstance(match, str) else match[0]
                try:
                    # Try multiple date parsing strategies
                    parsed_date = self._parse_date_string(date_str)
                    if parsed_date:
                        dates.append({
                            'raw_text': date_str,
                            'parsed_date': parsed_date,
                            'iso_format': parsed_date.isoformat()
                        })
                except Exception as e:
                    self.logger.debug(f"Failed to parse date '{date_str}': {e}")
        
        return dates
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse a date string into a datetime object."""
        formats = [
            '%m/%d/%Y', '%Y-%m-%d', '%B %d, %Y', '%b %d, %Y',
            '%d %B %Y', '%d %b %Y', '%Y/%m/%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_ratings(self, text: str) -> List[Dict[str, Any]]:
        """Extract rating information from text."""
        ratings = []
        
        for pattern in self.rating_patterns:
            matches = pattern.findall(text)
            for match in matches:
                rating_str = match if isinstance(match, str) else match[0]
                try:
                    rating_value = float(rating_str)
                    if 0 <= rating_value <= 5:  # Reasonable rating range
                        ratings.append({
                            'raw_text': rating_str,
                            'value': rating_value,
                            'scale': 5.0
                        })
                except ValueError:
                    continue
        
        return ratings
    
    def _extract_prices(self, text: str) -> List[Dict[str, Any]]:
        """Extract price information from text."""
        prices = []
        
        for pattern in self.price_patterns:
            matches = pattern.findall(text)
            for match in matches:
                price_str = match if isinstance(match, str) else match[0]
                try:
                    # Remove commas and convert to float
                    price_value = float(price_str.replace(',', ''))
                    if 0 <= price_value <= 1000000:  # Reasonable price range
                        prices.append({
                            'raw_text': price_str,
                            'value': price_value,
                            'currency': 'USD'
                        })
                except ValueError:
                    continue
        
        return prices
    
    def _calculate_quality_score(self, result: Dict[str, Any], position: int) -> float:
        """Calculate a quality score for the result."""
        score = 0.5  # Base score
        
        # Domain reputation boost
        domain = result.get('domain', '')
        if domain in self.domain_quality_scores:
            score += self.domain_quality_scores[domain] * 0.3
        
        # Position penalty (later results get lower scores)
        position_penalty = max(0, (position - 1) * 0.05)
        score = max(0.1, score - position_penalty)
        
        # Title quality indicators
        title = result.get('title', '').lower()
        if 'tutorial' in title or 'guide' in title or 'documentation' in title:
            score += 0.1
        
        # Snippet quality indicators
        snippet = result.get('snippet', '').lower()
        if len(snippet) > 100:  # Longer snippets often indicate more content
            score += 0.05
        
        if any(word in snippet for word in ['example', 'examples', 'step-by-step', 'comprehensive']):
            score += 0.1
        
        return min(1.0, score)  # Cap at 1.0
    
    def _calculate_enhanced_relevance(self, result: Dict[str, Any], quality_score: float, position: int) -> float:
        """Calculate enhanced relevance score combining multiple factors."""
        base_relevance = result.get('relevance_score', 0.5)
        
        # Combine base relevance with quality score
        enhanced = (base_relevance * 0.7) + (quality_score * 0.3)
        
        # Bonus for recent dates
        if 'most_recent_date' in result:
            recent_date = result['most_recent_date']['parsed_date']
            days_old = (datetime.now() - recent_date).days
            if days_old < 365:  # Within a year
                freshness_bonus = max(0, (365 - days_old) / 365 * 0.1)
                enhanced += freshness_bonus
        
        # Bonus for high ratings
        if 'average_rating' in result:
            rating_bonus = (result['average_rating'] - 3.0) / 5.0 * 0.1
            enhanced += max(0, rating_bonus)
        
        return min(1.0, enhanced)
    
    def _detect_content_type(self, result: Dict[str, Any]) -> str:
        """Detect the type of content based on URL and snippet."""
        url = result.get('url', '')
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        domain = result.get('domain', '')
        
        # GitHub repository
        if 'github.com' in domain and '/blob/' not in url:
            return 'repository'
        
        # Documentation
        if any(word in domain for word in ['docs.', 'documentation', 'doc.']):
            return 'documentation'
        
        # Tutorial content
        if any(word in title + snippet for word in ['tutorial', 'guide', 'how to', 'learn']):
            return 'tutorial'
        
        # Stack Overflow questions
        if 'stackoverflow.com' in domain:
            return 'q_and_a'
        
        # Academic papers
        if any(word in domain for word in ['arxiv', 'papers', 'journal', 'academic']):
            return 'academic'
        
        # Blog posts
        if any(word in domain for word in ['medium.com', 'dev.to', 'blog']):
            return 'blog'
        
        return 'general'
    
    def _improve_snippet(self, snippet: str) -> str:
        """Improve snippet readability and formatting."""
        if not snippet:
            return snippet
        
        # Remove excessive whitespace
        improved = re.sub(r'\s+', ' ', snippet)
        
        # Ensure it ends with proper punctuation or ellipsis
        if improved and not improved.endswith(('.', '!', '?', '...')):
            improved += '...'
        
        # Capitalize first letter
        if improved:
            improved = improved[0].upper() + improved[1:]
        
        return improved.strip()
    
    def _rerank_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank results based on enhanced relevance scores."""
        try:
            # Sort by enhanced_relevance_score if available, otherwise by original relevance_score
            return sorted(
                results, 
                key=lambda r: r.get('enhanced_relevance_score', r.get('relevance_score', 0)),
                reverse=True
            )
        except Exception as e:
            self.logger.warning(f"Failed to rerank results: {e}")
            return results


class DuckDuckGoResultValidator:
    """Validates and filters DuckDuckGo search results."""
    
    def __init__(self, min_quality_score: float = 0.3):
        self.min_quality_score = min_quality_score
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def validate_and_filter_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and filter results based on quality and relevance criteria.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Filtered list of valid results
        """
        valid_results = []
        
        for result in results:
            validation_result = self._validate_single_result(result)
            if validation_result['valid']:
                valid_results.append(result)
            else:
                self.logger.debug(f"Filtered out result: {validation_result['reason']}")
        
        return valid_results
    
    def _validate_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single search result."""
        # Check required fields
        if not result.get('title'):
            return {'valid': False, 'reason': 'Missing title'}
        
        if not result.get('url'):
            return {'valid': False, 'reason': 'Missing URL'}
        
        # Check URL validity
        try:
            parsed_url = urlparse(result['url'])
            if not parsed_url.scheme or not parsed_url.netloc:
                return {'valid': False, 'reason': 'Invalid URL format'}
        except Exception:
            return {'valid': False, 'reason': 'URL parsing failed'}
        
        # Check quality score if available
        quality_score = result.get('quality_score', 1.0)
        if quality_score < self.min_quality_score:
            return {'valid': False, 'reason': f'Quality score too low: {quality_score}'}
        
        # Check for spam indicators
        title_lower = result.get('title', '').lower()
        if any(spam_word in title_lower for spam_word in ['click here', 'free download', 'get rich']):
            return {'valid': False, 'reason': 'Spam indicators detected'}
        
        return {'valid': True, 'reason': 'Passed all validation checks'}


# Convenience functions for direct usage
def enhance_duckduckgo_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enhance DuckDuckGo search results with additional metadata."""
    enhancer = DuckDuckGoResultEnhancer()
    return enhancer.enhance_results(results)


def validate_and_filter_duckduckgo_results(
    results: List[Dict[str, Any]], 
    min_quality_score: float = 0.3
) -> List[Dict[str, Any]]:
    """Validate and filter DuckDuckGo search results."""
    validator = DuckDuckGoResultValidator(min_quality_score)
    return validator.validate_and_filter_results(results)
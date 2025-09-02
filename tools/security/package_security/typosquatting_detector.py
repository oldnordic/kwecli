"""
Advanced Typosquatting Detection System.

This module implements comprehensive typosquatting detection using multiple
algorithms including Levenshtein distance, visual similarity, phonetic matching,
and keyboard distance analysis.
"""

import re
import unicodedata
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import math
import itertools
from collections import defaultdict


@dataclass
class TyposquattingResult:
    """Result structure for typosquatting detection."""
    package_name: str
    is_suspicious: bool
    overall_score: float
    similar_packages: List[str]
    detection_methods: Dict[str, Any]
    risk_level: str
    recommendations: List[str] = field(default_factory=list)


class StringSimilarityCalculator:
    """Calculate various string similarity metrics."""
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Number of single-character edits required
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer than s2
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def damerau_levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Damerau-Levenshtein distance (includes transpositions).
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Number of edits including transpositions
        """
        len1 = len(s1)
        len2 = len(s2)
        big_int = max(len1, len2) + 1
        
        # Create dictionary of unique characters
        da = {}
        for c in s1 + s2:
            da[c] = 0
        
        # Create distance matrix H with dimensions (len1+2, len2+2)
        max_dist = len1 + len2
        H = [[max_dist for _ in range(len2 + 2)] for _ in range(len1 + 2)]
        H[0][0] = max_dist
        
        for i in range(0, len1 + 1):
            H[i + 1][0] = max_dist
            H[i + 1][1] = i
        for j in range(0, len2 + 1):
            H[0][j + 1] = max_dist
            H[1][j + 1] = j
        
        for i in range(1, len1 + 1):
            db = 0
            for j in range(1, len2 + 1):
                k = da[s2[j - 1]]
                l = db
                if s1[i - 1] == s2[j - 1]:
                    cost = 0
                    db = j
                else:
                    cost = 1
                H[i + 1][j + 1] = min(
                    H[i][j] + cost,  # substitution
                    H[i + 1][j] + 1,  # insertion
                    H[i][j + 1] + 1,  # deletion
                    H[k][l] + (i - k - 1) + 1 + (j - l - 1),  # transposition
                )
            da[s1[i - 1]] = i
        
        return H[len1 + 1][len2 + 1]
    
    def jaro_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate Jaro similarity.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Maximum allowed distance
        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0
        
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        # Identify matches
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
    
    def jaro_winkler_similarity(self, s1: str, s2: str, prefix_scale: float = 0.1) -> float:
        """
        Calculate Jaro-Winkler similarity.
        
        Args:
            s1: First string
            s2: Second string
            prefix_scale: Scaling factor for common prefix (default 0.1)
            
        Returns:
            Similarity score between 0 and 1
        """
        jaro_score = self.jaro_similarity(s1, s2)
        
        if jaro_score == 0:
            return 0.0
        
        # Find common prefix up to 4 characters
        prefix = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        return jaro_score + prefix * prefix_scale * (1 - jaro_score)
    
    def ngram_similarity(self, s1: str, s2: str, n: int = 2) -> float:
        """
        Calculate N-gram similarity between two strings.
        
        Args:
            s1: First string
            s2: Second string
            n: Size of n-grams (default 2 for bigrams)
            
        Returns:
            Jaccard similarity coefficient between n-gram sets
        """
        if len(s1) < n or len(s2) < n:
            return 0.0 if s1 != s2 else 1.0
        
        # Generate n-grams
        ngrams1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
        ngrams2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        return len(intersection) / len(union) if union else 0.0


class VisualSimilarityDetector:
    """Detect visual similarity and homograph attacks."""
    
    # Unicode confusables mapping (simplified)
    CONFUSABLES = {
        'a': ['а', 'ạ', 'ά', 'ā', 'ă', 'ą'],  # Latin a vs Cyrillic а
        'e': ['е', 'ё', 'ė', 'ē', 'ę', 'ě'],  # Latin e vs Cyrillic е
        'o': ['о', '0', 'ο', 'ō', 'ŏ', 'ő'],  # Latin o vs Cyrillic о vs zero
        'p': ['р', 'ρ'],  # Latin p vs Cyrillic р vs Greek ρ
        'c': ['с', 'ç', 'ć', 'č'],  # Latin c vs Cyrillic с
        'x': ['х', 'χ'],  # Latin x vs Cyrillic х vs Greek χ
        'y': ['у', 'ý', 'ÿ', 'ŷ'],  # Latin y vs Cyrillic у
        'i': ['і', '1', 'l', 'ı', 'í', 'ī'],  # Various i/1/l confusables
        'l': ['1', 'I', 'ł', 'ĺ', 'ľ'],  # l vs 1 vs I
        '0': ['O', 'o', 'о', 'ο'],  # Zero vs O
        '1': ['l', 'I', 'i', '|'],  # One vs l/I
    }
    
    def visual_similarity_score(self, s1: str, s2: str) -> float:
        """
        Calculate visual similarity score between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Visual similarity score between 0 and 1
        """
        if s1 == s2:
            return 1.0
        
        if len(s1) != len(s2):
            # Different lengths, apply penalty
            base_score = 0.0
        else:
            # Check character-by-character visual similarity
            matches = 0
            for c1, c2 in zip(s1, s2):
                if c1 == c2:
                    matches += 1
                elif self._are_visually_similar(c1, c2):
                    matches += 0.9  # High score for visually similar chars
            
            base_score = matches / len(s1)
        
        # Check for mixed scripts (suspicious)
        if self.detect_mixed_scripts(s1) or self.detect_mixed_scripts(s2):
            base_score = min(base_score + 0.1, 1.0)  # Increase suspicion
        
        return base_score
    
    def _are_visually_similar(self, c1: str, c2: str) -> bool:
        """Check if two characters are visually similar."""
        c1_lower = c1.lower()
        c2_lower = c2.lower()
        
        for key, confusables in self.CONFUSABLES.items():
            chars = [key] + confusables
            if c1_lower in chars and c2_lower in chars:
                return True
        
        return False
    
    def detect_homograph_attack(self, text: str) -> bool:
        """
        Detect potential homograph attack using mixed scripts.
        
        Args:
            text: String to check
            
        Returns:
            True if suspicious homograph detected
        """
        # Check for mixed scripts
        if self.detect_mixed_scripts(text):
            return True
        
        # Check for known confusable characters
        for char in text:
            for confusables in self.CONFUSABLES.values():
                if char in confusables:
                    return True
        
        return False
    
    def detect_mixed_scripts(self, text: str) -> bool:
        """
        Detect if text contains mixed character scripts.
        
        Args:
            text: String to check
            
        Returns:
            True if mixed scripts detected
        """
        scripts = set()
        
        for char in text:
            if char.isalpha():
                # Get Unicode category and script
                category = unicodedata.category(char)
                try:
                    name = unicodedata.name(char)
                    if 'LATIN' in name:
                        scripts.add('LATIN')
                    elif 'CYRILLIC' in name:
                        scripts.add('CYRILLIC')
                    elif 'GREEK' in name:
                        scripts.add('GREEK')
                    elif 'ARABIC' in name:
                        scripts.add('ARABIC')
                    elif 'HEBREW' in name:
                        scripts.add('HEBREW')
                    elif 'CJK' in name or 'CHINESE' in name or 'JAPANESE' in name:
                        scripts.add('CJK')
                except ValueError:
                    # Character doesn't have a Unicode name
                    pass
        
        # Mixed scripts are suspicious
        return len(scripts) > 1
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode text to canonical form.
        
        Args:
            text: String to normalize
            
        Returns:
            Normalized string
        """
        # NFC = Canonical Decomposition, followed by Canonical Composition
        return unicodedata.normalize('NFC', text)


class PhoneticSimilarityDetector:
    """Detect phonetic similarity between strings."""
    
    def soundex(self, text: str) -> str:
        """
        Calculate Soundex code for phonetic comparison.
        
        Args:
            text: String to encode
            
        Returns:
            Soundex code (4 characters)
        """
        if not text:
            return "0000"
        
        text = text.upper()
        
        # Soundex character mappings
        soundex_map = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }
        
        # Keep first letter
        code = text[0]
        
        # Convert rest to numbers
        for char in text[1:]:
            if char in soundex_map:
                digit = soundex_map[char]
                # Don't add duplicate digits
                if len(code) == 1 or digit != code[-1]:
                    code += digit
        
        # Remove vowels represented as no code
        code = code.replace('0', '')
        
        # Pad with zeros or truncate to length 4
        return (code + "0000")[:4]
    
    def double_metaphone(self, text: str) -> Tuple[str, str]:
        """
        Calculate Double Metaphone codes for advanced phonetic matching.
        
        Args:
            text: String to encode
            
        Returns:
            Tuple of (primary, secondary) phonetic codes
        """
        if not text:
            return ("", "")
        
        text = text.upper()
        
        # Simplified Double Metaphone implementation
        # This is a basic version - full implementation would be more complex
        
        primary = ""
        secondary = ""
        
        # Handle common phonetic patterns
        text = text.replace("PH", "F")
        text = text.replace("WR", "R")
        text = text.replace("QU", "KW")
        text = text.replace("GH", "")
        text = text.replace("CK", "K")
        text = text.replace("CE", "SE")
        text = text.replace("CI", "SI")
        text = text.replace("CY", "SY")
        
        # Build phonetic code
        i = 0
        while i < len(text) and len(primary) < 4:
            char = text[i]
            
            if char in "AEIOU":
                if i == 0:
                    primary += "A"
            elif char in "BFPV":
                primary += "F"
            elif char in "CGJKQSXZ":
                primary += "S"
            elif char in "DT":
                primary += "T"
            elif char == "L":
                primary += "L"
            elif char in "MN":
                primary += "M"
            elif char == "R":
                primary += "R"
            elif char in "WY":
                if i == 0:
                    primary += char
            
            i += 1
        
        # For simplified version, secondary is same as primary
        secondary = primary
        
        # Pad to minimum length
        primary = (primary + "    ")[:4].strip()
        secondary = (secondary + "    ")[:4].strip()
        
        return (primary, secondary)
    
    def phonetic_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate phonetic similarity between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Use both Soundex and Double Metaphone
        soundex1 = self.soundex(s1)
        soundex2 = self.soundex(s2)
        
        metaphone1_primary, metaphone1_secondary = self.double_metaphone(s1)
        metaphone2_primary, metaphone2_secondary = self.double_metaphone(s2)
        
        score = 0.0
        
        # Check Soundex similarity
        if soundex1 == soundex2:
            score += 0.5
        elif soundex1[:2] == soundex2[:2]:
            score += 0.3
        
        # Check Double Metaphone similarity
        if metaphone1_primary == metaphone2_primary:
            score += 0.5
        elif metaphone1_primary == metaphone2_secondary or metaphone1_secondary == metaphone2_primary:
            score += 0.3
        elif metaphone1_primary[:2] == metaphone2_primary[:2]:
            score += 0.2
        
        return min(score, 1.0)


class KeyboardDistanceCalculator:
    """Calculate keyboard distance for typo detection."""
    
    MAX_DISTANCE = 10.0
    
    # QWERTY keyboard layout
    KEYBOARD_LAYOUT = {
        # Row 1 (numbers)
        '1': (0, 0), '2': (1, 0), '3': (2, 0), '4': (3, 0), '5': (4, 0),
        '6': (5, 0), '7': (6, 0), '8': (7, 0), '9': (8, 0), '0': (9, 0),
        '-': (10, 0), '=': (11, 0),
        
        # Row 2
        'q': (0.3, 1), 'w': (1.3, 1), 'e': (2.3, 1), 'r': (3.3, 1), 't': (4.3, 1),
        'y': (5.3, 1), 'u': (6.3, 1), 'i': (7.3, 1), 'o': (8.3, 1), 'p': (9.3, 1),
        '[': (10.3, 1), ']': (11.3, 1),
        
        # Row 3
        'a': (0.6, 2), 's': (1.6, 2), 'd': (2.6, 2), 'f': (3.6, 2), 'g': (4.6, 2),
        'h': (5.6, 2), 'j': (6.6, 2), 'k': (7.6, 2), 'l': (8.6, 2),
        ';': (9.6, 2), "'": (10.6, 2),
        
        # Row 4
        'z': (1.1, 3), 'x': (2.1, 3), 'c': (3.1, 3), 'v': (4.1, 3), 'b': (5.1, 3),
        'n': (6.1, 3), 'm': (7.1, 3), ',': (8.1, 3), '.': (9.1, 3), '/': (10.1, 3),
        
        # Space bar
        ' ': (5.5, 4)
    }
    
    def keyboard_distance(self, c1: str, c2: str) -> float:
        """
        Calculate physical distance between two keys on QWERTY keyboard.
        
        Args:
            c1: First character
            c2: Second character
            
        Returns:
            Euclidean distance between keys
        """
        c1_lower = c1.lower()
        c2_lower = c2.lower()
        
        if c1_lower == c2_lower:
            return 0.0
        
        pos1 = self.KEYBOARD_LAYOUT.get(c1_lower)
        pos2 = self.KEYBOARD_LAYOUT.get(c2_lower)
        
        if pos1 is None or pos2 is None:
            return self.MAX_DISTANCE
        
        # Calculate Euclidean distance
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def keyboard_weighted_distance(self, s1: str, s2: str) -> float:
        """
        Calculate Levenshtein distance weighted by keyboard distance.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Weighted edit distance
        """
        len1, len2 = len(s1), len(s2)
        
        # Create distance matrix
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize base cases
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Weight substitution by keyboard distance
                    kbd_dist = self.keyboard_distance(s1[i-1], s2[j-1])
                    substitution_cost = 1 + (kbd_dist / self.MAX_DISTANCE)
                    
                    dp[i][j] = min(
                        dp[i-1][j] + 1,  # deletion
                        dp[i][j-1] + 1,  # insertion
                        dp[i-1][j-1] + substitution_cost  # weighted substitution
                    )
        
        return dp[len1][len2]
    
    def generate_adjacent_typos(self, text: str) -> List[str]:
        """
        Generate possible typos based on adjacent keys.
        
        Args:
            text: Original string
            
        Returns:
            List of possible typos from adjacent key mistakes
        """
        typos = []
        
        # Find adjacent keys for each position
        for i, char in enumerate(text):
            char_lower = char.lower()
            if char_lower in self.KEYBOARD_LAYOUT:
                pos = self.KEYBOARD_LAYOUT[char_lower]
                
                # Find all keys within distance 1.5 (adjacent keys)
                for key, key_pos in self.KEYBOARD_LAYOUT.items():
                    if key != char_lower:
                        distance = math.sqrt((pos[0] - key_pos[0])**2 + (pos[1] - key_pos[1])**2)
                        if distance <= 1.5:
                            # Create typo with adjacent key
                            typo = text[:i] + key + text[i+1:]
                            if typo not in typos:
                                typos.append(typo)
        
        return typos


class AdvancedTyposquattingDetector:
    """Main typosquatting detection system."""
    
    def __init__(self, popular_packages: Dict[str, List[str]], config: Optional[Dict] = None):
        """
        Initialize the typosquatting detector.
        
        Args:
            popular_packages: Dictionary of popular packages by ecosystem
            config: Optional configuration dictionary
        """
        self.popular_packages = popular_packages
        self.config = self._merge_config(config)
        
        # Initialize sub-detectors
        self.string_calculator = StringSimilarityCalculator()
        self.visual_detector = VisualSimilarityDetector()
        self.phonetic_detector = PhoneticSimilarityDetector()
        self.keyboard_calculator = KeyboardDistanceCalculator()
        
        # Cache for performance
        self._cache = {}
    
    def _merge_config(self, custom_config: Optional[Dict]) -> Dict:
        """Merge custom config with defaults."""
        default_config = {
            "algorithms": {
                "levenshtein": {"enabled": True, "weight": 0.25, "threshold": 2},
                "jaro_winkler": {"enabled": True, "weight": 0.2, "threshold": 0.85},
                "keyboard_distance": {"enabled": True, "weight": 0.2},
                "phonetic": {"enabled": True, "weight": 0.15},
                "visual_similarity": {"enabled": True, "weight": 0.2}
            },
            "ecosystem_configs": {
                "npm": {"min_package_length": 3, "check_scoped": True},
                "pypi": {"min_package_length": 2, "check_underscores": True},
                "crates.io": {"check_hyphens": True}
            },
            "thresholds": {
                "suspicious_score": 0.7,
                "critical_score": 0.85
            }
        }
        
        if custom_config:
            # Deep merge
            for key, value in custom_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        # Validate weights
        for algo, settings in default_config["algorithms"].items():
            if "weight" in settings:
                settings["weight"] = max(0, min(1, settings["weight"]))
        
        return default_config
    
    async def detect_typosquatting(self, package_name: str, ecosystem: str) -> TyposquattingResult:
        """
        Detect if a package name is potentially typosquatting.
        
        Args:
            package_name: Package name to check
            ecosystem: Package ecosystem (npm, pypi, etc.)
            
        Returns:
            TyposquattingResult with detection details
        """
        # Normalize package name
        package_name = package_name.strip().lower()
        
        # Handle scoped packages (e.g., @scope/package)
        if "/" in package_name:
            # Extract the actual package name from scoped format
            package_parts = package_name.split("/")
            if len(package_parts) == 2:
                # Check both the full scoped name and just the package part
                scope_part = package_parts[0]
                package_part = package_parts[1]
                # We'll check the package part against popular packages
                base_package_name = package_part
            else:
                base_package_name = package_name
        else:
            base_package_name = package_name
        
        # Check cache
        cache_key = f"{ecosystem}:{package_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Handle empty or invalid input
        if not package_name or len(package_name) == 0:
            result = TyposquattingResult(
                package_name=package_name,
                is_suspicious=False,
                overall_score=0.0,
                similar_packages=[],
                detection_methods={},
                risk_level="NONE",
                recommendations=[]
            )
            self._cache[cache_key] = result
            return result
        
        # Check ecosystem-specific rules
        eco_config = self.config["ecosystem_configs"].get(ecosystem, {})
        min_length = eco_config.get("min_package_length", 0)
        
        if len(package_name) < min_length:
            result = TyposquattingResult(
                package_name=package_name,
                is_suspicious=False,
                overall_score=0.0,
                similar_packages=[],
                detection_methods={},
                risk_level="NONE",
                recommendations=["Package name too short for this ecosystem"]
            )
            self._cache[cache_key] = result
            return result
        
        # Get popular packages for this ecosystem
        ecosystem_packages = self.popular_packages.get(ecosystem, [])
        if not ecosystem_packages:
            # Unknown ecosystem
            result = TyposquattingResult(
                package_name=package_name,
                is_suspicious=False,
                overall_score=0.0,
                similar_packages=[],
                detection_methods={},
                risk_level="NONE",
                recommendations=["Unknown ecosystem"]
            )
            self._cache[cache_key] = result
            return result
        
        # Run detection algorithms
        detection_results = {}
        similar_packages = set()
        weighted_scores = []
        
        for popular_package in ecosystem_packages:
            scores = {}
            
            # Use base_package_name for comparisons if we have a scoped package
            check_name = base_package_name if "/" in package_name else package_name
            
            # Levenshtein distance
            if self.config["algorithms"]["levenshtein"]["enabled"]:
                distance = self.string_calculator.levenshtein_distance(check_name, popular_package.lower())
                threshold = self.config["algorithms"]["levenshtein"]["threshold"]
                if distance <= threshold and distance > 0:
                    score = 1.0 - (distance / max(len(check_name), len(popular_package)))
                    scores["levenshtein"] = score
                    similar_packages.add(popular_package)
            
            # Jaro-Winkler similarity
            if self.config["algorithms"]["jaro_winkler"]["enabled"]:
                similarity = self.string_calculator.jaro_winkler_similarity(check_name, popular_package.lower())
                threshold = self.config["algorithms"]["jaro_winkler"]["threshold"]
                if similarity >= threshold and check_name != popular_package.lower():
                    scores["jaro_winkler"] = similarity
                    similar_packages.add(popular_package)
            
            # Visual similarity
            if self.config["algorithms"]["visual_similarity"]["enabled"]:
                visual_score = self.visual_detector.visual_similarity_score(check_name, popular_package.lower())
                if visual_score > 0.8 and check_name != popular_package.lower():
                    scores["visual"] = visual_score
                    similar_packages.add(popular_package)
            
            # Phonetic similarity
            if self.config["algorithms"]["phonetic"]["enabled"]:
                phonetic_score = self.phonetic_detector.phonetic_similarity(check_name, popular_package.lower())
                if phonetic_score > 0.7 and check_name != popular_package.lower():
                    scores["phonetic"] = phonetic_score
                    similar_packages.add(popular_package)
            
            # Keyboard distance
            if self.config["algorithms"]["keyboard_distance"]["enabled"]:
                kbd_distance = self.keyboard_calculator.keyboard_weighted_distance(check_name, popular_package.lower())
                if kbd_distance <= 2 and kbd_distance > 0:
                    kbd_score = 1.0 - (kbd_distance / max(len(check_name), len(popular_package)))
                    scores["keyboard"] = kbd_score
                    similar_packages.add(popular_package)
            
            # If any algorithm detected similarity, record it
            if scores:
                detection_results[popular_package] = scores
                
                # Calculate weighted score for this match
                weighted_score = 0.0
                total_weight = 0.0
                for algo, score in scores.items():
                    algo_name = algo.replace("_", "")  # Handle both jaro_winkler and jarowinkler
                    if algo == "jaro_winkler":
                        weight = self.config["algorithms"]["jaro_winkler"].get("weight", 0.2)
                    elif algo == "keyboard":
                        weight = self.config["algorithms"]["keyboard_distance"].get("weight", 0.2)
                    elif algo == "visual":
                        weight = self.config["algorithms"]["visual_similarity"].get("weight", 0.2)
                    else:
                        weight = self.config["algorithms"].get(algo, {}).get("weight", 0.2)
                    weighted_score += score * weight
                    total_weight += weight
                # Normalize by total weight to ensure score is between 0 and 1
                if total_weight > 0:
                    weighted_score = weighted_score / total_weight
                weighted_scores.append(weighted_score)
        
        # Calculate overall score
        if weighted_scores:
            overall_score = max(weighted_scores)
        else:
            overall_score = 0.0
        
        # Determine risk level
        if overall_score >= self.config["thresholds"]["critical_score"]:
            risk_level = "CRITICAL"
        elif overall_score >= self.config["thresholds"]["suspicious_score"]:
            risk_level = "HIGH"
        elif overall_score >= 0.5:
            risk_level = "MEDIUM"
        elif overall_score >= 0.3:
            risk_level = "LOW"
        else:
            risk_level = "NONE"
        
        is_suspicious = overall_score >= self.config["thresholds"]["suspicious_score"]
        
        # Generate recommendations
        recommendations = []
        if is_suspicious and similar_packages:
            top_match = list(similar_packages)[0]
            recommendations.append(f"Use '{top_match}' instead of '{package_name}'")
            recommendations.append("Verify the package source and maintainer")
            recommendations.append("Check package download statistics and age")
        
        # Build result
        result = TyposquattingResult(
            package_name=package_name,
            is_suspicious=is_suspicious,
            overall_score=overall_score,
            similar_packages=list(similar_packages),
            detection_methods=detection_results,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
    
    async def detect_batch_typosquatting(self, packages: List[Dict[str, str]]) -> List[TyposquattingResult]:
        """
        Detect typosquatting for multiple packages.
        
        Args:
            packages: List of dicts with 'name' and 'ecosystem' keys
            
        Returns:
            List of TyposquattingResult objects
        """
        tasks = []
        for package in packages:
            name = package.get("name", "")
            ecosystem = package.get("ecosystem", "npm")
            tasks.append(self.detect_typosquatting(name, ecosystem))
        
        results = await asyncio.gather(*tasks)
        return results


# For backward compatibility - alias main class
TyposquattingDetector = AdvancedTyposquattingDetector
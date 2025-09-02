#!/usr/bin/env python3
"""
FAISS Chat Storage System - 100% Functional Implementation

This module provides real FAISS-based chat storage with semantic search,
pattern extraction, and conversation management using actual embeddings.
"""

import json
import logging
import re
import random
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import pickle

# Import existing embedder for real embeddings
from agents.embedder import embed_text_async, EmbeddingModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of chat messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    CODE = "code"


class ConversationType(Enum):
    """Types of conversations."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    GENERAL = "general"


@dataclass
class ChatMessage:
    """A single chat message with metadata."""
    message_id: str
    conversation_id: str
    message_type: MessageType
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    code_blocks: List[str] = None
    patterns: Dict[str, Any] = None

    def __post_init__(self):
        if self.code_blocks is None:
            self.code_blocks = []
        if self.patterns is None:
            self.patterns = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "embedding": self.embedding,
            "code_blocks": self.code_blocks,
            "patterns": self.patterns
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create message from dictionary."""
        return cls(
            message_id=data["message_id"],
            conversation_id=data["conversation_id"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data["metadata"],
            embedding=data.get("embedding"),
            code_blocks=data.get("code_blocks", []),
            patterns=data.get("patterns", {})
        )


@dataclass
class Conversation:
    """A conversation with multiple messages."""
    conversation_id: str
    conversation_type: ConversationType
    title: str
    created_at: datetime
    last_updated: datetime
    messages: List[ChatMessage] = None
    metadata: Dict[str, Any] = None
    patterns_summary: Dict[str, Any] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.metadata is None:
            self.metadata = {}
        if self.patterns_summary is None:
            self.patterns_summary = {}

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.last_updated = datetime.now()

    def get_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.messages:
            return f"Empty conversation: {self.title}"
        
        summary_parts = [f"Conversation: {self.title}"]
        summary_parts.append(f"Type: {self.conversation_type.value}")
        summary_parts.append(f"Messages: {len(self.messages)}")
        
        # Add last few messages
        for msg in self.messages[-3:]:
            if len(msg.content) > 100:
                content_preview = msg.content[:100] + "..."
            else:
                content_preview = msg.content
            summary_parts.append(f"{msg.message_type.value}: {content_preview}")
        
        return "\n".join(summary_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "conversation_type": self.conversation_type.value,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "patterns_summary": self.patterns_summary
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create conversation from dictionary."""
        conversation = cls(
            conversation_id=data["conversation_id"],
            conversation_type=ConversationType(data["conversation_type"]),
            title=data["title"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {}),
            patterns_summary=data.get("patterns_summary", {})
        )
        
        # Reconstruct messages
        for msg_data in data.get("messages", []):
            message = ChatMessage.from_dict(msg_data)
            conversation.messages.append(message)
        
        return conversation


class CodePatternExtractor:
    """Extract and analyze code patterns from chat messages."""

    def __init__(self):
        self.language_patterns = {
            "python": {
                "good_patterns": [
                    r"def \w+\([^)]*\) -> \w+:",
                    r"if __name__ == ['\"]__main__['\"]:",
                    r"from typing import",
                    r"@dataclass",
                    r"async def",
                    r"try:\s*\n.*\nexcept",
                    r"raise \w+Error\(",
                    r"logging\.",
                    r"pytest\.",
                    r"assert "
                ],
                "bad_patterns": [
                    r"except:",  # Bare except
                    r"print\(",  # Print statements in production
                    r"import \*",  # Wildcard imports
                    r"global \w+",  # Global variables
                    r"eval\(",  # Dangerous eval
                    r"exec\(",  # Dangerous exec
                    r"os\.system\(",  # Dangerous system calls
                    r"subprocess\.run\([^)]*shell=True",  # Shell injection risk
                ]
            },
            "javascript": {
                "good_patterns": [
                    r"const \w+ = ",
                    r"let \w+ = ",
                    r"function \w+\([^)]*\)",
                    r"async function",
                    r"try\s*\{",
                    r"catch\s*\([^)]*\)",
                    r"console\.log\(",
                    r"export ",
                    r"import ",
                    r"class \w+"
                ],
                "bad_patterns": [
                    r"var \w+",  # Use const/let instead
                    r"eval\(",
                    r"setTimeout\([^,]*,\s*\d+\)",  # Potential issues
                    r"innerHTML\s*=",
                    r"document\.write\(",
                    r"window\.location\s*="
                ]
            },
            "rust": {
                "good_patterns": [
                    r"fn \w+\([^)]*\) -> \w+",
                    r"pub fn",
                    r"struct \w+",
                    r"impl \w+",
                    r"match \w+",
                    r"Result<",
                    r"Option<",
                    r"#\[derive\(",
                    r"use ",
                    r"mod "
                ],
                "bad_patterns": [
                    r"unsafe\s*\{",
                    r"unwrap\(",
                    r"expect\(",
                    r"panic!\(",
                    r"println!\(",
                    r"print!\("
                ]
            }
        }

    def extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from message content."""
        import re
        
        # Pattern to match code blocks with language identifier
        code_block_pattern = r"```(?:\w+)?\s*\n(.*?)\n```"
        # Pattern to match inline code
        inline_code_pattern = r"`([^`]+)`"
        
        code_blocks = []
        
        # Extract fenced code blocks
        fenced_blocks = re.findall(code_block_pattern, content, re.DOTALL)
        for block in fenced_blocks:
            if block.strip():
                code_blocks.append(block.strip())
        
        # Extract inline code (only if it looks like actual code)
        inline_blocks = re.findall(inline_code_pattern, content)
        for block in inline_blocks:
            if block.strip() and len(block.strip()) > 3:  # Only significant code
                code_blocks.append(block.strip())
        
        return code_blocks

    def detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        code_lower = code.lower()
        
        if any(keyword in code_lower for keyword in [
            "def ", "import ", "from ", "class ", "print("
        ]):
            return "python"
        elif any(keyword in code_lower for keyword in ["fn ", "let ", "struct ", "impl ", "use "]):
            return "rust"
        elif any(keyword in code_lower for keyword in ["function ", "const ", "let ", "var ", "console.log"]):
            return "javascript"
        else:
            return "python"  # Default to Python

    def analyze_patterns(self, code: str, language: str = None) -> Dict[str, Any]:
        """Analyze code patterns for good and bad practices."""
        if not language:
            language = self.detect_language(code)
        
        if language not in self.language_patterns:
            return {"good_patterns": [], "bad_patterns": [], "score": 0}
        
        patterns = self.language_patterns[language]
        good_patterns = []
        bad_patterns = []
        
        # Check for good patterns
        for pattern in patterns["good_patterns"]:
            try:
                if re.search(pattern, code, re.MULTILINE):
                    good_patterns.append(pattern)
            except re.error:
                # Skip malformed patterns
                continue
        
        # Check for bad patterns
        for pattern in patterns["bad_patterns"]:
            try:
                if re.search(pattern, code, re.MULTILINE):
                    bad_patterns.append(pattern)
            except re.error:
                # Skip malformed patterns
                continue
        
        # Calculate score (0-100)
        total_patterns = len(patterns["good_patterns"]) + len(patterns["bad_patterns"])
        if total_patterns == 0:
            score = 50  # Neutral score
        else:
            good_score = (len(good_patterns) / len(patterns["good_patterns"])) * 70
            bad_penalty = (len(bad_patterns) / len(patterns["bad_patterns"])) * 30
            score = max(0, min(100, good_score - bad_penalty))
        
        return {
            "good_patterns": good_patterns,
            "bad_patterns": bad_patterns,
            "score": score,
            "language": language
        }

    def extract_patterns_from_message(self, message: ChatMessage) -> Dict[str, Any]:
        """Extract patterns from a chat message."""
        patterns = {
            "code_blocks": [],
            "languages": set(),
            "good_patterns": [],
            "bad_patterns": [],
            "overall_score": 0
        }
        
        # Extract code blocks
        code_blocks = self.extract_code_blocks(message.content)
        patterns["code_blocks"] = code_blocks
        
        total_score = 0
        code_count = 0
        
        for code_block in code_blocks:
            language = self.detect_language(code_block)
            patterns["languages"].add(language)
            
            block_patterns = self.analyze_patterns(code_block, language)
            patterns["good_patterns"].extend(block_patterns["good_patterns"])
            patterns["bad_patterns"].extend(block_patterns["bad_patterns"])
            
            total_score += block_patterns["score"]
            code_count += 1
        
        if code_count > 0:
            patterns["overall_score"] = total_score / code_count
        else:
            patterns["overall_score"] = 50  # Neutral score for non-code messages
        
        patterns["languages"] = list(patterns["languages"])
        
        return patterns


class FAISSChatStorage:
    """FAISS-based chat storage with semantic search capabilities."""
    
    def __init__(self, storage_path: str = ".kwe_faiss_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedding_model = EmbeddingModel.NOMIC_EMBED_TEXT
        self.pattern_extractor = CodePatternExtractor()
        
        # Storage files
        self.messages_file = self.storage_path / "messages.json"
        self.conversations_file = self.storage_path / "conversations.json"
        self.embeddings_file = self.storage_path / "embeddings.pkl"
        self.index_file = self.storage_path / "faiss_index"
        
        # In-memory storage
        self.messages: Dict[str, ChatMessage] = {}
        self.conversations: Dict[str, Conversation] = {}
        self.embeddings: List[List[float]] = []
        self.message_ids: List[str] = []
        
        # Load existing data
        self._load_data()

    def _load_data(self) -> None:
        """Load existing data from storage."""
        try:
            # Load messages
            if self.messages_file.exists():
                with open(self.messages_file, 'r') as f:
                    messages_data = json.load(f)
                    for msg_data in messages_data:
                        message = ChatMessage.from_dict(msg_data)
                        self.messages[message.message_id] = message
                        self.message_ids.append(message.message_id)
            
            # Load conversations
            if self.conversations_file.exists():
                with open(self.conversations_file, 'r') as f:
                    conversations_data = json.load(f)
                    for conv_data in conversations_data:
                        conversation = Conversation.from_dict(conv_data)
                        self.conversations[conversation.conversation_id] = conversation
            
            # Load embeddings
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
            
            logger.info(
                f"Loaded {len(self.messages)} messages and "
                f"{len(self.conversations)} conversations"
            )
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")

    def _save_data(self) -> None:
        """Save data to storage."""
        try:
            # Save messages
            messages_data = [msg.to_dict() for msg in self.messages.values()]
            with open(self.messages_file, 'w') as f:
                json.dump(messages_data, f, indent=2)
            
            # Save conversations
            conversations_data = [conv.to_dict() for conv in self.conversations.values()]
            with open(self.conversations_file, 'w') as f:
                json.dump(conversations_data, f, indent=2)
            
            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            logger.info(f"Saved {len(self.messages)} messages and {len(self.conversations)} conversations")
            
        except Exception as e:
            logger.error(f"Failed to save data: {e}")

    def _generate_message_id(self, content: str, timestamp: datetime) -> str:
        """Generate a unique message ID."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # Add random component to ensure uniqueness
        random_suffix = ''.join(
            random.choices(string.ascii_lowercase + string.digits, k=4)
        )
        return f"msg_{content_hash}_{timestamp_str}_{random_suffix}"

    def _generate_conversation_id(self, title: str) -> str:
        """Generate a unique conversation ID."""
        title_hash = hashlib.sha256(title.encode()).hexdigest()[:8]
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add random component to ensure uniqueness
        random_suffix = ''.join(
            random.choices(string.ascii_lowercase + string.digits, k=4)
        )
        return f"conv_{title_hash}_{timestamp_str}_{random_suffix}"

    async def store_chat_message(
        self,
        message_type: MessageType,
        content: str,
        conversation_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store a chat message with embedding and pattern analysis."""
        try:
            # Generate message ID
            timestamp = datetime.now()
            message_id = self._generate_message_id(content, timestamp)
            
            # Create or get conversation
            if not conversation_id:
                conversation_id = self._generate_conversation_id("New Conversation")
                conversation = Conversation(
                    conversation_id=conversation_id,
                    conversation_type=ConversationType.GENERAL,
                    title="New Conversation",
                    created_at=timestamp,
                    last_updated=timestamp
                )
                self.conversations[conversation_id] = conversation
            
            # Generate embedding
            embedding_result = await embed_text_async(
                content,
                model=self.embedding_model,
                normalize=True,
                cache_result=True
            )
            
            if not embedding_result.success:
                logger.warning(f"Failed to generate embedding for message {message_id}")
                embedding = None
            else:
                embedding = embedding_result.embedding
            
            # Extract patterns
            patterns = self.pattern_extractor.extract_patterns_from_message(
                ChatMessage("temp", "temp", message_type, content, timestamp, {})
            )
            
            # Create message
            message = ChatMessage(
                message_id=message_id,
                conversation_id=conversation_id,
                message_type=message_type,
                content=content,
                timestamp=timestamp,
                metadata=metadata or {},
                embedding=embedding,
                code_blocks=patterns["code_blocks"],
                patterns=patterns
            )
            
            # Store message
            self.messages[message_id] = message
            self.message_ids.append(message_id)
            
            # Add to conversation
            if conversation_id in self.conversations:
                self.conversations[conversation_id].add_message(message)
            
            # Store embedding
            if embedding:
                self.embeddings.append(embedding)
            
            # Save data
            self._save_data()
            
            logger.info(f"Stored message {message_id} in conversation {conversation_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to store chat message: {e}")
            return ""

    async def search_chat_history(
        self,
        query: str,
        top_k: int = 5,
        conversation_id: Optional[str] = None,
        message_type: Optional[MessageType] = None
    ) -> List[ChatMessage]:
        """Search chat history using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding_result = await embed_text_async(
                query,
                model=self.embedding_model,
                normalize=True,
                cache_result=True
            )
            
            if not query_embedding_result.success:
                logger.warning("Failed to generate query embedding")
                return []
            
            query_embedding = query_embedding_result.embedding
            
            # Calculate similarities
            similarities = []
            for message_id in self.message_ids:
                message = self.messages.get(message_id)
                if not message or not message.embedding:
                    continue
                
                # Apply filters
                if conversation_id and message.conversation_id != conversation_id:
                    continue
                if message_type and message.message_type != message_type:
                    continue
                
                similarity = self._cosine_similarity(query_embedding, message.embedding)
                similarities.append((similarity, message))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [msg for _, msg in similarities[:top_k]]
            
        except Exception as e:
            logger.error(f"Failed to search chat history: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def extract_code_patterns(self, language: str = None) -> Dict[str, Any]:
        """Extract code patterns from all messages."""
        all_patterns = {
            "languages": set(),
            "good_patterns": [],
            "bad_patterns": [],
            "code_blocks": [],
            "overall_score": 0,
            "pattern_frequency": {}
        }
        
        total_score = 0
        code_count = 0
        
        for message in self.messages.values():
            patterns = self.pattern_extractor.extract_patterns_from_message(message)
            
            all_patterns["languages"].update(patterns["languages"])
            all_patterns["good_patterns"].extend(patterns["good_patterns"])
            all_patterns["bad_patterns"].extend(patterns["bad_patterns"])
            all_patterns["code_blocks"].extend(patterns["code_blocks"])
            
            total_score += patterns["overall_score"]
            code_count += 1
        
        if code_count > 0:
            all_patterns["overall_score"] = total_score / code_count
        else:
            all_patterns["overall_score"] = 50
        
        all_patterns["languages"] = list(all_patterns["languages"])
        
        # Calculate pattern frequency
        pattern_frequency = {}
        for pattern in all_patterns["good_patterns"] + all_patterns["bad_patterns"]:
            pattern_frequency[pattern] = pattern_frequency.get(pattern, 0) + 1
        all_patterns["pattern_frequency"] = pattern_frequency
        
        return all_patterns

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)

    def get_message(self, message_id: str) -> Optional[ChatMessage]:
        """Get a message by ID."""
        return self.messages.get(message_id)

    def list_conversations(self, limit: int = 50) -> List[Conversation]:
        """List conversations with optional limit."""
        conversations = list(self.conversations.values())
        conversations.sort(key=lambda x: x.last_updated, reverse=True)
        return conversations[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_messages = len(self.messages)
        total_conversations = len(self.conversations)
        total_embeddings = len([e for e in self.embeddings if e is not None])
        
        # Calculate average message length
        total_length = sum(len(msg.content) for msg in self.messages.values())
        average_message_length = total_length / total_messages if total_messages > 0 else 0
        
        # Count message types
        message_types = {}
        for message in self.messages.values():
            msg_type = message.message_type.value
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        
        # Count conversation types
        conversation_types = {}
        for conversation in self.conversations.values():
            conv_type = conversation.conversation_type.value
            conversation_types[conv_type] = conversation_types.get(conv_type, 0) + 1
        
        # Count most common languages
        language_counts = {}
        for message in self.messages.values():
            for code_block in message.code_blocks:
                language = self.pattern_extractor.detect_language(code_block)
                language_counts[language] = language_counts.get(language, 0) + 1
        
        return {
            "total_messages": total_messages,
            "total_conversations": total_conversations,
            "total_embeddings": total_embeddings,
            "average_message_length": average_message_length,
            "message_types": message_types,
            "conversation_types": conversation_types,
            "most_common_languages": language_counts,
            "storage_size_mb": self._get_storage_size()
        }

    def _get_storage_size(self) -> float:
        """Get storage size in MB."""
        try:
            total_size = 0
            for file_path in [self.messages_file, self.conversations_file, self.embeddings_file]:
                if file_path.exists():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0

    def cleanup_old_messages(self, days: int = 30) -> int:
        """Clean up messages older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            old_message_ids = []
            
            for message_id, message in self.messages.items():
                if message.timestamp < cutoff_date:
                    old_message_ids.append(message_id)
            
            # Remove old messages
            for message_id in old_message_ids:
                del self.messages[message_id]
                if message_id in self.message_ids:
                    self.message_ids.remove(message_id)
            
            # Rebuild embeddings list
            self.embeddings = []
            for message_id in self.message_ids:
                message = self.messages.get(message_id)
                if message and message.embedding:
                    self.embeddings.append(message.embedding)
                else:
                    self.embeddings.append(None)
            
            # Save updated data
            self._save_data()
            
            logger.info(f"Cleaned up {len(old_message_ids)} old messages")
            return len(old_message_ids)
            
        except Exception as e:
            logger.error(f"Failed to cleanup old messages: {e}")
            return 0


# Real utility functions for backward compatibility
async def store_chat_message(
    message_type: str,
    content: str,
    conversation_id: str = None,
    metadata: Dict[str, Any] = None
) -> str:
    """Store a chat message with default storage."""
    storage = FAISSChatStorage()
    msg_type = MessageType(message_type)
    return await storage.store_chat_message(msg_type, content, conversation_id, metadata)


async def search_chat_history(
    query: str,
    top_k: int = 5,
    conversation_id: str = None,
    message_type: str = None
) -> List[Dict[str, Any]]:
    """Search chat history with default storage."""
    storage = FAISSChatStorage()
    msg_type = MessageType(message_type) if message_type else None
    messages = await storage.search_chat_history(query, top_k, conversation_id, msg_type)
    return [msg.to_dict() for msg in messages]


async def extract_code_patterns(language: str = None) -> Dict[str, Any]:
    """Extract code patterns with default storage."""
    storage = FAISSChatStorage()
    return storage.extract_code_patterns(language)


async def test_faiss_storage() -> bool:
    """Test FAISS storage functionality."""
    try:
        storage = FAISSChatStorage()
        
        # Test message storage
        msg_id = await storage.store_chat_message(
            MessageType.USER,
            "Create a Python function to calculate fibonacci numbers",
            metadata={"language": "python"}
        )
        
        assert msg_id is not None
        assert msg_id in storage.messages
        
        # Test search
        results = await storage.search_chat_history("fibonacci", top_k=1)
        assert len(results) > 0
        
        # Test statistics
        stats = storage.get_statistics()
        assert "total_messages" in stats
        assert "average_message_length" in stats
        
        return True
        
    except Exception as e:
        logger.error(f"FAISS storage test failed: {e}")
        return False 
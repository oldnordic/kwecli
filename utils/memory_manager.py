#!/usr/bin/env python3
"""
Memory Management System for KWE CLI

This module provides memory management functionality including:
- Conversation memory storage and retrieval
- Context window optimization
- Memory persistence and backup
- Pattern learning and analysis
"""

import json
import logging
import sqlite3
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory that can be stored."""
    CONVERSATION = "conversation"
    CONTEXT_WINDOW = "context_window"
    CODE_PATTERN = "code_pattern"
    SYSTEM_STATE = "system_state"
    USER_PREFERENCE = "user_preference"


class MemoryPriority(Enum):
    """Priority levels for memory entries."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryNode:
    """Individual memory node storing content and metadata."""
    node_id: str
    content: str
    timestamp: datetime
    memory_type: str
    priority: int
    metadata: Dict[str, Any]
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory node to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'memory_type': self.memory_type,
            'priority': self.priority,
            'metadata': self.metadata,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create memory node from dictionary."""
        return cls(
            node_id=data['node_id'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']) if data['timestamp'] else datetime.now(),
            memory_type=data['memory_type'],
            priority=data['priority'],
            metadata=data.get('metadata', {}),
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None
        )


@dataclass
class ConversationMemory:
    """Manages conversation-specific memory."""
    conversation_id: str
    user_id: str
    max_messages: int = 100
    max_age_hours: int = 24
    messages: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to conversation memory."""
        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
        self._cleanup_old_messages()

    def _cleanup_old_messages(self) -> None:
        """Clean up old messages based on limits."""
        # Remove messages over max count
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Remove messages over max age
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
        self.messages = [
            msg for msg in self.messages
            if datetime.fromisoformat(msg['timestamp']) > cutoff_time
        ]

    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from conversation."""
        return self.messages[-count:] if self.messages else []


@dataclass 
class ContextWindow:
    """Manages context window for AI interactions."""
    window_id: str
    max_tokens: int = 4000
    current_content: List[Dict[str, Any]] = None
    token_count: int = 0

    def __post_init__(self):
        if self.current_content is None:
            self.current_content = []

    def add_content(self, content: str, content_type: str = "text", priority: int = 1) -> None:
        """Add content to context window."""
        # Simple token estimation (rough)
        estimated_tokens = len(content.split()) * 1.3
        
        # Remove low priority content if needed
        while (self.token_count + estimated_tokens) > self.max_tokens and self.current_content:
            removed = min(self.current_content, key=lambda x: x.get('priority', 0))
            self.current_content.remove(removed)
            self.token_count -= removed.get('estimated_tokens', 0)

        # Add new content
        content_entry = {
            'id': str(uuid.uuid4()),
            'content': content,
            'content_type': content_type,
            'priority': priority,
            'estimated_tokens': estimated_tokens,
            'timestamp': datetime.now().isoformat()
        }
        self.current_content.append(content_entry)
        self.token_count += estimated_tokens

    def get_context_string(self) -> str:
        """Get formatted context string."""
        sorted_content = sorted(self.current_content, key=lambda x: x.get('priority', 0), reverse=True)
        return "\n".join([item['content'] for item in sorted_content])


class MemoryIndex:
    """Provides indexing and search capabilities for memory."""
    
    def __init__(self):
        self.indices = {}
        
    def index_memory_node(self, node: MemoryNode) -> None:
        """Add memory node to indices."""
        # Index by type
        if node.memory_type not in self.indices:
            self.indices[node.memory_type] = []
        self.indices[node.memory_type].append(node.node_id)
        
        # Index by keywords (simple keyword extraction)
        keywords = self._extract_keywords(node.content)
        for keyword in keywords:
            if keyword not in self.indices:
                self.indices[keyword] = []
            self.indices[keyword].append(node.node_id)
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        # Simple keyword extraction - split and filter
        words = content.lower().split()
        keywords = [word.strip('.,!?;:') for word in words if len(word) > 3]
        return list(set(keywords))
    
    def search_by_keywords(self, keywords: List[str]) -> List[str]:
        """Search for memory nodes by keywords."""
        matching_ids = set()
        for keyword in keywords:
            if keyword.lower() in self.indices:
                matching_ids.update(self.indices[keyword.lower()])
        return list(matching_ids)


class MemoryStorage:
    """Handles persistent storage of memory."""
    
    def __init__(self, storage_path: Union[str, Path]):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "memory.db"
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for memory storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    node_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    metadata TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                )
            """)
            conn.commit()
    
    def store_memory_node(self, node: MemoryNode) -> None:
        """Store memory node in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_nodes 
                (node_id, content, timestamp, memory_type, priority, metadata, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.node_id,
                node.content,
                node.timestamp.isoformat(),
                node.memory_type,
                node.priority,
                json.dumps(node.metadata),
                node.access_count,
                node.last_accessed.isoformat() if node.last_accessed else None
            ))
            conn.commit()
    
    def load_memory_node(self, node_id: str) -> Optional[MemoryNode]:
        """Load memory node from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, content, timestamp, memory_type, priority, metadata, access_count, last_accessed
                FROM memory_nodes WHERE node_id = ?
            """, (node_id,))
            row = cursor.fetchone()
            
            if row:
                return MemoryNode(
                    node_id=row[0],
                    content=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    memory_type=row[3],
                    priority=row[4],
                    metadata=json.loads(row[5]),
                    access_count=row[6],
                    last_accessed=datetime.fromisoformat(row[7]) if row[7] else None
                )
            return None
    
    def search_by_type(self, memory_type: str, limit: int = 100) -> List[MemoryNode]:
        """Search memory nodes by type."""
        nodes = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, content, timestamp, memory_type, priority, metadata, access_count, last_accessed
                FROM memory_nodes WHERE memory_type = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (memory_type, limit))
            
            for row in cursor.fetchall():
                nodes.append(MemoryNode(
                    node_id=row[0],
                    content=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    memory_type=row[3],
                    priority=row[4],
                    metadata=json.loads(row[5]),
                    access_count=row[6],
                    last_accessed=datetime.fromisoformat(row[7]) if row[7] else None
                ))
        return nodes


class MemoryBackup:
    """Handles memory backup and restoration."""
    
    def __init__(self, backup_path: Union[str, Path]):
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, storage: MemoryStorage) -> str:
        """Create backup of memory storage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_path / f"memory_backup_{timestamp}.json"
        
        # Export all memory nodes
        backup_data = {
            'timestamp': timestamp,
            'nodes': []
        }
        
        # Read all nodes from storage
        with sqlite3.connect(storage.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, content, timestamp, memory_type, priority, metadata, access_count, last_accessed
                FROM memory_nodes
            """)
            
            for row in cursor.fetchall():
                node_data = {
                    'node_id': row[0],
                    'content': row[1],
                    'timestamp': row[2],
                    'memory_type': row[3],
                    'priority': row[4],
                    'metadata': json.loads(row[5]),
                    'access_count': row[6],
                    'last_accessed': row[7]
                }
                backup_data['nodes'].append(node_data)
        
        # Write backup file
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        logger.info(f"Created memory backup: {backup_file}")
        return str(backup_file)
    
    def restore_backup(self, backup_file: Union[str, Path], storage: MemoryStorage) -> None:
        """Restore memory from backup file."""
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
        
        for node_data in backup_data['nodes']:
            node = MemoryNode.from_dict(node_data)
            storage.store_memory_node(node)
        
        logger.info(f"Restored memory from backup: {backup_file}")


class MemoryOptimizer:
    """Optimizes memory usage and performance."""
    
    def __init__(self):
        self.optimization_stats = {}
    
    def optimize_memory(self, storage: MemoryStorage) -> Dict[str, Any]:
        """Optimize memory storage."""
        optimization_results = {
            'nodes_cleaned': 0,
            'space_saved': 0,
            'duration_ms': 0
        }
        
        start_time = datetime.now()
        
        # Clean up old, low-priority nodes
        cutoff_date = datetime.now() - timedelta(days=30)
        
        with sqlite3.connect(storage.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM memory_nodes 
                WHERE timestamp < ? AND priority = 1 AND access_count = 0
            """, (cutoff_date.isoformat(),))
            
            optimization_results['nodes_cleaned'] = cursor.rowcount
            conn.commit()
        
        # Vacuum database
        with sqlite3.connect(storage.db_path) as conn:
            conn.execute("VACUUM")
        
        duration = datetime.now() - start_time
        optimization_results['duration_ms'] = int(duration.total_seconds() * 1000)
        
        logger.info(f"Memory optimization completed: {optimization_results}")
        return optimization_results


class MemoryManager:
    """Main memory management interface."""
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None, backup_enabled: bool = False):
        # Set up storage path
        if storage_path is None:
            storage_path = Path.home() / ".kwecli" / "memory"
        self.storage_path = Path(storage_path)
        
        # Initialize components
        self.storage = MemoryStorage(self.storage_path)
        self.index = MemoryIndex()
        self.optimizer = MemoryOptimizer()
        
        # Optional backup
        self.backup_enabled = backup_enabled
        if backup_enabled:
            backup_path = self.storage_path / "backups"
            self.backup = MemoryBackup(backup_path)
        else:
            self.backup = None
        
        # Memory caches
        self.conversation_memories = {}
        self.context_windows = {}
        
        logger.info(f"Memory manager initialized at: {self.storage_path}")
    
    async def add_conversation_memory(self, conversation_id: str, user_id: str, 
                                   role: str, content: str, 
                                   metadata: Optional[Dict] = None) -> None:
        """Add conversation memory."""
        if conversation_id not in self.conversation_memories:
            self.conversation_memories[conversation_id] = ConversationMemory(
                conversation_id, user_id
            )
        
        self.conversation_memories[conversation_id].add_message(role, content, metadata)
        
        # Store in persistent storage
        node = MemoryNode(
            node_id=str(uuid.uuid4()),
            content=f"[{role}] {content}",
            timestamp=datetime.now(),
            memory_type=MemoryType.CONVERSATION.value,
            priority=MemoryPriority.NORMAL.value,
            metadata={
                'conversation_id': conversation_id,
                'user_id': user_id,
                'role': role,
                **(metadata or {})
            }
        )
        
        self.storage.store_memory_node(node)
        self.index.index_memory_node(node)
    
    async def get_conversation_memory(self, conversation_id: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get conversation memory."""
        if conversation_id in self.conversation_memories:
            return self.conversation_memories[conversation_id].get_recent_messages(count)
        
        # Load from storage
        nodes = self.storage.search_by_type(MemoryType.CONVERSATION.value, count * 2)
        messages = []
        
        for node in nodes:
            if node.metadata.get('conversation_id') == conversation_id:
                messages.append({
                    'content': node.content,
                    'timestamp': node.timestamp.isoformat(),
                    'metadata': node.metadata
                })
                
                if len(messages) >= count:
                    break
        
        return messages
    
    async def add_context_window(self, window_id: str, content: str, 
                               content_type: str = "text", priority: int = 1) -> None:
        """Add content to context window."""
        if window_id not in self.context_windows:
            self.context_windows[window_id] = ContextWindow(window_id)
        
        self.context_windows[window_id].add_content(content, content_type, priority)
    
    async def get_context_window(self, window_id: str) -> Optional[str]:
        """Get context window content."""
        if window_id in self.context_windows:
            return self.context_windows[window_id].get_context_string()
        return None
    
    async def search_memory(self, query: str, memory_type: Optional[str] = None,
                          limit: int = 10) -> List[MemoryNode]:
        """Search memory by query."""
        keywords = query.lower().split()
        matching_ids = self.index.search_by_keywords(keywords)
        
        # Load matching nodes
        nodes = []
        for node_id in matching_ids[:limit]:
            node = self.storage.load_memory_node(node_id)
            if node and (memory_type is None or node.memory_type == memory_type):
                # Update access statistics
                node.access_count += 1
                node.last_accessed = datetime.now()
                self.storage.store_memory_node(node)
                nodes.append(node)
        
        # Sort by relevance (simple scoring)
        nodes.sort(key=lambda n: n.access_count + n.priority, reverse=True)
        return nodes[:limit]
    
    async def cleanup_old_memory(self, days: int = 30) -> Dict[str, Any]:
        """Clean up old memory entries."""
        return self.optimizer.optimize_memory(self.storage)
    
    async def create_backup(self) -> Optional[str]:
        """Create memory backup."""
        if self.backup:
            return self.backup.create_backup(self.storage)
        return None
    
    async def restore_backup(self, backup_file: Union[str, Path]) -> None:
        """Restore from backup."""
        if self.backup:
            self.backup.restore_backup(backup_file, self.storage)
        else:
            logger.warning("Backup not enabled, cannot restore")


# Export main classes
__all__ = [
    'MemoryManager',
    'ConversationMemory', 
    'ContextWindow',
    'MemoryNode',
    'MemoryIndex',
    'MemoryStorage',
    'MemoryBackup',
    'MemoryOptimizer',
    'MemoryType',
    'MemoryPriority'
]
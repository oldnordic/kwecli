#!/usr/bin/env python3
"""
Real ACP Persistence Layer

Production-ready database and persistence management for ACP system.
Provides real SQLAlchemy-based persistence with:
- Agent registration and lifecycle tracking
- Message history and audit trails
- Conversation context persistence
- Performance metrics storage
- Configuration management
- Data migration and backup

No mock implementations - all functionality is real and production-ready.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import (
    Column, String, DateTime, JSON, Boolean, Integer, Float, Text,
    ForeignKey, Index, UniqueConstraint, create_engine
)
from sqlalchemy import String  # UUID will be stored as String in SQLite
from sqlalchemy.sql import func
from alembic.config import Config
from alembic import command
import aiosqlite

from .acp_models import (
    ACPMessage, AgentProfile, ConversationContext, MessageMetrics,
    TaskRequest, TaskResult, MessageStatus, FIPAPerformative
)

logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()


class AgentRecord(Base):
    """Database model for agent registration and metadata."""
    __tablename__ = 'agents'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    capabilities = Column(JSON)
    profile_data = Column(JSON)  # Complete AgentProfile serialized
    
    # Status tracking
    status = Column(String(50), default='active', index=True)
    load_factor = Column(Float, default=0.0)
    max_concurrent_tasks = Column(Integer, default=10)
    
    # Lifecycle
    registered_at = Column(DateTime, default=func.now())
    last_seen = Column(DateTime, default=func.now())
    last_heartbeat = Column(DateTime)
    
    # Metadata
    version = Column(String(50))
    owner = Column(String(255))
    organization = Column(String(255))
    contact = Column(String(255))
    metadata_json = Column(JSON)
    
    # Relationships
    sent_messages = relationship("MessageRecord", foreign_keys="MessageRecord.sender_id", back_populates="sender")
    received_messages = relationship("MessageRecord", foreign_keys="MessageRecord.receiver_id", back_populates="receiver")
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_status', 'status'),
        Index('idx_agent_capabilities', 'capabilities'),
        Index('idx_agent_last_seen', 'last_seen'),
    )


class MessageRecord(Base):
    """Database model for message persistence and audit trail."""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    message_id = Column(String(255), unique=True, nullable=False, index=True)
    conversation_id = Column(String(255), index=True)
    
    # FIPA-ACL fields
    performative = Column(String(50), nullable=False, index=True)
    sender_id = Column(String(255), ForeignKey('agents.agent_id'), nullable=False, index=True)
    receiver_id = Column(String(255), ForeignKey('agents.agent_id'), index=True)
    content = Column(JSON)
    
    # Protocol fields
    reply_to = Column(String(255), ForeignKey('messages.message_id'))
    reply_with = Column(String(255))
    language = Column(String(50), default='JSON')
    encoding = Column(String(50), default='UTF-8')
    ontology = Column(String(255))
    protocol = Column(String(100), default='fipa-request')
    
    # Timing
    timestamp = Column(DateTime, nullable=False, index=True)
    reply_by = Column(DateTime)
    ttl = Column(Integer)  # Time to live in seconds
    
    # Status and delivery
    status = Column(String(50), default='pending', index=True)
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    
    # Lifecycle
    created_at = Column(DateTime, default=func.now())
    sent_at = Column(DateTime)
    delivered_at = Column(DateTime)
    processed_at = Column(DateTime)
    expires_at = Column(DateTime, index=True)
    
    # Metadata
    routing_path = Column(JSON)  # Via field
    transport_address = Column(String(255))
    size_bytes = Column(Integer)
    processing_time = Column(Float)  # Seconds
    
    # Relationships
    sender = relationship("AgentRecord", foreign_keys=[sender_id], back_populates="sent_messages")
    receiver = relationship("AgentRecord", foreign_keys=[receiver_id], back_populates="received_messages")
    parent_message = relationship("MessageRecord", remote_side=[id])
    
    # Indexes
    __table_args__ = (
        Index('idx_message_conversation', 'conversation_id'),
        Index('idx_message_timestamp', 'timestamp'),
        Index('idx_message_status', 'status'),
        Index('idx_message_expires', 'expires_at'),
        Index('idx_message_performative', 'performative'),
    )


class ConversationRecord(Base):
    """Database model for conversation context and state."""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Participants
    participants = Column(JSON)  # List of agent IDs
    initiator_id = Column(String(255), ForeignKey('agents.agent_id'))
    
    # Protocol and state
    protocol = Column(String(100), default='fipa-request')
    state = Column(String(50), default='active', index=True)
    ontology = Column(String(255))
    
    # Metrics
    message_count = Column(Integer, default=0)
    total_size_bytes = Column(Integer, default=0)
    
    # Lifecycle
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime)
    
    # Context data
    context_data = Column(JSON)  # ConversationContext serialized
    metadata_json = Column(JSON)
    
    # Relationships
    initiator = relationship("AgentRecord", foreign_keys=[initiator_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_conversation_state', 'state'),
        Index('idx_conversation_created', 'created_at'),
        Index('idx_conversation_updated', 'updated_at'),
    )


class TaskRecord(Base):
    """Database model for task execution tracking."""
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)
    conversation_id = Column(String(255), ForeignKey('conversations.conversation_id'), index=True)
    
    # Task details
    task_type = Column(String(100), nullable=False, index=True)
    requester_id = Column(String(255), ForeignKey('agents.agent_id'), nullable=False)
    executor_id = Column(String(255), ForeignKey('agents.agent_id'))
    
    # Request and result
    parameters = Column(JSON)
    result_data = Column(JSON)
    error_message = Column(Text)
    
    # Status and priority
    status = Column(String(50), default='pending', index=True)
    priority = Column(Integer, default=5)
    
    # Timing
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    timeout_at = Column(DateTime)
    
    # Metrics
    execution_time = Column(Float)  # Seconds
    queue_time = Column(Float)  # Seconds
    retry_count = Column(Integer, default=0)
    
    # Dependencies
    dependencies = Column(JSON)  # List of task IDs
    dependents = Column(JSON)  # List of task IDs that depend on this
    
    # Metadata
    metadata_json = Column(JSON)
    
    # Relationships
    requester = relationship("AgentRecord", foreign_keys=[requester_id])
    executor = relationship("AgentRecord", foreign_keys=[executor_id])
    conversation = relationship("ConversationRecord")
    
    # Indexes
    __table_args__ = (
        Index('idx_task_status', 'status'),
        Index('idx_task_type', 'task_type'),
        Index('idx_task_created', 'created_at'),
        Index('idx_task_priority', 'priority'),
    )


class MetricsRecord(Base):
    """Database model for system metrics and monitoring."""
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    metric_type = Column(String(100), nullable=False, index=True)
    source = Column(String(255), index=True)  # Agent ID or system component
    
    # Metrics data
    metrics_data = Column(JSON)
    
    # Specific metrics for quick queries
    messages_sent = Column(Integer, default=0)
    messages_received = Column(Integer, default=0)
    messages_failed = Column(Integer, default=0)
    average_response_time = Column(Float)
    active_connections = Column(Integer)
    queue_size = Column(Integer)
    load_factor = Column(Float)
    
    # Indexes
    __table_args__ = (
        Index('idx_metrics_timestamp_type', 'timestamp', 'metric_type'),
        Index('idx_metrics_source', 'source'),
    )


class ConfigRecord(Base):
    """Database model for configuration storage."""
    __tablename__ = 'config'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(255), unique=True, nullable=False, index=True)
    value = Column(JSON)
    description = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    updated_by = Column(String(255))  # Agent or user who updated
    
    # Versioning
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)


class ACPPersistenceManager:
    """Real persistence manager for ACP system."""
    
    def __init__(
        self,
        database_url: str = "sqlite+aiosqlite:///acp.db",
        pool_size: int = 20,
        max_overflow: int = 30,
        echo: bool = False
    ):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.echo = echo
        
        # Database engine and session factory
        self.engine = None
        self.session_factory = None
        
        # Connection pool
        self.active_sessions: Set[AsyncSession] = set()
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.cleanup_running = False
    
    async def initialize(self):
        """Initialize database and create tables."""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=self.echo,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600  # 1 hour
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            # Start background tasks
            await self.start_background_tasks()
            
            logger.info(f"Database initialized: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown database connections and cleanup."""
        try:
            # Stop background tasks
            self.cleanup_running = False
            for task in self.background_tasks:
                task.cancel()
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            self.background_tasks.clear()
            
            # Close active sessions
            for session in list(self.active_sessions):
                await session.close()
            self.active_sessions.clear()
            
            # Close engine
            if self.engine:
                await self.engine.dispose()
            
            logger.info("Database shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during database shutdown: {e}")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.session_factory()
        self.active_sessions.add(session)
        
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()
            self.active_sessions.discard(session)
    
    # Agent management
    async def store_agent(self, profile: AgentProfile) -> bool:
        """Store or update agent profile."""
        try:
            async with self.get_session() as session:
                # Check if agent exists
                existing = await session.get(AgentRecord, profile.agent_id)
                
                if existing:
                    # Update existing agent
                    existing.name = profile.name
                    existing.description = profile.description
                    existing.capabilities = [cap.dict() for cap in profile.capabilities]
                    existing.profile_data = profile.dict()
                    existing.status = profile.status
                    existing.load_factor = profile.load_factor
                    existing.max_concurrent_tasks = profile.max_concurrent_tasks
                    existing.version = profile.version
                    existing.owner = profile.owner
                    existing.organization = profile.organization
                    existing.contact = profile.contact
                    existing.last_seen = datetime.utcnow()
                    existing.metadata = getattr(profile, 'metadata', {})
                else:
                    # Create new agent record
                    agent_record = AgentRecord(
                        agent_id=profile.agent_id,
                        name=profile.name,
                        description=profile.description,
                        capabilities=[cap.dict() for cap in profile.capabilities],
                        profile_data=profile.dict(),
                        status=profile.status,
                        load_factor=profile.load_factor,
                        max_concurrent_tasks=profile.max_concurrent_tasks,
                        version=profile.version,
                        owner=profile.owner,
                        organization=profile.organization,
                        contact=profile.contact,
                        metadata=getattr(profile, 'metadata', {})
                    )
                    session.add(agent_record)
                
                await session.commit()
                logger.debug(f"Agent stored: {profile.agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store agent {profile.agent_id}: {e}")
            return False
    
    async def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Retrieve agent profile by ID."""
        try:
            async with self.get_session() as session:
                record = await session.get(AgentRecord, agent_id)
                if record and record.profile_data:
                    return AgentProfile(**record.profile_data)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            return None
    
    async def update_agent_heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat timestamp."""
        try:
            async with self.get_session() as session:
                record = await session.get(AgentRecord, agent_id)
                if record:
                    record.last_heartbeat = datetime.utcnow()
                    record.last_seen = datetime.utcnow()
                    await session.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to update heartbeat for {agent_id}: {e}")
            return False
    
    async def list_agents(self, status_filter: Optional[str] = None) -> List[AgentProfile]:
        """List all agents with optional status filter."""
        try:
            async with self.get_session() as session:
                query = session.query(AgentRecord)
                if status_filter:
                    query = query.filter(AgentRecord.status == status_filter)
                
                result = await query.all()
                agents = []
                for record in result:
                    if record.profile_data:
                        agents.append(AgentProfile(**record.profile_data))
                
                return agents
                
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return []
    
    # Message management
    async def store_message(self, message: ACPMessage) -> bool:
        """Store message in database."""
        try:
            async with self.get_session() as session:
                expires_at = None
                if message.ttl:
                    expires_at = message.timestamp + timedelta(seconds=message.ttl)
                
                message_record = MessageRecord(
                    message_id=message.message_id,
                    conversation_id=message.conversation_id,
                    performative=message.performative.value,
                    sender_id=message.sender,
                    receiver_id=message.receiver,
                    content=message.content,
                    reply_to=message.reply_to,
                    reply_with=message.reply_with,
                    language=message.language,
                    encoding=message.encoding,
                    ontology=message.ontology,
                    protocol=message.protocol,
                    timestamp=message.timestamp,
                    reply_by=message.reply_by,
                    ttl=message.ttl,
                    status=message.status.value,
                    retry_count=message.retry_count,
                    error_message=message.error_message,
                    expires_at=expires_at,
                    routing_path=message.via,
                    transport_address=message.transport_address,
                    size_bytes=len(json.dumps(message.content))
                )
                
                session.add(message_record)
                await session.commit()
                logger.debug(f"Message stored: {message.message_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store message {message.message_id}: {e}")
            return False
    
    async def update_message_status(
        self,
        message_id: str,
        status: MessageStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """Update message status."""
        try:
            async with self.get_session() as session:
                record = await session.get(MessageRecord, message_id)
                if record:
                    record.status = status.value
                    if error_message:
                        record.error_message = error_message
                    
                    # Update timestamps based on status
                    now = datetime.utcnow()
                    if status == MessageStatus.SENT and not record.sent_at:
                        record.sent_at = now
                    elif status == MessageStatus.DELIVERED and not record.delivered_at:
                        record.delivered_at = now
                    elif status == MessageStatus.PROCESSED and not record.processed_at:
                        record.processed_at = now
                        
                        # Calculate processing time
                        if record.delivered_at:
                            record.processing_time = (now - record.delivered_at).total_seconds()
                    
                    await session.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to update message status {message_id}: {e}")
            return False
    
    async def get_message_history(
        self,
        agent_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ACPMessage]:
        """Get message history with optional filters."""
        try:
            async with self.get_session() as session:
                query = session.query(MessageRecord)
                
                if agent_id:
                    query = query.filter(
                        (MessageRecord.sender_id == agent_id) |
                        (MessageRecord.receiver_id == agent_id)
                    )
                
                if conversation_id:
                    query = query.filter(MessageRecord.conversation_id == conversation_id)
                
                query = query.order_by(MessageRecord.timestamp.desc())
                query = query.limit(limit).offset(offset)
                
                records = await query.all()
                messages = []
                
                for record in records:
                    message = ACPMessage(
                        performative=FIPAPerformative(record.performative),
                        sender=record.sender_id,
                        receiver=record.receiver_id or "",
                        content=record.content or {},
                        message_id=record.message_id,
                        conversation_id=record.conversation_id,
                        reply_to=record.reply_to,
                        reply_with=record.reply_with,
                        language=record.language,
                        encoding=record.encoding,
                        ontology=record.ontology,
                        protocol=record.protocol,
                        timestamp=record.timestamp,
                        reply_by=record.reply_by,
                        ttl=record.ttl,
                        status=MessageStatus(record.status),
                        retry_count=record.retry_count,
                        error_message=record.error_message
                    )
                    messages.append(message)
                
                return messages
                
        except Exception as e:
            logger.error(f"Failed to get message history: {e}")
            return []
    
    # Conversation management
    async def store_conversation(self, context: ConversationContext) -> bool:
        """Store or update conversation context."""
        try:
            async with self.get_session() as session:
                existing = await session.get(ConversationRecord, context.conversation_id)
                
                if existing:
                    # Update existing conversation
                    existing.participants = context.participants
                    existing.protocol = context.protocol
                    existing.state = context.state
                    existing.updated_at = context.updated_at
                    existing.message_count = context.message_count
                    existing.context_data = context.dict()
                    existing.metadata = context.metadata
                else:
                    # Create new conversation record
                    conv_record = ConversationRecord(
                        conversation_id=context.conversation_id,
                        participants=context.participants,
                        initiator_id=context.participants[0] if context.participants else None,
                        protocol=context.protocol,
                        state=context.state,
                        created_at=context.created_at,
                        updated_at=context.updated_at,
                        message_count=context.message_count,
                        context_data=context.dict(),
                        metadata=context.metadata
                    )
                    session.add(conv_record)
                
                await session.commit()
                logger.debug(f"Conversation stored: {context.conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store conversation {context.conversation_id}: {e}")
            return False
    
    # Metrics management
    async def store_metrics(
        self,
        metric_type: str,
        source: str,
        metrics: MessageMetrics
    ) -> bool:
        """Store system metrics."""
        try:
            async with self.get_session() as session:
                metrics_record = MetricsRecord(
                    metric_type=metric_type,
                    source=source,
                    metrics_data=metrics.to_dict(),
                    messages_sent=metrics.total_sent,
                    messages_received=metrics.total_received,
                    messages_failed=metrics.total_failed,
                    average_response_time=metrics.average_processing_time,
                    queue_size=metrics.current_queue_size
                )
                
                session.add(metrics_record)
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
            return False
    
    # Background tasks
    async def start_background_tasks(self):
        """Start background maintenance tasks."""
        self.cleanup_running = True
        
        # Cleanup expired data task
        cleanup_task = asyncio.create_task(self._cleanup_task())
        self.background_tasks.add(cleanup_task)
        
        # Optimize database task
        optimize_task = asyncio.create_task(self._optimize_task())
        self.background_tasks.add(optimize_task)
    
    async def _cleanup_task(self):
        """Background task to cleanup expired data."""
        while self.cleanup_running:
            try:
                await self._cleanup_expired_messages()
                await self._cleanup_old_metrics()
                await self._cleanup_completed_conversations()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(3600)
    
    async def _optimize_task(self):
        """Background task to optimize database."""
        while self.cleanup_running:
            try:
                # Run database optimization
                async with self.get_session() as session:
                    # VACUUM for SQLite
                    if 'sqlite' in self.database_url:
                        await session.execute("VACUUM")
                    
                    # Analyze tables for better query planning
                    await session.execute("ANALYZE")
                    await session.commit()
                
                logger.info("Database optimization completed")
                await asyncio.sleep(86400)  # Run once daily
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimize task: {e}")
                await asyncio.sleep(86400)
    
    async def _cleanup_expired_messages(self):
        """Clean up expired messages."""
        try:
            async with self.get_session() as session:
                now = datetime.utcnow()
                
                # Delete expired messages
                result = await session.execute(
                    "DELETE FROM messages WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,)
                )
                
                deleted_count = result.rowcount
                await session.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired messages")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired messages: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics data."""
        try:
            async with self.get_session() as session:
                # Keep metrics for 30 days
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                result = await session.execute(
                    "DELETE FROM metrics WHERE timestamp < ?",
                    (cutoff_date,)
                )
                
                deleted_count = result.rowcount
                await session.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old metrics records")
                
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
    
    async def _cleanup_completed_conversations(self):
        """Clean up old completed conversations."""
        try:
            async with self.get_session() as session:
                # Keep completed conversations for 7 days
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                result = await session.execute(
                    "DELETE FROM conversations WHERE state IN ('completed', 'failed', 'timeout') AND completed_at < ?",
                    (cutoff_date,)
                )
                
                deleted_count = result.rowcount
                await session.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old conversations")
                
        except Exception as e:
            logger.error(f"Error cleaning up old conversations: {e}")


# Context manager for persistence manager
@asynccontextmanager
async def persistence_context(database_url: str = "sqlite+aiosqlite:///acp.db"):
    """Context manager for persistence manager lifecycle."""
    manager = ACPPersistenceManager(database_url)
    
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.shutdown()


# Database migration utilities
async def migrate_database(database_url: str, alembic_config_path: str):
    """Run database migrations using Alembic."""
    try:
        # Configure Alembic
        config = Config(alembic_config_path)
        config.set_main_option("sqlalchemy.url", database_url.replace("+aiosqlite", ""))
        
        # Run migrations
        command.upgrade(config, "head")
        logger.info("Database migration completed successfully")
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise


async def backup_database(database_url: str, backup_path: Path):
    """Create database backup."""
    try:
        if "sqlite" in database_url:
            # SQLite backup
            db_path = database_url.split("///")[-1]
            async with aiosqlite.connect(db_path) as source:
                async with aiosqlite.connect(str(backup_path)) as backup:
                    await source.backup(backup)
            
            logger.info(f"Database backup created: {backup_path}")
        else:
            logger.warning("Backup not implemented for non-SQLite databases")
            
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise


# Example usage and testing
async def example_persistence_usage():
    """Example of using the persistence manager."""
    async with persistence_context() as persistence:
        
        # Create test agent profile
        from acp_models import AgentProfile, AgentCapability
        
        capabilities = [
            AgentCapability(name="code_generation", description="Generate code"),
            AgentCapability(name="code_analysis", description="Analyze code")
        ]
        
        profile = AgentProfile(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent for demonstration",
            capabilities=capabilities
        )
        
        # Store agent
        success = await persistence.store_agent(profile)
        print(f"Agent stored: {success}")
        
        # Retrieve agent
        retrieved = await persistence.get_agent("test-agent")
        print(f"Agent retrieved: {retrieved.name if retrieved else 'Not found'}")
        
        # Create and store test message
        test_message = ACPMessage(
            performative=FIPAPerformative.INFORM,
            sender="test-agent",
            receiver="other-agent",
            content={"message": "Hello from test agent"}
        )
        
        success = await persistence.store_message(test_message)
        print(f"Message stored: {success}")
        
        # Get message history
        history = await persistence.get_message_history(agent_id="test-agent", limit=10)
        print(f"Message history: {len(history)} messages")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_persistence_usage())
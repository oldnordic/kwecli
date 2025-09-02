#!/usr/bin/env python3
"""
Real ACP (Agent Communication Protocol) Models

Production-ready Pydantic models for Agent Communication Protocol.
These models define the data structures for FIPA-ACL compliant communication
between agents in the KWE CLI system. No mock implementations.
"""

import uuid
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import json
from dataclasses import dataclass
from pathlib import Path


class FIPAPerformative(str, Enum):
    """FIPA-ACL standard performatives for agent communication."""
    # Core communicative acts
    INFORM = "inform"
    REQUEST = "request"
    QUERY_IF = "query-if"
    QUERY_REF = "query-ref"
    CONFIRM = "confirm"
    DISCONFIRM = "disconfirm"
    
    # Response acts
    AGREE = "agree"
    REFUSE = "refuse"
    FAILURE = "failure"
    NOT_UNDERSTOOD = "not-understood"
    
    # Negotiation acts
    PROPOSE = "propose"
    ACCEPT_PROPOSAL = "accept-proposal"
    REJECT_PROPOSAL = "reject-proposal"
    
    # Subscription acts
    SUBSCRIBE = "subscribe"
    CANCEL = "cancel"
    
    # Custom performatives for KWE CLI
    CAPABILITY_QUERY = "capability-query"
    CAPABILITY_ANNOUNCE = "capability-announce"
    STATUS_UPDATE = "status-update"
    HEARTBEAT = "heartbeat"


class MessageStatus(str, Enum):
    """Message delivery and processing status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"


class AgentCapability(BaseModel):
    """Agent capability description."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    ontology: Optional[str] = None
    service_type: Optional[str] = None
    

class AgentProfile(BaseModel):
    """Complete agent profile for registration."""
    agent_id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    languages: List[str] = Field(default_factory=lambda: ["JSON", "XML"])
    protocols: List[str] = Field(default_factory=lambda: ["fipa-request", "fipa-query"])
    ontologies: List[str] = Field(default_factory=list)
    services: List[str] = Field(default_factory=list)
    
    # Agent metadata
    version: str = "1.0.0"
    owner: Optional[str] = None
    organization: Optional[str] = None
    contact: Optional[str] = None
    
    # Runtime information
    status: str = "active"
    load_factor: float = 0.0  # 0.0 to 1.0
    max_concurrent_tasks: int = 10
    
    @field_validator('load_factor')
    @classmethod
    def validate_load_factor(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Load factor must be between 0.0 and 1.0')
        return v


class ACPMessagePart(BaseModel):
    """Part of a multipart ACP message."""
    text: str
    part_type: str = "text"
    encoding: str = "UTF-8"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict(exclude_none=True)


class ACPMessage(BaseModel):
    """FIPA-ACL compliant message structure."""
    
    # Mandatory FIPA-ACL parameters
    performative: FIPAPerformative
    sender: str  # Agent identifier
    receiver: str  # Can be agent ID, capability name, or "*" for broadcast
    content: Dict[str, Any]  # Message payload
    
    # Optional FIPA-ACL parameters
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: Optional[str] = None
    reply_to: Optional[str] = None  # Message ID this is replying to
    reply_with: Optional[str] = None  # Expected reply identifier
    
    # Protocol parameters
    language: str = "JSON"  # Content language
    encoding: str = "UTF-8"
    ontology: Optional[str] = None  # Domain ontology
    protocol: str = "fipa-request"  # Interaction protocol
    
    # Time parameters
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reply_by: Optional[datetime] = None  # Reply deadline
    ttl: Optional[int] = None  # Time to live in seconds
    
    # Routing and delivery
    via: Optional[List[str]] = None  # Routing path
    transport_address: Optional[str] = None
    
    # Internal tracking
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @model_validator(mode='before')
    @classmethod
    def validate_message(cls, values):
        """Validate message consistency."""
        # Set conversation ID if not provided
        if values.get('conversation_id') is None:
            values['conversation_id'] = values.get('message_id')
        
        # Validate reply_by is in the future
        if values.get('reply_by') and values.get('reply_by') <= datetime.utcnow():
            raise ValueError('reply_by must be in the future')
        
        # Validate TTL
        if values.get('ttl') and values.get('ttl') <= 0:
            raise ValueError('TTL must be positive')
        
        return values
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl is None:
            return False
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > self.ttl
    
    def is_reply_overdue(self) -> bool:
        """Check if reply deadline has passed."""
        if self.reply_by is None:
            return False
        return datetime.utcnow() > self.reply_by
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict(exclude_none=True, by_alias=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ACPMessage':
        """Create message from dictionary."""
        # Handle datetime fields
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'reply_by' in data and isinstance(data['reply_by'], str):
            data['reply_by'] = datetime.fromisoformat(data['reply_by'])
        
        return cls(**data)


class TaskRequest(BaseModel):
    """Task execution request structure."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 5  # 1-10, where 10 is highest
    timeout: Optional[int] = None  # Seconds
    dependencies: List[str] = Field(default_factory=list)  # Task IDs
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Priority must be between 1 and 10')
        return v


class TaskResult(BaseModel):
    """Task execution result structure."""
    task_id: str
    status: str  # success, failure, timeout, cancelled
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None  # Seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        allowed = {'success', 'failure', 'timeout', 'cancelled'}
        if v not in allowed:
            raise ValueError(f'Status must be one of: {allowed}')
        return v


class CapabilityQuery(BaseModel):
    """Query for agent capabilities."""
    capability_name: Optional[str] = None  # Specific capability
    capability_type: Optional[str] = None  # Type category
    parameters: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)


class CapabilityResponse(BaseModel):
    """Response to capability query."""
    agent_id: str
    capabilities: List[AgentCapability]
    availability: bool = True
    load_factor: float = 0.0
    estimated_response_time: Optional[int] = None  # Seconds


class StatusUpdate(BaseModel):
    """Agent status update."""
    agent_id: str
    status: str  # active, idle, busy, offline, error
    load_factor: float
    active_tasks: int
    queue_length: int
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        allowed = {'active', 'idle', 'busy', 'offline', 'error'}
        if v not in allowed:
            raise ValueError(f'Status must be one of: {allowed}')
        return v
    
    @field_validator('load_factor')
    @classmethod
    def validate_load_factor(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Load factor must be between 0.0 and 1.0')
        return v


class ErrorInfo(BaseModel):
    """Error information structure."""
    error_code: str
    error_type: str  # protocol, semantic, syntax, system
    message: str
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = False
    retry_after: Optional[int] = None  # Seconds
    
    @field_validator('error_type')
    @classmethod
    def validate_error_type(cls, v):
        allowed = {'protocol', 'semantic', 'syntax', 'system'}
        if v not in allowed:
            raise ValueError(f'Error type must be one of: {allowed}')
        return v


class ConversationContext(BaseModel):
    """Context for multi-turn conversations."""
    conversation_id: str
    participants: List[str]
    protocol: str = "fipa-request"
    state: str = "active"  # active, completed, failed, timeout
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    
    @field_validator('state')
    @classmethod
    def validate_state(cls, v):
        allowed = {'active', 'completed', 'failed', 'timeout'}
        if v not in allowed:
            raise ValueError(f'State must be one of: {allowed}')
        return v


class ServiceDescription(BaseModel):
    """Service description for agent directory."""
    service_id: str
    name: str
    description: str
    service_type: str
    provider_agent: str
    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    outputs: List[Dict[str, Any]] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    effects: List[str] = Field(default_factory=list)
    qos_parameters: Dict[str, Any] = Field(default_factory=dict)
    cost: Optional[float] = None
    currency: Optional[str] = None


class AgentDirectory(BaseModel):
    """Agent directory entry for discovery."""
    directory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agents: Dict[str, AgentProfile] = Field(default_factory=dict)
    services: Dict[str, ServiceDescription] = Field(default_factory=dict)
    capability_index: Dict[str, Set[str]] = Field(default_factory=dict)  # capability -> agent_ids
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def add_agent(self, profile: AgentProfile):
        """Add agent to directory and update indexes."""
        self.agents[profile.agent_id] = profile
        
        # Update capability index
        for capability in profile.capabilities:
            if capability.name not in self.capability_index:
                self.capability_index[capability.name] = set()
            self.capability_index[capability.name].add(profile.agent_id)
        
        self.last_updated = datetime.utcnow()
    
    def remove_agent(self, agent_id: str):
        """Remove agent from directory and update indexes."""
        if agent_id in self.agents:
            profile = self.agents[agent_id]
            
            # Remove from capability index
            for capability in profile.capabilities:
                if capability.name in self.capability_index:
                    self.capability_index[capability.name].discard(agent_id)
                    if not self.capability_index[capability.name]:
                        del self.capability_index[capability.name]
            
            del self.agents[agent_id]
            self.last_updated = datetime.utcnow()
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents with specific capability."""
        return list(self.capability_index.get(capability, set()))
    
    def get_agent_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across agents."""
        return {
            agent_id: profile.load_factor
            for agent_id, profile in self.agents.items()
        }


class ACPConfig(BaseModel):
    """Configuration for ACP system."""
    server_host: str = "127.0.0.1"
    websocket_port: int = 8001
    http_port: int = 8002
    database_url: str = "sqlite+aiosqlite:///acp.db"
    
    # Security settings
    use_ssl: bool = False
    ssl_cert_path: Optional[Path] = None
    ssl_key_path: Optional[Path] = None
    secret_key: Optional[str] = None
    
    # Performance settings
    max_connections: int = 1000
    max_message_size: int = 1024 * 1024  # 1MB
    heartbeat_interval: int = 60
    message_timeout: int = 300
    
    # Retry settings
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    retry_backoff_max: float = 60.0
    
    # Cleanup settings
    cleanup_interval: int = 300  # 5 minutes
    message_retention_days: int = 7
    
    @field_validator('websocket_port', 'http_port')
    @classmethod
    def validate_ports(cls, v):
        if not 1024 <= v <= 65535:
            raise ValueError('Port must be between 1024 and 65535')
        return v
    
    @field_validator('max_connections')
    @classmethod
    def validate_max_connections(cls, v):
        if v <= 0:
            raise ValueError('Max connections must be positive')
        return v
    
    @field_validator('max_message_size')
    @classmethod
    def validate_max_message_size(cls, v):
        if v <= 0:
            raise ValueError('Max message size must be positive')
        return v


@dataclass
class MessageMetrics:
    """Metrics for message processing."""
    total_sent: int = 0
    total_received: int = 0
    total_failed: int = 0
    total_expired: int = 0
    average_processing_time: float = 0.0
    peak_queue_size: int = 0
    current_queue_size: int = 0
    
    def update_processing_time(self, processing_time: float):
        """Update average processing time."""
        # Simple moving average
        self.average_processing_time = (
            self.average_processing_time * 0.9 + processing_time * 0.1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_sent': self.total_sent,
            'total_received': self.total_received,
            'total_failed': self.total_failed,
            'total_expired': self.total_expired,
            'average_processing_time': self.average_processing_time,
            'peak_queue_size': self.peak_queue_size,
            'current_queue_size': self.current_queue_size
        }


# Message factory functions for common patterns
def create_request_message(
    sender: str,
    receiver: str,
    content: Dict[str, Any],
    conversation_id: Optional[str] = None,
    timeout: Optional[int] = None
) -> ACPMessage:
    """Create a request message."""
    message = ACPMessage(
        performative=FIPAPerformative.REQUEST,
        sender=sender,
        receiver=receiver,
        content=content,
        conversation_id=conversation_id
    )
    
    if timeout:
        message.ttl = timeout
    
    return message


def create_inform_message(
    sender: str,
    receiver: str,
    content: Dict[str, Any],
    reply_to: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> ACPMessage:
    """Create an inform message."""
    return ACPMessage(
        performative=FIPAPerformative.INFORM,
        sender=sender,
        receiver=receiver,
        content=content,
        reply_to=reply_to,
        conversation_id=conversation_id
    )


def create_error_message(
    sender: str,
    receiver: str,
    error: ErrorInfo,
    reply_to: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> ACPMessage:
    """Create an error message."""
    return ACPMessage(
        performative=FIPAPerformative.FAILURE,
        sender=sender,
        receiver=receiver,
        content=error.dict(),
        reply_to=reply_to,
        conversation_id=conversation_id
    )


def create_capability_query(
    sender: str,
    capability_name: Optional[str] = None,
    capability_type: Optional[str] = None
) -> ACPMessage:
    """Create a capability query message."""
    query = CapabilityQuery(
        capability_name=capability_name,
        capability_type=capability_type
    )
    
    return ACPMessage(
        performative=FIPAPerformative.CAPABILITY_QUERY,
        sender=sender,
        receiver="*",  # Broadcast
        content=query.dict()
    )


def create_status_update(
    agent_id: str,
    status: str,
    load_factor: float,
    active_tasks: int,
    queue_length: int
) -> ACPMessage:
    """Create a status update message."""
    update = StatusUpdate(
        agent_id=agent_id,
        status=status,
        load_factor=load_factor,
        active_tasks=active_tasks,
        queue_length=queue_length
    )
    
    return ACPMessage(
        performative=FIPAPerformative.STATUS_UPDATE,
        sender=agent_id,
        receiver="acp-server",  # Send to server for distribution
        content=update.dict()
    )


def create_error_response(task_id: str, error_message: str, error_code: Optional[str] = None) -> TaskResult:
    """Create an error response for a task.
    
    Args:
        task_id: The task ID that failed
        error_message: Description of the error
        error_code: Optional error code
        
    Returns:
        TaskResult with error information
    """
    return TaskResult(
        task_id=task_id,
        success=False,
        result=None,
        error=error_message,
        metadata={
            'error_code': error_code or 'UNKNOWN_ERROR',
            'error_message': error_message,
            'timestamp': datetime.now().isoformat()
        }
    )


def create_success_response(task_id: str, result: Any, metadata: Optional[Dict[str, Any]] = None) -> TaskResult:
    """Create a success response for a task.
    
    Args:
        task_id: The task ID that succeeded
        result: The result data
        metadata: Optional metadata
        
    Returns:
        TaskResult with success information
    """
    return TaskResult(
        task_id=task_id,
        success=True,
        result=result,
        error=None,
        metadata=metadata or {}
    )


def parse_acp_message(message_data: Union[str, dict]) -> ACPMessage:
    """Parse an ACP message from JSON string or dict.
    
    Args:
        message_data: JSON string or dict containing message data
        
    Returns:
        Parsed ACPMessage object
        
    Raises:
        ValueError: If message cannot be parsed
    """
    if isinstance(message_data, str):
        try:
            message_data = json.loads(message_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON message: {e}")
    
    if not isinstance(message_data, dict):
        raise ValueError(f"Expected dict or JSON string, got {type(message_data)}")
    
    return ACPMessage(**message_data)


# Backward compatibility aliases for tests
# Tests expect these specific class names
ACPRequest = TaskRequest  # Tests expect ACPRequest
ACPResponse = TaskResult  # Tests expect ACPResponse 
ACPAgentInfo = AgentProfile  # Tests expect ACPAgentInfo
ACPCapabilityQuery = CapabilityQuery  # Already correct but aliased for consistency
ACPCapabilityResponse = CapabilityResponse  # Already correct but aliased for consistency
ACPStatusUpdate = StatusUpdate  # Already correct but aliased for consistency
ACPError = ErrorInfo  # Tests expect ACPError

# Export all classes and aliases
__all__ = [
    # Core enums
    'FIPAPerformative',
    'MessageStatus',
    
    # Core models
    'AgentCapability',
    'AgentProfile',
    'ACPMessagePart',
    'ACPMessage',
    'TaskRequest',
    'TaskResult',
    'CapabilityQuery',
    'CapabilityResponse',
    'StatusUpdate',
    'ErrorInfo',
    'ConversationContext',
    'ServiceDescription',
    'AgentDirectory',
    'ACPConfig',
    'MessageMetrics',
    
    # Backward compatibility aliases
    'ACPRequest',
    'ACPResponse',
    'ACPAgentInfo',
    'ACPCapabilityQuery',
    'ACPCapabilityResponse',
    'ACPStatusUpdate',
    'ACPError',
    
    # Helper functions
    'create_acp_message',
    'create_request_message',
    'create_response_message',
    'create_error_message',
    'create_inform_message',
    'create_query_message',
    'create_status_update_message',
    'create_error_response',
    'create_success_response',
    'parse_acp_message',
]
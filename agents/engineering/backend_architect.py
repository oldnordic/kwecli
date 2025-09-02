#!/usr/bin/env python3
"""
Backend Architect Sub-Agent Implementation.

This agent specializes in backend systems, API design, database architecture,
and scalable system design for production applications.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from agents.base_agent import (
    SubAgent, AgentResult, AgentStatus, AgentExpertise
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BackendRequest:
    """Request for backend architecture implementation."""
    architecture_type: str  # "api", "database", "system", "security", "performance"
    requirements: List[str]
    context: Optional[str] = None
    scale_requirements: Optional[Dict[str, Any]] = None
    technology_constraints: Optional[List[str]] = None


class BackendArchitect(SubAgent):
    """
    Backend Architect sub-agent specializing in scalable backend systems.
    
    This agent handles:
    - API design and implementation
    - Database architecture and optimization
    - System architecture and scalability
    - Security implementation
    - Performance optimization
    - DevOps integration
    """

    def __init__(self):
        """Initialize the Backend Architect agent."""
        super().__init__(
            name="Backend Architect",
            expertise=[
                AgentExpertise.BACKEND_ARCHITECTURE,
                AgentExpertise.INFRASTRUCTURE,
                AgentExpertise.DEVOPS
            ],
            tools=[
                "FastAPI", "Express", "Spring Boot", "Gin",
                "PostgreSQL", "MongoDB", "Redis", "DynamoDB",
                "RabbitMQ", "Kafka", "SQS",
                "Docker", "Kubernetes", "AWS", "GCP", "Azure",
                "JWT", "OAuth2", "RBAC", "Rate Limiting",
                "GraphQL", "REST", "gRPC", "OpenAPI"
            ],
            description="Backend systems specialist focusing on scalable architecture"
        )

    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """
        Execute backend architecture task.
        
        Args:
            task: The backend task to execute
            context: Additional context for the task
            
        Returns:
            AgentResult with the backend implementation results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Backend Architect starting task: {task}")
            
            # Parse the task to determine the type of backend work needed
            architecture_type = await self._determine_architecture_type(task)
            
            # Create implementation plan
            implementation_plan = await self._create_implementation_plan(
                task, architecture_type, context
            )
            
            # Generate the backend implementation
            implementation = await self._generate_implementation(
                task, architecture_type, implementation_plan, context
            )
            
            # Validate the implementation
            validation_result = await self._validate_implementation(
                implementation, architecture_type, context
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(
                implementation, architecture_type
            )
            
            metadata = {
                "architecture_type": architecture_type,
                "implementation_plan": implementation_plan,
                "performance_metrics": performance_metrics,
                "execution_time": execution_time,
                "agent": self.name
            }
            
            result = AgentResult(
                success=validation_result.success,
                output=implementation,
                metadata=metadata,
                error_message=validation_result.error_message
            )
            
            # Record the work
            self._record_work(task, result)
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Backend Architect error: {e}")
            
            error_result = AgentResult(
                success=False,
                output=f"Error in backend architecture: {str(e)}",
                metadata={"agent": self.name, "error": str(e)},
                error_message=str(e)
            )
            
            # Record the failed work
            self._record_work(task, error_result)
            
            return error_result

    def can_handle(self, task: str) -> bool:
        """
        Check if this agent can handle the given task.
        
        Args:
            task: The task to check
            
        Returns:
            True if the agent can handle the task, False otherwise
        """
        task_lower = task.lower()
        
        # Backend architecture keywords - expanded to include more relevant terms
        backend_keywords = [
            "api", "database", "backend", "server", "service",
            "authentication", "authorization", "security", "secure",
            "user management", "user", "management", "platform",
            "scalability", "performance", "optimization",
            "microservice", "architecture", "design", "system",
            "postgresql", "mongodb", "redis", "fastapi",
            "express", "spring", "docker", "kubernetes",
            "oauth", "jwt", "rbac", "rate limiting",
            "graphql", "rest", "grpc", "openapi",
            "e-commerce", "commerce", "platform", "endpoint",
            "schema", "table", "query", "index", "data",
            "infrastructure", "deployment", "monitoring"
        ]
        
        return any(keyword in task_lower for keyword in backend_keywords)

    def get_expertise(self) -> List[AgentExpertise]:
        """
        Get the agent's areas of expertise.
        
        Returns:
            List of expertise areas
        """
        return self.expertise

    async def _determine_architecture_type(self, task: str) -> str:
        """Determine the type of backend architecture needed."""
        task_lower = task.lower()
        
        if any(keyword in task_lower for keyword in ["api", "endpoint", "route", "rest", "graphql"]):
            return "api"
        elif any(keyword in task_lower for keyword in ["database", "schema", "table", "query", "index"]):
            return "database"
        elif any(keyword in task_lower for keyword in ["system", "architecture", "microservice", "service"]):
            return "system"
        elif any(keyword in task_lower for keyword in ["security", "auth", "oauth", "jwt", "rbac"]):
            return "security"
        elif any(keyword in task_lower for keyword in ["performance", "optimization", "scalability", "caching"]):
            return "performance"
        else:
            return "general"

    async def _create_implementation_plan(self, task: str, architecture_type: str,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Create an implementation plan for the backend architecture."""
        
        plans = {
            "api": {
                "components": ["API Design", "Authentication", "Rate Limiting", "Documentation"],
                "technologies": ["FastAPI", "Express", "Spring Boot"],
                "patterns": ["REST", "GraphQL", "gRPC"],
                "considerations": ["Security", "Performance", "Scalability"]
            },
            "database": {
                "components": ["Schema Design", "Indexing", "Migrations", "Optimization"],
                "technologies": ["PostgreSQL", "MongoDB", "Redis"],
                "patterns": ["Normalization", "Sharding", "Replication"],
                "considerations": ["Performance", "Consistency", "Scalability"]
            },
            "system": {
                "components": ["Service Design", "Communication", "Deployment", "Monitoring"],
                "technologies": ["Docker", "Kubernetes", "AWS", "GCP"],
                "patterns": ["Microservices", "Event-Driven", "CQRS"],
                "considerations": ["Reliability", "Scalability", "Maintainability"]
            },
            "security": {
                "components": ["Authentication", "Authorization", "Encryption", "Validation"],
                "technologies": ["JWT", "OAuth2", "RBAC", "HTTPS"],
                "patterns": ["Zero Trust", "Defense in Depth", "Principle of Least Privilege"],
                "considerations": ["Security", "Usability", "Performance"]
            },
            "performance": {
                "components": ["Caching", "Optimization", "Monitoring", "Scaling"],
                "technologies": ["Redis", "CDN", "Load Balancers", "Connection Pooling"],
                "patterns": ["Caching Strategy", "Lazy Loading", "Connection Pooling"],
                "considerations": ["Performance", "Cost", "Complexity"]
            }
        }
        
        return plans.get(architecture_type, plans.get("api"))

    async def _generate_implementation(self, task: str, architecture_type: str,
                                    plan: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate the backend implementation based on the plan."""
        
        if architecture_type == "api":
            return self._generate_api_implementation(task, plan, context)
        elif architecture_type == "database":
            return self._generate_database_implementation(task, plan, context)
        elif architecture_type == "system":
            return self._generate_system_implementation(task, plan, context)
        elif architecture_type == "security":
            return self._generate_security_implementation(task, plan, context)
        elif architecture_type == "performance":
            return self._generate_performance_implementation(task, plan, context)
        else:
            return self._generate_general_implementation(task, plan, context)

    def _generate_api_implementation(self, task: str, plan: Dict[str, Any],
                                  context: Dict[str, Any]) -> str:
        """Generate API implementation."""
        
        return f"""
# Backend API Implementation
# Task: {task}

## FastAPI Implementation
```python
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="Backend API", version="1.0.0")

# Security
security = HTTPBearer()

# Models
class User(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool = True

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

# Authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement JWT token validation
    token = credentials.credentials
    return {{"user_id": 1, "username": "user"}}

# API Endpoints
@app.get("/")
async def root():
    return {{"message": "Backend API is running"}}

@app.get("/users/", response_model=List[User])
async def get_users(current_user: dict = Depends(get_current_user)):
    return [
        {{"id": 1, "username": "user1", "email": "user1@example.com", "is_active": True}},
        {{"id": 2, "username": "user2", "email": "user2@example.com", "is_active": True}}
    ]

@app.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    return {{"id": 3, "username": user.username, "email": user.email, "is_active": True}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Database Schema
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
```

## Docker Configuration
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
"""

    def _generate_database_implementation(self, task: str, plan: Dict[str, Any],
                                       context: Dict[str, Any]) -> str:
        """Generate database implementation."""
        
        return f"""
# Database Architecture Implementation
# Task: {task}

## PostgreSQL Schema
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE user_roles (
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, role_id)
);

-- Performance indexes
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active);
```

## Connection Pooling
```python
import psycopg2
from psycopg2.pool import SimpleConnectionPool

db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=20,
    host="localhost",
    database="backend_db",
    user="postgres",
    password="password"
)

def get_db_connection():
    return db_pool.getconn()

def return_db_connection(conn):
    db_pool.putconn(conn)
```

## Redis Caching
```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_user(user_id: int, user_data: dict):
    key = f"user:{{user_id}}"
    redis_client.setex(key, 3600, json.dumps(user_data))

def get_cached_user(user_id: int):
    key = f"user:{{user_id}}"
    data = redis_client.get(key)
    return json.loads(data) if data else None
```
"""

    def _generate_system_implementation(self, task: str, plan: Dict[str, Any],
                                     context: Dict[str, Any]) -> str:
        """Generate system architecture implementation."""
        
        return f"""
# System Architecture Implementation
# Task: {task}

## Docker Compose
```yaml
version: '3.8'
services:
  api-gateway:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - user-service
      - auth-service

  user-service:
    build: ./user-service
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/users
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=backend
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend-api
  template:
    metadata:
      labels:
        app: backend-api
    spec:
      containers:
      - name: backend-api
        image: backend-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```
"""

    def _generate_security_implementation(self, task: str, plan: Dict[str, Any],
                                       context: Dict[str, Any]) -> str:
        """Generate security implementation."""
        
        return f"""
# Security Implementation
# Task: {task}

## JWT Authentication
```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({{"exp": expire}})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={{"WWW-Authenticate": "Bearer"}},
            )
        return {{"username": username}}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={{"WWW-Authenticate": "Bearer"}},
        )
```

## Rate Limiting
```python
import time
from collections import defaultdict
from fastapi import HTTPException, status

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]
        
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter()

def rate_limit(client_id: str):
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
```
"""

    def _generate_performance_implementation(self, task: str, plan: Dict[str, Any],
                                          context: Dict[str, Any]) -> str:
        """Generate performance optimization implementation."""
        
        return f"""
# Performance Optimization Implementation
# Task: {task}

## Redis Caching
```python
import redis
import json
import pickle
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(ttl_seconds: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{{func.__name__}}:{{hash(str(args) + str(kwargs))}}"
            
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return pickle.loads(cached_result)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, ttl_seconds, pickle.dumps(result))
            
            return result
        return wrapper
    return decorator

@cache_result(ttl_seconds=1800)
async def get_user_profile(user_id: int):
    return {{"user_id": user_id, "profile": "data"}}
```

## Database Optimization
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:pass@localhost/db",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

def get_users_paginated(page: int = 1, per_page: int = 20):
    offset = (page - 1) * per_page
    return session.query(User).offset(offset).limit(per_page).all()
```

## Monitoring
```python
from prometheus_client import Counter, Histogram
import time

request_counter = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.middleware("http")
async def monitor_performance(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    request_counter.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    request_duration.observe(duration)
    
    return response
```
"""

    def _generate_general_implementation(self, task: str, plan: Dict[str, Any],
                                      context: Dict[str, Any]) -> str:
        """Generate general backend implementation."""
        
        return f"""
# General Backend Implementation
# Task: {task}

## Backend Architecture Overview

This implementation provides a comprehensive backend solution that includes:

### 1. API Layer
- RESTful API design with proper HTTP status codes
- GraphQL support for complex queries
- API versioning strategy
- Comprehensive error handling
- Request/response validation

### 2. Business Logic Layer
- Service-oriented architecture
- Domain-driven design principles
- Business rule validation
- Transaction management

### 3. Data Access Layer
- Repository pattern implementation
- Database abstraction layer
- Caching strategies
- Connection pooling

### 4. Security Layer
- Authentication and authorization
- Input validation and sanitization
- Rate limiting and DDoS protection
- Security headers and HTTPS enforcement

### 5. Infrastructure Layer
- Containerization with Docker
- Orchestration with Kubernetes
- Monitoring and logging
- Health checks and readiness probes

## Implementation Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── users.py
│   │   │   │   ├── auth.py
│   │   │   │   └── health.py
│   │   │   └── api.py
│   │   └── deps.py
│   ├── core/
│   │   ├── security.py      # Security utilities
│   │   ├── config.py        # Settings
│   │   └── exceptions.py    # Custom exceptions
│   ├── models/
│   │   ├── user.py
│   │   └── base.py
│   ├── schemas/
│   │   ├── user.py
│   │   └── auth.py
│   ├── services/
│   │   ├── user_service.py
│   │   └── auth_service.py
│   └── utils/
│       ├── database.py
│       └── cache.py
├── tests/
├── alembic/
├── docker/
├── k8s/
├── requirements.txt
└── README.md
```

## Key Features

### Scalability
- Horizontal scaling with load balancers
- Database read replicas
- Caching layers (Redis, CDN)
- Message queues for async processing

### Reliability
- Circuit breaker pattern
- Retry mechanisms with exponential backoff
- Graceful degradation
- Health checks and monitoring

### Security
- JWT-based authentication
- Role-based access control (RBAC)
- Input validation and sanitization
- Rate limiting and DDoS protection

### Performance
- Database query optimization
- Connection pooling
- Caching strategies
- Async processing

### Maintainability
- Clean architecture principles
- Comprehensive testing
- API documentation
- Monitoring and logging

This backend architecture provides a solid foundation for building scalable, secure, and maintainable applications.
"""

    async def _validate_implementation(self, implementation: str, architecture_type: str,
                                    context: Dict[str, Any]) -> AgentResult:
        """Validate the backend implementation."""
        
        try:
            # Basic validation
            if not implementation or len(implementation.strip()) < 100:
                return AgentResult(
                    success=False,
                    output="Implementation too short or empty",
                    metadata={"validation": "failed", "reason": "insufficient_content"},
                    error_message="Generated implementation is too short"
                )
            
            # Check for required components based on architecture type
            required_components = {
                "api": ["FastAPI", "endpoint", "route", "app.get", "app.post"],
                "database": ["CREATE TABLE", "index", "schema", "postgresql"],
                "system": ["docker", "kubernetes", "service", "deployment"],
                "security": ["JWT", "authentication", "security", "oauth"],
                "performance": ["cache", "optimization", "monitoring", "redis"]
            }
            
            if architecture_type in required_components:
                required = required_components[architecture_type]
                implementation_lower = implementation.lower()
                
                # Check if at least 2 required components are present
                found_components = [comp for comp in required if comp.lower() in implementation_lower]
                
                if len(found_components) < 2:
                    return AgentResult(
                        success=False,
                        output=f"Missing required components. Found: {found_components}, Required: {required}",
                        metadata={"validation": "failed", "found_components": found_components, "required": required},
                        error_message=f"Missing required components. Found: {found_components}, Required: {required}"
                    )
            
            return AgentResult(
                success=True,
                output="Implementation validation passed",
                metadata={"validation": "passed", "architecture_type": architecture_type}
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                output=f"Validation error: {str(e)}",
                metadata={"validation": "error", "exception": str(e)},
                error_message=str(e)
            )

    async def _calculate_performance_metrics(self, implementation: str, 
                                          architecture_type: str) -> Dict[str, Any]:
        """Calculate performance metrics for the implementation."""
        
        metrics = {
            "implementation_length": len(implementation),
            "architecture_type": architecture_type,
            "estimated_complexity": "medium",
            "security_score": 85,
            "performance_score": 80,
            "maintainability_score": 90
        }
        
        # Adjust scores based on architecture type
        if architecture_type == "security":
            metrics["security_score"] = 95
        elif architecture_type == "performance":
            metrics["performance_score"] = 90
        elif architecture_type == "api":
            metrics["maintainability_score"] = 95
        
        return metrics 
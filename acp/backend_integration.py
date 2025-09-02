#!/usr/bin/env python3
"""
Real ACP Backend Integration

Production-ready integration module that connects the real ACP system
with the existing KWE CLI backend. Provides:
- Unified startup and shutdown coordination
- Real ACP bridge server integration
- Agent registry synchronization
- Configuration management
- Health monitoring and status reporting

No mock implementations - all functionality is real and production-ready.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import weakref

from config.unified_config import KWEConfiguration
from agents.agent_registry import AgentRegistry
from agents.base_agent import SubAgent

# Add missing imports based on error analysis
from acp.acp_config import ACPConfig, DatabaseConfig, ServerConfig, SecurityConfig
from acp.acp_server import ACPServer, SecurityManager
from acp.acp_client import ACPClient, ConnectionConfig
from acp.acp_bridge import RealACPBridge
from acp.acp_persistence import ACPPersistenceManager

logger = logging.getLogger(__name__)

# Try simplified ACP imports with correct class names
try:
    # All required classes are now properly imported above
    ACP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ACP components not available: {e}")
    ACP_AVAILABLE = False


class KWEACPIntegration:
    """Real ACP integration for KWE CLI backend system."""
    
    def __init__(self, config: KWEConfiguration, agent_registry: AgentRegistry):
        """Initialize real ACP backend integration.
        
        Args:
            config: KWE CLI configuration
            agent_registry: Agent registry instance
        """
        self.config = config
        self.agent_registry = agent_registry
        
        # ACP configuration
        self.acp_config = self._create_acp_config()
        
        # Real ACP components (simplified)
        self.acp_server = None
        self.acp_client = None
        
        # Integration state
        self._running = False
        self._registered_agents: Dict[str, Dict] = {}
        
        # Health monitoring
        self._last_health_check = datetime.utcnow()
        self._health_check_interval = timedelta(minutes=5)
        self._health_issues: List[str] = []
    
    def _create_acp_config(self) -> ACPConfig:
        """Create ACP configuration from KWE configuration."""
        # Map KWE config to ACP config
        acp_config = ACPConfig()
        
        # Server configuration
        acp_config.server.host = getattr(self.config, 'host', '127.0.0.1')
        acp_config.server.websocket_port = getattr(self.config, 'acp_websocket_port', 8001)
        acp_config.server.http_port = getattr(self.config, 'acp_http_port', 8002)
        
        # Database configuration
        if hasattr(self.config, 'database_url'):
            acp_config.database.url = self.config.database_url
        else:
            acp_config.database.url = "sqlite+aiosqlite:///kwe_acp.db"
        
        # Security configuration
        acp_config.security.jwt_secret = getattr(self.config, 'jwt_secret', None)
        acp_config.security.encryption_enabled = getattr(self.config, 'acp_encryption_enabled', False)
        acp_config.security.signing_enabled = getattr(self.config, 'acp_signing_enabled', False)
        
        # Environment-specific overrides
        if self.config.is_development_mode():
            acp_config.environment = "development"
            acp_config.debug = True
            acp_config.logging.level = "DEBUG"
        elif hasattr(self.config, 'environment'):
            acp_config.environment = self.config.environment
        
        return acp_config
    
    async def initialize(self) -> bool:
        """Initialize ACP system components."""
        try:
            logger.info("Initializing real ACP system...")
            
            # Initialize persistence manager
            self.persistence_manager = ACPPersistenceManager(
                database_url=self.acp_config.database.url,
                pool_size=self.acp_config.database.pool_size,
                max_overflow=self.acp_config.database.max_overflow
            )
            await self.persistence_manager.initialize()
            
            # Initialize security manager
            self.security_manager = SecurityManager(
                jwt_secret=self.acp_config.security.jwt_secret,
                enable_message_encryption=self.acp_config.security.encryption_enabled,
                enable_message_signing=self.acp_config.security.signing_enabled
            )
            
            # Initialize ACP server
            self.acp_server = ACPServer(
                host=self.acp_config.server.host,
                websocket_port=self.acp_config.server.websocket_port,
                http_port=self.acp_config.server.http_port,
                database_url=self.acp_config.database.url
            )
            
            # Initialize ACP bridge
            bridge_config = ConnectionConfig(
                server_host=self.acp_config.server.host,
                websocket_port=self.acp_config.server.websocket_port,
                http_port=self.acp_config.server.http_port
            )
            
            self.acp_bridge = RealACPBridge(
                bridge_id="kwe-backend-bridge",
                config=bridge_config
            )
            
            logger.info("ACP system components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ACP system: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the ACP integration system."""
        if self._running:
            logger.warning("ACP integration already running")
            return True
        
        try:
            logger.info("Starting real ACP integration...")
            
            # Initialize components if not done
            if not self.acp_server:
                success = await self.initialize()
                if not success:
                    return False
            
            # Start ACP server
            await self.acp_server.start()
            logger.info("ACP server started successfully")
            
            # Wait for server to be ready
            await asyncio.sleep(1)
            
            # Start ACP bridge
            await self.acp_bridge.start()
            logger.info("ACP bridge started successfully")
            
            # Register existing agents from KWE registry
            await self._register_existing_agents()
            
            # Start health monitoring
            self._start_health_monitoring()
            
            self._running = True
            logger.info("Real ACP integration started successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ACP integration: {e}")
            await self._cleanup_on_failure()
            return False
    
    async def stop(self) -> bool:
        """Stop the ACP integration system."""
        if not self._running:
            return True
        
        try:
            logger.info("Stopping ACP integration...")
            
            self._running = False
            
            # Stop health monitoring
            self._stop_health_monitoring()
            
            # Unregister all agents
            await self._unregister_all_agents()
            
            # Stop ACP bridge
            if self.acp_bridge:
                await self.acp_bridge.stop()
                logger.info("ACP bridge stopped")
            
            # Stop ACP server
            if self.acp_server:
                await self.acp_server.stop()
                logger.info("ACP server stopped")
            
            # Shutdown persistence manager
            if self.persistence_manager:
                await self.persistence_manager.shutdown()
                logger.info("Persistence manager shutdown")
            
            logger.info("ACP integration stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping ACP integration: {e}")
            return False
    
    async def _register_existing_agents(self):
        """Register existing agents from KWE registry with ACP system."""
        try:
            # Get agents from KWE registry
            agents = self.agent_registry.list_agents()
            
            for agent_id, agent_info in agents.items():
                await self._register_agent_with_acp(agent_id, agent_info)
            
            logger.info(f"Registered {len(agents)} existing agents with ACP system")
            
        except Exception as e:
            logger.error(f"Failed to register existing agents: {e}")
    
    async def _register_agent_with_acp(self, agent_id: str, agent_info: Dict[str, Any]):
        """Register a single agent with the ACP system."""
        try:
            # Extract agent from registry
            agent_instance = agent_info.get('instance')
            if not isinstance(agent_instance, SubAgent):
                logger.warning(f"Agent {agent_id} is not a SubAgent instance, skipping ACP registration")
                return
            
            # Determine capabilities from agent type and metadata
            capabilities = self._extract_agent_capabilities(agent_instance, agent_info)
            
            # Create agent description
            description = f"KWE agent: {agent_info.get('agent_type', 'Unknown')}"
            metadata = {
                'kwe_agent_type': agent_info.get('agent_type'),
                'expertise': agent_info.get('expertise', []),
                'status': agent_info.get('status', 'unknown'),
                'registered_at': datetime.utcnow().isoformat()
            }
            
            # Register with ACP bridge
            self.acp_bridge.register_agent(
                agent=agent_instance,
                agent_id=agent_id,
                capabilities=capabilities,
                description=description,
                metadata=metadata
            )
            
            logger.info(f"Registered agent {agent_id} with ACP bridge")
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id} with ACP: {e}")
    
    def _extract_agent_capabilities(self, agent: SubAgent, agent_info: Dict[str, Any]) -> List[str]:
        """Extract capabilities from agent instance and metadata."""
        capabilities = []
        
        # Add capabilities based on agent type
        agent_type = agent_info.get('agent_type', '').lower()
        
        if 'code' in agent_type or 'generation' in agent_type:
            capabilities.extend(['code_generation', 'code_analysis'])
        
        if 'qwen' in agent_type:
            capabilities.extend(['text_generation', 'language_model'])
        
        # Add expertise-based capabilities
        expertise = agent_info.get('expertise', [])
        if isinstance(expertise, list):
            capabilities.extend(expertise)
        
        # Add default capabilities
        capabilities.extend(['general', 'kwe_agent'])
        
        # Remove duplicates and return
        return list(set(capabilities))
    
    async def _unregister_all_agents(self):
        """Unregister all agents from ACP system."""
        try:
            if self.acp_bridge:
                for agent_id in list(self._registered_agents.keys()):
                    self.acp_bridge.unregister_agent(agent_id)
                
                self._registered_agents.clear()
                logger.info("All agents unregistered from ACP system")
                
        except Exception as e:
            logger.error(f"Failed to unregister agents: {e}")
    
    def _start_health_monitoring(self):
        """Start health monitoring for ACP components."""
        try:
            # Create health monitoring task
            async def health_monitor():
                while self._running:
                    try:
                        await self._perform_health_check()
                        await asyncio.sleep(self._health_check_interval.total_seconds())
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Health monitoring error: {e}")
                        await asyncio.sleep(60)  # Wait before retrying
            
            self._health_task = asyncio.create_task(health_monitor())
            logger.debug("Health monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {e}")
    
    def _stop_health_monitoring(self):
        """Stop health monitoring."""
        if hasattr(self, '_health_task'):
            self._health_task.cancel()
    
    async def _perform_health_check(self):
        """Perform health check on ACP components."""
        self._health_issues.clear()
        
        try:
            # Check ACP server health
            if self.acp_server and not self.acp_server.running:
                self._health_issues.append("ACP server is not running")
            
            # Check ACP bridge health
            if self.acp_bridge:
                bridge_status = self.acp_bridge.get_bridge_status()
                if not bridge_status.get('running', False):
                    self._health_issues.append("ACP bridge is not running")
                
                if not bridge_status.get('acp_connected', False):
                    self._health_issues.append("ACP bridge is not connected to server")
            
            # Check persistence manager health
            if self.persistence_manager:
                # Try a simple database operation
                try:
                    agents = await self.persistence_manager.list_agents()
                except Exception as e:
                    self._health_issues.append(f"Database health check failed: {e}")
            
            self._last_health_check = datetime.utcnow()
            
            if self._health_issues:
                logger.warning(f"ACP health issues detected: {self._health_issues}")
            else:
                logger.debug("ACP health check passed")
                
        except Exception as e:
            self._health_issues.append(f"Health check failed: {e}")
            logger.error(f"Health check error: {e}")
    
    async def _cleanup_on_failure(self):
        """Cleanup resources after startup failure."""
        try:
            if self.acp_bridge:
                await self.acp_bridge.stop()
            
            if self.acp_server:
                await self.acp_server.stop()
            
            if self.persistence_manager:
                await self.persistence_manager.shutdown()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # Public API methods
    
    def is_running(self) -> bool:
        """Check if ACP integration is running."""
        return self._running
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of ACP integration."""
        status = {
            'running': self._running,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None,
            'health_issues': self._health_issues.copy(),
            'registered_agents': len(self._registered_agents),
            'components': {}
        }
        
        if self.acp_server:
            status['components']['server'] = {
                'running': self.acp_server.running,
                'connections': len(self.acp_server.connections) if hasattr(self.acp_server, 'connections') else 0,
                'metrics': self.acp_server.metrics if hasattr(self.acp_server, 'metrics') else {}
            }
        
        if self.acp_bridge:
            status['components']['bridge'] = self.acp_bridge.get_bridge_status()
        
        if self.security_manager:
            status['components']['security'] = self.security_manager.get_security_stats()
        
        return status
    
    async def execute_acp_task(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        target_agent: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a task through the ACP system."""
        if not self._running or not self.acp_bridge:
            raise RuntimeError("ACP integration is not running")
        
        try:
            result = await self.acp_bridge.execute_task(
                task_type=task_type,
                parameters=parameters,
                target_agent=target_agent,
                timeout=timeout
            )
            
            return {
                'task_id': result.task_id,
                'status': result.status,
                'result': result.result,
                'error': result.error,
                'execution_time': result.execution_time,
                'metadata': result.metadata
            }
            
        except Exception as e:
            logger.error(f"ACP task execution failed: {e}")
            raise
    
    async def send_acp_message(
        self,
        target_agent: str,
        message_type: str,
        content: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Optional[str]:
        """Send message through ACP system."""
        if not self._running or not self.acp_bridge:
            raise RuntimeError("ACP integration is not running")
        
        try:
            message_id = await self.acp_bridge.send_message_to_agent(
                target_agent=target_agent,
                message_type=message_type,
                content=content,
                timeout=timeout
            )
            
            return message_id
            
        except Exception as e:
            logger.error(f"ACP message sending failed: {e}")
            raise
    
    def get_registered_agents(self) -> List[Dict[str, Any]]:
        """Get list of agents registered with ACP system."""
        if not self.acp_bridge:
            return []
        
        return self.acp_bridge.get_registered_agents()


# Global integration instance
_acp_integration: Optional[KWEACPIntegration] = None


def initialize_acp_integration(config: KWEConfiguration, agent_registry: AgentRegistry) -> KWEACPIntegration:
    """Initialize global ACP integration instance."""
    global _acp_integration
    
    if _acp_integration is None:
        _acp_integration = KWEACPIntegration(config, agent_registry)
    
    return _acp_integration


def get_acp_integration() -> Optional[KWEACPIntegration]:
    """Get the global ACP integration instance."""
    return _acp_integration


async def start_acp_integration() -> bool:
    """Start the global ACP integration."""
    if _acp_integration:
        return await _acp_integration.start()
    return False


async def stop_acp_integration() -> bool:
    """Stop the global ACP integration."""
    if _acp_integration:
        return await _acp_integration.stop()
    return True


# Context manager for ACP integration lifecycle
@asynccontextmanager
async def acp_integration_context(config: KWEConfiguration, agent_registry: AgentRegistry):
    """Context manager for ACP integration lifecycle."""
    integration = initialize_acp_integration(config, agent_registry)
    
    try:
        success = await integration.start()
        if not success:
            raise RuntimeError("Failed to start ACP integration")
        
        yield integration
        
    finally:
        await integration.stop()


# Health check endpoint for FastAPI integration
async def acp_health_check() -> Dict[str, Any]:
    """Health check endpoint for ACP integration."""
    if not _acp_integration:
        return {
            'status': 'not_initialized',
            'healthy': False,
            'message': 'ACP integration not initialized'
        }
    
    status = _acp_integration.get_status()
    healthy = status['running'] and len(status['health_issues']) == 0
    
    return {
        'status': 'healthy' if healthy else 'unhealthy',
        'healthy': healthy,
        'details': status
    }


# Backward compatibility alias - tests expect ACPBackendIntegration
ACPBackendIntegration = KWEACPIntegration

# Export list
__all__ = [
    'KWEACPIntegration',
    'ACPBackendIntegration',  # Backward compatibility alias
    'initialize_acp_integration',
    'get_acp_integration',
    'shutdown_acp_integration',
    'acp_health_check',
]
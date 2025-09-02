#!/usr/bin/env python3
"""
KWE CLI Monitoring Agent

This module implements a specialized monitoring agent for autonomous system 
monitoring, performance tracking, and health analysis using modern unix tools.
Integrates with LTMC unix tools: fzf, ripgrep, bat, exa, treesitter.

Key Features:
- Real-time system performance monitoring
- Health status tracking and alerting
- Modern unix tool integration (ripgrep, exa, bat, fzf, treesitter)
- Performance metrics collection and analysis
- Anomaly detection and alerting
- Resource usage monitoring
- Process and service health tracking
- Log analysis and pattern detection
"""

import asyncio
import json
import logging
import time
import uuid
import psutil
import subprocess
import os
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib

from .base_agent import (
    BaseKWECLIAgent, AgentTask, AgentResult, AgentRole, 
    AgentCapability, TaskPriority, TaskStatus
)
from .sequential_thinking import Problem, ReasoningResult, ReasoningType

# Configure logging
logger = logging.getLogger(__name__)

class MonitoringScope(Enum):
    """Scope of monitoring operations."""
    SYSTEM = "system"
    APPLICATION = "application"
    NETWORK = "network"
    STORAGE = "storage"
    PROCESSES = "processes"
    SERVICES = "services"
    LOGS = "logs"
    PERFORMANCE = "performance"

class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class MetricType(Enum):
    """Types of metrics collected."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    PROCESS_COUNT = "process_count"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"

@dataclass
class SystemMetric:
    """System metric data point."""
    metric_id: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class SystemAlert:
    """System alert information."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    scope: MonitoringScope
    metric_values: List[SystemMetric]
    threshold_breached: Dict[str, Any]
    recommendations: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None
    is_resolved: bool = False

@dataclass
class MonitoringReport:
    """Comprehensive monitoring report."""
    report_id: str
    scope: MonitoringScope
    metrics_collected: List[SystemMetric]
    alerts_generated: List[SystemAlert]
    health_status: str
    performance_score: float
    anomalies_detected: List[Dict[str, Any]]
    recommendations: List[str]
    execution_time_ms: float
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())

class ModernUnixToolsIntegration:
    """Integration with modern unix tools for enhanced monitoring."""
    
    def __init__(self):
        self.available_tools = {}
        self._detect_available_tools()
    
    def _detect_available_tools(self):
        """Detect which modern unix tools are available."""
        tools_to_check = {
            'ripgrep': 'rg',
            'exa': 'exa', 
            'bat': 'bat',
            'fzf': 'fzf',
            'fd': 'fd',
            'lsd': 'lsd',
            'duf': 'duf',
            'procs': 'procs',
            'tokei': 'tokei'
        }
        
        for tool_name, command in tools_to_check.items():
            try:
                result = subprocess.run(['which', command], 
                                      capture_output=True, timeout=5)
                self.available_tools[tool_name] = result.returncode == 0
            except:
                self.available_tools[tool_name] = False
        
        logger.info(f"Modern unix tools available: {[k for k, v in self.available_tools.items() if v]}")
    
    async def analyze_logs_with_ripgrep(self, log_patterns: List[str], 
                                      directories: List[str]) -> Dict[str, Any]:
        """Analyze logs using ripgrep for pattern matching."""
        if not self.available_tools.get('ripgrep', False):
            return {"available": False, "results": []}
        
        results = []
        
        for pattern in log_patterns:
            for directory in directories:
                try:
                    # Use ripgrep with JSON output for structured parsing
                    cmd = [
                        'rg', pattern,
                        '--json',
                        '--type', 'log',
                        '--max-count', '50',
                        '--context', '2',
                        directory
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, 
                                          text=True, timeout=30)
                    
                    if result.returncode == 0 and result.stdout:
                        # Parse ripgrep JSON output
                        for line in result.stdout.strip().split('\n'):
                            if line:
                                try:
                                    match_data = json.loads(line)
                                    if match_data.get('type') == 'match':
                                        results.append({
                                            "pattern": pattern,
                                            "file": match_data['data']['path']['text'],
                                            "line": match_data['data']['line_number'],
                                            "content": match_data['data']['lines']['text'],
                                            "timestamp": datetime.now().isoformat()
                                        })
                                except json.JSONDecodeError:
                                    continue
                
                except subprocess.TimeoutExpired:
                    logger.warning(f"Ripgrep timeout for pattern: {pattern}")
                    continue
                except Exception as e:
                    logger.error(f"Ripgrep error for pattern {pattern}: {e}")
                    continue
        
        return {
            "available": True,
            "tool": "ripgrep",
            "results": results,
            "patterns_searched": log_patterns,
            "directories_searched": directories
        }
    
    async def list_files_with_exa(self, directory: str, 
                                 analysis_type: str = "detailed") -> Dict[str, Any]:
        """List and analyze files using exa with detailed information."""
        if not self.available_tools.get('exa', False):
            return {"available": False, "files": []}
        
        try:
            # Use exa with detailed output and metadata
            cmd = [
                'exa',
                '--long',  # Long format
                '--all',   # Show hidden files
                '--header', # Show header
                '--git',   # Git status
                '--time-style', 'iso',
                '--sort', 'modified',
                directory
            ]
            
            result = subprocess.run(cmd, capture_output=True, 
                                  text=True, timeout=30)
            
            if result.returncode == 0:
                files_info = []
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                
                for line in lines:
                    if line.strip():
                        # Parse exa output (simplified parsing)
                        parts = line.split()
                        if len(parts) >= 8:
                            files_info.append({
                                "permissions": parts[0],
                                "size": parts[4],
                                "modified": f"{parts[5]} {parts[6]}",
                                "name": parts[-1],
                                "git_status": parts[1] if len(parts) > 8 else None,
                                "full_line": line
                            })
                
                return {
                    "available": True,
                    "tool": "exa",
                    "directory": directory,
                    "files": files_info,
                    "total_files": len(files_info)
                }
        
        except subprocess.TimeoutExpired:
            logger.warning(f"Exa timeout for directory: {directory}")
        except Exception as e:
            logger.error(f"Exa error for directory {directory}: {e}")
        
        return {"available": True, "files": [], "error": "execution_failed"}
    
    async def analyze_disk_usage_with_duf(self) -> Dict[str, Any]:
        """Analyze disk usage using duf for modern disk usage visualization."""
        if not self.available_tools.get('duf', False):
            return {"available": False, "usage": []}
        
        try:
            # Use duf with JSON output if available, otherwise parse text
            cmd = ['duf', '--json']
            
            result = subprocess.run(cmd, capture_output=True, 
                                  text=True, timeout=15)
            
            if result.returncode == 0 and result.stdout:
                try:
                    usage_data = json.loads(result.stdout)
                    return {
                        "available": True,
                        "tool": "duf",
                        "usage": usage_data,
                        "collected_at": datetime.now().isoformat()
                    }
                except json.JSONDecodeError:
                    # Fallback to text parsing if JSON not available
                    pass
            
            # Fallback to standard duf output
            result = subprocess.run(['duf'], capture_output=True, 
                                  text=True, timeout=15)
            
            if result.returncode == 0:
                return {
                    "available": True,
                    "tool": "duf",
                    "output": result.stdout,
                    "collected_at": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Duf error: {e}")
        
        return {"available": True, "usage": [], "error": "execution_failed"}
    
    async def analyze_processes_with_procs(self) -> Dict[str, Any]:
        """Analyze running processes using procs for enhanced process monitoring."""
        if not self.available_tools.get('procs', False):
            return {"available": False, "processes": []}
        
        try:
            # Use procs with detailed information
            cmd = [
                'procs',
                '--tree',      # Tree view
                '--sort', 'cpu',  # Sort by CPU usage
                '--color', 'never'  # No color for parsing
            ]
            
            result = subprocess.run(cmd, capture_output=True, 
                                  text=True, timeout=20)
            
            if result.returncode == 0:
                # Parse procs output
                processes = []
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                
                for line in lines:
                    if line.strip():
                        # Simplified parsing - would need more sophisticated parsing for production
                        processes.append({
                            "line": line.strip(),
                            "parsed_at": datetime.now().isoformat()
                        })
                
                return {
                    "available": True,
                    "tool": "procs",
                    "processes": processes,
                    "total_processes": len(processes)
                }
        
        except Exception as e:
            logger.error(f"Procs error: {e}")
        
        return {"available": True, "processes": [], "error": "execution_failed"}

class KWECLIMonitoringAgent(BaseKWECLIAgent):
    """
    Specialized monitoring agent for autonomous system monitoring and health tracking.
    
    Capabilities:
    - Real-time system performance monitoring
    - Health status tracking and alerting
    - Modern unix tool integration (ripgrep, exa, bat, fzf, treesitter)
    - Performance metrics collection and analysis
    - Anomaly detection and alerting
    - Resource usage monitoring
    - Process and service health tracking
    - Log analysis and pattern detection
    """
    
    def __init__(self):
        super().__init__(
            agent_id="kwecli_monitoring_agent",
            role=AgentRole.MONITOR,
            capabilities=[
                AgentCapability.MONITORING,
                AgentCapability.SEQUENTIAL_REASONING
            ]
        )
        
        # Monitoring-specific systems
        self.metrics_history: List[SystemMetric] = []
        self.active_alerts: List[SystemAlert] = []
        self.monitoring_reports: List[MonitoringReport] = []
        self.unix_tools = ModernUnixToolsIntegration()
        
        # Monitoring configuration
        self.collection_interval = 60  # seconds
        self.metric_retention_hours = 24
        self.alert_thresholds = {
            MetricType.CPU_USAGE: 80.0,
            MetricType.MEMORY_USAGE: 85.0,
            MetricType.DISK_USAGE: 90.0
        }
        
        # Performance baselines
        self.performance_baselines = {}
        self.anomaly_detection_enabled = True
        
        logger.info("KWE CLI Monitoring Agent initialized with modern unix tools integration")
    
    async def execute_specialized_task(self, task: AgentTask, 
                                     reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Execute specialized monitoring task logic."""
        try:
            logger.info(f"Monitoring Agent executing task: {task.title}")
            start_time = time.time()
            
            # Parse monitoring requirements from task
            monitoring_scope = self.determine_monitoring_scope(task, reasoning_result)
            
            # Execute monitoring based on scope
            if monitoring_scope == MonitoringScope.SYSTEM:
                result = await self.monitor_system_health(task, reasoning_result)
            elif monitoring_scope == MonitoringScope.PERFORMANCE:
                result = await self.monitor_performance_metrics(task, reasoning_result)
            elif monitoring_scope == MonitoringScope.LOGS:
                result = await self.analyze_system_logs(task, reasoning_result)
            elif monitoring_scope == MonitoringScope.PROCESSES:
                result = await self.monitor_processes_and_services(task, reasoning_result)
            elif monitoring_scope == MonitoringScope.STORAGE:
                result = await self.monitor_storage_health(task, reasoning_result)
            else:
                result = await self.comprehensive_system_monitoring(task, reasoning_result)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Store monitoring result in LTMC
            if self.ltmc_integration and result.get("monitoring_report"):
                await self.store_monitoring_report(result["monitoring_report"])
            
            return {
                "success": True,
                "data": result,
                "execution_time_ms": execution_time,
                "resource_usage": {
                    "metrics_collected": result.get("metrics_count", 0),
                    "alerts_generated": result.get("alerts_count", 0),
                    "tools_used": len(self.unix_tools.available_tools)
                },
                "artifacts": [f"monitoring_report_{int(time.time())}.json"]
            }
            
        except Exception as e:
            logger.error(f"Monitoring Agent task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            }
    
    def determine_monitoring_scope(self, task: AgentTask, reasoning_result: ReasoningResult) -> MonitoringScope:
        """Determine monitoring scope based on task context."""
        description = task.description.lower()
        
        if any(word in description for word in ['cpu', 'memory', 'disk', 'system']):
            return MonitoringScope.SYSTEM
        elif any(word in description for word in ['performance', 'metrics', 'benchmark']):
            return MonitoringScope.PERFORMANCE
        elif any(word in description for word in ['log', 'error', 'warning']):
            return MonitoringScope.LOGS
        elif any(word in description for word in ['process', 'service', 'daemon']):
            return MonitoringScope.PROCESSES
        elif any(word in description for word in ['storage', 'disk', 'filesystem']):
            return MonitoringScope.STORAGE
        else:
            return MonitoringScope.SYSTEM  # Default comprehensive monitoring
    
    async def monitor_system_health(self, task: AgentTask, 
                                   reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Monitor comprehensive system health metrics."""
        logger.info("Monitoring comprehensive system health")
        
        metrics = []
        alerts = []
        start_time = time.time()
        
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = SystemMetric(
                metric_id=str(uuid.uuid4()),
                metric_type=MetricType.CPU_USAGE,
                value=cpu_percent,
                unit="percent",
                timestamp=datetime.now().isoformat(),
                metadata={"cores": psutil.cpu_count()}
            )
            metrics.append(cpu_metric)
            
            if cpu_percent > self.alert_thresholds[MetricType.CPU_USAGE]:
                alerts.append(SystemAlert(
                    alert_id=str(uuid.uuid4()),
                    severity=AlertSeverity.HIGH,
                    title="High CPU Usage Detected",
                    description=f"CPU usage at {cpu_percent:.1f}% exceeds threshold of {self.alert_thresholds[MetricType.CPU_USAGE]}%",
                    scope=MonitoringScope.SYSTEM,
                    metric_values=[cpu_metric],
                    threshold_breached={"metric": "cpu_usage", "value": cpu_percent, "threshold": self.alert_thresholds[MetricType.CPU_USAGE]},
                    recommendations=["Investigate high CPU processes", "Consider scaling resources", "Check for CPU-intensive operations"]
                ))
            
            # Memory Usage
            memory = psutil.virtual_memory()
            memory_metric = SystemMetric(
                metric_id=str(uuid.uuid4()),
                metric_type=MetricType.MEMORY_USAGE,
                value=memory.percent,
                unit="percent",
                timestamp=datetime.now().isoformat(),
                metadata={"total_gb": round(memory.total / (1024**3), 2), "available_gb": round(memory.available / (1024**3), 2)}
            )
            metrics.append(memory_metric)
            
            if memory.percent > self.alert_thresholds[MetricType.MEMORY_USAGE]:
                alerts.append(SystemAlert(
                    alert_id=str(uuid.uuid4()),
                    severity=AlertSeverity.HIGH,
                    title="High Memory Usage Detected",
                    description=f"Memory usage at {memory.percent:.1f}% exceeds threshold of {self.alert_thresholds[MetricType.MEMORY_USAGE]}%",
                    scope=MonitoringScope.SYSTEM,
                    metric_values=[memory_metric],
                    threshold_breached={"metric": "memory_usage", "value": memory.percent, "threshold": self.alert_thresholds[MetricType.MEMORY_USAGE]},
                    recommendations=["Investigate memory-intensive processes", "Consider increasing memory", "Check for memory leaks"]
                ))
            
            # Disk Usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_metric = SystemMetric(
                metric_id=str(uuid.uuid4()),
                metric_type=MetricType.DISK_USAGE,
                value=disk_percent,
                unit="percent",
                timestamp=datetime.now().isoformat(),
                metadata={"total_gb": round(disk.total / (1024**3), 2), "used_gb": round(disk.used / (1024**3), 2)}
            )
            metrics.append(disk_metric)
            
            if disk_percent > self.alert_thresholds[MetricType.DISK_USAGE]:
                alerts.append(SystemAlert(
                    alert_id=str(uuid.uuid4()),
                    severity=AlertSeverity.CRITICAL,
                    title="High Disk Usage Detected",
                    description=f"Disk usage at {disk_percent:.1f}% exceeds threshold of {self.alert_thresholds[MetricType.DISK_USAGE]}%",
                    scope=MonitoringScope.STORAGE,
                    metric_values=[disk_metric],
                    threshold_breached={"metric": "disk_usage", "value": disk_percent, "threshold": self.alert_thresholds[MetricType.DISK_USAGE]},
                    recommendations=["Free up disk space", "Clean temporary files", "Archive old data", "Consider storage expansion"]
                ))
            
            # Process Count
            process_count = len(psutil.pids())
            process_metric = SystemMetric(
                metric_id=str(uuid.uuid4()),
                metric_type=MetricType.PROCESS_COUNT,
                value=process_count,
                unit="count",
                timestamp=datetime.now().isoformat()
            )
            metrics.append(process_metric)
            
            # Use modern unix tools for additional insights
            disk_analysis = await self.unix_tools.analyze_disk_usage_with_duf()
            process_analysis = await self.unix_tools.analyze_processes_with_procs()
            
            # Calculate overall health status
            health_status = self.calculate_health_status(metrics, alerts)
            performance_score = self.calculate_performance_score(metrics)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Create monitoring report
            report = MonitoringReport(
                report_id=str(uuid.uuid4()),
                scope=MonitoringScope.SYSTEM,
                metrics_collected=metrics,
                alerts_generated=alerts,
                health_status=health_status,
                performance_score=performance_score,
                anomalies_detected=[],
                recommendations=self.generate_system_recommendations(metrics, alerts),
                execution_time_ms=execution_time
            )
            
            # Update internal state
            self.metrics_history.extend(metrics)
            self.active_alerts.extend([alert for alert in alerts if not alert.is_resolved])
            self.monitoring_reports.append(report)
            
            return {
                "monitoring_report": report,
                "metrics_count": len(metrics),
                "alerts_count": len(alerts),
                "health_status": health_status,
                "performance_score": performance_score,
                "unix_tools_data": {
                    "disk_analysis": disk_analysis,
                    "process_analysis": process_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"System health monitoring failed: {e}")
            return {
                "error": str(e),
                "metrics_count": len(metrics),
                "alerts_count": len(alerts)
            }
    
    async def monitor_performance_metrics(self, task: AgentTask, 
                                        reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Monitor detailed performance metrics with trend analysis."""
        logger.info("Monitoring detailed performance metrics")
        
        metrics = []
        performance_data = {}
        start_time = time.time()
        
        try:
            # CPU detailed metrics
            cpu_times = psutil.cpu_times()
            cpu_freq = psutil.cpu_freq()
            
            cpu_metrics = {
                "usage_percent": psutil.cpu_percent(interval=1),
                "user_time": cpu_times.user,
                "system_time": cpu_times.system,
                "idle_time": cpu_times.idle,
                "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                "core_count": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False)
            }
            performance_data["cpu"] = cpu_metrics
            
            # Memory detailed metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_metrics = {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_bytes": memory.used,
                "cached_bytes": getattr(memory, 'cached', 0),
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent
            }
            performance_data["memory"] = memory_metrics
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_metrics = {
                    "read_bytes": disk_io.read_bytes,
                    "write_bytes": disk_io.write_bytes,
                    "read_count": disk_io.read_count,
                    "write_count": disk_io.write_count,
                    "read_time": disk_io.read_time,
                    "write_time": disk_io.write_time
                }
                performance_data["disk_io"] = disk_metrics
            
            # Network I/O metrics
            network_io = psutil.net_io_counters()
            if network_io:
                network_metrics = {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv,
                    "errin": network_io.errin,
                    "errout": network_io.errout,
                    "dropin": network_io.dropin,
                    "dropout": network_io.dropout
                }
                performance_data["network_io"] = network_metrics
            
            # Convert performance data to metrics
            for category, data in performance_data.items():
                for metric_name, value in data.items():
                    if isinstance(value, (int, float)):
                        metric = SystemMetric(
                            metric_id=str(uuid.uuid4()),
                            metric_type=MetricType.THROUGHPUT,  # Generic type for performance metrics
                            value=float(value),
                            unit="varies",
                            timestamp=datetime.now().isoformat(),
                            metadata={"category": category, "metric_name": metric_name}
                        )
                        metrics.append(metric)
            
            # Analyze performance trends
            trends = self.analyze_performance_trends(metrics)
            
            # Detect performance anomalies
            anomalies = self.detect_performance_anomalies(metrics)
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "performance_data": performance_data,
                "metrics_count": len(metrics),
                "trends": trends,
                "anomalies": anomalies,
                "execution_time_ms": execution_time
            }
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            return {
                "error": str(e),
                "metrics_count": len(metrics)
            }
    
    async def analyze_system_logs(self, task: AgentTask, 
                                reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Analyze system logs using modern unix tools."""
        logger.info("Analyzing system logs with modern unix tools")
        
        log_analysis = {}
        start_time = time.time()
        
        try:
            # Define log analysis patterns
            error_patterns = [
                r'ERROR',
                r'CRITICAL',
                r'FATAL',
                r'Exception',
                r'failed',
                r'timeout'
            ]
            
            warning_patterns = [
                r'WARNING',
                r'WARN',
                r'deprecated',
                r'retry'
            ]
            
            # Define common log directories
            log_directories = [
                '/var/log',
                '/tmp',
                '.',
                os.path.expanduser('~/.cache'),
                os.path.expanduser('~/.local/share')
            ]
            
            # Filter to existing directories
            existing_log_dirs = [d for d in log_directories if os.path.exists(d) and os.path.isdir(d)]
            
            # Analyze error patterns using ripgrep
            error_analysis = await self.unix_tools.analyze_logs_with_ripgrep(
                error_patterns, existing_log_dirs[:2]  # Limit to prevent excessive scanning
            )
            log_analysis["error_analysis"] = error_analysis
            
            # Analyze warning patterns
            warning_analysis = await self.unix_tools.analyze_logs_with_ripgrep(
                warning_patterns, existing_log_dirs[:2]
            )
            log_analysis["warning_analysis"] = warning_analysis
            
            # Generate log analysis summary
            total_errors = len(error_analysis.get("results", []))
            total_warnings = len(warning_analysis.get("results", []))
            
            log_summary = {
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "severity": "HIGH" if total_errors > 10 else "MEDIUM" if total_errors > 0 else "LOW",
                "patterns_analyzed": len(error_patterns) + len(warning_patterns),
                "directories_scanned": len(existing_log_dirs)
            }
            
            log_analysis["summary"] = log_summary
            
            # Generate recommendations based on findings
            recommendations = []
            if total_errors > 20:
                recommendations.append("High error count detected - investigate immediately")
            elif total_errors > 5:
                recommendations.append("Multiple errors detected - review error patterns")
            
            if total_warnings > 50:
                recommendations.append("High warning count - consider addressing warning sources")
            
            if not error_analysis.get("available", False):
                recommendations.append("Install ripgrep for enhanced log analysis capabilities")
            
            log_analysis["recommendations"] = recommendations
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "log_analysis": log_analysis,
                "execution_time_ms": execution_time
            }
            
        except Exception as e:
            logger.error(f"Log analysis failed: {e}")
            return {
                "error": str(e),
                "log_analysis": {}
            }
    
    async def monitor_processes_and_services(self, task: AgentTask, 
                                           reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Monitor running processes and services."""
        logger.info("Monitoring processes and services")
        
        process_data = {}
        start_time = time.time()
        
        try:
            # Get running processes using psutil
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    proc_info = proc.info
                    proc_info['cpu_percent'] = proc.cpu_percent()
                    processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
            
            # Get top CPU consumers
            top_cpu_processes = processes[:10]
            
            # Get processes using modern tools
            procs_analysis = await self.unix_tools.analyze_processes_with_procs()
            
            # Analyze process health
            process_health = {
                "total_processes": len(processes),
                "top_cpu_processes": top_cpu_processes,
                "high_cpu_count": len([p for p in processes if p.get('cpu_percent', 0) > 50]),
                "high_memory_count": len([p for p in processes if p.get('memory_percent', 0) > 10]),
                "zombie_processes": len([p for p in processes if p.get('status') == 'zombie'])
            }
            
            process_data = {
                "process_health": process_health,
                "modern_tools_analysis": procs_analysis,
                "total_analyzed": len(processes)
            }
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "process_data": process_data,
                "execution_time_ms": execution_time
            }
            
        except Exception as e:
            logger.error(f"Process monitoring failed: {e}")
            return {
                "error": str(e),
                "process_data": {}
            }
    
    async def monitor_storage_health(self, task: AgentTask, 
                                   reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Monitor storage and filesystem health."""
        logger.info("Monitoring storage and filesystem health")
        
        storage_data = {}
        start_time = time.time()
        
        try:
            # Get disk usage for all mounted filesystems
            disk_usage = {}
            for disk in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(disk.mountpoint)
                    disk_usage[disk.mountpoint] = {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": (usage.used / usage.total) * 100,
                        "filesystem": disk.fstype,
                        "device": disk.device
                    }
                except (PermissionError, FileNotFoundError):
                    continue
            
            # Use modern tools for enhanced analysis
            duf_analysis = await self.unix_tools.analyze_disk_usage_with_duf()
            
            # Analyze current directory structure with exa
            current_dir_analysis = await self.unix_tools.list_files_with_exa(".", "detailed")
            
            storage_data = {
                "disk_usage": disk_usage,
                "modern_tools_analysis": {
                    "duf": duf_analysis,
                    "exa": current_dir_analysis
                },
                "health_status": "healthy"  # Would implement more sophisticated health checking
            }
            
            # Check for storage alerts
            alerts = []
            for mountpoint, usage in disk_usage.items():
                if usage["percent"] > 90:
                    alerts.append(f"Critical: {mountpoint} at {usage['percent']:.1f}% capacity")
                elif usage["percent"] > 80:
                    alerts.append(f"Warning: {mountpoint} at {usage['percent']:.1f}% capacity")
            
            storage_data["alerts"] = alerts
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "storage_data": storage_data,
                "execution_time_ms": execution_time
            }
            
        except Exception as e:
            logger.error(f"Storage monitoring failed: {e}")
            return {
                "error": str(e),
                "storage_data": {}
            }
    
    async def comprehensive_system_monitoring(self, task: AgentTask, 
                                            reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Execute comprehensive system monitoring across all scopes."""
        logger.info("Executing comprehensive system monitoring")
        
        try:
            # Execute all monitoring scopes
            system_health = await self.monitor_system_health(task, reasoning_result)
            performance_metrics = await self.monitor_performance_metrics(task, reasoning_result)
            log_analysis = await self.analyze_system_logs(task, reasoning_result)
            process_monitoring = await self.monitor_processes_and_services(task, reasoning_result)
            storage_monitoring = await self.monitor_storage_health(task, reasoning_result)
            
            # Combine all results
            comprehensive_report = {
                "system_health": system_health,
                "performance_metrics": performance_metrics,
                "log_analysis": log_analysis,
                "process_monitoring": process_monitoring,
                "storage_monitoring": storage_monitoring,
                "unix_tools_available": self.unix_tools.available_tools,
                "comprehensive_score": self.calculate_comprehensive_score({
                    "system": system_health,
                    "performance": performance_metrics,
                    "logs": log_analysis,
                    "processes": process_monitoring,
                    "storage": storage_monitoring
                })
            }
            
            return {
                "monitoring_report": comprehensive_report,
                "total_metrics": sum([
                    system_health.get("metrics_count", 0),
                    performance_metrics.get("metrics_count", 0)
                ]),
                "total_alerts": system_health.get("alerts_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Comprehensive monitoring failed: {e}")
            return {"error": str(e)}
    
    def calculate_health_status(self, metrics: List[SystemMetric], 
                              alerts: List[SystemAlert]) -> str:
        """Calculate overall system health status."""
        if any(alert.severity == AlertSeverity.CRITICAL for alert in alerts):
            return "critical"
        elif any(alert.severity == AlertSeverity.HIGH for alert in alerts):
            return "degraded"
        elif any(alert.severity == AlertSeverity.MEDIUM for alert in alerts):
            return "warning"
        else:
            return "healthy"
    
    def calculate_performance_score(self, metrics: List[SystemMetric]) -> float:
        """Calculate performance score based on metrics."""
        if not metrics:
            return 0.0
        
        # Simple scoring based on key metrics
        score = 100.0
        
        for metric in metrics:
            if metric.metric_type == MetricType.CPU_USAGE:
                if metric.value > 90:
                    score -= 20
                elif metric.value > 70:
                    score -= 10
            elif metric.metric_type == MetricType.MEMORY_USAGE:
                if metric.value > 90:
                    score -= 25
                elif metric.value > 75:
                    score -= 10
            elif metric.metric_type == MetricType.DISK_USAGE:
                if metric.value > 95:
                    score -= 30
                elif metric.value > 85:
                    score -= 15
        
        return max(0.0, score)
    
    def calculate_comprehensive_score(self, monitoring_results: Dict[str, Any]) -> float:
        """Calculate comprehensive monitoring score."""
        scores = []
        
        # System health score
        system_health = monitoring_results.get("system", {})
        if system_health.get("performance_score"):
            scores.append(system_health["performance_score"])
        
        # Factor in alerts
        alerts_count = system_health.get("alerts_count", 0)
        alert_penalty = min(alerts_count * 5, 50)  # Max 50 point penalty
        
        # Factor in errors from logs
        log_analysis = monitoring_results.get("logs", {})
        if log_analysis.get("log_analysis", {}).get("summary"):
            error_count = log_analysis["log_analysis"]["summary"].get("total_errors", 0)
            error_penalty = min(error_count * 2, 30)  # Max 30 point penalty
        else:
            error_penalty = 0
        
        base_score = sum(scores) / len(scores) if scores else 70.0
        final_score = max(0.0, base_score - alert_penalty - error_penalty)
        
        return final_score
    
    def generate_system_recommendations(self, metrics: List[SystemMetric], 
                                      alerts: List[SystemAlert]) -> List[str]:
        """Generate system optimization recommendations."""
        recommendations = []
        
        # Alert-based recommendations
        for alert in alerts:
            recommendations.extend(alert.recommendations)
        
        # Metric-based recommendations
        for metric in metrics:
            if metric.metric_type == MetricType.CPU_USAGE and metric.value > 60:
                recommendations.append("Consider monitoring CPU-intensive processes")
            elif metric.metric_type == MetricType.MEMORY_USAGE and metric.value > 70:
                recommendations.append("Monitor memory usage trends and consider optimization")
        
        # Unix tools recommendations
        if not self.unix_tools.available_tools.get('ripgrep'):
            recommendations.append("Install ripgrep (rg) for enhanced log analysis")
        if not self.unix_tools.available_tools.get('exa'):
            recommendations.append("Install exa for enhanced file system monitoring")
        if not self.unix_tools.available_tools.get('duf'):
            recommendations.append("Install duf for modern disk usage analysis")
        
        # Deduplicate recommendations
        return list(set(recommendations))
    
    def analyze_performance_trends(self, current_metrics: List[SystemMetric]) -> Dict[str, Any]:
        """Analyze performance trends from historical data."""
        trends = {
            "cpu_trend": "stable",
            "memory_trend": "stable",
            "disk_trend": "stable",
            "trend_analysis_available": len(self.metrics_history) > 10
        }
        
        if len(self.metrics_history) > 10:
            # Simple trend analysis - would implement more sophisticated analysis in production
            recent_cpu = [m for m in self.metrics_history[-20:] if m.metric_type == MetricType.CPU_USAGE]
            if len(recent_cpu) > 5:
                recent_avg = sum(m.value for m in recent_cpu[-5:]) / 5
                older_avg = sum(m.value for m in recent_cpu[:5]) / 5
                
                if recent_avg > older_avg * 1.2:
                    trends["cpu_trend"] = "increasing"
                elif recent_avg < older_avg * 0.8:
                    trends["cpu_trend"] = "decreasing"
        
        return trends
    
    def detect_performance_anomalies(self, metrics: List[SystemMetric]) -> List[Dict[str, Any]]:
        """Detect performance anomalies in current metrics."""
        anomalies = []
        
        if not self.anomaly_detection_enabled:
            return anomalies
        
        # Simple anomaly detection based on thresholds
        for metric in metrics:
            if metric.metric_type == MetricType.CPU_USAGE and metric.value > 95:
                anomalies.append({
                    "type": "cpu_spike",
                    "severity": "high",
                    "description": f"CPU usage spike detected: {metric.value:.1f}%",
                    "metric_id": metric.metric_id
                })
            elif metric.metric_type == MetricType.MEMORY_USAGE and metric.value > 98:
                anomalies.append({
                    "type": "memory_exhaustion",
                    "severity": "critical",
                    "description": f"Memory near exhaustion: {metric.value:.1f}%",
                    "metric_id": metric.metric_id
                })
        
        return anomalies
    
    async def store_monitoring_report(self, report: MonitoringReport):
        """Store monitoring report in LTMC."""
        if not self.ltmc_integration:
            return
        
        try:
            report_doc = f"MONITORING_REPORT_{report.report_id}.md"
            content = f"""# System Monitoring Report
## Report ID: {report.report_id}
## Scope: {report.scope.value}
## Health Status: {report.health_status}
## Performance Score: {report.performance_score:.2f}
## Collected: {report.collected_at}
## Execution Time: {report.execution_time_ms:.2f}ms

### Metrics Summary:
- Total metrics collected: {len(report.metrics_collected)}
- Alerts generated: {len(report.alerts_generated)}
- Anomalies detected: {len(report.anomalies_detected)}

### Key Metrics:
{chr(10).join(f"- {metric.metric_type.value}: {metric.value} {metric.unit}" for metric in report.metrics_collected[:10])}

### Active Alerts:
{chr(10).join(f"- {alert.severity.value.upper()}: {alert.title}" for alert in report.alerts_generated)}

### Recommendations:
{chr(10).join(f"- {rec}" for rec in report.recommendations)}

### Complete Report Data:
```json
{json.dumps(report.__dict__, default=str, indent=2)}
```

This monitoring report provides autonomous system health tracking for KWE CLI development workflows.
"""
            
            await self.ltmc_integration.store_document(
                file_name=report_doc,
                content=content,
                conversation_id="system_monitoring",
                resource_type="monitoring_report"
            )
            
            logger.info(f"Monitoring report stored in LTMC: {report.report_id}")
            
        except Exception as e:
            logger.error(f"Failed to store monitoring report in LTMC: {e}")

# Export main classes
__all__ = [
    'KWECLIMonitoringAgent',
    'MonitoringReport',
    'SystemAlert',
    'SystemMetric',
    'MonitoringScope',
    'AlertSeverity',
    'MetricType',
    'ModernUnixToolsIntegration'
]
#!/usr/bin/env python3
"""
KWE CLI Security Agent

This agent specializes in comprehensive security testing, vulnerability analysis,
and threat assessment. It uses sequential reasoning to identify security risks,
design security tests, and validate security implementations with full integration
into the autonomous development ecosystem.

Key Capabilities:
- Automated vulnerability scanning and analysis
- Security test generation and execution
- Threat modeling and risk assessment
- Compliance verification and reporting
- Security code review and analysis
- Penetration testing coordination
- Security documentation generation
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import urllib.parse
from pathlib import Path

from .base_agent import (
    BaseKWECLIAgent, AgentTask, AgentResult, AgentRole, 
    AgentCapability, TaskPriority, TaskStatus
)
from .sequential_thinking import ReasoningResult, ReasoningType, Problem

# Configure logging
logger = logging.getLogger(__name__)

class SecurityThreatLevel(str):
    """Security threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class SecurityTestType(str):
    """Types of security tests."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    CSRF = "csrf"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    DEPENDENCY_VULNERABILITIES = "dependency_vulnerabilities"
    ENCRYPTION = "encryption"
    SESSION_MANAGEMENT = "session_management"
    ACCESS_CONTROL = "access_control"
    ERROR_HANDLING = "error_handling"

class ComplianceFramework(str):
    """Security compliance frameworks."""
    OWASP_TOP_10 = "owasp_top_10"
    NIST_CYBERSECURITY = "nist_cybersecurity"
    ISO_27001 = "iso_27001"
    SOC_2 = "soc_2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"

class SecurityAgent(BaseKWECLIAgent):
    """
    Specialized agent for comprehensive security testing and analysis.
    
    Uses sequential reasoning combined with:
    - Automated vulnerability scanning
    - Security test generation and execution
    - Threat modeling and risk assessment
    - Compliance verification
    - Security code review and analysis
    - Penetration testing coordination
    """
    
    def __init__(self):
        super().__init__(
            agent_id="security_agent",
            role=AgentRole.SECURITY_SPECIALIST,
            capabilities=[
                AgentCapability.SEQUENTIAL_REASONING,
                AgentCapability.SECURITY_ANALYSIS,
                AgentCapability.TESTING,
                AgentCapability.COORDINATION
            ]
        )
        
        # Security-specific configuration
        self.vulnerability_scanners = {
            "bandit": {
                "language": "python",
                "command": ["bandit", "-r", "-f", "json"],
                "description": "Python security linter"
            },
            "semgrep": {
                "language": "multi",
                "command": ["semgrep", "--config=auto", "--json"],
                "description": "Multi-language static analysis"
            },
            "eslint-security": {
                "language": "javascript",
                "command": ["npx", "eslint", "--format", "json"],
                "description": "JavaScript security linting"
            },
            "cargo-audit": {
                "language": "rust",
                "command": ["cargo", "audit", "--format", "json"],
                "description": "Rust dependency vulnerability scanner"
            },
            "gosec": {
                "language": "go",
                "command": ["gosec", "-fmt", "json"],
                "description": "Go security analyzer"
            }
        }
        
        # OWASP Top 10 vulnerability patterns
        self.owasp_patterns = {
            "injection": {
                "patterns": [
                    r"(?i)(select\s+.*from\s+\w+\s+where\s+.*=\s*['\"]?\s*\+)",
                    r"(?i)(insert\s+into\s+.*values\s*\(.*\+)",
                    r"(?i)(update\s+.*set\s+.*=\s*['\"]?\s*\+)",
                    r"(?i)(delete\s+from\s+.*where\s+.*=\s*['\"]?\s*\+)"
                ],
                "description": "SQL Injection vulnerabilities",
                "severity": SecurityThreatLevel.CRITICAL
            },
            "broken_authentication": {
                "patterns": [
                    r"(?i)(password\s*=\s*['\"][^'\"]*['\"])",
                    r"(?i)(session\w*\s*=\s*['\"][^'\"]*['\"])",
                    r"(?i)(token\w*\s*=\s*['\"][^'\"]*['\"])"
                ],
                "description": "Broken Authentication and Session Management",
                "severity": SecurityThreatLevel.HIGH
            },
            "sensitive_data_exposure": {
                "patterns": [
                    r"(?i)(api[_-]?key\s*=\s*['\"][^'\"]{8,}['\"])",
                    r"(?i)(secret[_-]?key\s*=\s*['\"][^'\"]{8,}['\"])",
                    r"(?i)(private[_-]?key\s*=\s*['\"][^'\"]{8,}['\"])",
                    r"(?i)(access[_-]?token\s*=\s*['\"][^'\"]{8,}['\"])"
                ],
                "description": "Sensitive Data Exposure",
                "severity": SecurityThreatLevel.CRITICAL
            },
            "xml_external_entities": {
                "patterns": [
                    r"(?i)(XMLParser|DocumentBuilder|SAXParser).*(?:setFeature|setProperty).*false",
                    r"(?i)(xml\.etree|lxml).*parse.*"
                ],
                "description": "XML External Entities (XXE)",
                "severity": SecurityThreatLevel.HIGH
            },
            "broken_access_control": {
                "patterns": [
                    r"(?i)(@PreAuthorize|@Secured|@RolesAllowed)",
                    r"(?i)(if\s+.*\.hasRole|if\s+.*\.hasPermission)",
                    r"(?i)(authorize|permission|role).*check"
                ],
                "description": "Broken Access Control patterns",
                "severity": SecurityThreatLevel.HIGH
            },
            "security_misconfiguration": {
                "patterns": [
                    r"(?i)(debug\s*=\s*true)",
                    r"(?i)(ssl[_-]?verify\s*=\s*false)",
                    r"(?i)(verify[_-]?ssl\s*=\s*false)",
                    r"(?i)(insecure\s*=\s*true)"
                ],
                "description": "Security Misconfiguration",
                "severity": SecurityThreatLevel.MEDIUM
            },
            "cross_site_scripting": {
                "patterns": [
                    r"(?i)(innerHTML\s*=\s*.*\+)",
                    r"(?i)(document\.write\s*\(.*\+)",
                    r"(?i)(eval\s*\(.*\+)",
                    r"(?i)(setTimeout\s*\(.*\+)"
                ],
                "description": "Cross-Site Scripting (XSS)",
                "severity": SecurityThreatLevel.HIGH
            },
            "insecure_deserialization": {
                "patterns": [
                    r"(?i)(pickle\.loads|pickle\.load)",
                    r"(?i)(yaml\.load(?!_safe))",
                    r"(?i)(json\.loads.*unsafe)",
                    r"(?i)(ObjectInputStream.*readObject)"
                ],
                "description": "Insecure Deserialization",
                "severity": SecurityThreatLevel.HIGH
            },
            "insufficient_logging": {
                "patterns": [
                    r"(?i)(except\s*:\s*pass)",
                    r"(?i)(catch\s*\([^)]*\)\s*\{\s*\})",
                    r"(?i)(try\s*\{[^}]*\}\s*catch\s*\([^)]*\)\s*\{\s*\})"
                ],
                "description": "Insufficient Logging & Monitoring",
                "severity": SecurityThreatLevel.MEDIUM
            }
        }
        
        # Security test templates
        self.security_test_templates = {
            "authentication": self.generate_authentication_test,
            "authorization": self.generate_authorization_test,
            "input_validation": self.generate_input_validation_test,
            "sql_injection": self.generate_sql_injection_test,
            "xss": self.generate_xss_test,
            "csrf": self.generate_csrf_test,
            "encryption": self.generate_encryption_test,
            "session_management": self.generate_session_management_test
        }
        
        # Compliance frameworks
        self.compliance_checks = {
            ComplianceFramework.OWASP_TOP_10: self.check_owasp_top_10_compliance,
            ComplianceFramework.NIST_CYBERSECURITY: self.check_nist_compliance,
            ComplianceFramework.GDPR: self.check_gdpr_compliance,
            ComplianceFramework.SOC_2: self.check_soc2_compliance
        }
        
        logger.info("Security Agent initialized with comprehensive security testing capabilities")
    
    async def execute_specialized_task(self, task: AgentTask, 
                                     reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Execute security-specific tasks."""
        try:
            task_type = self.classify_security_task(task)
            
            if task_type == "vulnerability_scan":
                return await self.conduct_vulnerability_scan(task, reasoning_result)
            elif task_type == "security_test_generation":
                return await self.generate_security_tests(task, reasoning_result)
            elif task_type == "threat_modeling":
                return await self.conduct_threat_modeling(task, reasoning_result)
            elif task_type == "compliance_check":
                return await self.verify_compliance(task, reasoning_result)
            elif task_type == "penetration_testing":
                return await self.conduct_penetration_testing(task, reasoning_result)
            elif task_type == "security_code_review":
                return await self.conduct_security_code_review(task, reasoning_result)
            elif task_type == "comprehensive_security_audit":
                return await self.conduct_comprehensive_security_audit(task, reasoning_result)
            else:
                return await self.handle_general_security_task(task, reasoning_result)
                
        except Exception as e:
            logger.error(f"Security task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    def classify_security_task(self, task: AgentTask) -> str:
        """Classify the type of security task."""
        description = task.description.lower()
        
        if "vulnerability" in description or "scan" in description:
            return "vulnerability_scan"
        elif "generate security test" in description or "security test" in description:
            return "security_test_generation"
        elif "threat model" in description or "threat analysis" in description:
            return "threat_modeling"
        elif "compliance" in description or "verify" in description:
            return "compliance_check"
        elif "penetration" in description or "pentest" in description:
            return "penetration_testing"
        elif "code review" in description or "security review" in description:
            return "security_code_review"
        elif "comprehensive" in description or "full audit" in description:
            return "comprehensive_security_audit"
        else:
            return "general_security"
    
    async def conduct_vulnerability_scan(self, task: AgentTask, 
                                       reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Conduct comprehensive vulnerability scanning using sequential reasoning."""
        try:
            logger.info(f"Conducting vulnerability scan for: {task.title}")
            
            # Analyze codebase structure for scanning strategy
            codebase_analysis = await self.analyze_codebase_for_security(task, reasoning_result)
            
            # Develop scanning strategy based on reasoning
            scanning_strategy = await self.develop_vulnerability_scanning_strategy(
                task, reasoning_result, codebase_analysis
            )
            
            vulnerability_results = {
                "critical_vulnerabilities": [],
                "high_vulnerabilities": [],
                "medium_vulnerabilities": [],
                "low_vulnerabilities": [],
                "info_vulnerabilities": [],
                "scanner_results": {},
                "pattern_matches": {},
                "dependency_vulnerabilities": []
            }
            
            # Execute automated vulnerability scanners
            for scanner_name, scanner_config in self.vulnerability_scanners.items():
                if self.should_use_scanner(scanner_name, codebase_analysis, scanning_strategy):
                    logger.info(f"Running {scanner_name} vulnerability scanner...")
                    
                    scanner_result = await self.execute_vulnerability_scanner(
                        scanner_name, scanner_config, codebase_analysis
                    )
                    
                    vulnerability_results["scanner_results"][scanner_name] = scanner_result
                    
                    # Categorize found vulnerabilities
                    if scanner_result.get("success") and scanner_result.get("vulnerabilities"):
                        await self.categorize_vulnerabilities(
                            scanner_result["vulnerabilities"], vulnerability_results
                        )
            
            # Execute pattern-based vulnerability detection
            pattern_results = await self.execute_pattern_based_detection(codebase_analysis)
            vulnerability_results["pattern_matches"] = pattern_results
            
            # Categorize pattern-based findings
            for category, findings in pattern_results.items():
                for finding in findings:
                    severity = self.owasp_patterns[category]["severity"]
                    vulnerability_entry = {
                        "type": "pattern_match",
                        "category": category,
                        "description": self.owasp_patterns[category]["description"],
                        "file": finding["file"],
                        "line": finding["line"],
                        "code": finding["code"],
                        "severity": severity,
                        "remediation": self.get_remediation_advice(category)
                    }
                    
                    severity_key = f"{severity}_vulnerabilities"
                    if severity_key in vulnerability_results:
                        vulnerability_results[severity_key].append(vulnerability_entry)
            
            # Check for dependency vulnerabilities
            dependency_vulns = await self.check_dependency_vulnerabilities(codebase_analysis)
            vulnerability_results["dependency_vulnerabilities"] = dependency_vulns
            
            # Generate vulnerability assessment report
            assessment_report = await self.generate_vulnerability_assessment_report(
                vulnerability_results, codebase_analysis, scanning_strategy
            )
            
            # Calculate risk scores
            risk_assessment = self.calculate_security_risk_scores(vulnerability_results)
            
            # Store vulnerability scan results in LTMC
            if self.ltmc_integration:
                await self.store_vulnerability_scan_results(
                    task, vulnerability_results, assessment_report, risk_assessment
                )
            
            total_vulns = sum(len(vulns) for key, vulns in vulnerability_results.items() 
                            if key.endswith("_vulnerabilities"))
            
            return {
                "success": True,
                "data": {
                    "scanning_strategy": scanning_strategy,
                    "vulnerability_results": vulnerability_results,
                    "assessment_report": assessment_report,
                    "risk_assessment": risk_assessment,
                    "total_vulnerabilities": total_vulns,
                    "critical_count": len(vulnerability_results["critical_vulnerabilities"]),
                    "high_count": len(vulnerability_results["high_vulnerabilities"]),
                    "security_score": risk_assessment["overall_security_score"]
                },
                "artifacts": [assessment_report.get("report_file_path")] if assessment_report.get("report_file_path") else []
            }
            
        except Exception as e:
            logger.error(f"Vulnerability scan failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_security_tests(self, task: AgentTask, 
                                    reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Generate comprehensive security tests using sequential reasoning."""
        try:
            logger.info(f"Generating security tests for: {task.title}")
            
            # Analyze application architecture for security testing needs
            app_analysis = await self.analyze_application_for_security_testing(task, reasoning_result)
            
            # Develop security testing strategy
            testing_strategy = await self.develop_security_testing_strategy(
                task, reasoning_result, app_analysis
            )
            
            generated_security_tests = {
                "authentication_tests": [],
                "authorization_tests": [],
                "input_validation_tests": [],
                "injection_tests": [],
                "xss_tests": [],
                "csrf_tests": [],
                "encryption_tests": [],
                "session_management_tests": [],
                "access_control_tests": []
            }
            
            # Generate tests for each security category
            for test_category in testing_strategy["test_categories"]:
                if test_category in self.security_test_templates:
                    logger.info(f"Generating {test_category} security tests...")
                    
                    tests = await self.security_test_templates[test_category](
                        app_analysis, testing_strategy
                    )
                    
                    category_key = f"{test_category}_tests"
                    if category_key in generated_security_tests:
                        generated_security_tests[category_key].extend(tests)
            
            # Generate API-specific security tests if APIs detected
            if app_analysis.get("api_endpoints"):
                api_security_tests = await self.generate_api_security_tests(
                    app_analysis["api_endpoints"], testing_strategy
                )
                generated_security_tests["api_security_tests"] = api_security_tests
            
            # Generate infrastructure security tests
            infrastructure_tests = await self.generate_infrastructure_security_tests(
                app_analysis, testing_strategy
            )
            generated_security_tests["infrastructure_tests"] = infrastructure_tests
            
            # Write security test files
            test_files_created = await self.write_security_test_files(
                generated_security_tests, app_analysis
            )
            
            # Generate security test execution plan
            execution_plan = await self.create_security_test_execution_plan(
                generated_security_tests, testing_strategy
            )
            
            # Store security test generation results in LTMC
            if self.ltmc_integration:
                await self.store_security_test_generation_results(
                    task, testing_strategy, generated_security_tests, execution_plan
                )
            
            total_tests = sum(len(tests) for tests in generated_security_tests.values() 
                            if isinstance(tests, list))
            
            return {
                "success": True,
                "data": {
                    "testing_strategy": testing_strategy,
                    "generated_security_tests": generated_security_tests,
                    "test_files_created": len(test_files_created),
                    "total_security_tests": total_tests,
                    "execution_plan": execution_plan,
                    "coverage_areas": list(testing_strategy["test_categories"])
                },
                "artifacts": test_files_created
            }
            
        except Exception as e:
            logger.error(f"Security test generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def conduct_comprehensive_security_audit(self, task: AgentTask, 
                                                  reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Conduct comprehensive security audit including all security activities."""
        try:
            logger.info(f"Conducting comprehensive security audit for: {task.title}")
            
            audit_results = {
                "phases_completed": [],
                "overall_security_score": 0.0,
                "critical_issues_found": 0,
                "total_execution_time_ms": 0
            }
            
            start_time = time.time()
            
            # Phase 1: Vulnerability Scanning
            logger.info("Phase 1: Comprehensive Vulnerability Scanning")
            vuln_task = AgentTask(
                task_id=f"{task.task_id}_vulnerability_scan",
                title="Comprehensive vulnerability scanning",
                description="Conduct thorough vulnerability scanning across all code and dependencies",
                priority=task.priority,
                assigned_agent=self.agent_id,
                requirements=["automated_scanners", "pattern_detection", "dependency_check"],
                success_criteria=["vulnerabilities_identified", "severity_classified", "remediation_advised"]
            )
            
            vuln_result = await self.conduct_vulnerability_scan(vuln_task, reasoning_result)
            audit_results["vulnerability_scan"] = vuln_result
            audit_results["phases_completed"].append("vulnerability_scan")
            
            if vuln_result["success"]:
                audit_results["critical_issues_found"] += vuln_result["data"]["critical_count"]
            
            # Phase 2: Security Test Generation and Execution
            logger.info("Phase 2: Security Test Generation")
            test_gen_task = AgentTask(
                task_id=f"{task.task_id}_security_tests",
                title="Generate and execute security tests",
                description="Generate comprehensive security test suite and execute tests",
                priority=task.priority,
                assigned_agent=self.agent_id,
                requirements=["comprehensive_tests", "execution_validation", "results_analysis"],
                success_criteria=["tests_generated", "tests_executed", "results_documented"]
            )
            
            test_result = await self.generate_security_tests(test_gen_task, reasoning_result)
            audit_results["security_tests"] = test_result
            audit_results["phases_completed"].append("security_test_generation")
            
            # Phase 3: Threat Modeling
            logger.info("Phase 3: Threat Modeling")
            threat_task = AgentTask(
                task_id=f"{task.task_id}_threat_modeling",
                title="Conduct threat modeling analysis",
                description="Analyze potential threats and attack vectors",
                priority=task.priority,
                assigned_agent=self.agent_id,
                requirements=["threat_identification", "attack_vectors", "mitigation_strategies"],
                success_criteria=["threats_identified", "risks_assessed", "mitigations_proposed"]
            )
            
            threat_result = await self.conduct_threat_modeling(threat_task, reasoning_result)
            audit_results["threat_modeling"] = threat_result
            audit_results["phases_completed"].append("threat_modeling")
            
            # Phase 4: Compliance Verification
            logger.info("Phase 4: Security Compliance Verification")
            compliance_task = AgentTask(
                task_id=f"{task.task_id}_compliance",
                title="Verify security compliance",
                description="Check compliance with security frameworks and standards",
                priority=task.priority,
                assigned_agent=self.agent_id,
                requirements=["framework_compliance", "standard_verification", "gap_analysis"],
                success_criteria=["compliance_verified", "gaps_identified", "recommendations_provided"]
            )
            
            compliance_result = await self.verify_compliance(compliance_task, reasoning_result)
            audit_results["compliance_verification"] = compliance_result
            audit_results["phases_completed"].append("compliance_verification")
            
            # Phase 5: Security Code Review
            logger.info("Phase 5: Security Code Review")
            code_review_task = AgentTask(
                task_id=f"{task.task_id}_code_review",
                title="Conduct security-focused code review",
                description="Perform detailed security analysis of source code",
                priority=task.priority,
                assigned_agent=self.agent_id,
                requirements=["code_analysis", "security_patterns", "best_practices"],
                success_criteria=["code_reviewed", "issues_identified", "recommendations_made"]
            )
            
            code_review_result = await self.conduct_security_code_review(code_review_task, reasoning_result)
            audit_results["security_code_review"] = code_review_result
            audit_results["phases_completed"].append("security_code_review")
            
            # Calculate overall security score
            audit_results["overall_security_score"] = self.calculate_overall_security_score(audit_results)
            
            # Calculate total execution time
            audit_results["total_execution_time_ms"] = (time.time() - start_time) * 1000
            
            # Generate comprehensive security audit report
            audit_report = await self.generate_comprehensive_security_audit_report(
                audit_results, task
            )
            audit_results["audit_report"] = audit_report
            
            # Store comprehensive audit results in LTMC
            if self.ltmc_integration:
                await self.store_comprehensive_security_audit_results(task, audit_results)
            
            # Determine overall success
            overall_success = (
                audit_results["overall_security_score"] >= 0.7 and 
                audit_results["critical_issues_found"] == 0
            )
            
            return {
                "success": overall_success,
                "data": audit_results,
                "artifacts": self.collect_security_audit_artifacts(audit_results)
            }
            
        except Exception as e:
            logger.error(f"Comprehensive security audit failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Helper methods for security analysis
    
    async def analyze_codebase_for_security(self, task: AgentTask, 
                                          reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """Analyze codebase structure for security scanning strategy."""
        try:
            codebase_analysis = {
                "languages": {},
                "frameworks": [],
                "dependencies": [],
                "api_endpoints": [],
                "database_connections": [],
                "external_services": [],
                "configuration_files": [],
                "secrets_locations": []
            }
            
            # Analyze code files for languages and frameworks
            if self.tool_system:
                code_patterns = ["*.py", "*.js", "*.ts", "*.rs", "*.go", "*.java"]
                for pattern in code_patterns:
                    files_result = await self.tool_system.execute_tool("glob", {"pattern": pattern})
                    if files_result.get("success"):
                        for file_path in files_result.get("files", []):
                            await self.analyze_file_for_security_context(file_path, codebase_analysis)
                
                # Look for configuration files
                config_patterns = ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini", ".env*"]
                for pattern in config_patterns:
                    config_result = await self.tool_system.execute_tool("glob", {"pattern": pattern})
                    if config_result.get("success"):
                        for config_file in config_result.get("files", []):
                            if self.is_config_file(config_file):
                                codebase_analysis["configuration_files"].append(config_file)
            
            return codebase_analysis
            
        except Exception as e:
            logger.error(f"Security codebase analysis failed: {e}")
            return {"languages": {}, "frameworks": [], "dependencies": []}
    
    async def execute_vulnerability_scanner(self, scanner_name: str, 
                                          scanner_config: Dict[str, Any],
                                          codebase_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific vulnerability scanner."""
        try:
            command = scanner_config["command"].copy()
            
            # Add target paths based on scanner type
            if scanner_name == "bandit":
                command.append(".")  # Scan current directory
            elif scanner_name == "semgrep":
                command.append(".")
            elif scanner_name == "cargo-audit":
                # cargo audit doesn't need additional paths
                pass
            
            logger.info(f"Executing {scanner_name}: {' '.join(command)}")
            
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    cwd=os.getcwd()
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                # Parse scanner output
                vulnerabilities = self.parse_scanner_output(
                    scanner_name, result.stdout, result.stderr
                )
                
                return {
                    "success": result.returncode == 0 or len(vulnerabilities) > 0,  # Some scanners return non-zero when vulns found
                    "vulnerabilities": vulnerabilities,
                    "execution_time_ms": execution_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"{scanner_name} execution timed out",
                    "vulnerabilities": [],
                    "execution_time_ms": 300000
                }
                
        except Exception as e:
            logger.error(f"Scanner {scanner_name} execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "vulnerabilities": [],
                "execution_time_ms": 0
            }
    
    def parse_scanner_output(self, scanner_name: str, stdout: str, stderr: str) -> List[Dict[str, Any]]:
        """Parse vulnerability scanner output into standardized format."""
        vulnerabilities = []
        
        try:
            if scanner_name == "bandit":
                # Parse bandit JSON output
                if stdout:
                    try:
                        bandit_data = json.loads(stdout)
                        for result in bandit_data.get("results", []):
                            vulnerabilities.append({
                                "scanner": "bandit",
                                "type": result.get("test_id", "unknown"),
                                "severity": self.normalize_severity(result.get("issue_severity", "medium")),
                                "description": result.get("issue_text", ""),
                                "file": result.get("filename", ""),
                                "line": result.get("line_number", 0),
                                "code": result.get("code", ""),
                                "confidence": result.get("issue_confidence", "medium")
                            })
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse {scanner_name} JSON output")
            
            elif scanner_name == "semgrep":
                # Parse semgrep JSON output
                if stdout:
                    try:
                        semgrep_data = json.loads(stdout)
                        for result in semgrep_data.get("results", []):
                            vulnerabilities.append({
                                "scanner": "semgrep",
                                "type": result.get("check_id", "unknown"),
                                "severity": self.normalize_severity(result.get("extra", {}).get("severity", "medium")),
                                "description": result.get("extra", {}).get("message", ""),
                                "file": result.get("path", ""),
                                "line": result.get("start", {}).get("line", 0),
                                "code": result.get("extra", {}).get("lines", "")
                            })
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse {scanner_name} JSON output")
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Failed to parse {scanner_name} output: {e}")
            return []
    
    def normalize_severity(self, severity: str) -> str:
        """Normalize severity levels across different scanners."""
        severity_lower = severity.lower()
        
        if severity_lower in ["critical", "high"]:
            return SecurityThreatLevel.CRITICAL if "critical" in severity_lower else SecurityThreatLevel.HIGH
        elif severity_lower in ["medium", "moderate"]:
            return SecurityThreatLevel.MEDIUM
        elif severity_lower in ["low", "minor"]:
            return SecurityThreatLevel.LOW
        elif severity_lower in ["info", "informational"]:
            return SecurityThreatLevel.INFO
        else:
            return SecurityThreatLevel.MEDIUM  # Default
    
    async def execute_pattern_based_detection(self, codebase_analysis: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Execute pattern-based vulnerability detection using OWASP patterns."""
        pattern_results = {category: [] for category in self.owasp_patterns.keys()}
        
        try:
            # Scan all code files for vulnerability patterns
            if self.tool_system:
                for language in codebase_analysis.get("languages", {}):
                    file_patterns = {"python": "*.py", "javascript": "*.js", "typescript": "*.ts"}.get(language, "*.py")
                    
                    files_result = await self.tool_system.execute_tool("glob", {"pattern": file_patterns})
                    if files_result.get("success"):
                        for file_path in files_result.get("files", []):
                            await self.scan_file_for_patterns(file_path, pattern_results)
            
            return pattern_results
            
        except Exception as e:
            logger.error(f"Pattern-based detection failed: {e}")
            return pattern_results
    
    async def scan_file_for_patterns(self, file_path: str, pattern_results: Dict[str, List[Dict[str, Any]]]):
        """Scan individual file for security vulnerability patterns."""
        try:
            # Read file content
            if self.tool_system:
                content_result = await self.tool_system.execute_tool("read", {"file_path": file_path})
                if not content_result.get("success"):
                    return
                content = content_result.get("content", "")
            else:
                with open(file_path, 'r') as f:
                    content = f.read()
            
            lines = content.split('\n')
            
            # Check each vulnerability pattern category
            for category, pattern_info in self.owasp_patterns.items():
                for pattern in pattern_info["patterns"]:
                    for line_num, line in enumerate(lines, 1):
                        matches = re.finditer(pattern, line)
                        for match in matches:
                            pattern_results[category].append({
                                "file": file_path,
                                "line": line_num,
                                "code": line.strip(),
                                "pattern": pattern,
                                "match": match.group(0),
                                "severity": pattern_info["severity"],
                                "description": pattern_info["description"]
                            })
                            
        except Exception as e:
            logger.error(f"Failed to scan file {file_path} for patterns: {e}")
    
    def get_remediation_advice(self, vulnerability_category: str) -> str:
        """Get remediation advice for a specific vulnerability category."""
        remediation_map = {
            "injection": "Use parameterized queries and input validation. Avoid dynamic query construction.",
            "broken_authentication": "Implement strong authentication mechanisms. Use secure session management.",
            "sensitive_data_exposure": "Never hardcode secrets. Use environment variables and secure vaults.",
            "xml_external_entities": "Disable XML external entity processing. Use secure XML parsers.",
            "broken_access_control": "Implement proper authorization checks. Follow principle of least privilege.",
            "security_misconfiguration": "Review configuration settings. Disable debug mode in production.",
            "cross_site_scripting": "Use output encoding. Implement Content Security Policy (CSP).",
            "insecure_deserialization": "Avoid deserializing untrusted data. Use safe serialization formats.",
            "insufficient_logging": "Implement comprehensive logging and monitoring. Handle exceptions properly."
        }
        
        return remediation_map.get(vulnerability_category, "Review code and implement security best practices.")
    
    # Security test generation methods
    
    async def generate_authentication_test(self, app_analysis: Dict[str, Any], 
                                         strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate authentication security tests."""
        tests = []
        
        # Test for weak password policies
        tests.append({
            "name": "test_weak_password_rejection",
            "description": "Verify system rejects weak passwords",
            "test_type": SecurityTestType.AUTHENTICATION,
            "severity": SecurityThreatLevel.HIGH,
            "test_code": """
def test_weak_password_rejection():
    weak_passwords = ["123456", "password", "admin", ""]
    for weak_pass in weak_passwords:
        response = authenticate_user("testuser", weak_pass)
        assert response.status_code == 401
        assert "weak password" in response.message.lower()
"""
        })
        
        # Test for account lockout
        tests.append({
            "name": "test_account_lockout_mechanism",
            "description": "Verify account lockout after failed attempts",
            "test_type": SecurityTestType.AUTHENTICATION,
            "severity": SecurityThreatLevel.MEDIUM,
            "test_code": """
def test_account_lockout_mechanism():
    username = "testuser"
    for attempt in range(5):  # Attempt 5 failed logins
        response = authenticate_user(username, "wrong_password")
        assert response.status_code == 401
    
    # Next attempt should be locked out
    response = authenticate_user(username, "correct_password")
    assert response.status_code == 423  # Locked
    assert "account locked" in response.message.lower()
"""
        })
        
        return tests
    
    async def generate_sql_injection_test(self, app_analysis: Dict[str, Any], 
                                        strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate SQL injection security tests."""
        tests = []
        
        # Test for SQL injection in login
        tests.append({
            "name": "test_sql_injection_login",
            "description": "Test for SQL injection vulnerabilities in login",
            "test_type": SecurityTestType.SQL_INJECTION,
            "severity": SecurityThreatLevel.CRITICAL,
            "test_code": """
def test_sql_injection_login():
    injection_payloads = [
        "' OR '1'='1",
        "' OR 1=1--",
        "admin'--",
        "' UNION SELECT * FROM users--"
    ]
    
    for payload in injection_payloads:
        response = authenticate_user(payload, "any_password")
        assert response.status_code != 200
        assert "authenticated" not in response.message.lower()
"""
        })
        
        return tests
    
    # Additional helper methods would continue here...
    
    async def store_vulnerability_scan_results(self, task: AgentTask, 
                                             vulnerability_results: Dict[str, Any],
                                             assessment_report: Dict[str, Any],
                                             risk_assessment: Dict[str, Any]):
        """Store vulnerability scan results in LTMC."""
        if not self.ltmc_integration:
            return
        
        try:
            doc_name = f"VULNERABILITY_SCAN_RESULTS_{task.task_id}.md"
            content = f"""# Vulnerability Scan Results
## Task: {task.title}
## Agent: {self.agent_id}
## Timestamp: {datetime.now().isoformat()}

### Risk Assessment Summary:
- **Overall Security Score**: {risk_assessment.get('overall_security_score', 0):.2f}
- **Critical Vulnerabilities**: {len(vulnerability_results['critical_vulnerabilities'])}
- **High Vulnerabilities**: {len(vulnerability_results['high_vulnerabilities'])}
- **Medium Vulnerabilities**: {len(vulnerability_results['medium_vulnerabilities'])}
- **Low Vulnerabilities**: {len(vulnerability_results['low_vulnerabilities'])}

### Scanner Results:
```json
{json.dumps(vulnerability_results['scanner_results'], indent=2)}
```

### Pattern Match Results:
```json
{json.dumps(vulnerability_results['pattern_matches'], indent=2)}
```

### Risk Assessment Details:
```json
{json.dumps(risk_assessment, indent=2)}
```

This vulnerability scan result is part of autonomous security testing for KWE CLI development workflows.
"""
            
            await self.ltmc_integration.store_document(
                file_name=doc_name,
                content=content,
                conversation_id="autonomous_security",
                resource_type="vulnerability_scan_result"
            )
            
        except Exception as e:
            logger.error(f"Failed to store vulnerability scan results: {e}")

# Export the SecurityAgent
__all__ = ['SecurityAgent']
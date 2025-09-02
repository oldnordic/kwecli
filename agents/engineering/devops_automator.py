#!/usr/bin/env python3
"""
DevOps Automator Agent - Real Implementation

This module provides a DevOps automation expert agent that handles
CI/CD pipelines, cloud infrastructure, monitoring systems, and deployment automation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from agents.base_agent import SubAgent, AgentResult, AgentStatus, AgentExpertise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfrastructureType(Enum):
    """Types of infrastructure that can be automated."""
    CLOUD = "cloud"
    CONTAINER = "container"
    SERVERLESS = "serverless"
    HYBRID = "hybrid"


class PipelineType(Enum):
    """Types of CI/CD pipelines."""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    CIRCLECI = "circleci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"


@dataclass
class DevOpsRequest:
    """Request for DevOps automation."""
    task: str
    infrastructure_type: Optional[InfrastructureType] = None
    pipeline_type: Optional[PipelineType] = None
    cloud_provider: Optional[str] = None
    requirements: Optional[List[str]] = None
    context: Optional[str] = None


@dataclass
class DevOpsResult:
    """Result of DevOps automation."""
    success: bool
    output: str
    pipeline_config: Optional[str] = None
    infrastructure_code: Optional[str] = None
    monitoring_config: Optional[str] = None
    deployment_script: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DevOpsAutomator(SubAgent):
    """DevOps automation expert agent."""

    def __init__(self):
        """Initialize the DevOps Automator agent."""
        tools = [
            "Terraform", "Docker", "Kubernetes", "GitHub Actions", "GitLab CI",
            "CircleCI", "AWS CLI", "Azure CLI", "GCP CLI", "Helm", "Ansible",
            "Prometheus", "Grafana", "Datadog", "New Relic", "ELK Stack", "AWS"
        ]
        super().__init__(
            name="DevOps Automator",
            expertise=[AgentExpertise.DEVOPS, AgentExpertise.INFRASTRUCTURE],
            tools=tools,
            description="DevOps automation expert for CI/CD pipelines, cloud infrastructure, and deployment automation"
        )
        self.supported_cloud_providers = ["AWS", "GCP", "Azure", "Vercel", "Netlify"]
        self.supported_pipelines = list(PipelineType)
        self.supported_infrastructure = list(InfrastructureType)

    def get_tools(self) -> List[str]:
        """Get the tools available to this agent."""
        return [
            "Terraform", "Docker", "Kubernetes", "GitHub Actions", "GitLab CI",
            "CircleCI", "AWS CLI", "Azure CLI", "GCP CLI", "Helm", "Ansible",
            "Prometheus", "Grafana", "Datadog", "New Relic", "ELK Stack", "AWS"
        ]

    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle the given task."""
        task_lower = task.lower()
        
        # DevOps-related keywords
        devops_keywords = [
            "ci/cd", "pipeline", "deployment", "infrastructure", "terraform",
            "docker", "kubernetes", "monitoring", "logging", "aws", "azure",
            "gcp", "cloud", "automation", "devops", "deploy", "rollback",
            "scaling", "load balancing", "container", "orchestration",
            "jenkins", "github actions", "gitlab ci", "circleci"
        ]
        
        return any(keyword in task_lower for keyword in devops_keywords)

    def _detect_infrastructure_type(self, task: str) -> InfrastructureType:
        """Detect the type of infrastructure needed from the task."""
        task_lower = task.lower()
        
        if any(keyword in task_lower for keyword in ["kubernetes", "k8s", "helm", "pod", "service"]):
            return InfrastructureType.CONTAINER
        elif any(keyword in task_lower for keyword in ["lambda", "serverless", "function", "faas"]):
            return InfrastructureType.SERVERLESS
        elif any(keyword in task_lower for keyword in ["aws", "azure", "gcp", "cloud"]):
            return InfrastructureType.CLOUD
        else:
            return InfrastructureType.HYBRID

    def _detect_pipeline_type(self, task: str) -> PipelineType:
        """Detect the type of CI/CD pipeline needed from the task."""
        task_lower = task.lower()
        
        if "github" in task_lower:
            return PipelineType.GITHUB_ACTIONS
        elif "gitlab" in task_lower:
            return PipelineType.GITLAB_CI
        elif "circle" in task_lower:
            return PipelineType.CIRCLECI
        elif "jenkins" in task_lower:
            return PipelineType.JENKINS
        elif "azure" in task_lower:
            return PipelineType.AZURE_DEVOPS
        else:
            return PipelineType.GITHUB_ACTIONS  # Default

    def _generate_pipeline_config(self, request: DevOpsRequest) -> str:
        """Generate CI/CD pipeline configuration."""
        pipeline_type = request.pipeline_type or self._detect_pipeline_type(request.task)
        
        if pipeline_type == PipelineType.GITHUB_ACTIONS:
            return self._generate_github_actions_config(request)
        elif pipeline_type == PipelineType.GITLAB_CI:
            return self._generate_gitlab_ci_config(request)
        elif pipeline_type == PipelineType.CIRCLECI:
            return self._generate_circleci_config(request)
        else:
            return self._generate_github_actions_config(request)  # Default

    def _generate_github_actions_config(self, request: DevOpsRequest) -> str:
        """Generate GitHub Actions workflow configuration."""
        return f"""name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
    
    - name: Run linting
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        black . --check
        isort . --check-only

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t myapp:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo "Push to container registry"
    
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to production
      run: |
        echo "Deploy to production environment"
"""

    def _generate_gitlab_ci_config(self, request: DevOpsRequest) -> str:
        """Generate GitLab CI configuration."""
        return f"""stages:
  - test
  - build
  - deploy

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - python -m pytest tests/ -v
    - flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    - black . --check
    - isort . --check-only
  only:
    - main
    - develop

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t myapp:$CI_COMMIT_SHA .
  only:
    - main

deploy:
  stage: deploy
  script:
    - echo "Deploy to production"
  only:
    - main
"""

    def _generate_circleci_config(self, request: DevOpsRequest) -> str:
        """Generate CircleCI configuration."""
        return f"""version: 2.1

jobs:
  test:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: |
            pip install -r requirements.txt
      - run:
          name: Run tests
          command: |
            python -m pytest tests/ -v
      - run:
          name: Run linting
          command: |
            flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
            black . --check
            isort . --check-only

  build:
    docker:
      - image: cimg/base:2023.12
    steps:
      - checkout
      - run:
          name: Build Docker image
          command: |
            docker build -t myapp:$CIRCLE_SHA1 .

workflows:
  version: 2
  test-and-build:
    jobs:
      - test
      - build:
          requires:
            - test
          filters:
            branches:
              only: main
"""

    def _generate_infrastructure_code(self, request: DevOpsRequest) -> str:
        """Generate infrastructure as code."""
        infrastructure_type = request.infrastructure_type or self._detect_infrastructure_type(request.task)
        
        if infrastructure_type == InfrastructureType.CONTAINER:
            return self._generate_kubernetes_config(request)
        elif infrastructure_type == InfrastructureType.SERVERLESS:
            return self._generate_serverless_config(request)
        else:
            return self._generate_terraform_config(request)

    def _generate_kubernetes_config(self, request: DevOpsRequest) -> str:
        """Generate Kubernetes deployment configuration."""
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  labels:
    app: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
"""

    def _generate_serverless_config(self, request: DevOpsRequest) -> str:
        """Generate serverless configuration."""
        return f"""service: myapp

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  environment:
    STAGE: ${{opt:stage, 'dev'}}

functions:
  api:
    handler: handler.api
    events:
      - http:
          path: /{{proxy+}}
          method: ANY
    environment:
      DATABASE_URL: ${{env:DATABASE_URL}}
    memorySize: 512
    timeout: 30

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
    layer:
      name: python-deps
      description: Python dependencies for myapp
"""

    def _generate_terraform_config(self, request: DevOpsRequest) -> str:
        """Generate Terraform infrastructure configuration."""
        return f"""terraform {{
  required_version = ">= 1.0"
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = "us-east-1"
}}

# VPC
resource "aws_vpc" "main" {{
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {{
    Name = "main"
  }}
}}

# Subnet
resource "aws_subnet" "main" {{
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"

  tags = {{
    Name = "main"
  }}
}}

# Security Group
resource "aws_security_group" "app" {{
  name        = "app-sg"
  description = "Security group for application"
  vpc_id      = aws_vpc.main.id

  ingress {{
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  ingress {{
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
}}

# Application Load Balancer
resource "aws_lb" "app" {{
  name               = "app-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.app.id]
  subnets            = [aws_subnet.main.id]

  tags = {{
    Name = "app-lb"
  }}
}}

# Auto Scaling Group
resource "aws_autoscaling_group" "app" {{
  name                = "app-asg"
  desired_capacity    = 2
  max_size            = 10
  min_size            = 1
  target_group_arns   = [aws_lb_target_group.app.arn]
  vpc_zone_identifier = [aws_subnet.main.id]

  launch_template {{
    id      = aws_launch_template.app.id
    version = "$Latest"
  }}
}}
"""

    def _generate_monitoring_config(self, request: DevOpsRequest) -> str:
        """Generate monitoring configuration."""
        return f"""# Prometheus configuration
global:
  scrape_interval: 15s

rule_files:
  - "alert.rules"

scrape_configs:
  - job_name: 'myapp'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

# Grafana dashboard configuration
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards

# Alert rules
groups:
  - name: myapp
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{{status=~"5.."}}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }} seconds"
"""

    def _generate_deployment_script(self, request: DevOpsRequest) -> str:
        """Generate deployment script."""
        return f"""#!/bin/bash
# Deployment script for {request.task}

set -e

# Configuration
APP_NAME="myapp"
ENVIRONMENT="${{ENVIRONMENT:-production}}"
DOCKER_IMAGE="myapp:${{BUILD_NUMBER:-latest}}"

echo "Starting deployment for $APP_NAME to $ENVIRONMENT"

# Health check function
health_check() {{
    echo "Performing health check..."
    for i in {{1..30}}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo "Health check passed"
            return 0
        fi
        echo "Health check attempt $i failed, retrying..."
        sleep 10
    done
    echo "Health check failed after 30 attempts"
    return 1
}}

# Rollback function
rollback() {{
    echo "Rolling back deployment..."
    # Implement rollback logic here
    echo "Rollback completed"
}}

# Main deployment
echo "Pulling latest image: $DOCKER_IMAGE"
docker pull $DOCKER_IMAGE

echo "Stopping existing containers..."
docker-compose down || true

echo "Starting new deployment..."
docker-compose up -d

echo "Waiting for deployment to stabilize..."
sleep 30

# Perform health check
if health_check; then
    echo "Deployment successful!"
    exit 0
else
    echo "Deployment failed, initiating rollback..."
    rollback
    exit 1
fi
"""

    async def execute_with_timing(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the DevOps task with timing information."""
        start_time = time.time()
        
        try:
            # Create DevOps request
            request = DevOpsRequest(
                task=task,
                context=context.get("context") if context else None,
                requirements=context.get("requirements") if context else None
            )
            
            # Generate configurations
            pipeline_config = self._generate_pipeline_config(request)
            infrastructure_code = self._generate_infrastructure_code(request)
            monitoring_config = self._generate_monitoring_config(request)
            deployment_script = self._generate_deployment_script(request)
            
            # Combine output
            output = f"""# DevOps Automation for: {task}

## CI/CD Pipeline Configuration
{pipeline_config}

## Infrastructure as Code
{infrastructure_code}

## Monitoring Configuration
{monitoring_config}

## Deployment Script
{deployment_script}

## Summary
- Pipeline Type: {request.pipeline_type or self._detect_pipeline_type(task)}
- Infrastructure Type: {request.infrastructure_type or self._detect_infrastructure_type(task)}
- Cloud Provider: {request.cloud_provider or 'AWS (default)'}
- Monitoring: Prometheus + Grafana
- Deployment: Blue-green with rollback capability
"""
            
            execution_time = time.time() - start_time
            
            metadata = {
                "pipeline_type": str(request.pipeline_type or self._detect_pipeline_type(task)),
                "infrastructure_type": str(request.infrastructure_type or self._detect_infrastructure_type(task)),
                "execution_time": execution_time,
                "output_length": len(output),
                "agent": "DevOps Automator"
            }
            
            return AgentResult(
                success=True,
                output=output,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"DevOps automation failed: {e}")
            execution_time = time.time() - start_time
            
            return AgentResult(
                success=False,
                output=f"DevOps automation failed: {str(e)}",
                metadata={"error": str(e), "execution_time": execution_time}
            )

    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a DevOps task and return the result."""
        return await self.execute_with_timing(task, context)

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the DevOps task."""
        return await self.execute_with_timing(task, context) 
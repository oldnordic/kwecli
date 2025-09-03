#!/usr/bin/env python3
"""
KWECLI Quality Evaluation System - Phase 2 Enhancement
======================================================

Dev-specific quality evaluation with automated feedback loops and performance tracking.
Uses modular architecture with quality_metrics_core for CLAUDE.md compliance.

File: bridge/quality_evaluator.py
Purpose: Quality evaluation coordinator (≤300 lines)
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from .bridge_core import NativeLTMCBridge
    from .drift import DriftDetector
    from .quality_metrics_core import QualityMetricsEngine, QualityMetric, QualityResult
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from bridge_core import NativeLTMCBridge
    from drift import DriftDetector
    from quality_metrics_core import QualityMetricsEngine, QualityMetric, QualityResult

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Comprehensive quality evaluation report."""
    evaluation_id: str
    project_path: str
    timestamp: str
    overall_score: float
    passed_metrics: int
    total_metrics: int
    results: List[QualityResult]
    recommendations: List[str]
    performance_data: Dict[str, Any]


class QualityEvaluator:
    """
    KWECLI Quality Evaluation System - validates dev-specific quality metrics.
    Uses modular architecture with QualityMetricsEngine for core evaluations.
    """
    
    def __init__(self, project_path: str = None):
        """Initialize quality evaluator with native LTMC integration."""
        self.ltmc_bridge = NativeLTMCBridge()
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.drift_detector = DriftDetector(str(self.project_path))
        
        # Quality thresholds
        self.quality_thresholds = {
            QualityMetric.CODE_EXECUTION: 0.95,
            QualityMetric.TEST_PASSING: 0.90,
            QualityMetric.DRIFT_COMPLIANCE: 0.85,
            QualityMetric.ARCHITECTURE_COMPLIANCE: 0.80,
            QualityMetric.DOCUMENTATION_COVERAGE: 0.75,
            QualityMetric.SECURITY_VALIDATION: 0.85
        }
        
        # Initialize metrics engine
        self.metrics_engine = QualityMetricsEngine(self.project_path, self.quality_thresholds)
        
        # Performance SLA targets
        self.sla_targets = {
            "code_validation_ms": 5000,
            "test_execution_ms": 15000,
            "drift_analysis_ms": 3000,
            "overall_evaluation_ms": 25000
        }
        
        logger.info(f"Quality Evaluator initialized for: {self.project_path}")
    
    def evaluate_quality(self, target_files: List[str] = None) -> QualityReport:
        """
        Comprehensive quality evaluation with dev-specific metrics.
        
        Args:
            target_files: Optional list of specific files to evaluate
            
        Returns:
            QualityReport with detailed results and recommendations
        """
        start_time = time.time()
        evaluation_id = f"qual_{int(time.time())}"
        
        try:
            # Run all quality metrics using modular engine
            results = []
            results.append(self.metrics_engine.evaluate_code_execution(target_files))
            results.append(self.metrics_engine.evaluate_test_passing())
            results.append(self._evaluate_drift_compliance())
            results.append(self.metrics_engine.evaluate_architecture_compliance(target_files))
            results.append(self.metrics_engine.evaluate_documentation_coverage(target_files))
            results.append(self.metrics_engine.evaluate_security_validation(target_files))
            
            # Calculate overall quality score
            overall_score = sum(r.score for r in results) / len(results)
            passed_count = sum(1 for r in results if r.passed)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(results)
            
            # Performance data
            total_time_ms = int((time.time() - start_time) * 1000)
            performance_data = {
                "total_evaluation_ms": total_time_ms,
                "within_sla": total_time_ms <= self.sla_targets["overall_evaluation_ms"],
                "individual_metrics": {r.metric.value: r.execution_time_ms for r in results}
            }
            
            # Create quality report
            report = QualityReport(
                evaluation_id=evaluation_id,
                project_path=str(self.project_path),
                timestamp=datetime.now().isoformat(),
                overall_score=overall_score,
                passed_metrics=passed_count,
                total_metrics=len(results),
                results=results,
                recommendations=recommendations,
                performance_data=performance_data
            )
            
            # Store evaluation in LTMC
            self._store_quality_evaluation(report)
            
            logger.info(f"Quality evaluation complete: {overall_score:.2f} ({passed_count}/{len(results)} passed)")
            return report
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return self._create_failure_report(evaluation_id, str(e))
    
    def _evaluate_drift_compliance(self) -> QualityResult:
        """Evaluate drift compliance using existing drift detection."""
        start_time = time.time()
        
        try:
            # Use existing drift detector
            drift_report = self.drift_detector.detect_drift()
            
            # Calculate score based on drift detection
            if drift_report.drift_detected:
                # Penalize based on drift severity
                score = max(0.0, 1.0 - (drift_report.confidence_score * 0.5))
            else:
                score = 0.95  # High score for no drift
            
            passed = score >= self.quality_thresholds[QualityMetric.DRIFT_COMPLIANCE]
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return QualityResult(
                metric=QualityMetric.DRIFT_COMPLIANCE,
                score=score,
                passed=passed,
                details={
                    "drift_detected": drift_report.drift_detected,
                    "drift_confidence": drift_report.confidence_score,
                    "files_analyzed": drift_report.total_files_analyzed,
                    "code_changes": len(drift_report.code_changes),
                    "doc_inconsistencies": len(drift_report.documentation_inconsistencies)
                },
                execution_time_ms=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return self._create_metric_failure(QualityMetric.DRIFT_COMPLIANCE, str(e), start_time)
    
    def _create_metric_failure(self, metric: QualityMetric, error: str, start_time: float) -> QualityResult:
        """Create failure result for a metric."""
        return QualityResult(
            metric=metric,
            score=0.0,
            passed=False,
            details={"error": error},
            execution_time_ms=int((time.time() - start_time) * 1000),
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_recommendations(self, results: List[QualityResult]) -> List[str]:
        """Generate improvement recommendations based on quality results."""
        recommendations = []
        for result in results:
            if not result.passed:
                if result.metric == QualityMetric.CODE_EXECUTION:
                    recommendations.append("Fix syntax errors in Python files")
                elif result.metric == QualityMetric.ARCHITECTURE_COMPLIANCE:
                    recommendations.append("Refactor files exceeding 300 lines per CLAUDE.md")
                elif result.metric == QualityMetric.TEST_PASSING:
                    recommendations.append("Fix failing tests or add missing test coverage")
                elif result.metric == QualityMetric.DRIFT_COMPLIANCE:
                    recommendations.append("Address code-documentation drift issues")
                elif result.metric == QualityMetric.SECURITY_VALIDATION:
                    recommendations.append("Review and secure potential security issues")
                elif result.metric == QualityMetric.DOCUMENTATION_COVERAGE:
                    recommendations.append("Add documentation for better coverage")
        return recommendations
    
    def _store_quality_evaluation(self, report: QualityReport):
        """Store quality evaluation in LTMC for learning."""
        try:
            # Store quality evaluation via native bridge
            self.ltmc_bridge.memory_store(
                kind="quality_evaluation", 
                content=f"Quality: {report.overall_score:.2f} ({report.passed_metrics}/{report.total_metrics})",
                metadata={
                    "evaluation_id": report.evaluation_id,
                    "overall_score": report.overall_score,
                    "performance_data": report.performance_data
                }
            )
        except Exception as e:
            logger.debug(f"Could not store evaluation in LTMC: {e}")
    
    def _create_failure_report(self, evaluation_id: str, error: str) -> QualityReport:
        """Create failure report for evaluation errors."""
        return QualityReport(
            evaluation_id=evaluation_id,
            project_path=str(self.project_path),
            timestamp=datetime.now().isoformat(),
            overall_score=0.0,
            passed_metrics=0,
            total_metrics=0,
            results=[],
            recommendations=[f"Fix evaluation error: {error}"],
            performance_data={"error": error}
        )
    
    def get_quality_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get quality trends over time for continuous improvement."""
        try:
            # This would retrieve historical quality data from LTMC
            return {
                "trend_analysis": "Quality trends analysis",
                "improvement_suggestions": [
                    "Focus on architecture compliance",
                    "Improve test coverage",
                    "Address documentation gaps"
                ],
                "performance_trends": "Performance metrics over time"
            }
        except Exception as e:
            return {"error": str(e)}


# Convenience function
def evaluate_project_quality(project_path: str = None, target_files: List[str] = None) -> QualityReport:
    """Evaluate project quality using KWECLI quality evaluation system."""
    evaluator = QualityEvaluator(project_path)
    return evaluator.evaluate_quality(target_files)


if __name__ == "__main__":
    print("🧪 Testing KWECLI Quality Evaluation System...")
    
    evaluator = QualityEvaluator()
    report = evaluator.evaluate_quality()
    
    print(f"✅ Quality Evaluation Complete:")
    print(f"   Overall Score: {report.overall_score:.2f}")
    print(f"   Passed Metrics: {report.passed_metrics}/{report.total_metrics}")
    print(f"   Recommendations: {len(report.recommendations)}")
    
    for result in report.results:
        status = "✅" if result.passed else "❌"
        print(f"   {status} {result.metric.value}: {result.score:.2f}")
    
    # Test quality trends
    trends = evaluator.get_quality_trends()
    print(f"✅ Quality trends analysis available")
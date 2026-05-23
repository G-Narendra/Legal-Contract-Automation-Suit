"""
End-to-end integration tests for legal contract automation workflow.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.chunking import chunk_document
from src.retrieval.hybrid_search import BM25
from src.utils.cache import TTLCache
from src.utils.monitoring import MetricsCollector
from src.core.audit_logger import AuditLogger
from src.subsystems.risk_compliance import RiskComplianceSystem
from src.subsystems.contract_analysis import ContractAnalysisSystem
from src.subsystems.lifecycle_management import LifecycleManagementSystem
from src.evaluation.metrics import calculate_extraction_metrics, calculate_accuracy

import tempfile


class TestEndToEnd:
    """End-to-end workflow tests."""

    def test_full_analysis_workflow(self):
        """Test: Contract text → Chunking → Analysis → Risk Assessment"""
        contract = (
            "EMPLOYMENT CONTRACT\n\n"
            "BETWEEN: Gulf Tech Solutions LLC ('Employer')\n"
            "AND: Ahmed Hassan ('Employee')\n\n"
            "1. The Employee shall serve as Senior Software Engineer.\n"
            "2. Compensation: AED 25,000 per month.\n"
            "3. Working hours: 40 hours per week.\n"
            "4. Leave: 30 working days annual leave.\n"
            "5. Termination: 30 days notice period.\n"
            "6. Governing Law: UAE Federal Law.\n"
        )

        # Chunking
        chunks = chunk_document(contract, chunk_size=512)
        assert len(chunks) > 0
        assert all(c["text"] for c in chunks)

        # Analysis (rule-based, no LLM needed)
        analysis = ContractAnalysisSystem()
        result = analysis.analyze(contract, "employment", "english")
        assert result["success"] is True
        data = result["structured_data"]
        assert "parties" in data
        assert "governing_law" in data

        # Risk assessment
        risk = RiskComplianceSystem()
        assessment = risk.assess(contract, result)
        assert "risk_findings" in assessment
        assert "compliance_score" in assessment
        assert 0 <= assessment["compliance_score"] <= 1

    def test_bm25_retrieval_workflow(self):
        """Test: Document index → Search → Rerank"""
        docs = [
            "Employment contract under UAE Labour Law",
            "Non-disclosure agreement for confidential information",
            "Partnership agreement governed by UAE Commercial Code",
            "Service agreement with payment terms and conditions",
        ]

        bm25 = BM25()
        bm25.fit(docs)

        results = bm25.search("employment UAE", top_k=2)
        assert len(results) >= 1

    def test_audit_logging_workflow(self):
        """Test: Action → Audit log → Query audit trail"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(db_path=f"{tmpdir}/test_audit.db")
            trace_id = audit.log(
                trace_id="TEST-001",
                action="test_action",
                subsystem="test",
                user="test_user",
                summary="Test audit entry",
            )
            assert trace_id == "TEST-001"

            trace = audit.get_trace("TEST-001")
            assert trace is not None
            assert trace["action"] == "test_action"

            stats = audit.get_stats()
            assert stats["total_actions"] >= 1

    def test_cache_workflow(self):
        """Test: Set cache → Get cache → Cache stats"""
        cache = TTLCache(ttl_seconds=60)
        assert cache.get("test", "key") is None
        cache.set("test", "key", "value")
        assert cache.get("test", "key") == "value"
        assert cache.size >= 1

    def test_metrics_workflow(self):
        """Test: Record metrics → Query stats → Cost estimation"""
        metrics = MetricsCollector()
        metrics.record("test_system", 100.0, tokens=500)
        metrics.record("test_system", 200.0, error=True, tokens=500)

        stats = metrics.get_stats()
        assert "test_system" in stats
        assert stats["test_system"]["requests"] == 2
        assert stats["test_system"]["errors"] == 1

        cost = metrics.estimate_cost()
        assert "estimated_cost_usd" in cost

    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        expected = [{"parties": "Company A"}, {"parties": "Company B"}]
        actual = [{"parties": "Company A"}, {"parties": "Company C"}]

        result = calculate_extraction_metrics(expected, actual, key_field="parties")
        assert "precision" in result
        assert "recall" in result
        assert 0 <= result["f1"] <= 1

    def test_lifecycle_registration(self):
        """Test contract lifecycle registration."""
        lifecycle = LifecycleManagementSystem(llm=None)
        result = lifecycle.register_contract(
            title="Test Employment Contract",
            contract_type="employment",
            parties=["Company A", "Employee B"],
            value=250000,
        )
        assert "contract_id" in result
        assert result["status"] == "active"

        # Clean up
        if result.get("contract_id"):
            lifecycle.conn.execute(
                "DELETE FROM contracts WHERE contract_id = ?",
                (result["contract_id"],)
            )
            lifecycle.conn.commit()

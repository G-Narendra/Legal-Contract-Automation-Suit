"""
Central Contract Processing Engine — orchestrates all contract operations.

Routes contracts through the full pipeline:
1. Document extraction (PDF/DOCX/TXT)
2. Language detection (Arabic/English/Mixed)
3. Contract type classification
4. Subsystem dispatch based on task
"""

from typing import Dict, List, Optional, Generator
from enum import Enum
import io
import re
import time
import logging
import hashlib
import json
from pathlib import Path

logger = logging.getLogger("contract_processor")


class ContractType(Enum):
    EMPLOYMENT = "employment"
    NDA = "nda"
    PARTNERSHIP = "partnership"
    LEASE = "lease"
    SALES = "sales"
    SERVICE = "service"
    UNKNOWN = "unknown"


class TaskType(Enum):
    ANALYZE = "analyze"
    DRAFT = "draft"
    REVIEW = "review"
    RESEARCH = "research"


class ContractProcessor:
    """Main contract processing orchestrator.

    Design:
    - Lazy initialization: subsystems created on first use
    - Streaming for long operations (drafting, analysis)
    - Audit logging with trace tracking
    - Cache-friendly for repeated operations
    """

    def __init__(self, llm=None, knowledge_base=None, audit_logger=None):
        self.llm = llm
        if knowledge_base is None:
            from src.core.legal_knowledge_base import LegalKnowledgeBase
            knowledge_base = LegalKnowledgeBase()
        self.knowledge_base = knowledge_base
        self.audit_logger = audit_logger

        # Lazy-loaded subsystems
        self._analysis = None
        self._research = None
        self._drafting = None
        self._risk = None
        self._lifecycle = None

    # ------------------------------------------------------------------
    # Subsystem Lazy Loading
    # ------------------------------------------------------------------

    @property
    def analysis_system(self):
        if self._analysis is None:
            from src.subsystems.contract_analysis import ContractAnalysisSystem
            self._analysis = ContractAnalysisSystem(llm=self.llm, kb=self.knowledge_base)
        return self._analysis

    @property
    def research_system(self):
        if self._research is None:
            from src.subsystems.legal_research import LegalResearchSystem
            self._research = LegalResearchSystem(llm=self.llm, kb=self.knowledge_base)
        return self._research

    @property
    def drafting_system(self):
        if self._drafting is None:
            from src.subsystems.contract_drafting import ContractDraftingSystem
            self._drafting = ContractDraftingSystem(llm=self.llm, kb=self.knowledge_base)
        return self._drafting

    @property
    def risk_system(self):
        if self._risk is None:
            from src.subsystems.risk_compliance import RiskComplianceSystem
            self._risk = RiskComplianceSystem(llm=self.llm, kb=self.knowledge_base)
        return self._risk

    @property
    def lifecycle_system(self):
        if self._lifecycle is None:
            from src.subsystems.lifecycle_management import LifecycleManagementSystem
            self._lifecycle = LifecycleManagementSystem(llm=self.llm, kb=self.knowledge_base)
        return self._lifecycle

    # ------------------------------------------------------------------
    # Main Processing Pipeline
    # ------------------------------------------------------------------

    def process_contract(self, contract_file: bytes, file_type: str,
                         task: str = "analyze",
                         user_context: Optional[Dict] = None,
                         params: Optional[Dict] = None) -> Dict:
        """Main entry point for contract processing.

        Args:
            contract_file: Raw file bytes
            file_type: 'pdf', 'docx', or 'txt'
            task: 'analyze', 'draft', 'review', 'research'
            user_context: User info for audit
            params: Additional parameters (template params, etc.)

        Returns:
            Dict with results, metadata, and audit info
        """
        user_context = user_context or {}
        params = params or {}
        start_time = time.time()

        # Step 1: Extract text
        contract_text = self._extract_text(contract_file, file_type)
        if not contract_text:
            return {"error": "Could not extract text from document"}

        # Step 2: Detect language
        language = self._detect_language(contract_text)

        # Step 3: Classify contract type
        contract_type = self._classify_contract(contract_text)

        # Step 4: Dispatch to subsystem
        if task == "analyze":
            result = self._handle_analysis(contract_text, contract_type, language, params)
        elif task == "draft":
            result = self._handle_drafting(contract_type, language, params)
        elif task == "review":
            result = self._handle_review(contract_text, contract_type, language, params)
        elif task == "research":
            result = self._handle_research(params.get("query", ""), contract_text, params)
        else:
            result = {"error": f"Unknown task: {task}"}

        duration = (time.time() - start_time) * 1000

        # Audit log
        if self.audit_logger:
            trace_id = f"CT-{int(start_time)}-{hashlib.md5(contract_text[:100].encode()).hexdigest()[:6]}"
            self.audit_logger.log(
                trace_id=trace_id,
                action=task,
                subsystem=task,
                user=user_context.get("user", "anonymous"),
                contract_type=contract_type.value,
                language=language,
                duration_ms=round(duration, 1),
                success="error" not in result,
            )
        else:
            trace_id = f"CT-{int(start_time)}-local"

        return {
            "trace_id": trace_id,
            "contract_type": contract_type.value,
            "language": language,
            "duration_ms": round(duration, 1),
            "result": result,
        }

    def stream_process(self, contract_file: bytes, file_type: str,
                       task: str = "analyze",
                       user_context: Optional[Dict] = None,
                       params: Optional[Dict] = None) -> Generator[Dict, None, Dict]:
        """Stream processing results token by token (for drafting/analysis)."""
        user_context = user_context or {}
        params = params or {}

        contract_text = self._extract_text(contract_file, file_type)
        if not contract_text:
            yield {"error": "Could not extract text"}
            return {"error": "Could not extract text"}

        language = self._detect_language(contract_text)
        contract_type = self._classify_contract(contract_text)

        if task == "draft":
            yield from self.drafting_system.draft_stream(
                contract_type=contract_type,
                language=language,
                params=params,
            )
        elif task == "analyze":
            result = self.analysis_system.analyze(contract_text, contract_type, language)
            yield result
        else:
            yield {"error": f"Streaming not supported for task: {task}"}

        return {"completed": True}

    # ------------------------------------------------------------------
    # Task Handlers
    # ------------------------------------------------------------------

    def _handle_analysis(self, text: str, ctype: ContractType,
                         language: str, params: Dict) -> Dict:
        """Analyze contract and extract structured data."""
        analysis = self.analysis_system.analyze(text, ctype, language)
        clauses = self.analysis_system.identify_clauses(text, language)
        return {
            "analysis": analysis,
            "clauses": clauses,
            "clause_count": len(clauses),
        }

    def _handle_drafting(self, ctype: ContractType, language: str,
                         params: Dict) -> Dict:
        """Draft a new contract from template parameters."""
        draft = self.drafting_system.draft(ctype, params, language)
        return {
            "draft": draft,
            "requires_lawyer_review": True,
            "template_used": ctype.value,
        }

    def _handle_review(self, text: str, ctype: ContractType,
                       language: str, params: Dict) -> Dict:
        """Full review: analysis + risk assessment."""
        analysis = self.analysis_system.analyze(text, ctype, language)
        risk = self.risk_system.assess(text, analysis)
        return {
            "analysis": analysis,
            "risk_assessment": risk,
            "requires_lawyer_review": risk.get("requires_lawyer_review", False),
        }

    def _handle_research(self, query: str, context: str, params: Dict) -> Dict:
        """Legal research with context from the contract."""
        research = self.research_system.search(query, context)
        return {"research": research}

    # ------------------------------------------------------------------
    # Document Processing Utilities
    # ------------------------------------------------------------------

    def _extract_text(self, file_bytes: bytes, file_type: str) -> str:
        """Extract text from PDF, DOCX, or TXT."""
        try:
            if file_type == "pdf":
                from PyPDF2 import PdfReader
                pdf = PdfReader(io.BytesIO(file_bytes))
                return "\n".join(page.extract_text() or "" for page in pdf.pages)

            elif file_type == "docx":
                from docx import Document
                doc = Document(io.BytesIO(file_bytes))
                return "\n".join(p.text for p in doc.paragraphs)

            elif file_type == "txt":
                return file_bytes.decode("utf-8", errors="replace")

            else:
                return file_bytes.decode("utf-8", errors="replace")
        except ImportError as e:
            logger.warning(f"Missing library for {file_type}: {e}")
            return file_bytes.decode("utf-8", errors="replace")[:1000]
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return ""

    def _detect_language(self, text: str) -> str:
        """Detect primary language: arabic, english, or mixed."""
        if not text:
            return "unknown"

        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total = arabic_chars + english_chars

        if total == 0:
            return "unknown"

        arabic_ratio = arabic_chars / total
        if arabic_ratio > 0.7:
            return "arabic"
        elif arabic_ratio < 0.3:
            return "english"
        else:
            return "mixed"

    def _classify_contract(self, text: str) -> ContractType:
        """Classify contract type based on keywords (rule-based + LLM fallback)."""
        text_lower = text.lower()

        # Rule-based classification
        if any(term in text_lower for term in
               ['employment', 'employee', 'employer', 'عمل', 'موظف']):
            return ContractType.EMPLOYMENT

        if any(term in text_lower for term in
               ['nda', 'confidential', 'non-disclosure', 'سرية', 'عدم الإفصاح']):
            return ContractType.NDA

        if any(term in text_lower for term in
               ['partnership', 'partner', 'شراكة', 'شريك']):
            return ContractType.PARTNERSHIP

        if any(term in text_lower for term in
               ['lease', 'rent', 'إيجار', 'استئجار']):
            return ContractType.LEASE

        if any(term in text_lower for term in
               ['sale', 'purchase', 'بيع', 'شراء']):
            return ContractType.SALES

        if any(term in text_lower for term in
               ['service', 'services', 'خدمات']):
            return ContractType.SERVICE

        return ContractType.UNKNOWN


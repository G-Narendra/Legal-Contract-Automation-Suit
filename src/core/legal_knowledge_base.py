"""
Legal Knowledge Base — shared across all subsystems.
Provides access to vector stores, templates, and compliance rules.
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger("legal_kb")


class LegalKnowledgeBase:
    """Centralized legal knowledge accessible by all subsystems.

    Manages:
    - Vector DB collections (laws, cases, templates, terminology)
    - Contract templates for drafting
    - Compliance rules and checklists
    - Legal terminology dictionary (Arabic/English)
    """

    def __init__(self, vector_store=None, embedder=None):
        self.vector_store = vector_store
        self.embedder = embedder
        self._templates_cache = None
        self._terminology_cache = None
        self._compliance_rules_cache = None

    # ------------------------------------------------------------------ 
    # Vector Store Queries
    # ------------------------------------------------------------------

    def query(self, query_text: str, collection_name: str = "legal_laws",
              top_k: int = 5) -> List[Dict]:
        """Query specific knowledge collection."""
        if not self.vector_store:
            logger.warning("Vector store not initialized — returning empty results")
            return []

        try:
            results = self.vector_store.similarity_search(
                query=query_text,
                collection_name=collection_name,
                k=top_k,
            )
            return results
        except Exception as e:
            logger.error(f"Vector store query error: {e}")
            return []

    def hybrid_search(self, query: str, collection_name: str = "legal_laws",
                      top_k: int = 5) -> List[Dict]:
        """Hybrid search (dense + sparse) for better retrieval."""
        if not self.vector_store:
            return self.query(query, collection_name, top_k)

        return self.vector_store.hybrid_search(
            query=query,
            collection_name=collection_name,
            k=top_k,
        )

    # ------------------------------------------------------------------
    # Contract Templates
    # ------------------------------------------------------------------

    def get_template(self, contract_type: str, language: str = "english") -> str:
        """Get a contract template by type and language."""
        templates = self._load_templates()
        key = f"{contract_type}_{language}"
        return templates.get(key, templates.get(f"{contract_type}_english", ""))

    def list_templates(self) -> List[str]:
        """List all available template types."""
        templates = self._load_templates()
        return list(set(k.split("_")[0] for k in templates.keys()))

    def _load_templates(self) -> Dict[str, str]:
        """Lazy-load templates from templates directory."""
        if self._templates_cache is not None:
            return self._templates_cache

        templates = {}
        templates_dir = Path("data/templates")
        if templates_dir.exists():
            for f in templates_dir.glob("*.yaml"):
                with open(f, "r") as fh:
                    data = yaml.safe_load(fh) or {}
                    templates.update(data)

        # Fallback: minimal built-in templates
        if not templates:
            templates = self._default_templates()

        self._templates_cache = templates
        return templates

    def _default_templates(self) -> Dict[str, str]:
        """Minimal built-in templates as fallback."""
        return {
            "nda_english": (
                "CONFIDENTIALITY AGREEMENT\n\n"
                "This Confidentiality Agreement ('Agreement') is made on [DATE]\n"
                "BETWEEN:\n"
                "1. [PARTY_A_NAME] ('Disclosing Party')\n"
                "2. [PARTY_B_NAME] ('Receiving Party')\n\n"
                "1. DEFINITION OF CONFIDENTIAL INFORMATION\n"
                "2. OBLIGATIONS OF RECEIVING PARTY\n"
                "3. EXCLUSIONS\n"
                "4. TERM AND TERMINATION\n"
                "5. GOVERNING LAW\n"
            ),
            "employment_english": (
                "EMPLOYMENT CONTRACT\n\n"
                "This Employment Contract is made on [DATE]\n"
                "BETWEEN:\n"
                "1. [EMPLOYER_NAME] ('Employer')\n"
                "2. [EMPLOYEE_NAME] ('Employee')\n\n"
                "1. POSITION AND DUTIES\n"
                "2. COMPENSATION AND BENEFITS\n"
                "3. WORKING HOURS\n"
                "4. LEAVE ENTITLEMENT\n"
                "5. TERMINATION\n"
                "6. GOVERNING LAW: UAE Federal Law No. 8 of 1980\n"
            ),
            "nda_arabic": (
                "اتفاقية السرية\n\n"
                "تم إبرام هذه الاتفاقية في [التاريخ]\n"
                "بين:\n"
                "1. [اسم الطرف الأول] ('الطرف المفصح')\n"
                "2. [اسم الطرف الثاني] ('الطرف المتلقي')\n\n"
                "1. تعريف المعلومات السرية\n"
                "2. التزامات الطرف المتلقي\n"
                "3. الاستثناءات\n"
            ),
        }

    # ------------------------------------------------------------------
    # Legal Terminology
    # ------------------------------------------------------------------

    def get_terminology(self, term: str, language: str = "en") -> Dict:
        """Look up a legal term definition."""
        terms = self._load_terminology()
        return terms.get(term.lower(), terms.get(term.lower(), {}))

    def _load_terminology(self) -> Dict:
        if self._terminology_cache is not None:
            return self._terminology_cache

        # Built-in common legal terms
        self._terminology_cache = {
            "force majeure": {
                "en": "Unforeseeable circumstances preventing contract fulfillment",
                "ar": "ظروف قاهرة غير متوقعة تمنع تنفيذ العقد",
            },
            "indemnification": {
                "en": "Compensation for harm or loss",
                "ar": "تعويض عن الضرر أو الخسارة",
            },
            "arbitration": {
                "en": "Settlement of dispute by an impartial tribunal",
                "ar": "تسوية النزاع بواسطة هيئة تحكيم محايدة",
            },
            "liquidated damages": {
                "en": "Predetermined damages for breach of contract",
                "ar": "تعويضات محددة سلفًا للإخلال بالعقد",
            },
        }
        return self._terminology_cache

    # ------------------------------------------------------------------
    # Compliance Rules
    # ------------------------------------------------------------------

    def get_compliance_rules(self, contract_type: str = "") -> List[Dict]:
        """Get UAE compliance rules relevant to the contract type."""
        rules = self._load_compliance_rules()
        if contract_type:
            return [r for r in rules if contract_type in r.get("applies_to", [])]
        return rules

    def _load_compliance_rules(self) -> List[Dict]:
        if self._compliance_rules_cache is not None:
            return self._compliance_rules_cache

        self._compliance_rules_cache = [
            {
                "rule": "UAE Labour Law compliance",
                "law": "Federal Decree-Law No. 33 of 2021",
                "applies_to": ["employment"],
                "check": "Must include working hours, leave, termination per UAE Labour Law",
            },
            {
                "rule": "Data Protection",
                "law": "Federal Decree-Law No. 45 of 2021 (PDPL)",
                "applies_to": ["employment", "service", "nda"],
                "check": "Must include data processing and confidentiality clauses",
            },
            {
                "rule": "Commercial Transactions",
                "law": "Federal Law No. 18 of 1993 (Commercial Code)",
                "applies_to": ["sales", "service", "partnership"],
                "check": "Must specify payment terms, delivery, and dispute resolution",
            },
        ]
        return self._compliance_rules_cache

    def check_uae_compliance(self, contract_type: str, clauses: List[str]) -> List[Dict]:
        """Check if contract clauses comply with UAE law."""
        rules = self.get_compliance_rules(contract_type)
        findings = []

        for rule in rules:
            found = any(rule["check"].lower() in c.lower() for c in clauses)
            findings.append({
                "rule": rule["rule"],
                "compliant": found,
                "severity": "high" if not found else "info",
                "recommendation": rule["check"] if not found else "Compliant",
            })

        return findings

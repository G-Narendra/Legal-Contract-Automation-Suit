"""
Risk & Compliance Subsystem (RAG + Human-in-Loop).

Assesses contracts for legal risks and UAE compliance issues.
Routes high-risk contracts for mandatory lawyer review.
"""

from typing import Dict, List, Optional
import json
import re
import logging

logger = logging.getLogger("risk_compliance")


class RiskComplianceSystem:
    """Assess contracts for legal risks and UAE compliance.

    Human-in-Loop Design:
    - Low risk: Auto-approved (configurable)
    - Medium risk: Flagged for optional review
    - High risk: Requires mandatory lawyer review
    - Critical risk: Blocked until legal review

    Cost-effective:
    - Lite model for initial screening
    - Flash model only for complex risk analysis
    - Cached compliance checks per jurisdiction
    """

    def __init__(self, llm=None, kb=None):
        self.llm = llm
        self.kb = kb

    def assess(self, contract_text: str, analysis: Dict) -> Dict:
        """Full contract risk assessment.

        Args:
            contract_text: Full contract text
            analysis: Pre-computed analysis from ContractAnalysisSystem

        Returns:
            {
                "risk_findings": [{...}],
                "high_risk_clauses_found": True,
                "requires_lawyer_review": True,
                "compliance_score": 0.65,
                "review_priority": "high"
            }
        """
        risk_findings = []
        clauses = analysis.get("structured_data", {}).get("key_clauses", [])

        if not clauses:
            clauses = self._extract_clauses_simple(contract_text)

        for clause in clauses:
            if isinstance(clause, str):
                clause_text = clause
            elif isinstance(clause, dict):
                clause_text = clause.get("clause_text", clause.get("text", str(clause)))
            else:
                clause_text = str(clause)

            if len(clause_text) < 10:
                continue

            risk = self._analyze_clause(clause_text, contract_text)
            if risk and risk.get("risk_level") != "low":
                risk_findings.append(risk)

            if len(risk_findings) >= 10:
                break

        # Rule-based checks
        rule_findings = self._rule_based_checks(contract_text, analysis)
        risk_findings.extend(rule_findings)

        high_risk_count = sum(1 for r in risk_findings if r.get("risk_level") == "high")
        critical_count = sum(1 for r in risk_findings if r.get("risk_level") == "critical")

        requires_review = critical_count > 0 or high_risk_count > 0 or len(risk_findings) >= 3
        if critical_count > 0:
            review_priority = "critical"
        elif high_risk_count > 0:
            review_priority = "high"
        elif requires_review:
            review_priority = "medium"
        else:
            review_priority = "low"

        return {
            "risk_findings": risk_findings,
            "high_risk_clauses_found": high_risk_count > 0,
            "critical_clauses_found": critical_count > 0,
            "requires_lawyer_review": requires_review,
            "compliance_score": self._calculate_score(risk_findings),
            "review_priority": review_priority,
        }

    def _analyze_clause(self, clause_text: str, contract_text: str) -> Optional[Dict]:
        """Analyze a single clause for legal risk.

        Uses rule-based detection first (fast, free), then LLM for complex cases.
        """
        # Rule-based risk detection
        risk_keywords = {
            "critical": [
                "unlimited liability", "indemnify.*all", "waive.*all rights",
                "no liability", "unconscionable", "illegal",
                "مسؤولية غير محدودة", "غير قانوني", "مخالف للقانون",
            ],
            "high": [
                "indemnification", "limitation of liability", "termination.*cause",
                "force majeure.*exclude", "non-compete.*broad",
                "تعويض", "حد المسؤولية", "الإنهاء", "القوة القاهرة",
            ],
            "medium": [
                "confidentiality", "governing law", "jurisdiction",
                "assignment", "severability",
                "سرية", "القانون الحاكم", "الاختصاص القضائي",
            ],
        }

        clause_lower = clause_text.lower()
        for level, keywords in risk_keywords.items():
            for kw in keywords:
                if re.search(kw, clause_lower, re.IGNORECASE):
                    return {
                        "clause": clause_text[:200],
                        "risk_level": level,
                        "detected_by": "rule",
                        "compliance_issues": [f"Keyword match: {kw}"],
                        "recommendations": f"Review {level}-risk clause for UAE legal compliance",
                        "uae_law_citations": ["UAE Federal Law reference required"],
                    }

        # LLM analysis for complex clauses
        if self.llm and len(clause_text) > 100:
            return self._llm_clause_analysis(clause_text)

        return None

    def _llm_clause_analysis(self, clause_text: str) -> Dict:
        """Use LLM for deeper clause analysis."""
        prompt = (
            f"[INST] Analyze this clause for UAE legal risk:\n\n"
            f"Clause: {clause_text[:500]}\n\n"
            f"Assess risk level (low/medium/high/critical), compliance issues, "
            f"and recommendations. Return JSON:\n"
            f"{{\"risk_level\": \"...\", \"compliance_issues\": [...], "
            f"\"recommendations\": \"...\", \"uae_law_citations\": [...]}} [/INST]"
        )

        try:
            result = self.llm.generate(prompt, temperature=0.1)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                parsed["clause"] = clause_text[:200]
                parsed["detected_by"] = "llm"
                return parsed
        except Exception as e:
            logger.warning(f"LLM clause analysis error: {e}")

        return None

    def _rule_based_checks(self, text: str, analysis: Dict) -> List[Dict]:
        """Rule-based compliance checks (no LLM cost)."""
        findings = []
        text_lower = text.lower()

        # Check for missing governing law
        if not any(term in text_lower for term in
                   ["governing law", "applicable law", "القانون الحاكم"]):
            findings.append({
                "clause": "General",
                "risk_level": "medium",
                "detected_by": "rule",
                "compliance_issues": ["Missing governing law clause"],
                "recommendations": "Add UAE governing law clause",
                "uae_law_citations": ["Recommended for all UAE contracts"],
            })

        # Check for missing dispute resolution
        if not any(term in text_lower for term in
                   ["arbitration", "dispute", "jurisdiction", "تحكيم", "نزاع"]):
            findings.append({
                "clause": "General",
                "risk_level": "medium",
                "detected_by": "rule",
                "compliance_issues": ["Missing dispute resolution clause"],
                "recommendations": "Add arbitration or court jurisdiction clause",
                "uae_law_citations": ["UAE Civil Procedure Law"],
            })

        # Check for missing termination clause
        if not any(term in text_lower for term in
                   ["termination", "notice period", "إنهاء", "إخطار"]):
            findings.append({
                "clause": "General",
                "risk_level": "medium",
                "detected_by": "rule",
                "compliance_issues": ["Missing termination clause"],
                "recommendations": "Add termination and notice period provisions",
                "uae_law_citations": ["Depends on contract type"],
            })

        return findings

    def _calculate_score(self, findings: List[Dict]) -> float:
        """Calculate overall compliance score (0-1)."""
        if not findings:
            return 1.0

        risk_weights = {"critical": 0.8, "high": 0.5, "medium": 0.2, "low": 0.05}
        total_risk = sum(risk_weights.get(f.get("risk_level", "low"), 0.1) for f in findings)

        # Normalize: more findings = lower score
        max_possible = len(findings) * 0.8
        score = 1.0 - (total_risk / max_possible) if max_possible > 0 else 1.0
        return round(max(0.0, min(1.0, score)), 2)

    def _extract_clauses_simple(self, text: str) -> List[str]:
        """Simple clause extraction from numbered paragraphs."""
        clauses = []
        lines = text.split("\n")
        current = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    clauses.append(" ".join(current))
                    current = []
                continue

            # Starts with number?
            if re.match(r'^\d+[\.\)]', stripped):
                if current:
                    clauses.append(" ".join(current))
                current = [stripped]
            else:
                current.append(stripped)

        if current:
            clauses.append(" ".join(current))

        return [c for c in clauses if len(c) > 20][:20]

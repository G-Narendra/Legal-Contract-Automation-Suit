"""
LLM-as-Judge evaluation for contract quality assessment.

Evaluates:
- Extraction accuracy (compared to golden dataset)
- Draft quality (legal completeness, clarity)
- Risk detection (recall of high-risk clauses)
- Compliance checking accuracy
"""

from typing import Dict, List, Optional
import json
import re
import logging

logger = logging.getLogger("llm_judge")


class LLMJudge:
    """Model-as-a-Judge for evaluating legal AI outputs.

    Uses an LLM (typically a stronger model) to evaluate outputs
    of the contract automation subsystems against ground truth.
    """

    def __init__(self, llm=None):
        self.llm = llm

    def evaluate_extraction(self, expected: Dict, actual: Dict) -> Dict:
        """Evaluate contract extraction accuracy."""
        if not self.llm or not expected or not actual:
            return self._rule_eval(expected, actual)

        prompt = (
            f"[INST] Compare the EXTRACTED data with EXPECTED data for a contract.\n\n"
            f"EXPECTED (ground truth):\n{json.dumps(expected, indent=2, ensure_ascii=False)}\n\n"
            f"EXTRACTED (system output):\n{json.dumps(actual, indent=2, ensure_ascii=False)}\n\n"
            f"Score each field as CORRECT (1.0), PARTIAL (0.5), or INCORRECT (0.0).\n"
            f"Return JSON:\n"
            f"{{\"field_scores\": {{\"parties\": 0.0, ...}}, "
            f"\"overall_score\": 0.0, \"issues\": [...]}} [/INST]"
        )

        try:
            result = self.llm.generate(prompt, temperature=0.0)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"LLM judge error: {e}")

        return self._rule_eval(expected, actual)

    def _rule_eval(self, expected: Dict, actual: Dict) -> Dict:
        """Fallback: simple rule-based evaluation."""
        scores = {}
        total = 0
        matched = 0

        for key in expected:
            if key in actual:
                total += 1
                exp_val = str(expected[key]).lower()
                act_val = str(actual[key]).lower()
                if exp_val == act_val:
                    scores[key] = 1.0
                    matched += 1
                elif exp_val[:20] in act_val or act_val[:20] in exp_val:
                    scores[key] = 0.5
                else:
                    scores[key] = 0.0

        return {
            "field_scores": scores,
            "overall_score": round(matched / total, 2) if total > 0 else 0,
            "issues": [],
            "method": "rule_based",
        }

    def evaluate_risk_detection(self, expected_risks: List[Dict],
                                 detected_risks: List[Dict]) -> Dict:
        """Evaluate risk detection recall and precision."""
        expected_texts = [r.get("clause", "")[:50] for r in expected_risks]
        detected_texts = [r.get("clause", "")[:50] for r in detected_risks]

        # Simple overlap
        true_positives = sum(1 for e in expected_texts if any(e in d or d in e for d in detected_texts))
        false_negatives = len(expected_texts) - true_positives
        false_positives = len(detected_texts) - true_positives

        recall = true_positives / len(expected_texts) if expected_texts else 1.0
        precision = true_positives / len(detected_texts) if detected_texts else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "recall": round(recall, 2),
            "precision": round(precision, 2),
            "f1": round(f1, 2),
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "false_positives": false_positives,
        }

    def evaluate_draft_quality(self, draft: str, requirements: str) -> Dict:
        """Evaluate the quality of a contract draft."""
        if not self.llm:
            return {"quality_score": 0.5, "method": "estimated"}

        prompt = (
            f"[INST] Evaluate this contract draft quality.\n\n"
            f"Requirements: {requirements[:500]}\n\n"
            f"Draft (first 2000 chars): {draft[:2000]}\n\n"
            f"Score (0-1): completeness, clarity, legal correctness, clause coverage.\n"
            f"Return JSON:\n"
            f"{{\"completeness\": 0.0, \"clarity\": 0.0, "
            f"\"legal_correctness\": 0.0, \"clause_coverage\": 0.0, "
            f"\"overall\": 0.0, \"missing_clauses\": [...]}} [/INST]"
        )

        try:
            result = self.llm.generate(prompt, temperature=0.0)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"Draft quality eval error: {e}")

        return {"overall": 0.5, "method": "estimated"}

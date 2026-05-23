"""
Contract Writer Agent — generates and refines contract drafts.

Capabilities:
- Generate complete contracts from templates
- Refine drafts based on feedback
- Insert specific clauses into existing contracts
- Format contracts for legal standards
"""

from typing import Dict, List, Optional
import logging

from src.agents.base_agent import BaseAgent

logger = logging.getLogger("agent_writer")


class ContractWriterAgent(BaseAgent):
    """Agent specialized in drafting and refining legal contracts.

    Tools:
    - draft_contract: Generate contract from parameters
    - refine_draft: Refine an existing draft based on feedback
    - insert_clause: Add a clause to an existing contract
    - format_contract: Format for legal standards
    """

    def __init__(self, llm=None, knowledge_base=None, audit_logger=None):
        super().__init__(name="contract_writer", llm=llm, audit_logger=audit_logger)
        self.knowledge_base = knowledge_base

        self.register_tool("draft_contract", self._draft, "Generate contract draft")
        self.register_tool("refine_draft", self._refine, "Refine draft from feedback")
        self.register_tool("insert_clause", self._insert_clause, "Add clause to contract")
        self.register_tool("format_contract", self._format, "Format for legal standards")

    def _draft(self, contract_type: str, params: Dict,
               language: str = "english", **kwargs) -> Dict:
        """Generate a contract draft."""
        template = ""
        if self.knowledge_base:
            template = self.knowledge_base.get_template(contract_type, language)

        if self.llm:
            prompt = self._build_drafting_prompt(contract_type, params, language, template)
            try:
                draft = self.llm.generate(prompt, temperature=0.2, max_tokens=4096)
                draft = self._clean_draft(draft)
            except Exception as e:
                logger.error(f"Draft error: {e}")
                draft = self._fill_template(contract_type, params, language, template)
        else:
            draft = self._fill_template(contract_type, params, language, template)

        return {
            "contract_draft": draft,
            "contract_type": contract_type,
            "language": language,
            "requires_lawyer_review": True,
            "model": self.llm.get_provider_name() if self.llm else "template",
        }

    def _refine(self, draft: str, feedback: str, **kwargs) -> Dict:
        """Refine an existing draft based on feedback."""
        if not self.llm:
            return {"draft": draft, "changes": "No LLM available for refinement"}

        prompt = (
            f"[INST] Refine this contract draft based on feedback.\\n\\n"
            f"DRAFT:\\n{draft[:3000]}\\n\\n"
            f"FEEDBACK:\\n{feedback}\\n\\n"
            f"Apply the feedback and return the improved draft. "
            f"Preserve all legal language and structure. "
            f"Note any significant changes made. [/INST]"
        )

        try:
            refined = self.llm.generate(prompt, temperature=0.2)
            return {
                "draft": refined,
                "feedback_applied": True,
                "requires_review": True,
            }
        except Exception as e:
            return {"draft": draft, "error": str(e)}

    def _insert_clause(self, contract_text: str, clause_type: str,
                       clause_params: Dict, **kwargs) -> Dict:
        """Insert a specific clause into an existing contract."""
        if not self.llm:
            return {"error": "LLM not available for clause insertion"}

        prompt = (
            f"[INST] Insert a {clause_type} clause into this contract.\\n\\n"
            f"CONTRACT:\\n{contract_text[:3000]}\\n\\n"
            f"CLAUSE PARAMETERS:\\n{clause_params}\\n\\n"
            f"Insert the clause in the correct position. "
            f"Return the full updated contract. [/INST]"
        )

        try:
            updated = self.llm.generate(prompt, temperature=0.1)
            return {"updated_contract": updated, "clause_added": clause_type}
        except Exception as e:
            return {"error": str(e)}

    def _format(self, contract_text: str, style: str = "legal", **kwargs) -> Dict:
        """Format contract for legal standards."""
        # Simple formatting: ensure proper structure
        formatted = contract_text.strip()
        if not formatted.startswith(("#", "CONTRACT", "عقد")):
            formatted = f"# CONTRACT\n\n{formatted}"

        return {
            "formatted_contract": formatted,
            "style": style,
            "word_count": len(formatted.split()),
        }

    def _build_drafting_prompt(self, ctype: str, params: Dict,
                                language: str, template: str) -> str:
        """Build optimized drafting prompt."""
        params_str = "\n".join([f"- {k}: {v}" for k, v in params.items()])

        if language == "arabic":
            return (
                f"[INST] اكتب عقد {ctype} باللغة العربية القانونية.\\n\\n"
                f"المعلومات:\\n{params_str}\\n\\n"
                f"تأكد من تضمين جميع البنود القانونية القياسية. [/INST]"
            )
        return (
            f"[INST] Draft a complete {ctype} contract.\\n\\n"
            f"Details:\\n{params_str}\\n\\n"
            f"Template:\\n{template[:500] if template else 'Standard format'}\\n\\n"
            f"Include ALL standard clauses. Use clear section headings. [/INST]"
        )

    def _clean_draft(self, draft: str) -> str:
        """Clean the LLM output to extract just the contract."""
        # Remove potential instructions at the end
        lines = draft.split("\n")
        clean_lines = []
        for line in lines:
            if line.lower().strip().startswith(("note:", "disclaimer:", "this is a draft")):
                if len(clean_lines) > 10:
                    break
            clean_lines.append(line)
        return "\n".join(clean_lines).strip()

    def _fill_template(self, ctype: str, params: Dict,
                       language: str, template: str) -> str:
        """Fallback: fill template without LLM."""
        if not template:
            template = self._default_template(ctype, language)

        filled = template
        replacements = {
            "[DATE]": params.get("effective_date", "[DATE]"),
            "[PARTY_A_NAME]": params.get("party1_name", params.get("party_a", "[Party A]")),
            "[PARTY_B_NAME]": params.get("party2_name", params.get("party_b", "[Party B]")),
            "[COMPENSATION]": params.get("compensation", "[Compensation]"),
            "[DURATION]": params.get("duration", "[Duration]"),
        }
        for placeholder, value in replacements.items():
            filled = filled.replace(placeholder, str(value))

        closing = "\n\n____________________\n"
        closing += f"{replacements['[PARTY_A_NAME]']}\n\n____________________\n"
        closing += f"{replacements['[PARTY_B_NAME]']}"
        return filled + closing

    def _default_template(self, ctype: str, language: str = "english") -> str:
        templates = {
            "employment": (
                "EMPLOYMENT CONTRACT\n\nDate: [DATE]\n\n"
                "BETWEEN:\n[PARTY_A_NAME] ('Employer')\nAND\n[PARTY_B_NAME] ('Employee')\n\n"
                "1. POSITION AND DUTIES\n2. COMPENSATION: [COMPENSATION]\n"
                "3. WORKING HOURS\n4. LEAVE\n5. TERMINATION\n6. GOVERNING LAW: UAE"
            ),
            "nda": (
                "CONFIDENTIALITY AGREEMENT\n\nDate: [DATE]\n\n"
                "BETWEEN:\n[PARTY_A_NAME] ('Disclosing Party')\nAND\n[PARTY_B_NAME] ('Receiving Party')\n\n"
                "1. CONFIDENTIAL INFORMATION\n2. OBLIGATIONS\n3. EXCLUSIONS\n"
                "4. TERM: [DURATION]\n5. GOVERNING LAW: UAE"
            ),
        }
        return templates.get(ctype, f"{ctype.upper()} CONTRACT\n\nDate: [DATE]")

"""
Legal Research Subsystem (RAG-based).

Searches UAE legal knowledge base for relevant laws, cases, and precedents.
Uses hybrid search (dense + BM25) for high recall.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger("legal_research")


class LegalResearchSystem:
    """RAG-based legal research system.

    Cost-effective design:
    - Hybrid search: dense (semantic) + sparse (keyword) for best recall
    - Enhanced queries with contract context for relevance
    - Source-grounded answers with citations
    - Lite model for synthesis (saves 60% cost)
    """

    def __init__(self, llm=None, kb=None):
        self.llm = llm
        self.kb = kb

    def search(self, query: str, contract_context: str = "",
               top_k: int = 10) -> Dict:
        """Search legal knowledge base and synthesize answer.

        Args:
            query: Legal question to research
            contract_context: Context from the contract being analyzed
            top_k: Number of sources to retrieve

        Returns:
            {
                "answer": "Synthesized answer with citations",
                "sources": [{"text": "...", "source_type": "...", "citation": "...", ...}],
                "total_found": 5
            }
        """
        # Enhance query with contract context
        enhanced_query = query
        if contract_context:
            enhanced_query = f"{query}\n\nContract Context: {contract_context[:500]}"

        # Retrieve from knowledge base
        sources = self._retrieve_sources(enhanced_query, top_k)

        # Synthesize answer
        answer = self._synthesize_answer(query, sources) if self.llm else self._simple_answer(query, sources)

        return {
            "answer": answer,
            "sources": sources,
            "total_found": len(sources),
        }

    def _retrieve_sources(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve legal sources from knowledge base."""
        sources = []

        # Try hybrid search first
        if self.kb:
            try:
                results = self.kb.hybrid_search(query, "legal_laws", top_k)
                sources.extend(results)
            except Exception as e:
                logger.warning(f"Hybrid search error: {e}")

        # Fallback: return guidance
        if not sources:
            sources = [
                {
                    "text": f"Search for: {query}",
                    "source_type": "guidance",
                    "relevance_score": 0.8,
                    "citation": "UAE Legal Knowledge Base",
                }
            ]

        return sources

    def _synthesize_answer(self, query: str, sources: List[Dict]) -> str:
        """Synthesize a coherent answer from retrieved sources."""
        context = "\n\n".join([
            f"[{s.get('source_type', 'SOURCE').upper()}] {s.get('text', '')}"
            for s in sources[:5]
        ])

        prompt = (
            f"[INST] Research this UAE legal question:\n\n"
            f"Question: {query}\n\n"
            f"Legal Sources:\n{context}\n\n"
            f"Provide a clear answer citing specific laws/articles.\n"
            f"Structure: Summary | Legal Basis | Application | Citations [/INST]"
        )

        try:
            return self.llm.generate(prompt, temperature=0.1)
        except Exception as e:
            logger.error(f"Research synthesis error: {e}")
            return self._simple_answer(query, sources)

    def _simple_answer(self, query: str, sources: List[Dict]) -> str:
        """Fallback: simple answer without LLM."""
        citations = "\n".join([
            f"- {s.get('citation', s.get('text', '')[:80])}"
            for s in sources[:3]
        ])
        return (
            f"**Legal Research: {query}**\n\n"
            f"Based on available UAE legal sources:\n\n"
            f"{citations}\n\n"
            f"*Note: This is a preliminary research result. "
            f"Consult a qualified UAE lawyer for definitive legal advice.*"
        )

    def find_similar_cases(self, clause_text: str, top_k: int = 5) -> List[Dict]:
        """Find similar case law for a given clause."""
        if not self.kb:
            return []

        try:
            results = self.kb.query(
                f"UAE court case about: {clause_text[:200]}",
                collection_name="case_law",
                top_k=top_k,
            )
            return results
        except Exception as e:
            logger.warning(f"Case law search error: {e}")
            return []

    def check_law_compliance(self, clause: str, law_ref: str = "") -> Dict:
        """Check if a clause complies with a specific UAE law."""
        prompt = (
            f"[INST] Check UAE legal compliance:\n\n"
            f"Clause: {clause[:500]}\n"
            f"Relevant Law: {law_ref or 'UAE Federal Law'}\n\n"
            f"Analyze: Is this clause compliant? "
            f"Cite specific articles if violated.\n"
            f"Format: Compliant/Non-Compliant | Reasoning | Recommended Fix [/INST]"
        )

        if self.llm:
            try:
                result = self.llm.generate(prompt, temperature=0.1)
                return {"analysis": result, "checked_by": "LLM"}
            except Exception:
                pass

        return {"analysis": "Compliance check requires legal review", "checked_by": "rule"}

"""
Legal Research Agent — conducts UAE legal research using RAG + web search.

Capabilities:
- Search legal knowledge base (vector store)
- Conduct web searches for recent legal updates
- Synthesize findings into structured reports
- Cite sources with specific law articles
"""

from typing import Dict, List, Optional
import logging

from src.agents.base_agent import BaseAgent

logger = logging.getLogger("agent_researcher")


class LegalResearchAgent(BaseAgent):
    """Agent specialized in UAE legal research.

    Tools:
    - search_knowledge_base: Search vector store for legal documents
    - search_web: Search the web for recent legal updates
    - synthesize_findings: Combine multiple sources into coherent answer
    - check_compliance: Check clauses against UAE law
    """

    def __init__(self, llm=None, knowledge_base=None, audit_logger=None):
        super().__init__(name="legal_researcher", llm=llm, audit_logger=audit_logger)
        self.knowledge_base = knowledge_base

        # Register tools
        self.register_tool("search_knowledge_base", self._search_kb, "Search legal knowledge base")
        self.register_tool("search_web", self._search_web, "Search web for legal updates")
        self.register_tool("conduct_research", self._conduct_research, "Full research workflow")
        self.register_tool("check_compliance", self._check_compliance, "Check UAE legal compliance")
        self.register_tool("synthesize_findings", self._synthesize, "Synthesize research findings")

    def _search_kb(self, query: str, top_k: int = 5, **kwargs) -> List[Dict]:
        """Search the legal knowledge base."""
        if not self.knowledge_base:
            return [{"text": "Knowledge base not available", "source_type": "error"}]

        try:
            return self.knowledge_base.hybrid_search(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"KB search error: {e}")
            return []

    def _search_web(self, query: str, **kwargs) -> List[Dict]:
        """Search the web for legal information (stub for real integration)."""
        # In production, integrate with Tavily, SerpAPI, or similar
        logger.info(f"Web search requested: {query[:80]}...")
        return [{"text": "Web search requires API key configuration", "source_type": "info"}]

    def _conduct_research(self, query: str, context_text: str = "",
                          top_k: int = 10, **kwargs) -> Dict:
        """Full research workflow: search → synthesize → report."""
        # 1. Search knowledge base
        kb_results = self._search_kb(query, top_k=top_k)
        web_results = self._search_web(query)

        # 2. Use LLM to synthesize
        if self.llm and kb_results:
            context = "\n\n".join([
                f"[{r.get('source_type', 'SOURCE')}] {r.get('text', '')}"
                for r in kb_results[:5]
            ])

            prompt = (
                f"[INST] UAE Legal Research.\\n\\n"
                f"Query: {query}\\n"
                f"Context: {context_text[:500] if context_text else 'N/A'}\\n\\n"
                f"Sources:\\n{context}\\n\\n"
                f"Provide: 1) Legal answer with citations 2) Relevant laws 3) Practical implications\\n"
                f"Format as structured report. [/INST]"
            )

            try:
                answer = self.llm.generate(prompt, temperature=0.1)
            except Exception as e:
                logger.error(f"Research synthesis error: {e}")
                answer = self._fallback_answer(query, kb_results)
        else:
            answer = self._fallback_answer(query, kb_results)

        return {
            "answer": answer,
            "sources": kb_results + web_results,
            "total_sources": len(kb_results) + len(web_results),
        }

    def _check_compliance(self, clause_text: str, law_ref: str = "", **kwargs) -> Dict:
        """Check compliance of a clause against UAE law."""
        if not self.llm:
            return {"analysis": "Compliance check requires LLM configuration", "compliant": "unknown"}

        prompt = (
            f"[INST] UAE compliance check.\\n\\n"
            f"Clause: {clause_text[:500]}\\n"
            f"Law: {law_ref or 'UAE Federal Law'}\\n\\n"
            f"Is this clause compliant? Cite specific provisions.\\n"
            f"Format: Compliant/Non-Compliant | Reasoning | Recommended Fix [/INST]"
        )

        try:
            analysis = self.llm.generate(prompt, temperature=0.1)
            compliant = "non-compliant" not in analysis.lower()
            return {"analysis": analysis, "compliant": compliant}
        except Exception as e:
            return {"analysis": f"Error: {e}", "compliant": "error"}

    def _synthesize(self, sources: List[Dict], query: str, **kwargs) -> str:
        """Synthesize multiple sources into a coherent answer."""
        if not self.llm or not sources:
            return self._fallback_answer(query, sources)

        context = "\n".join([f"- {s.get('text', '')[:200]}" for s in sources[:5]])

        prompt = (
            f"[INST] Synthesize these sources into a clear answer.\\n\\n"
            f"Question: {query}\\n\\n"
            f"Sources:\\n{context}\\n\\n"
            f"Provide concise, well-cited answer. [/INST]"
        )

        try:
            return self.llm.generate(prompt, temperature=0.1)
        except Exception:
            return self._fallback_answer(query, sources)

    def _fallback_answer(self, query: str, sources: List[Dict]) -> str:
        """Fallback answer without LLM."""
        citations = "\n".join([
            f"- {s.get('citation', s.get('text', '')[:80])}"
            for s in sources[:3]
        ])
        return (
            f"**Legal Research: {query}**\\n\\n"
            f"Based on available sources:\\n{citations}\\n\\n"
            f"*Note: This is preliminary. Consult a qualified UAE lawyer.*"
        )

"""
Orchestrator Agent — coordinates specialized agents for complex legal tasks.

Routes tasks to the appropriate agent subsystem:
- ContractAnalysisAgent → analyzes contracts
- LegalResearchAgent → researches UAE laws
- ContractDraftingAgent → drafts contracts
- RiskComplianceAgent → assesses risks
- LifecycleAgent → manages contract lifecycle
"""

from typing import Dict, List, Optional, Any
import time
import logging
import uuid

from src.agents.base_agent import BaseAgent

logger = logging.getLogger("orchestrator")


class OrchestratorAgent(BaseAgent):
    """Orchestrates multi-agent workflows for complex legal tasks.

    Design:
    - Evaluates task requirements and selects best agent
    - For complex tasks, chains multiple agents together
    - Each step is audited for full traceability
    - Fallback strategies when primary agent fails
    """

    def __init__(self, llm=None, audit_logger=None):
        super().__init__(name="orchestrator", llm=llm, audit_logger=audit_logger)
        self._agents: Dict[str, BaseAgent] = {}

    def register_agent(self, name: str, agent: BaseAgent):
        """Register a specialized agent."""
        self._agents[name] = agent
        logger.info(f"Registered agent: {name}")

    def execute_task(self, task: str, context: Dict[str, Any]) -> Dict:
        """Execute a task by routing to appropriate agent(s).

        Args:
            task: Task type ('analyze', 'draft', 'review', 'research', 'manage')
            context: Task context including contract text, params, user info

        Returns:
            Dict with results, trace_id, agent_chain
        """
        trace_id = f"ORCH-{uuid.uuid4().hex[:8]}"
        start = time.time()

        task_map = {
            "analyze": self._execute_analysis,
            "draft": self._execute_drafting,
            "review": self._execute_review,
            "research": self._execute_research,
            "manage": self._execute_lifecycle,
        }

        handler = task_map.get(task)
        if not handler:
            return {"error": f"Unknown task: {task}", "trace_id": trace_id}

        try:
            result = handler(context)
            duration = (time.time() - start) * 1000

            if self.audit_logger:
                self.audit_logger.log(
                    trace_id=trace_id,
                    action=f"orchestrate:{task}",
                    subsystem="orchestrator",
                    duration_ms=round(duration, 1),
                    summary=f"Orchestrated {task} task",
                )

            return {
                "trace_id": trace_id,
                "task": task,
                "result": result,
                "duration_ms": round(duration, 1),
            }
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Orchestration failed for task '{task}': {e}")

            if self.audit_logger:
                self.audit_logger.log(
                    trace_id=trace_id,
                    action=f"orchestrate:{task}",
                    subsystem="orchestrator",
                    success=False,
                    duration_ms=round(duration, 1),
                    error=str(e),
                )

            return {"error": str(e), "trace_id": trace_id}

    def _execute_analysis(self, context: Dict) -> Dict:
        """Execute contract analysis using analysis + research agents."""
        results = {}

        if "analysis" in self._agents:
            analysis_agent = self._agents["analysis"]
            results["analysis"] = analysis_agent.dispatch(
                "analyze_contract",
                contract_text=context.get("contract_text", ""),
                contract_type=context.get("contract_type", ""),
            )

        if "research" in self._agents:
            research_agent = self._agents["research"]
            results["research_context"] = research_agent.dispatch(
                "search_law",
                query=f"Contract type: {context.get('contract_type', '')}",
            )

        return results

    def _execute_drafting(self, context: Dict) -> Dict:
        """Execute contract drafting using drafting agent."""
        if "drafting" not in self._agents:
            return {"error": "Drafting agent not available"}

        drafting_agent = self._agents["drafting"]
        result = drafting_agent.dispatch(
            "draft_contract",
            contract_type=context.get("contract_type"),
            params=context.get("params", {}),
            language=context.get("language", "english"),
        )
        return {"draft": result, "requires_review": True}

    def _execute_review(self, context: Dict) -> Dict:
        """Execute full review: analysis + risk assessment."""
        results = {}

        if "analysis" in self._agents:
            analysis_agent = self._agents["analysis"]
            analysis = analysis_agent.dispatch(
                "analyze_contract",
                contract_text=context.get("contract_text", ""),
                contract_type=context.get("contract_type", ""),
            )
            results["analysis"] = analysis

        if "risk" in self._agents:
            risk_agent = self._agents["risk"]
            risk = risk_agent.dispatch(
                "assess_risk",
                contract_text=context.get("contract_text", ""),
                analysis=analysis or {},
            )
            results["risk_assessment"] = risk

        requires_review = (
            results.get("risk_assessment", {}).get("requires_lawyer_review", False)
        )
        return {**results, "requires_lawyer_review": requires_review}

    def _execute_research(self, context: Dict) -> Dict:
        """Execute legal research."""
        if "research" not in self._agents:
            return {"error": "Research agent not available"}

        research_agent = self._agents["research"]
        result = research_agent.dispatch(
            "conduct_research",
            query=context.get("query", ""),
            context_text=context.get("contract_text", ""),
        )
        return {"research": result}

    def _execute_lifecycle(self, context: Dict) -> Dict:
        """Execute lifecycle management tasks."""
        if "lifecycle" not in self._agents:
            return {"error": "Lifecycle agent not available"}

        lifecycle_agent = self._agents["lifecycle"]
        action = context.get("lifecycle_action", "register")
        result = lifecycle_agent.dispatch(
            action,
            **context.get("params", {}),
        )
        return {"lifecycle": result}

    def get_agent_status(self) -> Dict:
        """Get status of all registered agents."""
        return {
            name: agent.get_stats()
            for name, agent in self._agents.items()
        }

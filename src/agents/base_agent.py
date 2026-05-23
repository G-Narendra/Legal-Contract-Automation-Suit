"""
Base agent class for Legal Contract Automation Suite.

Provides the foundation for specialized agents with:
- Tool registration and execution
- State management
- Audit logging
- Cost tracking
"""

from typing import Dict, List, Optional, Any, Callable
import time
import logging
import uuid

logger = logging.getLogger("legal_agent")


class BaseAgent:
    """Foundation for all specialized agents in the system.

    Design:
    - Tool-based architecture: agents use registered tools via dispatch()
    - State persists across the agent's lifecycle
    - Every action is audited for traceability
    - Cost per action is tracked for optimization
    """

    def __init__(self, name: str, llm=None, audit_logger=None):
        self.name = name
        self.llm = llm
        self.audit_logger = audit_logger
        self._tools: Dict[str, Callable] = {}
        self._state: Dict[str, Any] = {}
        self._start_time = time.time()

    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a tool that this agent can use."""
        self._tools[name] = {"func": func, "description": description, "calls": 0}

    def dispatch(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool with audit logging."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not registered with agent '{self.name}'")

        tool_info = self._tools[tool_name]
        tool_info["calls"] += 1
        start = time.time()

        trace_id = f"AGT-{uuid.uuid4().hex[:8]}"
        try:
            result = tool_info["func"](**kwargs)
            duration = (time.time() - start) * 1000

            if self.audit_logger:
                self.audit_logger.log(
                    trace_id=trace_id,
                    action=f"tool:{tool_name}",
                    subsystem=f"agent:{self.name}",
                    duration_ms=round(duration, 1),
                    summary=f"Agent {self.name} executed tool {tool_name}",
                )

            logger.debug(f"Agent '{self.name}' executed tool '{tool_name}' in {duration:.0f}ms")
            return result
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Agent '{self.name}' tool '{tool_name}' failed: {e}")

            if self.audit_logger:
                self.audit_logger.log(
                    trace_id=trace_id,
                    action=f"tool:{tool_name}",
                    subsystem=f"agent:{self.name}",
                    success=False,
                    duration_ms=round(duration, 1),
                    error=str(e),
                )
            raise

    def set_state(self, key: str, value: Any):
        """Set agent state variable."""
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get agent state variable."""
        return self._state.get(key, default)

    def get_stats(self) -> Dict:
        """Get agent usage statistics."""
        runtime = time.time() - self._start_time
        tool_calls = {name: info["calls"] for name, info in self._tools.items()}
        return {
            "agent": self.name,
            "runtime_seconds": round(runtime, 1),
            "tool_calls": tool_calls,
            "total_tool_calls": sum(tool_calls.values()),
            "state_variables": len(self._state),
        }

    def list_tools(self) -> Dict[str, str]:
        """List available tools with descriptions."""
        return {name: info["description"] for name, info in self._tools.items()}

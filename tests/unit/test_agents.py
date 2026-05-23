"""
Unit tests for agent system.
"""

import pytest
from src.agents.base_agent import BaseAgent


class TestBaseAgent:
    """Test suite for BaseAgent."""

    def test_agent_creation(self):
        agent = BaseAgent(name="test_agent")
        assert agent.name == "test_agent"
        assert agent.list_tools() == {}

    def test_register_tool(self):
        agent = BaseAgent(name="test")
        def my_tool(x: int) -> int:
            return x * 2
        agent.register_tool("double", my_tool, "Double a number")
        tools = agent.list_tools()
        assert "double" in tools
        assert tools["double"] == "Double a number"

    def test_dispatch_tool(self):
        agent = BaseAgent(name="test")
        def add(a, b):
            return a + b
        agent.register_tool("add", add)
        result = agent.dispatch("add", a=3, b=4)
        assert result == 7

    def test_dispatch_unknown_tool(self):
        agent = BaseAgent(name="test")
        with pytest.raises(ValueError, match="not registered"):
            agent.dispatch("nonexistent")

    def test_state_management(self):
        agent = BaseAgent(name="test")
        agent.set_state("key1", "value1")
        agent.set_state("key2", 42)
        assert agent.get_state("key1") == "value1"
        assert agent.get_state("key2") == 42
        assert agent.get_state("nonexistent") is None
        assert agent.get_state("nonexistent", "default") == "default"

    def test_get_stats(self):
        agent = BaseAgent(name="test_agent")
        agent.register_tool("tool1", lambda: None)
        stats = agent.get_stats()
        assert stats["agent"] == "test_agent"
        assert "tool_calls" in stats
        assert "runtime_seconds" in stats

    def test_tool_call_counting(self):
        agent = BaseAgent(name="test")
        def my_tool():
            return "done"
        agent.register_tool("my_tool", my_tool)
        agent.dispatch("my_tool")
        agent.dispatch("my_tool")
        stats = agent.get_stats()
        assert stats["tool_calls"]["my_tool"] == 2

    def test_dispatch_with_kwargs(self):
        agent = BaseAgent(name="test")
        def greet(greeting, name):
            return f"{greeting}, {name}!"
        agent.register_tool("greet", greet)
        result = agent.dispatch("greet", greeting="Hello", name="World")
        assert result == "Hello, World!"

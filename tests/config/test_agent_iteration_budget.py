from nanobot.config.schema import AgentDefaults


def test_default_agent_tool_iteration_budget_is_bounded() -> None:
    assert AgentDefaults().max_tool_iterations == 30


def test_agent_tool_iteration_budget_keeps_explicit_override() -> None:
    assert AgentDefaults(maxToolIterations=120).max_tool_iterations == 120

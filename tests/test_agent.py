import json

from src.agent.agent import run_rca_agent, AnomalyContext
from src.agent.llm_client import FakeLLMClient


def _sample_context() -> AnomalyContext:
    return AnomalyContext(
        anomaly_type="cpu_spike",
        metric="cpu_pct",
        start_idx=100,
        end_idx=112,
        peak_values={"cpu_pct": 88.0, "latency_ms": 210.0, "memory_pct": 47.0, "error_rate": 0.02, "throughput_rps": 180.0},
        baseline_values={"cpu_pct": 35.0, "latency_ms": 120.0, "memory_pct": 45.0, "error_rate": 0.01, "throughput_rps": 200.0},
    )


def test_agent_calls_tool_then_concludes():
    """Two-step script: check the runbook, then conclude with a diagnosis."""
    scripted = [
        json.dumps({"thought": "Let's check known causes", "action": "check_runbook", "action_input": {}}),
        json.dumps(
            {
                "thought": "Runbook points to deploys or retry storms; CPU and latency both elevated",
                "action": "conclude",
                "action_input": {
                    "diagnosis": "Likely an inefficient query introduced recently, given correlated CPU/latency rise.",
                    "confidence": 0.7,
                },
            }
        ),
    ]
    llm = FakeLLMClient(scripted)
    result = run_rca_agent(_sample_context(), llm)

    assert result.stopped_reason == "concluded"
    assert result.confidence == 0.7
    assert "inefficient query" in result.diagnosis
    assert len(result.trace) == 2
    assert result.trace[0].action == "check_runbook"
    assert result.trace[1].action == "conclude"


def test_agent_handles_markdown_fenced_json():
    """Models often wrap JSON in ```json fences even when told not to — must still parse."""
    scripted = [
        "```json\n" + json.dumps(
            {"thought": "checking", "action": "conclude", "action_input": {"diagnosis": "fenced test", "confidence": 0.5}}
        ) + "\n```"
    ]
    llm = FakeLLMClient(scripted)
    result = run_rca_agent(_sample_context(), llm)

    assert result.diagnosis == "fenced test"
    assert result.confidence == 0.5


def test_agent_stops_after_max_steps_without_concluding():
    """If the model never concludes, the agent must stop honestly rather than loop forever."""
    non_concluding_step = json.dumps(
        {"thought": "still investigating", "action": "get_deployment_history", "action_input": {}}
    )
    scripted = [non_concluding_step] * 5  # MAX_STEPS = 5, never concludes
    llm = FakeLLMClient(scripted)
    result = run_rca_agent(_sample_context(), llm)

    assert result.stopped_reason == "max_steps_reached"
    assert result.confidence == 0.0
    assert len(result.trace) == 5


def test_agent_handles_unparseable_response_gracefully():
    scripted = ["this is not json at all, sorry"]
    llm = FakeLLMClient(scripted)
    result = run_rca_agent(_sample_context(), llm)

    assert result.stopped_reason == "parse_error"
    assert result.confidence == 0.0


def test_agent_reports_low_confidence_when_model_says_evidence_is_thin():
    """The agent should be able to conclude with low confidence rather than fabricate certainty."""
    scripted = [
        json.dumps(
            {
                "thought": "No strong signal from runbook or logs",
                "action": "conclude",
                "action_input": {
                    "diagnosis": "Insufficient evidence to confirm root cause; recommend checking downstream dependency health directly.",
                    "confidence": 0.2,
                },
            }
        )
    ]
    llm = FakeLLMClient(scripted)
    result = run_rca_agent(_sample_context(), llm)

    assert result.confidence == 0.2
    assert "Insufficient evidence" in result.diagnosis

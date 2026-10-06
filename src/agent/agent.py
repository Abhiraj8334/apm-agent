"""
agent.py

The root-cause analysis agent. Given a flagged anomaly, it runs a
multi-step loop: hypothesize a cause, decide which tool would gather
useful evidence, call it, and either revise its hypothesis or conclude
with a diagnosis and a confidence score. The full reasoning trace is
returned alongside the diagnosis so *why* the agent reached its
conclusion is visible, not just the conclusion itself.

This is deliberately not "one prompt in, one answer out" — that would be
prompt engineering, not an agent. The distinguishing property here is that
the model chooses which evidence to gather next based on what it has
already seen, and can decide it lacks enough evidence to conclude at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.agent.llm_client import LLMClient
from src.agent.tools import (
    TOOL_REGISTRY,
    check_runbook,
    query_recent_logs,
    get_correlated_metrics,
    get_deployment_history,
)

MAX_STEPS = 5  # one extra step of headroom, since an empty-diagnosis retry now consumes a step

TOOL_DESCRIPTIONS = """\
- check_runbook: look up known causes and remediation guidance for this anomaly type. No input needed.
- query_recent_logs: fetch log lines from the anomaly's time window. No input needed.
- get_correlated_metrics: see the other metrics' values at the same point, to check if they moved together. No input needed.
- get_deployment_history: list recent deploys (optionally: {"lookback_minutes": <int>}, default 360).
- conclude: stop investigating and report your diagnosis. Requires "diagnosis" (string) and "confidence" (0.0-1.0).\
"""


@dataclass
class AnomalyContext:
    anomaly_type: str
    metric: str
    start_idx: int
    end_idx: int
    peak_values: dict[str, float]   # all metrics' values at the most extreme point in the window
    baseline_values: dict[str, float]  # typical/mean values for comparison


@dataclass
class TraceStep:
    step: int
    thought: str
    action: str
    action_input: dict
    observation: str


@dataclass
class RCAResult:
    diagnosis: str
    confidence: float
    trace: list[TraceStep] = field(default_factory=list)
    stopped_reason: str = "concluded"  # "concluded" | "max_steps_reached"


def _extract_json(text: str) -> dict:
    """
    Pull a JSON object out of the model's response, tolerating markdown
    code fences (```json ... ```) which models commonly add even when
    asked not to.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort: find the first {...} span in the text.
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))
        raise


def _build_initial_prompt(ctx: AnomalyContext) -> str:
    peak_str = ", ".join(f"{k}={v:.2f}" for k, v in ctx.peak_values.items())
    baseline_str = ", ".join(f"{k}={v:.2f}" for k, v in ctx.baseline_values.items())
    return f"""You are a senior site reliability engineer investigating a performance anomaly.

Anomaly detected:
  Type: {ctx.anomaly_type}
  Primary metric: {ctx.metric}
  Window: index {ctx.start_idx} to {ctx.end_idx}
  Values during anomaly: {peak_str}
  Normal baseline values: {baseline_str}

Available actions:
{TOOL_DESCRIPTIONS}

Investigate step by step. At each step, respond with ONLY a JSON object
(no markdown fences, no extra text) in this exact shape:
{{"thought": "<your reasoning>", "action": "<one of the action names above>", "action_input": {{}}}}

IMPORTANT: "action" must be EXACTLY one of these strings — check_runbook,
query_recent_logs, get_correlated_metrics, get_deployment_history, conclude
— copied verbatim. Do not invent a new action name, do not describe what
you want to do instead of naming a real action, and do not put a nested
object or function-call structure in "action" — it is always a plain string.

If you have enough evidence, use action "conclude" with:
{{"thought": "<summary>", "action": "conclude", "action_input": {{"diagnosis": "<root cause and recommended fix>", "confidence": <0.0-1.0>}}}}

If the evidence is genuinely inconclusive, still conclude, but with low
confidence and a diagnosis that says what's uncertain and what additional
evidence would help — do not fabricate certainty you don't have.

Respond now with your first step."""


def _build_followup_prompt(observation: str) -> str:
    return f"""Observation from your last action:
{observation}

Based on this, respond with your next JSON step (same format as before —
either another action to gather more evidence, or "conclude" if you have enough)."""


def run_rca_agent(ctx: AnomalyContext, llm: LLMClient) -> RCAResult:
    trace: list[TraceStep] = []
    prompt = _build_initial_prompt(ctx)

    for step_num in range(1, MAX_STEPS + 1):
        raw_response = llm.call(prompt)
        try:
            decision = _extract_json(raw_response)
        except (json.JSONDecodeError, AttributeError):
            trace.append(
                TraceStep(
                    step=step_num,
                    thought="(unparseable response)",
                    action="error",
                    action_input={},
                    observation=f"Could not parse model response as JSON: {raw_response[:200]}",
                )
            )
            return RCAResult(
                diagnosis="Investigation halted: model response could not be parsed.",
                confidence=0.0,
                trace=trace,
                stopped_reason="parse_error",
            )

        thought = decision.get("thought", "")
        action = decision.get("action", "")
        action_input = decision.get("action_input", {}) or {}

        if action == "conclude":
            diagnosis = (action_input.get("diagnosis") or "").strip()
            if not diagnosis:
                observation = (
                    "Your 'conclude' action had an empty or missing 'diagnosis'. "
                    "You must write an actual diagnosis string — even a low-confidence "
                    "one describing what's uncertain — before concluding."
                )
                trace.append(
                    TraceStep(step=step_num, thought=thought, action=action, action_input=action_input, observation=observation)
                )
                print(f"Step {step_num} [{action}] — empty diagnosis rejected, asking model to retry")
                prompt = _build_followup_prompt(observation)
                continue

            confidence = float(action_input.get("confidence", 0.0))
            trace.append(
                TraceStep(
                    step=step_num,
                    thought=thought,
                    action="conclude",
                    action_input=action_input,
                    observation="(investigation concluded)",
                )
            )
            print(f"Step {step_num} [conclude] confidence={confidence:.0%}")
            return RCAResult(diagnosis=diagnosis, confidence=confidence, trace=trace, stopped_reason="concluded")

        observation = _execute_tool(action, action_input, ctx)
        trace.append(
            TraceStep(step=step_num, thought=thought, action=action, action_input=action_input, observation=observation)
        )
        print(f"Step {step_num} [{action}] -> {observation[:80]}{'...' if len(observation) > 80 else ''}")
        prompt = _build_followup_prompt(observation)

    return RCAResult(
        diagnosis="Investigation inconclusive after maximum steps: insufficient evidence gathered to confirm a root cause.",
        confidence=0.0,
        trace=trace,
        stopped_reason="max_steps_reached",
    )


def _execute_tool(action: str, action_input: dict, ctx: AnomalyContext) -> str:
    if action == "check_runbook":
        return check_runbook(ctx.anomaly_type)
    elif action == "query_recent_logs":
        logs = query_recent_logs(ctx.metric, ctx.start_idx, ctx.end_idx)
        return "\n".join(logs) if logs else "(no relevant log lines found)"
    elif action == "get_correlated_metrics":
        correlated = get_correlated_metrics(ctx.peak_values, exclude=ctx.metric)
        return ", ".join(f"{k}={v:.2f}" for k, v in correlated.items())
    elif action == "get_deployment_history":
        lookback = int(action_input.get("lookback_minutes", 360))
        deploys = get_deployment_history(lookback)
        if not deploys:
            return f"No deploys found in the last {lookback} minutes."
        return "\n".join(f"{d['service']} {d['version']} ({d['minutes_ago']} min ago)" for d in deploys)
    else:
        valid_actions = ", ".join(list(TOOL_REGISTRY.keys()) + ["conclude"])
        return (
            f"Unknown action '{action}' — no tool with that name exists. "
            f"You must use the exact action name from this list: {valid_actions}."
        )

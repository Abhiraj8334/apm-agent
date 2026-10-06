# APM-Agent

An AI-driven application performance monitoring tool: ingests system metrics,
detects anomalies with ML, and diagnoses root causes with an LLM agent.

## Status: Phase 3 complete (LLM root-cause agent)

**Phase 1 — Metrics simulator**: generates a realistic synthetic metrics time
series (latency, CPU, memory, error rate, throughput) with injectable,
**ground-truth-labeled** anomalies (CPU spikes, latency spikes, gradual
memory leaks, error bursts).

**Phase 2 — Detection + evaluation**: two detectors, scored against each
other on the same ground truth, not just demoed in isolation:
- `ThresholdBaseline` — flags a point if any metric's z-score exceeds a cutoff.
  Simple, and a useful sanity check for whether the fancier approach is
  actually earning its complexity.
- `IsolationForestDetector` — first removes trend/seasonality per metric via
  STL decomposition, then runs Isolation Forest on the residuals *jointly*
  across all metrics, so it can catch correlated multivariate anomalies a
  per-metric threshold would miss or under-weight.
- `evaluate.py` — computes precision, recall, F1, and false-positive rate for
  both detectors against `data/anomaly_labels.json`, at two levels:
  - **Point-level** (strict): every point inside an anomaly window must be flagged.
  - **Window-level**: an alert anywhere inside the window counts as a catch —
    closer to how real alerting/APM tools are actually judged.
  Isolation Forest's `contamination` is set to the dataset's true anomaly
  rate (measured, not guessed) for a fair comparison.

**Honest finding from this dataset**: `ThresholdBaseline` currently
outperforms `IsolationForest+STL` on this dataset at both point- and
window-level. The injected anomalies are sharp, sudden shifts — close to
the ideal case for a simple z-score cutoff — so STL's seasonality-removal
isn't buying much here, and residual estimation adds noise instead.

At window-level, Isolation Forest's recall improves substantially (0.90,
catching 9/10 windows) but precision drops sharply (0.18) because it fires
40 separate single-point false-alarm episodes scattered outside the true
anomaly windows. This is a known, well-understood failure mode: the
detector is finding real signal, but isolated single-point flags are noisy.

**Identified next step (not yet built)**: alert debouncing / hysteresis —
only firing an alert if N consecutive points are flagged, not a single
blip. This is standard practice in real monitoring systems specifically to
avoid alert fatigue, and would likely close most of the precision gap here
without hurting recall much. Reported as a concrete, scoped improvement
rather than left as an unexplained weak result.

**Phase 3 — RCA agent**: given a flagged anomaly, the agent runs a genuine
multi-step investigation rather than a single prompt-in-answer-out call:
- `src/agent/tools.py` — evidence-gathering tools (runbook lookup, log
  query, correlated-metrics check, deployment history), mocked/synthetic
  but structured exactly like real integrations would be.
- `src/agent/llm_client.py` — a pluggable `LLMClient` interface. The agent
  never imports Gemini directly, so swapping backends later means adding
  one class here, not touching the agent's reasoning loop.
- `src/agent/agent.py` — the loop: hypothesize → pick a tool → observe →
  revise or conclude, up to 4 steps. The full reasoning trace is returned
  alongside the diagnosis, and the agent can conclude with **low
  confidence and an honest "insufficient evidence" statement** rather than
  fabricating certainty — this was a deliberate design goal, not an
  afterthought.
- `src/agent/run_rca.py` — end-to-end CLI: loads a real flagged anomaly
  from the detector's output and runs the agent against it via Gemini.

The agent loop is tested with a scripted fake LLM (`FakeLLMClient`), so the
control flow (tool execution, JSON parsing including markdown-fenced
responses, the max-steps fallback, low-confidence conclusions) is verified
without needing a real API key or network access.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For the LLM agent (Phase 4), you'll need a Gemini API key:
```bash
cp .env.example .env
# edit .env and add your GEMINI_API_KEY
```

## Usage

Generate a fresh synthetic dataset:
```bash
python -m src.simulator.metrics_simulator
```

This writes:
- `data/metrics.db` — SQLite table of all metric points
- `data/anomaly_labels.json` — ground-truth anomaly windows (type, start, end)

Run the detector evaluation (after generating data):
```bash
python -m src.detector.evaluate
```

Run the RCA agent end-to-end (requires a Gemini API key in `.env`):
```bash
python -m src.agent.run_rca
```

Run tests (agent tests use a scripted fake LLM — no API key needed):
```bash
pytest tests/ -v
```

## Architecture (planned)

```
Simulator -> SQLite -> Anomaly Detector (Isolation Forest + STL) -> Alert Queue
                                                                        |
                                                                        v
                                                          RCA Agent (Gemini, tool-calling)
                                                                        |
                                                                        v
                                                              Streamlit Dashboard
                                          Evaluation Harness scores the Detector against
                                          data/anomaly_labels.json (precision/recall/F1)
```

## Roadmap

- [x] Phase 1 — Metrics simulator with labeled synthetic anomalies
- [x] Phase 2 — Anomaly detection (STL + Isolation Forest) with evaluation harness (precision/recall/F1 vs. threshold baseline)
- [x] Phase 3 — LLM root-cause agent (Gemini, multi-step tool-calling, tested with a scripted fake LLM)
- [ ] Phase 4 — Resilience (async ingestion, retries, circuit breaker)
- [ ] Phase 5 — Streamlit dashboard

## Design notes

- **Ground truth first**: the simulator labels every anomaly it injects, so
  detector quality is measured, not assumed.
- **Pluggable LLM backend**: the agent calls a single `call_llm()` interface
  so the backend (Gemini today) can be swapped without touching agent logic.
- **Synthetic data, honestly labeled as such**: this is a portfolio/learning
  project, not a claim of production-scale validation.

# Autonomous Artificial Brain

This repository now contains the **Phase 1 foundation** of an experimental
Autonomous Artificial Brain (AAB). It is deliberately small: a PyTorch model
learns the four examples in the XOR truth table, an evaluation function
measures its performance, and a SQLite experience store persists the measured
result with provenance and verification status.

It does **not** claim consciousness, general intelligence, autonomous research,
or self-improvement. Those capabilities are future phases with separate
interfaces and evaluation requirements.

## Architecture

| Component | Purpose |
| --- | --- |
| `aab/models/neural_network.py` | Defines the small `XORNetwork` PyTorch model. |
| `aab/learning/training_pipeline.py` | Trains, evaluates, and records one experiment. |
| `aab/memory/database.py` | Stores structured experiences in SQLite. |
| `main.py` | Runs the Phase 1 demonstration. |
| `aab/brain/` | Provider-neutral reasoning, curiosity, and reversible decision selection. |
| `aab/autonomy/` | Bounded control loop, task history, lifecycle hooks, and resource limits. |
| `aab/research/` | Read-only public search, bounded page retrieval, source assessment, and citations. |
| `aab/learning/` | Prediction/outcome tracking, recurring-error analysis, and benchmark-gated strategy comparison. |
| `tests/` | Verifies learning, cognition, evaluation, and persistence. |

The storage interface is intentionally independent of the model so later
semantic and episodic memory implementations can be added without coupling
them to neural-network training.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the demonstration

```bash
python main.py
```

This trains a fresh model, reports its measured accuracy and loss, and writes
the experience to `data/experiences.db`.

## Run tests

```bash
python -m pytest
```

## Scope and next phase

Phase 1 demonstrates measurable supervised learning and persistent experiment
memory. Phase 2 adds offline structured reasoning, ranked question proposals,
and configurable action scoring. Phase 3 adds bounded autonomous coordination
with duplicate-task prevention, pause/resume hooks, shutdown, and finite
resource limits. It does not implement web research, fine-tuning, or
self-modification.

Phase 5 adds experience-based evaluation without claiming unrestricted learning:
predictions are compared with separately supplied outcomes, recurring errors are
counted, alternative strategies are proposed, and benchmark results determine
whether a candidate strategy outperforms a baseline. Neural-network training
and model replacement remain separate concerns.

The default `LocalMockProvider` is deterministic and requires no credentials.
Future providers must read credentials from environment variables such as
`AAB_MODEL_API_KEY`; secrets are never stored in source files.
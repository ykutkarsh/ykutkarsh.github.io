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
| `tests/` | Verifies learning, evaluation, and persistence. |

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
memory. It does not yet implement a language-model reasoning engine, curiosity,
autonomous control loop, web research, decision-making, or self-modification.
Phase 2 should add provider-neutral reasoning and question-generation
interfaces, backed by tests and explicit uncertainty handling, without merging
those concerns into the training pipeline.
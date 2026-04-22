# Docs Index (Implementation-Only)

This folder documents the code that currently exists in this repository.
It intentionally avoids roadmap material, speculative architecture, and
historical notes that are not reflected in the current code.

## Why Two Architecture Docs Exist

- `docs/architecture/PROJECT_ARCHITECTURE.md` is the system-level architecture.
  It explains components across `scripts/`, `src/lexisflow/`, `artifacts/`,
  and `results/`.
- `docs/SWEEP_ARCHITECTURE.md` is the operator view for sweep execution only:
  commands, lifecycle, outputs, and resume behavior.

They overlap by design at a high level, but answer different questions:
whole-system structure vs. sweep execution details.

## Canonical Docs

- `docs/architecture/PROJECT_ARCHITECTURE.md`
  - End-to-end architecture and module ownership.
- `docs/SWEEP_ARCHITECTURE.md`
  - Sweep pipeline behavior (`scripts/run_sweep.py` and dataset drivers).
- `docs/data/processing.md`
  - Dataset preparation and preprocessing outputs for MIMIC and Challenge 2012.
- `docs/data/features.md`
  - Static/dynamic feature semantics and autoregressive lag layout.
- (intentionally removed) old schema/status docs that mixed implementation with
  historical/project-management notes.

## Entry Points

- Public command:
  - `uv run python scripts/run_sweep.py --dataset mimic`
  - `uv run python scripts/run_sweep.py --dataset challenge2012`
- Visualization:
  - `uv run python scripts/common/analyze_sweep.py`
  - `uv run python scripts/common/analyze_sweep.py --dataset challenge2012`

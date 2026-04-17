## Learned User Preferences

- Prefers extending or rewriting existing workflow scripts and evaluation modules rather than adding parallel entry points or one-off scripts, to keep the repository surface small.
- Prefers fixing data preparation and outputs by changing scripts (not by hand-editing datasets or generated artifacts).
- For pipeline steps that should always run together (for example coupling in the main sweep driver), prefers default-on behavior and a small CLI surface over many optional flags.
- For substantive multi-step writing or code changes, expects a short plan first, then implementation, followed by a self-review pass; asks the agent to surface questions when requirements are unclear.
- For explanations, status updates, thesis-style summaries, and requested architecture or quality reviews, prefers a small set of high-signal points developed in depth (including candid identification of gaps where asked) over long enumerations.
- On implementation-heavy tasks, may request skipping ancillary documentation beyond what was explicitly asked for, to keep iterations lightweight.
- When heavy pipelines have already been run and verified locally, may direct the agent not to re-run those scripts.
- Wants multi-seed robustness wired into the main sweep or sampling flows where practical, rather than many ad-hoc flags or duplicate driver scripts.

## Learned Workspace Facts

- `synth-gen` is a KCL dissertation codebase for synthetic tabular ICU trajectories: primary models include ForestFlow and HS3F, with shared preprocessing and driver scripts under `scripts/` and evaluation logic under `packages/evaluation/` (utility, trajectory, TSTR-style, and privacy-related metrics).
- External baseline comparison for the thesis is oriented around CTGAN via SDV, trained and scored on the same processed data and metric stack as the in-repo models.
- The LaTeX report draft used for submission-style writing lives under `report_latex/`, alongside thesis-oriented material under `docs/Chapters/`.
- Python unit tests are co-located with their packages under `packages/*/tests/` (for example `packages/data/tests`, `packages/evaluation/tests`, `packages/models/tests`).

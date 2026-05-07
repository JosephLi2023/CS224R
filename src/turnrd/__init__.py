"""TurnRD (Method B) — learned per-turn reward decomposer.

Surface (M1):
- `TurnRDConfig`, `TurnRDOutput`, `TurnRD` from `src.turnrd.model`.

Adds `dataset.py` (replay buffer reader) and `train.py` (standalone
trainer entrypoint), plus the production policy-hidden-state embedder.
See `MEDIUM_FIXES.md::M1`.
"""

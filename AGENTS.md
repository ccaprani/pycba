# AGENTS.md

Notes for AI agents and human contributors working on PyCBA. Keep changes consistent with the points below.

## What PyCBA is (scope)

PyCBA is a focused **1-D continuous-beam engine** built on the matrix (direct) stiffness method, for the
design, assessment and teaching of beams and bridges. It is deliberately **not** a general 2-D/3-D finite
element package — that space is well served by other tools (anaStruct, PyNite, …). New features should fit
the 1-D continuous-beam framing (linear/nonlinear statics, moving loads, modal, foundations, prestress,
visualisation). Keep the core dependency set light: **numpy, scipy, matplotlib**.

## Environment & tests

- Use the project's virtualenv/conda interpreter (don't rely on the system Python).
- Run the suite before opening a PR:
  ```bash
  python -m pytest tests/ -v
  ```
- **Optional dependencies** must be guarded so the suite passes without them:
  - `pandas` is in the `test` extra; `plotly` is separate (`pip install pycba[plotly]`).
  - Tests that touch them should `pytest.importorskip("pandas")` / `("plotly")`.

## Code style

- Formatted with **black**, pinned to `black==23.12.1`. Newer black versions reformat the whole repo —
  match the pinned version.

## Branching & merging

- `main` is protected (review required). Work on feature branches and open a PR.
- PRs are **squash-merged**. Rebase sequential PRs on `origin/main` first (CHANGELOG/tutorial conflicts are
  usually additive).

## Documentation

- The Sphinx docs render **committed notebook outputs** (`nbsphinx` is configured with `execute="never"`).
  After changing plotting or API code, **re-execute the affected notebooks** (e.g. via `nbconvert`'s
  `ExecutePreprocessor`, run with the project interpreter) and commit the updated `.ipynb` — otherwise the
  rendered docs won't reflect the change.

## Versioning & releases

- The version is dynamic: `pyproject.toml` reads `pycba.__version__` (in `src/pycba/__init__.py`).
- Release flow: **bump the version + update `CHANGELOG.md` in a PR → merge to `main` → tag → create the
  GitHub release.** Always merge consequential changes (version, licence) to `main` *before* creating the
  release.
- Creating a GitHub release triggers the **PyPI publish** workflow. **PyPI versions are immutable** — a
  version can be yanked/deleted but never re-uploaded, so make sure `main` is exactly what you want to ship
  before tagging.

## Licence

PyCBA is licensed **AGPL-3.0-or-later**. Contributions are accepted under the same licence.

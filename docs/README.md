# docs/

Full technical documentation for the **Hybrid Movie Recommender** (MSc AI · Εφαρμογές
Τεχνητής Νοημοσύνης · Θέμα 2). Each file is self-contained and cross-links the others.
For the project overview, repo layout, and quick-start, see the root
[`README.md`](../README.md).

## Contents

| Doc | What it covers |
|---|---|
| [**evaluation.md**](evaluation.md) | The **full evaluation record** — protocol, every metric formula, the master results table (12 models × all K), per-model analysis, robustness checks (bootstrap/segmented/cold-start/full-catalogue), the hybrid-fusion analysis, the case study, threats to validity, and the verdict — with ~30 figures embedded. |
| [**models.md**](models.md) | Every one of the **12 models** in depth — philosophy, the actual maths, training mechanics, prediction formula, hyperparameters, complexity/memory, strengths & weaknesses, **background mechanics** (Surprise predict pipeline, Pearson-baseline shrinkage, fallback/edge-case semantics for all 12). Includes the **full results table** on the held-out test set. |
| [**notebooks.md**](notebooks.md) | The **14-notebook pipeline** — what each notebook reads/writes/produces, the train-and-evaluate-per-model structure, and the shared evaluation protocol (temporal split, sampled-negatives ranking). |
| [**backend.md**](backend.md) | The **FastAPI service** — the on-demand loading lifecycle, the one-heavy-model-at-a-time memory model (RAM estimates, the 507 guard, eviction), the `RecommenderBundle`, and every endpoint. |
| [**app.md**](app.md) | The **Streamlit front-end** — the thin-HTTP-client architecture, request plumbing & error handling, the on-demand loading UX, and all **6 tabs** in detail. |
| [**project-walkthrough.md**](project-walkthrough.md) | The **honest end-to-end narrative** — the problem, the data, the models, the evaluation, an honest assessment of the (corrected) results, and what other approaches (graph/Neo4j/KGE) could add. Good basis for the report. |

> **All the results in one place:** the consolidated, image-rich results writeup is the
> notebook [`notebooks/16_evaluation_report.ipynb`](../notebooks/16_evaluation_report.ipynb) —
> read-only (no model loading), it renders every metric table from `all_metrics.json` and
> references all ~41 figures with commentary. Start there for "what were the results?"; this
> `docs/` set explains the *how* and *why* behind them.

## How they fit together

```text
notebooks (train + evaluate)  ─►  artifacts/  ─►  backend (serves)  ─►  app (renders)
   notebooks.md                   models.md        backend.md            app.md
                         project-walkthrough.md = the story over all of it
```

## Reading order

- **New to the project?** Start with [project-walkthrough.md](project-walkthrough.md), then
  [models.md](models.md).
- **"What were the results?"** [evaluation.md](evaluation.md) — the complete record with every
  number, figure and caveat.
- **Writing the report?** [project-walkthrough.md](project-walkthrough.md) (narrative),
  [evaluation.md](evaluation.md) (the Πειραματική-μελέτη material: protocol, tables, figures,
  commentary), then [models.md](models.md) (the maths and the "how the hybrid combines CB + CF"
  explanation the rubric asks for), then [notebooks.md](notebooks.md) (experimental method).
- **Running or extending the app?** [backend.md](backend.md) then [app.md](app.md).

> Project context, the assignment brief, and the implementation deep-dive for Claude live in
> [`CLAUDE.md`](../CLAUDE.md) at the repo root.

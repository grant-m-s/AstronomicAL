# Recreated active_learning plugin

This refactor keeps the existing plugin id (`core.active_learning`) and existing public action ids, but splits the implementation into focused modules.

## Layout

```text
active_learning/
  plugin.py          platform registration
  state.py           session/label/batch state helpers
  strategies.py      batch-capable query strategy API and built-ins
  acquisition.py     query-pool construction, exclusions, selection handoff
  actions.py         AL core actions and shared platform utilities
  core_ml_bridge.py  optional core.ml train/predict/query workflow
  contracts.py       lightweight compatibility contract wrapper
  analytics.py       lightweight summaries/history helpers
  panel.py           thin Panel UI that delegates to actions
```

## Query strategy extension point

External plugins should register strategies against:

```python
registry = context.services.get("core.active_learning.query_strategy_registry")
registry.register(MyStrategy())
```

Prefer implementing `QueryStrategy.acquire(pool, k, params, seed, cancel_token)` for advanced methods such as QUEST, Learning Loss, BatchBALD, BADGE, and CoreSet. Existing per-row strategies can implement only `score(record, ...)`; the base class handles exclusion, stable sorting, and top-k selection.

```python
from astronomicAL.plugins.active_learning.strategies import QueryCandidate, QueryStrategy

class BatchBALDStrategy(QueryStrategy):
    id = "batchbald"
    title = "BatchBALD"
    batch_aware = True
    requires_probabilities = True
    required_prediction_fields = ("mc_probabilities",)
    params_schema = {
        "type": "object",
        "properties": {
            "mc_samples": {"type": "integer", "default": 20},
        },
    }

    def acquire(self, pool, *, k, params, seed=None, cancel_token=None):
        # Inspect pool.records, pool.predictions_payload, pool.session, etc.
        # Return candidates in acquisition order.
        return [QueryCandidate(row_id="example", score=1.0, metadata={})]
```

## Core behaviour

The AL core now owns:

- sessions
- labels and `Unsure`
- query strategy registry
- query pool exclusion logic
- query batch artifacts
- ordered selection handoff
- training-set materialisation

`core_ml_bridge.py` owns optional behaviour:

- data contract profiling
- derived training dataset creation
- calling `core.ml.run_ml_recipe`
- optional `core.ml.predict`
- optional automatic next query batch

This lets the active-learning plugin run without `core.ml` while preserving the old workflow when `core.ml` is enabled.

## Panel data binding updates

The session-start UI now discovers available datasets and columns from `context.datasets`.
Users select a dataset first, then a label column. The label set is inferred from
that column and shown as a removable multi-select; the start button remains disabled
until a dataset, label column, and at least one label are selected.

The panel subscribes to dataset/plugin/service events and refreshes dataset,
column, strategy, and recipe menus when data or services appear after the panel has
already been opened.

The optional training tab now lists saved `core.ml` recipe profiles from
`core.ml.recipe_profile_store`. Users configure model parameters, bindings, and
train/validation/test protocol in the core_ml Recipe Launcher, save that as a
profile, then select the saved profile in the AL panel.

## Review bulk labelling

The Review tab includes a `Next N labels from column` control. It reads each next
unlabelled review row's value from the session label column and records that row's
own pre-assigned label. It does not repeat the currently selected dropdown label.
If a start row is provided, or a focused row is available, bulk labelling starts
there; otherwise it starts at the first unlabelled row in the latest batch. Rows
already labelled, marked `Unsure`, in the training set, blank in the source label
column, or outside the reduced session label set are skipped while scanning for the
next N usable labels.

The same behaviour is available through the `bulk_label_next` action. Pass
`session_artifact_id`, optional `row_id`, optional `label_column`, and `n`; no label
argument is required because labels are resolved per row from the source dataset.

## Status, training lifecycle, and performance feedback

The panel now gives explicit status feedback before and after button actions. The
core.ml training bridge runs from the panel in a background thread so the UI can
show that training is in progress instead of appearing frozen.

The panel status area includes the active session id, round, number of labelled
points since the last training round, total labelled points, reviewed/unsure/queued
counts, and the selected model/prediction artifact ids.

The bridge publishes both AL-specific and generic ML lifecycle events:

```text
al.round.training_started
al.round.training_finished
al.round.training_failed
ml.recipe_run.started
ml.recipe_run.finished
ml.recipe_run.failed
ml.training.started
ml.training.finished
ml.training.failed
```

Finished events include the session, round, recipe profile, training dataset/artifact,
labelled count, model/prediction/run/log/evaluation artifact ids when available,
and best-effort scalar metrics extracted from the core.ml result or referenced
report/log artifacts.

The panel subscribes to those lifecycle events. When a training round finishes it
refreshes the session summary, selects the Query tab, and fills the model and
prediction artifact fields.

The new AL Performance tab plots one point per completed AL round, using labelled
training rows on the x-axis and the selected scalar metric on the y-axis.

## Training/query separation

The Train tab now only trains and refreshes model/prediction artifacts. It does
not create the next query batch. After training finishes, the panel switches to
the Query tab with the model and predictions selected so the user can choose the
query strategy and query batch size explicitly before querying.

The bridge still accepts `auto_query=True` as a legacy/advanced action parameter,
but the panel leaves it false.

## XY diagnostics

The XY Diagnostics tab provides a lightweight matplotlib view of the current pool
in two numeric dataset columns. It overlays labelled/trained rows and the latest
query batch. The latest query batch can be coloured by query score, query rank,
source label, or training status, which is useful for seeing where a strategy is
finding informative points in the data space.

## Latest behaviour updates

- Starting a training round invalidates any queued query batch rows. Queued rows are returned to the pool because their acquisition scores were produced by the previous model/predictions and are no longer valid after retraining starts.
- AL performance now reads metrics from the session history, training-finished event payloads, and referenced core.ml artifacts such as `ml.evaluation_report`, `ml.run`, and `ml.training_log`.
- The Train tab only trains and predicts. Querying is intentionally handled only by the Query tab so the user explicitly chooses the query strategy and batch size for each round.
- The XY Diagnostics tab now exposes all columns that can be coerced to numeric values, uses smaller/lighter points for dense plots, and includes prediction-correctness colouring: green for correct, red for incorrect, and yellow/gold when the true class is the second-highest probability class for problems with more than three classes. Pool/training rows and validation rows are plotted with different markers when a validation dataset is recorded on the session.

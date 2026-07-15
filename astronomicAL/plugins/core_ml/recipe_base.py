from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .paths import ml_run_artifact_dir
from .protocol import DataBinding, ProtocolConfig, TrainingComponents
from .run_logging import MLRunLogger
from .runtime import check_cancelled, publish, put_artifact

def make_run_context(
    *,
    context: Any,
    dataset_id: str,
    recipe_spec: Any,
    params: Mapping[str, Any],
    cancel_token: Any = None,
    run_id: Optional[str] = None,
) -> MLRunContext:
    run_id = str(run_id or params.get("run_id") or uuid.uuid4().hex)

    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f"astronomical-ml-recipe-{run_id[:8]}-"
        )
    )

    training_log_artifact_id = (
        params.get("training_log_artifact_id")
        or None
    )

    logger = MLRunLogger(
        context=context,
        run_id=run_id,
        dataset_id=dataset_id,
        training_log_artifact_id=training_log_artifact_id,
        recipe_spec=recipe_spec,
        params=params,
    )

    return MLRunContext(
        context=context,
        dataset_id=dataset_id,
        recipe_id=recipe_spec.id,
        recipe_version=recipe_spec.version,
        params=dict(params),
        run_id=run_id,
        work_dir=work_dir,
        cancel_token=cancel_token,
        training_log_artifact_id=logger.training_log_artifact_id,
        logger=logger,
    )

class MLRecipe:
    # --- identity / spec (defaults; concrete recipes override) --------------
    id: str = ""
    title: str = ""
    version: str = "0.0.0"
    task: str = "custom"
    modality: str = "custom"
    framework: str = ""          # "" -> runner falls back to tags/id inference
    complexity: str = "expert"
    author: str = ""
    description: str = ""
    tags: list = []
    required_mappings: list = []
    optional_mappings: list = []
    produces: list = []
    params_schema: Dict[str, Any] = {"type": "object", "properties": {}}

    @classmethod
    def spec(cls) -> "RecipeSpec":
        from .registry import RecipeSpec
        return RecipeSpec.from_recipe_cls(cls)

    # --- internals: the expert's contribution (managed recipes) -------------
    # The harness calls these; it never lets the recipe touch the protocol.

    def build_model(self, run, *, num_classes: int):
        raise NotImplementedError(
            "build_model(run, *, num_classes) must be implemented by a managed recipe."
        )

    def configure_training(self, run, model) -> "TrainingComponents":
        raise NotImplementedError(
            "configure_training(run, model) must be implemented by a managed recipe."
        )

    def train_transform(self, run):
        return None      # applied to TRAIN rows only

    def eval_transform(self, run):
        return None      # applied to VAL and TEST rows

    def load_sample(self, run, row):
        """Map one dataframe row to a raw input. The harness handles record-id
        keying, batching, and the data binding (run.binding); the recipe only
        knows how to read a single sample."""
        raise NotImplementedError(
            "load_sample(run, row) must be implemented by a managed recipe."
        )

    def eval_forward(self, model, batch_inputs):
        """Default forward used by the harness for val/test eval and the
        output-dimension check. Inherited as-is by recipes (e.g.
        CIFARResNetRecipe) that do a plain model(x) forward pass."""
        return model(batch_inputs)

    # --- the loop: the expert owns it; selection is delegated ---------------
    def fit(self, run, *, model, components: "TrainingComponents", train_loader, harness):
        """Run the training loop. The recipe sees ONLY train_loader and must
        call harness.report_epoch(epoch, model, train_metrics=...) at least
        once so the harness can evaluate the validation partition and select the
        best epoch. It receives no val/test loader and computes no selection
        metric, by design."""
        raise NotImplementedError(
            "fit(run, *, model, components, train_loader, harness) must be "
            "implemented by a managed recipe."
        )

@dataclass
class MLRunContext:
    context: Any
    dataset_id: str
    recipe_id: str
    recipe_version: str
    params: Dict[str, Any]
    run_id: str
    work_dir: Path
    cancel_token: Any = None
    training_log_artifact_id: Optional[str] = None
    logger: Optional[MLRunLogger] = None

    # Set by the runner after construction (run.protocol / run.binding) and read
    # by the harness via getattr(run, "protocol"/"binding", None). Declared here
    # so the contract is explicit and typed. String annotations avoid coupling to
    # the definition order of ProtocolConfig / DataBinding in this module.
    protocol: Optional["ProtocolConfig"] = None
    binding: Optional["DataBinding"] = None

    def check_cancelled(self) -> None:
        check_cancelled(self.cancel_token)

    def publish(self, event_type: str, payload: Optional[Mapping[str, Any]] = None) -> None:
        publish(self.context, event_type, payload)

    def put_artifact(
        self,
        artifact_type: str,
        payload: Mapping[str, Any],
        *,
        row_ids: Optional[Sequence[Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        return put_artifact(
            self.context,
            artifact_type,
            payload,
            dataset_id=self.dataset_id,
            row_ids=row_ids,
            params=params or self.params,
        )

    def log(
        self,
        *,
        message: str,
        status: str = "running",
        step: Optional[int] = None,
        total: Optional[int] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self.logger is not None:
            self.logger.log(
                message=message,
                status=status,
                step=step,
                total=total,
                metrics=metrics,
                extra=extra,
            )

class ManagedMLRecipe(MLRecipe):  # noqa: F821  (MLRecipe defined above in this file)
    """Recipe whose run() is the harness, not hand-written.

    A managed recipe implements only internals (build_model / configure_training
    / transforms / load_sample / fit). It NEVER partitions, evaluates val/test,
    selects the best epoch, or writes the scientific artifacts. The harness does
    all of that and refuses to delegate it.
    """

    def run(self, run) -> Dict[str, Any]:
        from .harnesses import make_harness
        harness = make_harness(run, self)
        return harness.execute()

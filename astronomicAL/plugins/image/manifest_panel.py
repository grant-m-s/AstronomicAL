from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import panel as pn

try:
    from .manifest import (
        DEFAULT_IMAGE_EXTENSIONS,
        build_image_manifest_dataframe,
        register_image_manifest_dataset,
        slugify,
    )
except ImportError:
    from manifest import (
        DEFAULT_IMAGE_EXTENSIONS,
        build_image_manifest_dataframe,
        register_image_manifest_dataset,
        slugify,
    )


class ImageManifestBuilderPanel:
    """Turn a folder of images into a manifest-backed AstronomicAL dataset."""

    state_version = 1

    def __init__(self, context: Any) -> None:
        self.context = context
        self._job_handles: list[Any] = []

        self.folder = pn.widgets.TextInput(
            name="Image folder",
            placeholder="/path/to/images",
            sizing_mode="stretch_width",
        )
        self.dataset_id = pn.widgets.TextInput(
            name="Dataset ID",
            value="image_dataset",
            sizing_mode="stretch_width",
        )
        self.dataset_name = pn.widgets.TextInput(
            name="Dataset name",
            value="Image Dataset",
            sizing_mode="stretch_width",
        )
        self.extensions = pn.widgets.TextInput(
            name="Extensions",
            value=", ".join(DEFAULT_IMAGE_EXTENSIONS),
            sizing_mode="stretch_width",
        )
        self.recursive = pn.widgets.Checkbox(name="Scan recursively", value=True)
        self.label_from_parent = pn.widgets.Checkbox(
            name="Use parent folder as target_label",
            value=True,
        )
        self.relative_paths = pn.widgets.Checkbox(
            name="Store relative image paths",
            value=True,
        )
        self.write_parquet = pn.widgets.Checkbox(
            name="Write manifest to Parquet cache if possible",
            value=True,
        )
        self.set_active = pn.widgets.Checkbox(name="Set as active dataset", value=True)

        self.build_button = pn.widgets.Button(
            name="Build image manifest",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.build_button.on_click(self._on_build_clicked)

        self.status = pn.pane.Alert(
            "Choose a folder and build a manifest dataset.",
            alert_type="info",
            sizing_mode="stretch_width",
        )
        self.preview = pn.pane.DataFrame(None, sizing_mode="stretch_width", height=220)

        self._view = pn.Column(
            pn.pane.Markdown("### Image Manifest Builder"),
            self.folder,
            pn.Row(self.dataset_id, self.dataset_name, sizing_mode="stretch_width"),
            self.extensions,
            pn.Row(
                self.recursive,
                self.label_from_parent,
                self.relative_paths,
                sizing_mode="stretch_width",
            ),
            pn.Row(self.write_parquet, self.set_active, sizing_mode="stretch_width"),
            self.build_button,
            self.status,
            pn.pane.Markdown("#### Preview"),
            self.preview,
            sizing_mode="stretch_both",
            styles={
                "height": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "auto",
                "padding": "8px",
            },
        )

    def panel(self) -> pn.Column:
        return self._view

    def dispose(self) -> None:
        for handle in self._job_handles:
            try:
                handle.cancel()
            except Exception:
                pass
        self._job_handles.clear()

    def get_state(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "folder": self.folder.value,
            "dataset_id": self.dataset_id.value,
            "dataset_name": self.dataset_name.value,
            "extensions": self.extensions.value,
            "recursive": self.recursive.value,
            "label_from_parent": self.label_from_parent.value,
            "relative_paths": self.relative_paths.value,
            "write_parquet": self.write_parquet.value,
            "set_active": self.set_active.value,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        widgets = {
            "folder": self.folder,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "extensions": self.extensions,
            "recursive": self.recursive,
            "label_from_parent": self.label_from_parent,
            "relative_paths": self.relative_paths,
            "write_parquet": self.write_parquet,
            "set_active": self.set_active,
        }
        for key, widget in widgets.items():
            if key in state:
                try:
                    widget.value = state[key]
                except Exception:
                    pass

    def _on_build_clicked(self, event: Any) -> None:
        folder = str(self.folder.value or "").strip()
        if not folder:
            self._set_error("Please provide an image folder.")
            return

        root = Path(folder).expanduser()
        if not root.exists():
            self._set_error(f"Folder does not exist: {html.escape(str(root))}")
            return

        dataset_id = slugify(self.dataset_id.value or root.name)
        dataset_name = self.dataset_name.value or dataset_id

        self.build_button.disabled = True
        self._set_loading(f"Scanning `{html.escape(str(root))}`…")

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                result = self._build_and_register(
                    root=root,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    cancel_token=None,
                )
                self._on_done(result)
            except Exception as exc:
                self._on_error(exc)
            return

        handle = jobs.submit(
            self._build_and_register,
            title="Build image manifest",
            key=f"core.image.manifest:{root}:{dataset_id}",
            on_done=self._on_done,
            on_error=self._on_error,
            root=root,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
        )
        self._job_handles.append(handle)

    def _build_and_register(
        self,
        *,
        cancel_token: Any,
        root: Path,
        dataset_id: str,
        dataset_name: str,
    ) -> dict[str, Any]:
        df = build_image_manifest_dataframe(
            root,
            recursive=bool(self.recursive.value),
            extensions=str(self.extensions.value or ""),
            label_from_parent=bool(self.label_from_parent.value),
            relative_paths=bool(self.relative_paths.value),
            cancel_token=cancel_token,
        )
        if cancel_token is not None and cancel_token.cancelled():
            return {"cancelled": True}

        result = register_image_manifest_dataset(
            self.context,
            df,
            dataset_id=dataset_id,
            name=dataset_name,
            base_path=root,
            set_active=bool(self.set_active.value),
            write_parquet=bool(self.write_parquet.value),
        )
        result["preview"] = df.head(25)
        return result

    def _on_done(self, result: dict[str, Any]) -> None:
        self.build_button.disabled = False
        if result.get("cancelled"):
            self.status.alert_type = "warning"
            self.status.object = "Manifest build cancelled."
            return

        preview = result.pop("preview", None)
        if preview is not None:
            self.preview.object = preview

        self.status.alert_type = "success"
        self.status.object = (
            f"Registered image dataset `{html.escape(result['dataset_id'])}` "
            f"with **{result['rows']}** rows using `{html.escape(result['backend'])}`."
        )

    def _on_error(self, exc: BaseException) -> None:
        self.build_button.disabled = False
        self._set_error(str(exc))

    def _set_loading(self, message: str) -> None:
        self.status.alert_type = "primary"
        self.status.object = message

    def _set_error(self, message: str) -> None:
        self.status.alert_type = "danger"
        self.status.object = message

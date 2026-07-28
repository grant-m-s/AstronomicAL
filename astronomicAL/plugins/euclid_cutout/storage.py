from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from threading import RLock, get_ident
from typing import Any, Dict, Mapping, Set


_RUNTIME_DIR_NAME = ".astronomical_runtime"
_RETAINED_DIR_NAME = "artifacts"


def _remove_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:
        return


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@dataclass
class EuclidRequestScratch:
    """One isolated request directory owned by a panel storage lease."""

    lease: "EuclidPanelStorageLease"
    request_id: int
    root: Path
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.lease._finish_request(self)

    def __enter__(self) -> "EuclidRequestScratch":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class EuclidPanelStorageLease:
    """Counter-based scratch ownership for one open Euclid panel."""

    def __init__(
        self,
        *,
        manager: "EuclidScratchManager",
        owner_id: str,
        slot: int,
    ) -> None:
        self.manager = manager
        self.owner_id = str(owner_id)
        self.slot = int(slot)
        self._active_requests: Dict[int, EuclidRequestScratch] = {}
        self._panel_roots: Set[Path] = set()
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def begin_request(self, *, base_dir: str, request_id: int) -> EuclidRequestScratch:
        return self.manager._begin_request(
            lease=self,
            base_dir=base_dir,
            request_id=int(request_id),
        )

    def release(self) -> None:
        self.manager._release_lease(self)

    def _finish_request(self, request: EuclidRequestScratch) -> None:
        self.manager._finish_request(self, request)


class EuclidScratchManager:
    """Manage bounded, counter-based scratch directories for Euclid panels.

    Slots are process-wide, so separate runtime instances cannot assign the same
    ``panel_N`` directory. A released slot is reused only after all requests
    belonging to its previous lease have finished.
    """

    _process_lock = RLock()
    _process_slots: Set[int] = set()

    def __init__(self) -> None:
        self._lock = RLock()
        self._leases: Set[EuclidPanelStorageLease] = set()
        self._prepared_bases: Set[Path] = set()
        self._request_counter = 0
        self._closed = False

    def acquire_panel(self, owner_id: str) -> EuclidPanelStorageLease:
        with self._lock:
            if self._closed:
                raise RuntimeError("Euclid scratch manager has been closed")

        with self._process_lock:
            slot = 0
            while slot in self._process_slots:
                slot += 1
            self._process_slots.add(slot)

        lease = EuclidPanelStorageLease(
            manager=self,
            owner_id=str(owner_id),
            slot=slot,
        )
        with self._lock:
            self._leases.add(lease)
        return lease

    def next_request_id(self) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("Euclid scratch manager has been closed")
            self._request_counter += 1
            return self._request_counter

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            leases = list(self._leases)
        for lease in leases:
            lease.release()

    def promote_files(
        self,
        *,
        fits_paths: Mapping[str, str],
        base_dir: str,
        identity: Mapping[str, Any],
    ) -> Dict[str, str]:
        """Copy an explicitly retained result into a deterministic path."""

        if not fits_paths:
            return {}

        identity_json = json.dumps(
            dict(identity),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = sha256(identity_json.encode("utf-8")).hexdigest()[:20]

        with self._lock:
            base = Path(base_dir).expanduser().resolve()
            destination = base / _RETAINED_DIR_NAME / digest
            destination.mkdir(parents=True, exist_ok=True)

            promoted: Dict[str, str] = {}
            for band, raw_path in fits_paths.items():
                source = Path(raw_path)
                if not source.is_file():
                    continue
                final_path = destination / f"{band}.fits"
                temporary_path = destination / (
                    f".{band}.{os.getpid()}.{get_ident()}.fits.tmp"
                )
                try:
                    shutil.copy2(source, temporary_path)
                    os.replace(temporary_path, final_path)
                finally:
                    try:
                        temporary_path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
                promoted[str(band)] = str(final_path)
            return promoted

    def _begin_request(
        self,
        *,
        lease: EuclidPanelStorageLease,
        base_dir: str,
        request_id: int,
    ) -> EuclidRequestScratch:
        with self._lock:
            if self._closed:
                raise RuntimeError("Euclid scratch manager has been closed")
            if lease not in self._leases or lease._released:
                raise RuntimeError("Euclid panel storage lease has been released")
            if request_id in lease._active_requests:
                raise RuntimeError(
                    f"Euclid request {request_id} is already active for panel "
                    f"{lease.slot}"
                )

            session_root = self._prepare_base(Path(base_dir).expanduser())
            panel_root = session_root / f"panel_{lease.slot}"
            request_root = panel_root / f"request_{request_id}"
            _remove_tree(request_root)
            request_root.mkdir(parents=True, exist_ok=True)

            request = EuclidRequestScratch(
                lease=lease,
                request_id=request_id,
                root=request_root,
            )
            lease._active_requests[request_id] = request
            lease._panel_roots.add(panel_root)
            return request

    def _finish_request(
        self,
        lease: EuclidPanelStorageLease,
        request: EuclidRequestScratch,
    ) -> None:
        with self._lock:
            current = lease._active_requests.get(request.request_id)
            if current is request:
                del lease._active_requests[request.request_id]
            _remove_tree(request.root)
            self._remove_empty_parents(request.root.parent)
            self._finalise_released_lease(lease)

    def _release_lease(self, lease: EuclidPanelStorageLease) -> None:
        with self._lock:
            if lease._released:
                return
            lease._released = True
            self._finalise_released_lease(lease)

    def _finalise_released_lease(self, lease: EuclidPanelStorageLease) -> None:
        if not lease._released or lease._active_requests:
            return

        for panel_root in list(lease._panel_roots):
            _remove_tree(panel_root)
            self._remove_empty_parents(panel_root.parent)
        lease._panel_roots.clear()
        self._leases.discard(lease)

        with self._process_lock:
            self._process_slots.discard(lease.slot)

    def _prepare_base(self, base: Path) -> Path:
        base = base.resolve()
        if base not in self._prepared_bases:
            base.mkdir(parents=True, exist_ok=True)
            runtime_root = base / _RUNTIME_DIR_NAME / "euclid"
            runtime_root.mkdir(parents=True, exist_ok=True)
            self._remove_stale_sessions(runtime_root)
            self._remove_legacy_temp_files(base)
            self._prepared_bases.add(base)
        return base / _RUNTIME_DIR_NAME / "euclid" / f"session_{os.getpid()}"

    @staticmethod
    def _remove_stale_sessions(runtime_root: Path) -> None:
        for candidate in runtime_root.glob("session_*"):
            try:
                pid = int(candidate.name.removeprefix("session_"))
            except ValueError:
                continue
            if not _process_is_alive(pid):
                _remove_tree(candidate)

    @staticmethod
    def _remove_legacy_temp_files(base: Path) -> None:
        for band in ("VIS", "NIR_Y", "NIR_J", "NIR_H"):
            try:
                (base / f"tmp_{band}.fits").unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    @staticmethod
    def _remove_empty_parents(path: Path) -> None:
        current = path
        for _ in range(3):
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent


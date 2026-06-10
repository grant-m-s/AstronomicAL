from __future__ import annotations

import csv
import html
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import panel as pn


def _try_import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


class ResourceSampler:
    """Collect lightweight host and process resource snapshots.

    psutil is optional. Without psutil, the sampler still opens and shows:
    - host/Python/process identity
    - NVIDIA GPU metrics through nvidia-smi, if available
    - a best-effort Unix `ps` top-process fallback, if available
    """

    def __init__(self) -> None:
        self.psutil = _try_import_psutil()
        self.pid = os.getpid()
        self._previous_net = None
        self._previous_net_time = None
        self._nvidia_smi_path = shutil.which("nvidia-smi")

        if self.psutil is not None:
            try:
                self.psutil.cpu_percent(interval=None)
            except Exception:
                pass

            try:
                for proc in self.psutil.process_iter():
                    try:
                        proc.cpu_percent(interval=None)
                    except Exception:
                        pass
            except Exception:
                pass

    def sample(
        self,
        *,
        top_n: int = 20,
        include_command: bool = False,
    ) -> Dict[str, Any]:
        now = time.time()

        notes: List[str] = []

        if self.psutil is None:
            notes.append(
                "Install `psutil` for full CPU, memory, disk, network, and process metrics."
            )

        if self._nvidia_smi_path is None:
            notes.append("`nvidia-smi` was not found, so NVIDIA GPU metrics are unavailable.")

        snapshot: Dict[str, Any] = {
            "schema_version": 1,
            "sampled_at": now,
            "sampled_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "host": self._sample_host(),
            "system": self._sample_system(notes),
            "process": self._sample_current_process(),
            "disk": self._sample_disk(),
            "network": self._sample_network(now),
            "gpus": self._sample_gpus(notes),
            "gpu_processes": self._sample_gpu_processes(notes),
            "top_processes": self._sample_top_processes(
                top_n=max(1, int(top_n)),
                include_command=include_command,
                notes=notes,
            ),
            "notes": notes,
        }

        return _json_safe(snapshot)

    def _sample_host(self) -> Dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "pid": self.pid,
            "cwd": str(Path.cwd()),
        }

    def _sample_system(self, notes: List[str]) -> Dict[str, Any]:
        psutil = self.psutil
        result: Dict[str, Any] = {
            "cpu_percent": None,
            "cpu_count_logical": os.cpu_count(),
            "cpu_count_physical": None,
            "load_average": None,
            "memory_total_mb": None,
            "memory_used_mb": None,
            "memory_available_mb": None,
            "memory_percent": None,
            "swap_total_mb": None,
            "swap_used_mb": None,
            "swap_percent": None,
        }

        try:
            if hasattr(os, "getloadavg"):
                result["load_average"] = list(os.getloadavg())
        except Exception:
            pass

        if psutil is None:
            return result

        try:
            result["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        except Exception as exc:
            notes.append(f"Could not sample CPU usage: {exc}")

        try:
            result["cpu_count_physical"] = psutil.cpu_count(logical=False)
            result["cpu_count_logical"] = psutil.cpu_count(logical=True)
        except Exception:
            pass

        try:
            mem = psutil.virtual_memory()
            result.update(
                {
                    "memory_total_mb": _bytes_to_mb(mem.total),
                    "memory_used_mb": _bytes_to_mb(mem.used),
                    "memory_available_mb": _bytes_to_mb(mem.available),
                    "memory_percent": float(mem.percent),
                }
            )
        except Exception as exc:
            notes.append(f"Could not sample memory usage: {exc}")

        try:
            swap = psutil.swap_memory()
            result.update(
                {
                    "swap_total_mb": _bytes_to_mb(swap.total),
                    "swap_used_mb": _bytes_to_mb(swap.used),
                    "swap_percent": float(swap.percent),
                }
            )
        except Exception:
            pass

        return result

    def _sample_current_process(self) -> Dict[str, Any]:
        psutil = self.psutil

        result: Dict[str, Any] = {
            "pid": self.pid,
            "cpu_percent": None,
            "rss_mb": None,
            "vms_mb": None,
            "memory_percent": None,
            "num_threads": None,
            "open_files": None,
            "connections": None,
            "create_time": None,
            "status": None,
            "command": " ".join(sys.argv),
        }

        if psutil is None:
            return result

        try:
            proc = psutil.Process(self.pid)
            mem = proc.memory_info()

            result.update(
                {
                    "cpu_percent": float(proc.cpu_percent(interval=None)),
                    "rss_mb": _bytes_to_mb(mem.rss),
                    "vms_mb": _bytes_to_mb(mem.vms),
                    "memory_percent": float(proc.memory_percent()),
                    "num_threads": proc.num_threads(),
                    "create_time": proc.create_time(),
                    "status": proc.status(),
                }
            )

            try:
                result["open_files"] = len(proc.open_files())
            except Exception:
                result["open_files"] = None

            try:
                result["connections"] = len(proc.net_connections())
            except Exception:
                try:
                    result["connections"] = len(proc.connections())
                except Exception:
                    result["connections"] = None

        except Exception:
            pass

        return result

    def _sample_disk(self) -> Dict[str, Any]:
        psutil = self.psutil

        result: Dict[str, Any] = {
            "path": str(Path.cwd().anchor or "/"),
            "total_mb": None,
            "used_mb": None,
            "free_mb": None,
            "percent": None,
        }

        if psutil is None:
            return result

        try:
            path = Path.cwd().anchor or "/"
            usage = psutil.disk_usage(str(path))
            result.update(
                {
                    "path": str(path),
                    "total_mb": _bytes_to_mb(usage.total),
                    "used_mb": _bytes_to_mb(usage.used),
                    "free_mb": _bytes_to_mb(usage.free),
                    "percent": float(usage.percent),
                }
            )
        except Exception:
            pass

        return result

    def _sample_network(self, now: float) -> Dict[str, Any]:
        psutil = self.psutil

        result: Dict[str, Any] = {
            "bytes_sent": None,
            "bytes_recv": None,
            "send_rate_mb_s": None,
            "recv_rate_mb_s": None,
        }

        if psutil is None:
            return result

        try:
            counters = psutil.net_io_counters()
        except Exception:
            return result

        result["bytes_sent"] = int(counters.bytes_sent)
        result["bytes_recv"] = int(counters.bytes_recv)

        if self._previous_net is not None and self._previous_net_time is not None:
            dt = max(1e-6, now - self._previous_net_time)
            result["send_rate_mb_s"] = _bytes_to_mb(
                max(0, counters.bytes_sent - self._previous_net.bytes_sent)
            ) / dt
            result["recv_rate_mb_s"] = _bytes_to_mb(
                max(0, counters.bytes_recv - self._previous_net.bytes_recv)
            ) / dt

        self._previous_net = counters
        self._previous_net_time = now

        return result

    def _sample_gpus(self, notes: List[str]) -> List[Dict[str, Any]]:
        if self._nvidia_smi_path is None:
            return []

        query = (
            "index,name,uuid,driver_version,memory.total,memory.used,memory.free,"
            "utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit"
        )

        rows, error = self._run_nvidia_smi(
            [
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
        )

        if error:
            notes.append(f"Could not sample NVIDIA GPUs: {error}")
            return []

        gpus: List[Dict[str, Any]] = []

        for row in _csv_rows(rows):
            if len(row) < 12:
                continue

            gpus.append(
                {
                    "index": _to_int(row[0]),
                    "name": row[1].strip(),
                    "uuid": row[2].strip(),
                    "driver_version": row[3].strip(),
                    "memory_total_mb": _to_float(row[4]),
                    "memory_used_mb": _to_float(row[5]),
                    "memory_free_mb": _to_float(row[6]),
                    "gpu_util_percent": _to_float(row[7]),
                    "memory_util_percent": _to_float(row[8]),
                    "temperature_c": _to_float(row[9]),
                    "power_draw_w": _to_float(row[10]),
                    "power_limit_w": _to_float(row[11]),
                }
            )

        return gpus

    def _sample_gpu_processes(self, notes: List[str]) -> List[Dict[str, Any]]:
        if self._nvidia_smi_path is None:
            return []

        query = "gpu_uuid,pid,process_name,used_memory"

        rows, error = self._run_nvidia_smi(
            [
                f"--query-compute-apps={query}",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
        )

        if error:
            # This often fails harmlessly when there are no compute processes or
            # on driver/toolkit combinations that do not expose this query.
            return []

        processes: List[Dict[str, Any]] = []

        for row in _csv_rows(rows):
            if len(row) < 4:
                continue

            processes.append(
                {
                    "gpu_uuid": row[0].strip(),
                    "pid": _to_int(row[1]),
                    "process_name": row[2].strip(),
                    "used_memory_mb": _to_float(row[3]),
                }
            )

        return processes

    def _sample_top_processes(
        self,
        *,
        top_n: int,
        include_command: bool,
        notes: List[str],
    ) -> List[Dict[str, Any]]:
        psutil = self.psutil

        if psutil is not None:
            rows: List[Dict[str, Any]] = []

            try:
                iterator = psutil.process_iter(
                    [
                        "pid",
                        "name",
                        "username",
                        "status",
                        "cpu_percent",
                        "memory_percent",
                        "memory_info",
                        "cmdline",
                    ]
                )

                for proc in iterator:
                    try:
                        info = proc.info
                        memory_info = info.get("memory_info")
                        cmdline = info.get("cmdline") or []

                        rows.append(
                            {
                                "pid": info.get("pid"),
                                "name": info.get("name") or "",
                                "username": info.get("username") or "",
                                "status": info.get("status") or "",
                                "cpu_percent": _safe_float(info.get("cpu_percent")),
                                "memory_percent": _safe_float(info.get("memory_percent")),
                                "rss_mb": _bytes_to_mb(getattr(memory_info, "rss", 0) or 0),
                                "command": " ".join(cmdline) if include_command else "",
                            }
                        )
                    except Exception:
                        continue

                rows.sort(
                    key=lambda row: (
                        row.get("cpu_percent") or 0.0,
                        row.get("memory_percent") or 0.0,
                    ),
                    reverse=True,
                )

                return rows[:top_n]

            except Exception as exc:
                notes.append(f"Could not sample process table with psutil: {exc}")

        return self._sample_top_processes_with_ps(top_n=top_n, include_command=include_command)

    def _sample_top_processes_with_ps(
        self,
        *,
        top_n: int,
        include_command: bool,
    ) -> List[Dict[str, Any]]:
        ps_path = shutil.which("ps")
        if ps_path is None:
            return []

        command_col = "args" if include_command else "comm"

        commands = [
            [ps_path, "-eo", f"pid=,pcpu=,pmem=,rss=,{command_col}=", "--sort=-pcpu"],
            [ps_path, "-eo", f"pid=,pcpu=,pmem=,rss=,{command_col}="],
        ]

        output = ""
        for command in commands:
            try:
                output = subprocess.check_output(
                    command,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                break
            except Exception:
                output = ""

        rows: List[Dict[str, Any]] = []

        for line in output.splitlines():
            parts = line.strip().split(None, 4)
            if len(parts) < 4:
                continue

            pid, cpu, mem, rss = parts[:4]
            name = parts[4] if len(parts) > 4 else ""

            rows.append(
                {
                    "pid": _to_int(pid),
                    "name": name if not include_command else name.split()[0] if name else "",
                    "username": "",
                    "status": "",
                    "cpu_percent": _to_float(cpu),
                    "memory_percent": _to_float(mem),
                    "rss_mb": _safe_float(rss) / 1024.0 if _safe_float(rss) is not None else None,
                    "command": name if include_command else "",
                }
            )

        rows.sort(
            key=lambda row: (
                row.get("cpu_percent") or 0.0,
                row.get("memory_percent") or 0.0,
            ),
            reverse=True,
        )

        return rows[:top_n]

    def _run_nvidia_smi(
        self,
        args: List[str],
        *,
        timeout: int = 3,
    ) -> tuple[str, Optional[str]]:
        if self._nvidia_smi_path is None:
            return "", "nvidia-smi not found"

        try:
            completed = subprocess.run(
                [self._nvidia_smi_path, *args],
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except Exception as exc:
            return "", str(exc)

        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            return "", error or f"nvidia-smi exited with status {completed.returncode}"

        return completed.stdout.strip(), None


class ResourceMonitorPanel:
    """Top-like resource monitor panel."""

    def __init__(
        self,
        *,
        context: Any,
        sampler: ResourceSampler,
        restore_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.context = context
        self.sampler = sampler
        self._periodic = None
        self._last_snapshot: Dict[str, Any] = {}
        self._disposed = False
        self._root = None
        self._periodic_started_once = False

        self.auto_refresh = pn.widgets.Checkbox(
            name="Auto refresh",
            value=True,
            sizing_mode="stretch_width",
        )
        self.refresh_interval = pn.widgets.IntSlider(
            name="Refresh interval, seconds",
            start=1,
            end=30,
            step=1,
            value=2,
            sizing_mode="stretch_width",
        )
        self.top_n = pn.widgets.IntSlider(
            name="Top process count",
            start=5,
            end=100,
            step=5,
            value=20,
            sizing_mode="stretch_width",
        )
        self.include_command = pn.widgets.Checkbox(
            name="Show full command lines in process table",
            value=False,
            sizing_mode="stretch_width",
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh now",
            button_type="primary",
            sizing_mode="stretch_width",
            height=34,
        )

        self.status = pn.pane.Alert(
            "Waiting for first resource sample...",
            alert_type="info",
            sizing_mode="stretch_width",
        )
        self.overview = pn.pane.HTML("", sizing_mode="stretch_width")
        self.gpu_table = pn.pane.DataFrame(
            pd.DataFrame(),
            height=230,
            sizing_mode="stretch_width",
        )
        self.gpu_process_table = pn.pane.DataFrame(
            pd.DataFrame(),
            height=230,
            sizing_mode="stretch_width",
        )
        self.process_table = pn.pane.DataFrame(
            pd.DataFrame(),
            height=380,
            sizing_mode="stretch_width",
        )
        self.raw_json = pn.pane.JSON(
            {},
            depth=3,
            sizing_mode="stretch_width",
        )
        self.notes = pn.pane.Markdown("", sizing_mode="stretch_width")

        self.refresh_button.on_click(lambda *_: self.refresh())

        self.auto_refresh.param.watch(lambda *_: self._restart_periodic(), "value")
        self.refresh_interval.param.watch(lambda *_: self._restart_periodic(), "value")
        self.top_n.param.watch(lambda *_: self.refresh(), "value")
        self.include_command.param.watch(lambda *_: self.refresh(), "value")

        if restore_state:
            self.restore_state(restore_state)

    def _start_periodic_when_ready(self) -> None:
        """Start auto-refresh after the Panel/Bokeh document exists.

        Creating a periodic callback too early can silently fail during restored
        layout construction. Registering an onload/next-tick start makes the
        initial auto-refresh behave the same way as changing the interval later.
        """

        if self._disposed or not bool(self.auto_refresh.value):
            return

        def start() -> None:
            if self._disposed or not bool(self.auto_refresh.value):
                return
            self._restart_periodic()
            self.refresh()

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(start)
                return
        except Exception:
            pass

        try:
            pn.state.onload(start)
            return
        except Exception:
            pass

        start()

    def panel(self):
        controls = pn.Column(
            pn.pane.HTML("<h3 style='margin:0 0 8px 0'>Resource Monitor</h3>"),
            self.auto_refresh,
            self.refresh_interval,
            self.top_n,
            self.include_command,
            self.refresh_button,
            self.status,
            sizing_mode="stretch_height",
            width=280,
            styles={
                "box-sizing": "border-box",
                "padding": "10px 12px 10px 10px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "min-height": "0",
                "height": "100%",
                "border-right": "1px solid #ddd",
            },
        )

        overview_tab = pn.Column(
            self.overview,
            self.notes,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "box-sizing": "border-box",
                "padding": "0 8px 16px 0",
                "min-height": "0",
                "height": "100%",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            },
        )

        gpu_tab = pn.Column(
            pn.pane.Markdown("### NVIDIA GPUs"),
            self.gpu_table,
            pn.pane.Markdown("### NVIDIA compute processes"),
            self.gpu_process_table,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "box-sizing": "border-box",
                "padding": "0 8px 16px 0",
                "min-height": "0",
                "height": "100%",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            },
        )

        processes_tab = pn.Column(
            self.process_table,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "box-sizing": "border-box",
                "padding": "0 8px 16px 0",
                "min-height": "0",
                "height": "100%",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            },
        )

        raw_tab = pn.Column(
            self.raw_json,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "box-sizing": "border-box",
                "padding": "0 8px 16px 0",
                "min-height": "0",
                "height": "100%",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            },
        )

        content = pn.Tabs(
            ("Overview", overview_tab),
            ("GPU", gpu_tab),
            ("Processes", processes_tab),
            ("Raw snapshot", raw_tab),
            dynamic=True,
            sizing_mode="stretch_both",
            styles={
                "box-sizing": "border-box",
                "min-height": "0",
                "height": "100%",
                "overflow": "hidden",
            },
        )

        self._root = pn.Row(
            controls,
            content,
            sizing_mode="stretch_both",
            styles={
                "box-sizing": "border-box",
                "overflow": "hidden",
                "min-height": "0",
                "height": "100%",
                "width": "100%",
            },
        )

        self.refresh()
        self._start_periodic_when_ready()

        return self._root

    def get_state(self) -> Dict[str, Any]:
        return {
            "auto_refresh": bool(self.auto_refresh.value),
            "refresh_interval": int(self.refresh_interval.value),
            "top_n": int(self.top_n.value),
            "include_command": bool(self.include_command.value),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        if "auto_refresh" in state:
            self.auto_refresh.value = bool(state["auto_refresh"])

        if "refresh_interval" in state:
            value = int(state["refresh_interval"])
            self.refresh_interval.value = max(
                self.refresh_interval.start,
                min(self.refresh_interval.end, value),
            )

        if "top_n" in state:
            value = int(state["top_n"])
            self.top_n.value = max(self.top_n.start, min(self.top_n.end, value))

        if "include_command" in state:
            self.include_command.value = bool(state["include_command"])

    def dispose(self) -> None:
        self._disposed = True
        self._stop_periodic()

    def refresh(self) -> None:
        if self._disposed:
            return

        try:
            snapshot = self.sampler.sample(
                top_n=int(self.top_n.value),
                include_command=bool(self.include_command.value),
            )
            self._last_snapshot = snapshot
            self._render(snapshot)
            self._publish_snapshot(snapshot)
        except Exception as exc:
            self.status.alert_type = "danger"
            self.status.object = f"Could not sample resources: `{exc}`"

    def _restart_periodic(self) -> None:
        self._stop_periodic()

        if not bool(self.auto_refresh.value) or self._disposed:
            return

        try:
            self._periodic = pn.state.add_periodic_callback(
                self.refresh,
                period=int(self.refresh_interval.value) * 1000,
                start=False,
            )

            start = getattr(self._periodic, "start", None)
            if callable(start):
                start()

            self._periodic_started_once = True

        except Exception:
            self._periodic = None
            self._periodic_started_once = False

    def _stop_periodic(self) -> None:
        callback = self._periodic
        self._periodic = None

        if callback is None:
            return

        stop = getattr(callback, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass

    def _render(self, snapshot: Mapping[str, Any]) -> None:
        sampled_at = snapshot.get("sampled_at_iso", "")
        notes = list(snapshot.get("notes") or [])

        self.status.alert_type = "success"
        self.status.object = f"Last sample: `{sampled_at}`"

        self.overview.object = self._overview_html(snapshot)

        gpu_df = pd.DataFrame(snapshot.get("gpus") or [])
        gpu_proc_df = pd.DataFrame(snapshot.get("gpu_processes") or [])
        proc_df = pd.DataFrame(snapshot.get("top_processes") or [])

        self.gpu_table.object = gpu_df
        self.gpu_process_table.object = gpu_proc_df
        self.process_table.object = proc_df
        self.raw_json.object = dict(snapshot)

        if notes:
            self.notes.object = "### Notes\n" + "\n".join(f"- {note}" for note in notes)
        else:
            self.notes.object = ""

    def _overview_html(self, snapshot: Mapping[str, Any]) -> str:
        host = snapshot.get("host") or {}
        system = snapshot.get("system") or {}
        process = snapshot.get("process") or {}
        disk = snapshot.get("disk") or {}
        network = snapshot.get("network") or {}
        gpus = list(snapshot.get("gpus") or [])

        gpu_util = None
        gpu_mem = None
        if gpus:
            util_values = [
                gpu.get("gpu_util_percent")
                for gpu in gpus
                if gpu.get("gpu_util_percent") is not None
            ]
            mem_values = [
                _percent(gpu.get("memory_used_mb"), gpu.get("memory_total_mb"))
                for gpu in gpus
                if gpu.get("memory_used_mb") is not None
                and gpu.get("memory_total_mb") is not None
            ]
            gpu_util = max(util_values) if util_values else None
            gpu_mem = max(mem_values) if mem_values else None

        cards = [
            _metric_card(
                "CPU",
                _fmt_percent(system.get("cpu_percent")),
                _bar(system.get("cpu_percent")),
            ),
            _metric_card(
                "Memory",
                _fmt_percent(system.get("memory_percent")),
                (
                    f"{_fmt_mb(system.get('memory_used_mb'))} / "
                    f"{_fmt_mb(system.get('memory_total_mb'))}"
                    + _bar(system.get("memory_percent"))
                ),
            ),
            _metric_card(
                "GPU",
                _fmt_percent(gpu_util),
                (
                    f"{len(gpus)} GPU(s)<br>"
                    f"max memory {_fmt_percent(gpu_mem)}"
                    + _bar(gpu_mem)
                ),
            ),
            _metric_card(
                "Swap",
                _fmt_percent(system.get("swap_percent")),
                (
                    f"{_fmt_mb(system.get('swap_used_mb'))} / "
                    f"{_fmt_mb(system.get('swap_total_mb'))}"
                    + _bar(system.get("swap_percent"))
                ),
            ),
            _metric_card(
                "Disk",
                _fmt_percent(disk.get("percent")),
                (
                    f"{html.escape(str(disk.get('path') or ''))}<br>"
                    f"{_fmt_mb(disk.get('used_mb'))} / {_fmt_mb(disk.get('total_mb'))}"
                    + _bar(disk.get("percent"))
                ),
            ),
            _metric_card(
                "AstronomicAL process",
                _fmt_percent(process.get("cpu_percent")),
                (
                    f"PID {process.get('pid')}<br>"
                    f"RSS {_fmt_mb(process.get('rss_mb'))}<br>"
                    f"Threads {process.get('num_threads') or 'n/a'}"
                ),
            ),
            _metric_card(
                "Network",
                f"↓ {_fmt_mb(network.get('recv_rate_mb_s'))}/s",
                f"↑ {_fmt_mb(network.get('send_rate_mb_s'))}/s",
            ),
            _metric_card(
                "Load average",
                _fmt_load(system.get("load_average")),
                (
                    f"logical CPUs: {system.get('cpu_count_logical') or 'n/a'}<br>"
                    f"physical CPUs: {system.get('cpu_count_physical') or 'n/a'}"
                ),
            ),
        ]

        host_html = (
            "<div style='margin-bottom:10px;font-size:13px;color:#555'>"
            f"<b>Host:</b> {html.escape(str(host.get('hostname') or ''))} &nbsp; "
            f"<b>Platform:</b> {html.escape(str(host.get('platform') or ''))} &nbsp; "
            f"<b>Python:</b> {html.escape(str(host.get('python') or ''))}"
            "</div>"
        )

        return (
            host_html
            + "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));"
            + "gap:10px;width:100%;box-sizing:border-box'>"
            + "".join(cards)
            + "</div>"
        )

    def _publish_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        events = getattr(self.context, "events", None)
        publish = getattr(events, "publish", None)

        if callable(publish):
            try:
                publish(
                    "resources.snapshot",
                    {
                        "sampled_at": snapshot.get("sampled_at"),
                        "host": (snapshot.get("host") or {}).get("hostname"),
                        "cpu_percent": (snapshot.get("system") or {}).get("cpu_percent"),
                        "memory_percent": (snapshot.get("system") or {}).get("memory_percent"),
                        "gpu_count": len(snapshot.get("gpus") or []),
                    },
                )
            except Exception:
                pass


def _csv_rows(text: str) -> Iterable[List[str]]:
    if not text.strip():
        return []

    reader = csv.reader(StringIO(text))
    return [list(row) for row in reader]


def _metric_card(title: str, value: str, detail: str) -> str:
    return (
        "<div style='border:1px solid #ddd;border-radius:8px;padding:10px;"
        "background:rgba(0,0,0,0.025);box-sizing:border-box;min-height:110px'>"
        f"<div style='font-size:12px;color:#666'>{html.escape(title)}</div>"
        f"<div style='font-size:24px;font-weight:700;margin:4px 0'>{html.escape(value)}</div>"
        f"<div style='font-size:12px;color:#555'>{detail}</div>"
        "</div>"
    )


def _bar(percent: Any) -> str:
    value = _safe_float(percent)

    if value is None:
        return ""

    value = max(0.0, min(100.0, value))

    return (
        "<div style='height:8px;background:#e7e7e7;border-radius:4px;margin-top:8px;"
        "overflow:hidden'>"
        f"<div style='width:{value:.1f}%;height:8px;background:#666'></div>"
        "</div>"
    )


def _fmt_percent(value: Any) -> str:
    value = _safe_float(value)
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def _fmt_mb(value: Any) -> str:
    value = _safe_float(value)
    if value is None:
        return "n/a"

    if abs(value) >= 1024:
        return f"{value / 1024.0:.2f} GB"

    return f"{value:.1f} MB"


def _fmt_load(value: Any) -> str:
    if not value:
        return "n/a"

    try:
        return ", ".join(f"{float(v):.2f}" for v in value)
    except Exception:
        return "n/a"


def _percent(part: Any, total: Any) -> Optional[float]:
    part_value = _safe_float(part)
    total_value = _safe_float(total)

    if part_value is None or total_value in (None, 0):
        return None

    return 100.0 * part_value / total_value


def _bytes_to_mb(value: Any) -> float:
    try:
        return float(value) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _to_int(value: Any) -> Optional[int]:
    try:
        text = str(value).strip()
        if not text or text.upper() in {"N/A", "NA", "NONE"}:
            return None
        return int(float(text))
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    return _safe_float(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        text = str(value).strip()
        if not text or text.upper() in {"N/A", "NA", "NONE", "NAN"}:
            return None
        return float(text)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)
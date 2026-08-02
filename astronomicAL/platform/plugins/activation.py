from __future__ import annotations

from typing import Any

from .errors import PluginValidationError
from .sources import PluginOrigin
from .state import PluginStateStore


class PluginActivationService:
    """Host-owned activation policy and persistent enable/disable operations."""

    _AUTO_ENABLE_ORIGINS = {
        PluginOrigin.BUNDLED,
        PluginOrigin.DEVELOPMENT,
    }
    _COMMUNITY_ORIGINS = {
        PluginOrigin.USER,
        PluginOrigin.ENTRY_POINT,
    }

    def __init__(self, state: PluginStateStore) -> None:
        self.state = state

    def should_enable(self, plugin_info: Any) -> bool:
        origin = self._coerce_origin(getattr(plugin_info, "origin", PluginOrigin.UNKNOWN))

        if origin in self._AUTO_ENABLE_ORIGINS:
            return True

        if origin in self._COMMUNITY_ORIGINS:
            plugin_id = str(getattr(plugin_info, "id", "") or "")
            return (
                self.state.community_plugins_enabled
                and bool(plugin_id)
                and self.state.is_enabled(plugin_id)
            )

        if origin == PluginOrigin.RUNTIME:
            status = getattr(plugin_info, "status", None)
            status_value = getattr(status, "value", status)
            return status_value == "enabled"

        return False

    def blocked_reason(self, plugin_info: Any) -> str:
        origin = self._coerce_origin(getattr(plugin_info, "origin", PluginOrigin.UNKNOWN))
        plugin_id = str(getattr(plugin_info, "id", "") or "")

        if origin in self._COMMUNITY_ORIGINS:
            if not self.state.community_plugins_enabled:
                return "Community plugins are disabled."
            if not plugin_id or not self.state.is_enabled(plugin_id):
                return "Plugin is not enabled in persistent plugin state."

        if origin == PluginOrigin.UNKNOWN:
            return "Plugin origin is unknown."

        return "Plugin activation is blocked by the current host policy."

    def can_enable_as_dependency(self, plugin_info: Any) -> bool:
        """Return whether a required plugin may be started implicitly.

        Required dependencies inherit the user's decision to enable their parent
        plugin, but community code still cannot cross the global Community Plugins
        gate.
        """

        origin = self._coerce_origin(getattr(plugin_info, "origin", PluginOrigin.UNKNOWN))

        if origin in self._AUTO_ENABLE_ORIGINS:
            return True

        if origin in self._COMMUNITY_ORIGINS:
            return self.state.community_plugins_enabled

        if origin == PluginOrigin.RUNTIME:
            status = getattr(plugin_info, "status", None)
            status_value = getattr(status, "value", status)
            return status_value == "enabled"

        return False

    def dependency_blocked_reason(self, plugin_info: Any) -> str:
        origin = self._coerce_origin(getattr(plugin_info, "origin", PluginOrigin.UNKNOWN))

        if origin in self._COMMUNITY_ORIGINS and not self.state.community_plugins_enabled:
            return "Community plugins are disabled."

        if origin == PluginOrigin.UNKNOWN:
            return "Plugin origin is unknown."

        return "Required plugin is blocked by the current host activation policy."

    def enable(
        self,
        plugin_id: str,
        *,
        context: Any,
        astronomical_version: str | None = None,
        validate: bool = True,
    ) -> None:
        manager = self._require_manager(context)
        info = manager.plugin_info(plugin_id)
        origin = self._coerce_origin(getattr(info, "origin", PluginOrigin.UNKNOWN))

        if origin in self._COMMUNITY_ORIGINS and not self.state.community_plugins_enabled:
            raise PluginValidationError(
                "Community plugins are disabled. Enable community plugins before "
                f"enabling {plugin_id!r}."
            )

        manager.enable(
            plugin_id,
            context=context,
            astronomical_version=astronomical_version,
            validate=validate,
        )

        self._persist_enabled_community_graph(manager, plugin_id)

    def disable(
        self,
        plugin_id: str,
        *,
        context: Any,
        remove_panels: bool = True,
        cancel_jobs: bool = True,
    ) -> None:
        manager = self._require_manager(context)
        info = manager.plugin_info(plugin_id)
        origin = self._coerce_origin(getattr(info, "origin", PluginOrigin.UNKNOWN))

        # Dependency refusal is a preflight condition, not a teardown failure. Check
        # it before changing persistent state so a rejected disable leaves the user's
        # configuration unchanged.
        assert_can_disable = getattr(manager, "assert_can_disable", None)
        if callable(assert_can_disable):
            assert_can_disable(plugin_id)

        # Persist a community-plugin disable before runtime teardown so the next
        # startup remains safe even if teardown fails or the process exits midway.
        if origin in self._COMMUNITY_ORIGINS:
            self.state.set_enabled(plugin_id, False)

        status = getattr(info, "status", None)
        status_value = getattr(status, "value", status)
        if status_value != "enabled":
            return

        manager.disable(
            plugin_id,
            context=context,
            remove_panels=remove_panels,
            cancel_jobs=cancel_jobs,
        )

    def set_community_plugins_enabled(
        self,
        enabled: bool,
        *,
        context: Any | None = None,
        astronomical_version: str | None = None,
    ) -> dict[str, str]:
        """Set the global community-plugin gate and reconcile live plugin state.

        The individual per-plugin preferences are preserved when the global gate is
        disabled. Re-enabling the gate attempts to start each community plugin that
        is still configured as enabled, matching the behaviour on the next startup.

        Returns a mapping of plugin IDs to lifecycle errors. The global preference is
        still persisted even when one individual plugin fails to enable or disable.
        """

        enabled = bool(enabled)

        # Persist the policy before touching runtime plugins. In particular, turning
        # the gate off must fail closed if the process exits during teardown.
        self.state.set_community_plugins_enabled(enabled)

        if context is None:
            return {}

        manager = getattr(context, "plugins", None)
        if manager is None:
            return {"platform": "AppContext is missing PluginManager."}

        failures: dict[str, str] = {}

        if enabled:
            for info in manager.list_plugins():
                origin = self._coerce_origin(
                    getattr(info, "origin", PluginOrigin.UNKNOWN)
                )
                if origin not in self._COMMUNITY_ORIGINS:
                    continue

                plugin_id = str(getattr(info, "id", "") or "")
                if not plugin_id or not self.state.is_enabled(plugin_id):
                    continue

                status = getattr(info, "status", None)
                status_value = getattr(status, "value", status)
                if status_value == "enabled":
                    continue

                try:
                    manager.enable(
                        plugin_id,
                        context=context,
                        astronomical_version=astronomical_version,
                        validate=True,
                    )
                    self._persist_enabled_community_graph(manager, plugin_id)
                except Exception as exc:
                    failures[plugin_id] = str(exc)

            return failures

        # The global gate is stronger than individual plugin preferences. Cascading
        # teardown ensures required community dependencies cannot be left running
        # merely because another enabled plugin currently depends on them.
        community_ids = [
            str(getattr(info, "id", "") or "")
            for info in manager.list_plugins()
            if self._coerce_origin(
                getattr(info, "origin", PluginOrigin.UNKNOWN)
            )
            in self._COMMUNITY_ORIGINS
        ]

        for plugin_id in community_ids:
            if not plugin_id:
                continue

            try:
                manager.disable(
                    plugin_id,
                    context=context,
                    cascade=True,
                )
            except Exception as exc:
                failures[plugin_id] = str(exc)

        return failures

    def _persist_enabled_community_graph(self, manager: Any, plugin_id: str) -> None:
        """Persist community dependencies implied by a successful activation."""

        resolver = getattr(manager, "resolve_activation_order", None)
        if not callable(resolver):
            info = manager.plugin_info(plugin_id)
            origin = self._coerce_origin(
                getattr(info, "origin", PluginOrigin.UNKNOWN)
            )
            if origin in self._COMMUNITY_ORIGINS:
                self.state.set_enabled(plugin_id, True)
            return

        for dependency_id in resolver(plugin_id):
            info = manager.plugin_info(dependency_id)
            origin = self._coerce_origin(
                getattr(info, "origin", PluginOrigin.UNKNOWN)
            )
            if origin in self._COMMUNITY_ORIGINS:
                self.state.set_enabled(dependency_id, True)

    @staticmethod
    def _require_manager(context: Any) -> Any:
        manager = getattr(context, "plugins", None)
        if manager is None:
            raise RuntimeError("AppContext is missing PluginManager.")
        return manager

    @staticmethod
    def _coerce_origin(value: Any) -> PluginOrigin:
        if isinstance(value, PluginOrigin):
            return value
        try:
            return PluginOrigin(str(value))
        except ValueError:
            return PluginOrigin.UNKNOWN
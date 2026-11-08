from __future__ import annotations

# Layout styling is applied directly to Panel layout models so it remains
# reliable across Panel/Bokeh shadow-root changes.
ROOT_STYLES = {
    "box-sizing": "border-box",
    "width": "100%",
    "max-width": "100%",
    "min-width": "0",
    "padding": "10px",
    "background": "#f3f5f8",
    "color": "#263244",
    "overflow": "visible",
}

SHELL_STYLES = {
    "box-sizing": "border-box",
    "width": "100%",
    "max-width": "100%",
    "min-width": "0",
    "overflow": "visible",
}

SURFACE_STYLES = {
    "box-sizing": "border-box",
    "width": "100%",
    "max-width": "100%",
    "min-width": "0",
    "min-height": "0",
    "padding": "12px",
    "border": "1px solid #d8dee8",
    "border-radius": "10px",
    "background": "#ffffff",
    "box-shadow": "0 1px 2px rgba(15, 23, 42, 0.04)",
}

TOOLBAR_STYLES = {
    **SURFACE_STYLES,
    "padding": "12px 14px",
    "overflow": "visible",
}

SUBSURFACE_STYLES = {
    "box-sizing": "border-box",
    "width": "100%",
    "max-width": "100%",
    "min-width": "0",
    "min-height": "0",
    "padding": "12px",
    "border": "1px solid #e2e7ee",
    "border-radius": "8px",
    "background": "#fbfcfe",
    "overflow": "visible",
}

RESULTS_LIST_STYLES = {
    "box-sizing": "border-box",
    "width": "100%",
    "max-width": "100%",
    "min-width": "0",
    "padding": "2px 4px 2px 0",
    "overflow": "visible",
}

INSPECTOR_STYLES = {
    **SURFACE_STYLES,
    "overflow-x": "hidden",
    "overflow-y": "visible",
}

EMPTY_STATE_STYLES = {
    "box-sizing": "border-box",
    "min-height": "280px",
    "padding": "36px 24px",
    "border": "1px dashed #cbd5e1",
    "border-radius": "9px",
    "background": "#fbfcfe",
    "display": "flex",
    "align-items": "center",
    "justify-content": "center",
}

SEARCH_BAR_INPUT_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  display: block !important;
  width: 100% !important;
  min-width: 0 !important;
  height: 38px !important;
  min-height: 38px !important;
  max-height: 38px !important;
  margin: 0 !important;
  overflow: hidden !important;
  color: #263244 !important;
}
label {
  display: none !important;
  width: 0 !important;
  min-width: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
.bk-input-group {
  box-sizing: border-box !important;
  width: 100% !important;
  height: 38px !important;
  min-height: 38px !important;
  max-height: 38px !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
input,
input.bk-input,
.bk-input {
  box-sizing: border-box !important;
  display: block !important;
  width: 100% !important;
  height: 38px !important;
  min-height: 38px !important;
  max-height: 38px !important;
  margin: 0 !important;
  padding: 0 11px !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 7px !important;
  background: #ffffff !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12.5px !important;
  line-height: 36px !important;
}
input:hover,
.bk-input:hover {
  border-color: #aeb9c8 !important;
}
input:focus,
.bk-input:focus {
  border-color: #0f6fbd !important;
  box-shadow: 0 0 0 2px rgba(15, 111, 189, 0.14) !important;
  outline: none !important;
}
"""

SEARCH_INPUT_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  width: 100% !important;
  min-width: 0 !important;
  overflow: visible !important;
  color: #263244 !important;
}
label {
  display: block !important;
  min-height: 14px !important;
  margin: 0 0 5px 0 !important;
  color: #526071 !important;
  font-size: 10px !important;
  font-weight: 700 !important;
  letter-spacing: 0.025em !important;
  line-height: 14px !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
.bk-input-group {
  width: 100% !important;
  height: 38px !important;
  min-height: 38px !important;
  margin: 0 !important;
  overflow: visible !important;
}
input,
input.bk-input,
.bk-input {
  box-sizing: border-box !important;
  width: 100% !important;
  height: 38px !important;
  min-height: 38px !important;
  padding: 0 11px !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 7px !important;
  background: #ffffff !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12.5px !important;
  line-height: 36px !important;
}
input:hover,
.bk-input:hover {
  border-color: #aeb9c8 !important;
}
input:focus,
.bk-input:focus {
  border-color: #0f6fbd !important;
  box-shadow: 0 0 0 2px rgba(15, 111, 189, 0.14) !important;
  outline: none !important;
}
"""

SELECT_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  width: 100% !important;
  min-width: 0 !important;
  overflow: visible !important;
  color: #263244 !important;
}
label {
  display: block !important;
  min-height: 14px !important;
  margin: 0 0 5px 0 !important;
  color: #526071 !important;
  font-size: 10px !important;
  font-weight: 700 !important;
  letter-spacing: 0.025em !important;
  line-height: 14px !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
.bk-input-group {
  width: 100% !important;
  margin: 0 !important;
  overflow: visible !important;
}
select,
select.bk-input,
.bk-input {
  box-sizing: border-box !important;
  min-height: 34px !important;
  height: 34px !important;
  padding: 0 30px 0 9px !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 7px !important;
  background-color: #ffffff !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 11.5px !important;
}
select:focus,
.bk-input:focus {
  border-color: #0f6fbd !important;
  box-shadow: 0 0 0 2px rgba(15, 111, 189, 0.13) !important;
  outline: none !important;
}
"""

NUMBER_INPUT_STYLESHEET = SELECT_STYLESHEET + """
input,
input.bk-input,
.bk-input {
  padding-right: 8px !important;
}
"""

CHECKBOX_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  width: 100% !important;
  min-width: 0 !important;
  overflow: visible !important;
  color: #263244 !important;
}
.bk-input-group {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 6px !important;
  margin: 0 !important;
}
label {
  box-sizing: border-box !important;
  display: inline-flex !important;
  align-items: center !important;
  min-height: 30px !important;
  margin: 0 !important;
  padding: 5px 9px !important;
  border: 1px solid #d8dee8 !important;
  border-radius: 999px !important;
  background: #ffffff !important;
  color: #526071 !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  line-height: 18px !important;
}
input[type="checkbox"] {
  margin: 0 6px 0 0 !important;
  accent-color: #0f6fbd !important;
}
"""

FORM_CHECKBOX_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  display: block !important;
  width: 100% !important;
  min-width: 0 !important;
  overflow: visible !important;
  color: #263244 !important;
}
.bk-input-group {
  box-sizing: border-box !important;
  display: block !important;
  width: 100% !important;
  margin: 0 !important;
  overflow: visible !important;
}
label {
  box-sizing: border-box !important;
  display: flex !important;
  align-items: flex-start !important;
  width: 100% !important;
  min-width: 0 !important;
  min-height: 34px !important;
  margin: 0 !important;
  padding: 7px 9px !important;
  border: 1px solid #d8dee8 !important;
  border-radius: 7px !important;
  background: #ffffff !important;
  color: #526071 !important;
  font-size: 10.5px !important;
  font-weight: 600 !important;
  line-height: 17px !important;
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: normal !important;
}
input[type="checkbox"] {
  flex: 0 0 auto !important;
  margin: 2px 7px 0 0 !important;
  accent-color: #0f6fbd !important;
}
"""

BUTTON_PRIMARY_STYLESHEET = """
:host {
  box-sizing: border-box !important;
}
button.bk-btn,
button {
  box-sizing: border-box !important;
  height: 38px !important;
  min-height: 38px !important;
  padding: 0 14px !important;
  border: 1px solid #0f6fbd !important;
  border-radius: 7px !important;
  background: #0f6fbd !important;
  box-shadow: none !important;
  color: #ffffff !important;
  font-size: 12px !important;
  font-weight: 650 !important;
  line-height: 36px !important;
  white-space: nowrap !important;
}
button.bk-btn:hover:not(:disabled),
button:hover:not(:disabled) {
  border-color: #0c5f9f !important;
  background: #0c5f9f !important;
}
button.bk-btn:disabled,
button:disabled {
  border-color: #b7cfdf !important;
  background: #b7cfdf !important;
  color: #f8fbfd !important;
  opacity: 1 !important;
}
"""

BUTTON_SUCCESS_STYLESHEET = BUTTON_PRIMARY_STYLESHEET.replace(
    "#0f6fbd", "#1f7a4d"
).replace("#0c5f9f", "#17613d").replace("#b7cfdf", "#b8d8c7")

BUTTON_SECONDARY_STYLESHEET = """
:host {
  box-sizing: border-box !important;
}
button.bk-btn,
button {
  box-sizing: border-box !important;
  height: 38px !important;
  min-height: 38px !important;
  padding: 0 12px !important;
  border: 1px solid #d8dee8 !important;
  border-radius: 7px !important;
  background: #ffffff !important;
  box-shadow: none !important;
  color: #526071 !important;
  font-size: 11.5px !important;
  font-weight: 650 !important;
  line-height: 36px !important;
  white-space: nowrap !important;
}
button.bk-btn:hover:not(:disabled),
button:hover:not(:disabled) {
  border-color: #bdc7d4 !important;
  background: #f5f7fa !important;
  color: #263244 !important;
}
button.bk-btn:disabled,
button:disabled {
  border-color: #e2e7ee !important;
  background: #f7f8fa !important;
  color: #9aa3b2 !important;
  opacity: 1 !important;
}
"""

RESULT_BUTTON_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  min-width: 0 !important;
}
button.bk-btn,
button {
  box-sizing: border-box !important;
  display: flex !important;
  justify-content: flex-start !important;
  width: 100% !important;
  min-width: 0 !important;
  height: 24px !important;
  min-height: 24px !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  font-weight: 680 !important;
  line-height: 24px !important;
  text-align: left !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
}
button.bk-btn:hover,
button:hover {
  background: transparent !important;
  color: #0f6fbd !important;
}
"""

PROGRESS_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  display: block !important;
  width: 100% !important;
  min-width: 0 !important;
  height: 7px !important;
  min-height: 7px !important;
  max-height: 7px !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
progress,
.bk-progress {
  width: 100% !important;
  height: 7px !important;
  min-height: 7px !important;
  border: 0 !important;
  border-radius: 999px !important;
  overflow: hidden !important;
  accent-color: #0f6fbd !important;
}
"""

# Do not install plugin CSS through ``pn.config.raw_css``.  That registry is
# process-global and survives an individual panel instance, so plugin-owned
# styling must remain on HF-owned Panel models or inside their HTML content.
#
# These names remain for compatibility with development snapshots that imported
# them directly, but they intentionally have no global side effect.
HF_BROWSER_CSS = ""

def install_huggingface_styles() -> None:
    """Deprecated no-op; Hugging Face panel styling is instance-local."""
    return None

__all__ = [
    "BUTTON_PRIMARY_STYLESHEET",
    "BUTTON_SECONDARY_STYLESHEET",
    "BUTTON_SUCCESS_STYLESHEET",
    "CHECKBOX_STYLESHEET",
    "FORM_CHECKBOX_STYLESHEET",
    "EMPTY_STATE_STYLES",
    "HF_BROWSER_CSS",
    "INSPECTOR_STYLES",
    "NUMBER_INPUT_STYLESHEET",
    "PROGRESS_STYLESHEET",
    "RESULT_BUTTON_STYLESHEET",
    "RESULTS_LIST_STYLES",
    "ROOT_STYLES",
    "SEARCH_BAR_INPUT_STYLESHEET",
    "SEARCH_INPUT_STYLESHEET",
    "SELECT_STYLESHEET",
    "SHELL_STYLES",
    "SUBSURFACE_STYLES",
    "SURFACE_STYLES",
    "TOOLBAR_STYLES",
    "install_huggingface_styles",
]

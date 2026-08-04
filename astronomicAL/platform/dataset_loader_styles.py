from __future__ import annotations

DATASET_LOADER_CSS = r"""
.al-dataset-loader-card {
  box-sizing: border-box;
  width: 960px;
  height: 680px;
  overflow: hidden;
}

.al-dataset-loader-body {
  box-sizing: border-box;
  width: 100%;
  height: 540px;
  padding: 16px 18px 12px;
  overflow-x: hidden;
  overflow-y: auto;
  background: var(--al-chrome-page, #f3f5f8);
}

.al-dataset-loader-grid {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  align-items: stretch;
}

.al-dataset-loader-section {
  box-sizing: border-box;
  min-width: 0;
  padding: 14px;
  border: 1px solid var(--al-chrome-border, #d8dee8);
  border-radius: var(--al-chrome-radius, 11px);
  background: var(--al-chrome-card, #ffffff);
  box-shadow: var(--al-chrome-shadow, 0 1px 2px rgba(15, 23, 42, 0.05));
}

.al-dataset-loader-section-title {
  margin: 0 0 3px;
  color: var(--al-chrome-text, #263244);
  font-size: 14px;
  font-weight: 700;
  line-height: 1.3;
}

.al-dataset-loader-section-copy {
  margin: 0;
  color: var(--al-chrome-muted, #687386);
  font-size: 12px;
  line-height: 1.45;
}

.al-dataset-loader-field-label {
  margin: 0 0 5px;
  color: var(--al-chrome-text, #263244);
  font-size: 12px;
  font-weight: 650;
  line-height: 1.3;
}

.al-dataset-loader-source-summary,
.al-dataset-loader-behaviour,
.al-dataset-loader-result {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  color: var(--al-chrome-text, #263244);
  font-size: 12px;
  line-height: 1.45;
}

.al-dataset-loader-kv {
  display: grid;
  grid-template-columns: minmax(92px, 0.34fr) minmax(0, 1fr);
  gap: 6px 12px;
  width: 100%;
  margin: 0;
}

.al-dataset-loader-kv dt {
  margin: 0;
  color: var(--al-chrome-muted, #687386);
  font-weight: 600;
}

.al-dataset-loader-kv dd {
  min-width: 0;
  margin: 0;
  color: var(--al-chrome-text, #263244);
  overflow-wrap: anywhere;
}

.al-dataset-loader-code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
}

.al-dataset-loader-status {
  box-sizing: border-box;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  width: 100%;
  min-height: 58px;
  padding: 11px 12px;
  border: 1px solid var(--al-chrome-border, #d8dee8);
  border-radius: 8px;
  background: #fbfcfe;
}

.al-dataset-loader-status-dot {
  flex: 0 0 auto;
  width: 9px;
  height: 9px;
  margin-top: 4px;
  border-radius: 999px;
  background: var(--al-chrome-muted, #687386);
}

.al-dataset-loader-status.busy .al-dataset-loader-status-dot {
  background: var(--al-chrome-accent, #0f6fbd);
  box-shadow: 0 0 0 4px rgba(15, 111, 189, 0.12);
}

.al-dataset-loader-status.success .al-dataset-loader-status-dot {
  background: var(--al-chrome-success, #1f7a4d);
  box-shadow: 0 0 0 4px rgba(31, 122, 77, 0.12);
}

.al-dataset-loader-status.error .al-dataset-loader-status-dot {
  background: var(--al-chrome-danger, #b42318);
  box-shadow: 0 0 0 4px rgba(180, 35, 24, 0.12);
}

.al-dataset-loader-status.warning .al-dataset-loader-status-dot {
  background: var(--al-chrome-warning, #9a6700);
  box-shadow: 0 0 0 4px rgba(154, 103, 0, 0.12);
}

.al-dataset-loader-status-title {
  color: var(--al-chrome-text, #263244);
  font-size: 12px;
  font-weight: 700;
  line-height: 1.35;
}

.al-dataset-loader-status-copy {
  margin-top: 2px;
  color: var(--al-chrome-muted, #687386);
  font-size: 11.5px;
  line-height: 1.42;
}

.al-dataset-loader-footer {
  box-sizing: border-box;
  width: 100%;
  height: 64px;
  padding: 12px 18px;
  border-top: 1px solid var(--al-chrome-divider, #e2e7ee);
  background: var(--al-chrome-card, #ffffff);
  overflow: visible;
}
"""

DATASET_LOADER_INPUT_STYLESHEET = r"""
:host {
  box-sizing: border-box !important;
  color: #263244 !important;
}

.bk-input-group {
  box-sizing: border-box !important;
  width: 100% !important;
  margin: 0 !important;
}

input,
select,
textarea,
.bk-input {
  box-sizing: border-box !important;
  width: 100% !important;
  min-height: 36px !important;
  height: 36px !important;
  padding: 0 10px !important;
  border: 1px solid #c8d0dc !important;
  border-radius: 7px !important;
  background: #ffffff !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
}

select,
select.bk-input {
  padding-right: 30px !important;
}

input:hover:not(:disabled),
select:hover:not(:disabled),
textarea:hover:not(:disabled),
.bk-input:hover:not(:disabled) {
  border-color: #aeb9c8 !important;
}

input:focus,
select:focus,
textarea:focus,
.bk-input:focus {
  border-color: #0f6fbd !important;
  box-shadow: 0 0 0 2px rgba(15, 111, 189, 0.14) !important;
  outline: none !important;
}

input:disabled,
select:disabled,
textarea:disabled,
.bk-input:disabled {
  border-color: #e2e7ee !important;
  background: #f7f8fa !important;
  color: #9aa3b2 !important;
  opacity: 1 !important;
}

option {
  background: #ffffff !important;
  color: #263244 !important;
}
"""

__all__ = [
    "DATASET_LOADER_CSS",
    "DATASET_LOADER_INPUT_STYLESHEET",
]
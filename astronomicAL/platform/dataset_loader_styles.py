from __future__ import annotations

DATASET_LOADER_CSS = r"""
/* Give only the dataset loader more of the viewport. The shared modal helper
 * keeps its conservative 64px top offset for every other modal. Modern browser
 * :has() support lets this stylesheet lift only the wrapper that contains the
 * loader, without changing modal_utils.py or other modal call sites. */
.al-template-modal-card:has(.al-dataset-loader-card) {
  top: 2px !important;
  max-height: calc(100vh - 4px) !important;
}

.al-dataset-loader-card {
  box-sizing: border-box;
  width: 1060px;
  height: min(1072px, calc(100vh - 4px));
  padding: 12px 16px !important;
  overflow: hidden;
}

.al-dataset-loader-body {
  box-sizing: border-box;
  width: 100%;
  height: min(918px, calc(100vh - 158px));
  min-height: 0;
  padding: 14px 18px 12px;
  overflow-x: hidden;
  overflow-y: auto;
  background: var(--al-chrome-page, #f3f5f8);
}

.al-dataset-loader-grid {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  align-items: flex-start;
}

.al-dataset-loader-section {
  box-sizing: border-box;
  min-width: 0;
  padding: 14px;
  border: 1px solid var(--al-chrome-border, #d8dee8);
  border-radius: var(--al-chrome-radius, 11px);
  background: var(--al-chrome-card, #ffffff);
  box-shadow: var(--al-chrome-shadow, 0 1px 2px rgba(15, 23, 42, 0.05));
  margin-bottom: 12px;
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
  grid-template-columns: minmax(104px, 0.34fr) minmax(0, 1fr);
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


.al-dataset-loader-conversion-summary-host {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
}

.al-dataset-loader-conversion-summary {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  padding: 10px 11px;
  border: 1px solid #d8dee8;
  border-radius: 8px;
  background: #f8fafc;
}

.al-dataset-loader-conversion-summary.complete {
  border-color: #b8ddc8;
  background: #f4fbf7;
}

.al-dataset-loader-conversion-summary.error {
  border-color: #efc0bc;
  background: #fff7f6;
}

.al-dataset-loader-conversion-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  min-width: 0;
}

.al-dataset-loader-conversion-eyebrow {
  color: var(--al-chrome-muted, #687386);
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.055em;
  line-height: 1.3;
  text-transform: uppercase;
}

.al-dataset-loader-conversion-label {
  margin-top: 2px;
  color: var(--al-chrome-text, #263244);
  font-size: 12px;
  font-weight: 700;
  line-height: 1.35;
}

.al-dataset-loader-conversion-percent {
  flex: 0 0 auto;
  color: var(--al-chrome-text, #263244);
  font-size: 16px;
  font-weight: 750;
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}

.al-dataset-loader-progress-track {
  position: relative;
  width: 100%;
  height: 7px;
  margin: 8px 0 9px;
  border-radius: 999px;
  background: #e6ebf1;
  overflow: hidden;
}

.al-dataset-loader-progress-fill {
  height: 100%;
  border-radius: inherit;
  background: var(--al-chrome-accent, #0f6fbd);
  transition: width 160ms linear;
}

.al-dataset-loader-conversion-summary.complete .al-dataset-loader-progress-fill {
  background: var(--al-chrome-success, #1f7a4d);
}

.al-dataset-loader-conversion-summary.error .al-dataset-loader-progress-fill {
  background: var(--al-chrome-danger, #b42318);
}

.al-dataset-loader-progress-fill.is-indeterminate {
  position: absolute;
  width: 32% !important;
  animation: al-dataset-loader-progress-slide 1.2s ease-in-out infinite;
}

@keyframes al-dataset-loader-progress-slide {
  0% { left: -34%; }
  50% { left: 52%; }
  100% { left: 102%; }
}

.al-dataset-loader-progress-kv {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 4px 8px;
  width: 100%;
  margin: 0;
  font-size: 10.5px;
  line-height: 1.35;
}

.al-dataset-loader-progress-kv dt {
  margin: 0;
  color: var(--al-chrome-muted, #687386);
  font-weight: 650;
  white-space: nowrap;
}

.al-dataset-loader-progress-kv dd {
  min-width: 0;
  margin: 0;
  color: var(--al-chrome-text, #263244);
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}

.al-dataset-loader-footer {
  box-sizing: border-box;
  width: 100%;
  height: 54px;
  padding: 8px 18px;
  border-top: 1px solid var(--al-chrome-divider, #e2e7ee);
  background: var(--al-chrome-card, #ffffff);
  overflow: hidden;
}

/*
 * The shared overlay modal remains untouched. The dataset loader uses almost
 * the full viewport when available and contracts only its own scrollable body
 * on shorter displays, keeping the title and footer continuously accessible.
 */
.al-dataset-loader-card {
  height: min(1072px, calc(100vh - 4px)) !important;
}

.al-dataset-loader-body {
  height: min(918px, calc(100vh - 158px)) !important;
}

@media (max-width: 1120px) {
  .al-dataset-loader-card {
    width: min(1060px, calc(100vw - 24px));
  }
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
  /* Panel/Bokeh's normal select affordance can disappear once the loader's
   * custom input background is applied. Use an explicit chevron so file and
   * subresource selectors remain visually identifiable as dropdowns. */
  -webkit-appearance: none !important;
  appearance: none !important;
  padding-right: 38px !important;
  background-color: #ffffff !important;
  background-image:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 14 14'%3E%3Cpath d='M3.25 5.25 7 9l3.75-3.75' fill='none' stroke='%23687386' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") !important;
  background-repeat: no-repeat !important;
  background-position: right 11px center !important;
  background-size: 14px 14px !important;
  cursor: pointer !important;
}

input:hover:not(:disabled),
select:hover:not(:disabled),
.bk-input:hover:not(:disabled) {
  border-color: #aeb9c8 !important;
}

input:focus,
select:focus,
.bk-input:focus {
  border-color: #0f6fbd !important;
  box-shadow: 0 0 0 2px rgba(15, 111, 189, 0.14) !important;
  outline: none !important;
}

input:disabled,
select:disabled,
.bk-input:disabled {
  border-color: #e2e7ee !important;
  background-color: #f7f8fa !important;
  color: #9aa3b2 !important;
  opacity: 1 !important;
}

select:disabled,
select.bk-input:disabled {
  /* Keep the chevron visible so a disabled Select still reads as a Select. */
  background-image:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 14 14'%3E%3Cpath d='M3.25 5.25 7 9l3.75-3.75' fill='none' stroke='%239aa3b2' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") !important;
  background-repeat: no-repeat !important;
  background-position: right 11px center !important;
  background-size: 14px 14px !important;
  cursor: not-allowed !important;
}

option {
  background: #ffffff !important;
  color: #263244 !important;
}
"""

DATASET_LOADER_LOG_STYLESHEET = r"""
:host {
  box-sizing: border-box;
  display: block;
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
}

.al-dataset-loader-log-viewport {
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  padding: 9px 10px;
  border: 1px solid #d8dee8;
  border-radius: 7px;
  background: #111827;
  color: #e5e7eb;
  overflow: auto;
  overscroll-behavior: contain;
  scrollbar-gutter: stable;
  outline: none;
  cursor: text;
}

.al-dataset-loader-log-viewport:focus-visible {
  border-color: #0f6fbd;
  box-shadow: 0 0 0 2px rgba(15, 111, 189, 0.20);
}

.al-dataset-loader-log-pre {
  box-sizing: border-box;
  min-width: 100%;
  margin: 0;
  padding: 0;
  color: inherit;
  background: transparent;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 10.5px;
  line-height: 1.45;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  user-select: text;
}
"""

__all__ = [
    "DATASET_LOADER_CSS",
    "DATASET_LOADER_INPUT_STYLESHEET",
    "DATASET_LOADER_LOG_STYLESHEET",
]


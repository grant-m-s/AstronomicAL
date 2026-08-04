from __future__ import annotations

APPLICATION_CHROME_CSS = """
:root {
  --al-application-chrome-connected: 1;
  --al-application-header-height: 56px;
  --al-application-toolbar-height: 48px;
  --al-chrome-page: #f3f5f8;
  --al-chrome-card: #ffffff;
  --al-panel-titlebar: #e8edf3;
  --al-panel-titlebar-border: #d2d9e3;
  --al-chrome-toolbar-tier: #fbfcfe;
  --al-chrome-control: #f7f8fa;
  --al-chrome-control-hover: #edf1f5;
  --al-chrome-border: #d8dee8;
  --al-chrome-divider: #e2e7ee;
  --al-chrome-text: #263244;
  --al-chrome-muted: #687386;
  --al-chrome-accent: #0f6fbd;
  --al-chrome-accent-hover: #0c5f9f;
  --al-chrome-accent-soft: rgba(15, 111, 189, 0.10);
  --al-chrome-warning: #9a6700;
  --al-chrome-warning-soft: #fff7d6;
  --al-chrome-danger: #b42318;
  --al-chrome-danger-soft: #fff0ee;
  --al-chrome-success: #1f7a4d;
  --al-chrome-success-soft: #ecf8f1;
  --al-chrome-radius: 11px;
  --al-chrome-control-radius: 7px;
  --al-chrome-shadow:
    0 1px 2px rgba(15, 23, 42, 0.05),
    0 3px 10px rgba(15, 23, 42, 0.045);
}

/* Keep the complete application canvas consistent with the chrome backing.
 * Panels, menus and controls retain their own white surfaces; only the page,
 * template and workspace layers use the neutral grey canvas. */
html,
body,
body > .bk-root,
.pn-template,
.pn-template-base,
.pn-template-main,
#content,
#main,
#main-content,
.pn-main,
.react-grid-layout {
  background-color: var(--al-chrome-page) !important;
}

/* Keep workspace panel hosts visually distinct from the neutral page canvas.
 * DynamicReactGrid renders .tile, .tile-header and .tile-body inside its shadow
 * root, so those internal elements are styled through the component's own
 * stylesheet in load_config.py rather than from this global stylesheet. */
.react-grid-layout > .react-grid-item,
.react-grid-layout > .react-grid-item > div,
.react-grid-item .pn-card,
.react-grid-item .card,
.react-grid-item .bk-card,
.react-grid-item .bk-panel-models-layout-Card,
.react-grid-item .panel-widget-box,
.react-grid-item [class~="pn-card"] {
  background-color: var(--al-chrome-card) !important;
}

/* Avoid a white strip below short workspace layouts. */
html,
body,
.pn-template,
.pn-template-base,
#content,
#main {
  min-height: 100%;
}

/* The ReactTemplate's built-in title occupies a separate left column. The
 * application brand is rendered inside the card instead, so both chrome rows
 * can use the same full-width geometry. */
#header > .navbar-brand,
#header .navbar-brand,
#header #header-title,
#header #header-logo,
#header .app-logo,
#header .app-title,
#header .title {
  display: none !important;
}

#header {
  box-sizing: border-box;
  position: relative !important;
  height: var(--al-application-header-height);
  min-height: var(--al-application-header-height);
  max-height: var(--al-application-header-height);
  padding: 0 !important;
  border: 0 !important;
  background: var(--al-chrome-page) !important;
  color: var(--al-chrome-text) !important;
  overflow: visible !important;
}

#header-items {
  box-sizing: border-box;
  position: absolute !important;
  inset: 0 !important;
  grid-column: 1 / -1 !important;
  width: 100% !important;
  max-width: none !important;
  height: var(--al-application-header-height);
  min-height: var(--al-application-header-height);
  margin: 0 !important;
  padding: 6px 8px 0 !important;
  overflow: visible !important;
}

/* RuntimeStatusBox is the application-level activity indicator. The template
 * busy indicator is disabled from Python, while these selectors prevent a
 * stale template root from reappearing during a hot reload. */
#header #busy-indicator,
#header #busy_indicator,
#header .pn-busy-indicator,
#header .pn-loading-spinner,
#header .bk-LoadingSpinner,
#header [class*="busy-indicator"],
#header [class*="loading-spinner"] {
  display: none !important;
  width: 0 !important;
  min-width: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}

.al-application-header-shell {
  box-sizing: border-box;
  width: 100% !important;
  min-width: 0 !important;
  height: 50px !important;
  min-height: 50px !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
}

.al-application-header {
  box-sizing: border-box;
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100% !important;
  height: 50px !important;
  min-height: 50px !important;
  margin: 0 !important;
  padding: 8px 10px !important;
  border: 1px solid var(--al-chrome-border);
  border-bottom: 0;
  border-radius: var(--al-chrome-radius) var(--al-chrome-radius) 0 0;
  background: var(--al-chrome-card);
  box-shadow: none;
  color: var(--al-chrome-text);
  overflow: visible !important;
}

.al-application-header > div,
.al-application-header .bk-Row,
.al-header-group,
.al-header-group > div {
  min-width: 0 !important;
  overflow: visible !important;
}

.al-application-brand {
  box-sizing: border-box;
  flex: 0 0 176px !important;
  width: 176px !important;
  min-width: 176px !important;
  height: 32px !important;
  margin: 0 !important;
  overflow: hidden !important;
}

.al-brand-lockup {
  box-sizing: border-box;
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  height: 32px;
  color: var(--al-chrome-text);
  white-space: nowrap;
}

.al-brand-mark {
  box-sizing: border-box;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: var(--al-chrome-accent);
  color: #ffffff;
  font-size: 14px;
  font-weight: 750;
  letter-spacing: -0.03em;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.18);
}

.al-brand-wordmark {
  color: var(--al-chrome-text);
  font-size: 18px;
  font-weight: 650;
  letter-spacing: -0.025em;
  line-height: 32px;
}

.al-brand-wordmark strong {
  color: var(--al-chrome-accent);
  font-weight: 750;
}

.al-header-group {
  box-sizing: border-box;
  display: flex !important;
  align-items: center !important;
  flex-wrap: nowrap !important;
  gap: 4px !important;
  height: 32px !important;
  min-height: 32px !important;
  margin: 0 !important;
}

.al-header-data-group {
  flex: 0 1 auto !important;
  max-width: min(820px, 62vw) !important;
}

.al-header-status-group {
  flex: 0 0 auto !important;
}

.al-header-workspace-group {
  flex: 0 0 auto !important;
  margin-left: 0 !important;
}

.al-header-inner-divider {
  flex: 0 0 1px !important;
  width: 1px !important;
  min-width: 1px !important;
  height: 20px !important;
  margin: 0 3px !important;
  background: var(--al-chrome-divider);
}

.al-dataset-header {
  box-sizing: border-box;
  display: flex !important;
  align-items: center !important;
  flex-wrap: nowrap !important;
  gap: 5px !important;
  width: auto !important;
  min-width: 0 !important;
  height: 30px !important;
  margin: 0 !important;
  overflow: visible !important;
}

.al-dataset-context-label {
  box-sizing: border-box;
  flex: 0 0 auto !important;
  width: auto !important;
  height: 30px !important;
  margin: 0 3px 0 0 !important;
  color: var(--al-chrome-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.055em;
  line-height: 30px;
  text-transform: uppercase;
  white-space: nowrap;
}

.al-header-dataset-select {
  flex: 0 1 270px !important;
  width: 270px !important;
  min-width: 220px !important;
  max-width: 300px !important;
  height: 30px !important;
  max-height: 30px !important;
}

.al-dataset-stats {
  box-sizing: border-box;
  flex: 0 0 132px !important;
  width: 132px !important;
  min-width: 132px !important;
  max-width: 132px !important;
  height: 30px !important;
  margin: 0 2px 0 1px !important;
  padding: 0 2px !important;
  color: var(--al-chrome-muted);
  font-size: 10.5px;
  font-variant-numeric: tabular-nums;
  line-height: 30px;
  white-space: nowrap;
  overflow: hidden !important;
  text-overflow: ellipsis;
}

.al-header-mapping-slot {
  box-sizing: border-box;
  display: flex !important;
  align-items: center !important;
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
  height: 30px !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
}


.al-header-add-data {
  flex: 0 0 96px !important;
  width: 96px !important;
  min-width: 96px !important;
  height: 30px !important;
  max-height: 30px !important;
}

.al-header-mapping-control,
.al-header-mapping-control > div {
  flex: 0 0 120px !important;
  width: 120px !important;
  min-width: 120px !important;
  height: 30px !important;
  max-height: 30px !important;
  margin: 0 !important;
}

.al-header-runtime-control,
.al-header-runtime-control > div {
  height: 30px !important;
  max-height: 30px !important;
  margin: 0 !important;
}

.al-add-menu-btn {
  box-sizing: border-box !important;
  flex: 0 0 104px !important;
  width: 104px !important;
  min-width: 104px !important;
  max-width: 104px !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  margin: 0 !important;
  padding: 0 !important;
  font-size: 12px !important;
  font-weight: 650 !important;
  line-height: 30px !important;
}

/* Runtime status and its Details action are presented as one segmented pill. */
.al-runtime-status-box {
  box-sizing: border-box !important;
  display: flex !important;
  align-items: center !important;
  gap: 0 !important;
  width: auto !important;
  min-width: 0 !important;
  height: 30px !important;
  max-height: 30px !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 1px solid var(--al-chrome-border) !important;
  border-radius: 999px !important;
  background: var(--al-chrome-card) !important;
  color: var(--al-chrome-text) !important;
  overflow: hidden !important;
}

.al-runtime-status-box > div:first-child {
  box-sizing: border-box;
  flex: 0 0 180px !important;
  width: 180px !important;
  min-width: 180px !important;
  max-width: 180px !important;
  height: 28px !important;
  min-height: 28px !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 999px 0 0 999px !important;
  background: transparent !important;
  color: inherit !important;
  font-size: 11px;
  font-weight: 600;
  line-height: 28px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.al-runtime-status-box .al-runtime-status-pill {
  box-sizing: border-box !important;
  display: block !important;
  width: 220px !important;
  min-width: 220px !important;
  max-width: 220px !important;
  height: 28px !important;
  padding: 5px 10px !important;
  border: 0 !important;
  border-radius: 999px 0 0 999px !important;
  line-height: 18px !important;
}

.al-runtime-status-box > div:last-child {
  overflow: hidden !important;
  background: #ffffff !important;
  border-radius: 0 999px 999px 0 !important;
}

.al-runtime-status-box > div:last-child button,
.al-runtime-status-box > div:last-child .bk-btn {
  width: 100% !important;
  height: 28px !important;
  border: 0 !important;
  border-left: 1px solid var(--al-chrome-border, #d8dee8) !important;
  border-radius: 0 999px 999px 0 !important;
  background: #ffffff !important;
  background-color: #ffffff !important;
  opacity: 1 !important;
  color: var(--al-chrome-text, #263244) !important;
}

.al-runtime-status-box > div:last-child button:hover,
.al-runtime-status-box > div:last-child .bk-btn:hover {
  background: #f7f8fa !important;
  background-color: #f7f8fa !important;
}

.al-runtime-status-box .al-runtime-status-pill.idle {
  background: var(--al-chrome-success-soft) !important;
  color: var(--al-chrome-success) !important;
}

.al-runtime-status-box .al-runtime-status-pill.busy {
  background: var(--al-chrome-accent-soft) !important;
  color: var(--al-chrome-accent) !important;
}

.al-runtime-status-box .al-runtime-status-pill.slow {
  background: var(--al-chrome-warning-soft) !important;
  color: var(--al-chrome-warning) !important;
}

.al-runtime-status-box .al-runtime-status-pill.error {
  background: var(--al-chrome-danger-soft) !important;
  color: var(--al-chrome-danger) !important;
}

/* Both toolbar popovers are anchored to the right edge of their trigger.
 * The overflow rule also protects against Panel/Bokeh inline positioning that
 * would otherwise place the menu beyond the viewport edge. Popovers are flat
 * cards; no detached callout arrow is rendered beside the trigger. */
.al-toolbar-overflow-popover {
  position: absolute !important;
  top: 34px !important;
  left: auto !important;
  right: 0 !important;
  inset-inline-start: auto !important;
  inset-inline-end: 0 !important;
  width: 188px !important;
  min-width: 188px !important;
  max-width: calc(100vw - 24px) !important;
  transform: none !important;
  overflow: hidden !important;
}

.al-toolbar-popover::before,
.al-toolbar-popover::after,
.al-toolbar-menu-wrapper::before,
.al-toolbar-menu-wrapper::after,
.al-toolbar-layout-popover::before,
.al-toolbar-layout-popover::after,
.al-toolbar-overflow-popover::before,
.al-toolbar-overflow-popover::after {
  display: none !important;
  content: none !important;
  width: 0 !important;
  height: 0 !important;
  border: 0 !important;
  box-shadow: none !important;
}

.al-layout-controls-mount {
  box-sizing: border-box !important;
  flex: 0 0 0 !important;
  width: 0 !important;
  min-width: 0 !important;
  max-width: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  max-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
}

@media (max-width: 1460px) {
  .al-dataset-stats {
    display: none !important;
  }

  .al-header-data-group {
    max-width: 60vw !important;
  }
}

@media (max-width: 1220px) {
  .al-dataset-context-label {
    display: none !important;
  }

  .al-header-dataset-select {
    width: 235px !important;
    max-width: 235px !important;
  }

  .al-application-brand {
    flex-basis: 154px !important;
    width: 154px !important;
    min-width: 154px !important;
  }

  .al-brand-wordmark {
    font-size: 16px;
  }
}

@media (max-width: 1040px) {
  .al-header-mapping-slot,
  .al-header-mapping-control,
  .al-header-mapping-control > div {
    display: none !important;
  }

  .al-header-dataset-select {
    width: 210px !important;
    max-width: 210px !important;
  }
}

@media (max-width: 860px) {
  .al-application-brand {
    flex-basis: 34px !important;
    width: 34px !important;
    min-width: 34px !important;
  }

  .al-brand-wordmark {
    display: none !important;
  }

  .al-header-runtime-control {
    display: none !important;
  }
}

@media (max-width: 680px) {
  #header-items {
    padding-left: 8px !important;
    padding-right: 8px !important;
  }

  .al-application-header {
    gap: 5px !important;
    padding-left: 7px !important;
    padding-right: 7px !important;
  }

  .al-header-dataset-select {
    width: min(210px, 42vw) !important;
    max-width: min(210px, 42vw) !important;
    min-width: 105px !important;
  }

  .al-header-add-data {
    width: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
    flex-basis: 34px !important;
  }
}
"""

HEADER_BUTTON_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  color: #263244 !important;
}
button.bk-btn {
  box-sizing: border-box !important;
  min-height: 30px !important;
  height: 30px !important;
  max-height: 30px !important;
  padding: 0 10px !important;
  border: 1px solid #d8dee8 !important;
  border-radius: 7px !important;
  background: #f7f8fa !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  font-weight: 550 !important;
  line-height: 28px !important;
  white-space: nowrap !important;
}
button.bk-btn:hover:not(:disabled) {
  border-color: #c8d0dc !important;
  background: #edf1f5 !important;
  color: #263244 !important;
}
button.bk-btn:focus-visible {
  outline: 2px solid rgba(15, 111, 189, 0.35) !important;
  outline-offset: 1px !important;
}
button.bk-btn:disabled {
  border-color: #e2e7ee !important;
  background: #f7f8fa !important;
  color: #9aa3b2 !important;
  opacity: 1 !important;
}
"""

HEADER_PRIMARY_BUTTON_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  color: #ffffff !important;
}
button.bk-btn {
  box-sizing: border-box !important;
  min-height: 30px !important;
  height: 30px !important;
  max-height: 30px !important;
  padding: 0 11px !important;
  border: 1px solid #0f6fbd !important;
  border-radius: 7px !important;
  background: #0f6fbd !important;
  box-shadow: none !important;
  color: #ffffff !important;
  font-size: 12px !important;
  font-weight: 650 !important;
  line-height: 28px !important;
  white-space: nowrap !important;
}
button.bk-btn:hover:not(:disabled) {
  border-color: #0c5f9f !important;
  background: #0c5f9f !important;
  color: #ffffff !important;
}
button.bk-btn:focus-visible {
  outline: 2px solid rgba(15, 111, 189, 0.35) !important;
  outline-offset: 2px !important;
}
button.bk-btn:disabled {
  border-color: #aac8df !important;
  background: #aac8df !important;
  color: #f6fbff !important;
  opacity: 1 !important;
}
"""

HEADER_STATUS_BUTTON_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  color: #263244 !important;
}
button.bk-btn {
  box-sizing: border-box !important;
  min-height: 30px !important;
  height: 30px !important;
  max-height: 30px !important;
  padding: 0 9px !important;
  border: 1px solid #d8dee8 !important;
  border-radius: 999px !important;
  background: #ffffff !important;
  box-shadow: none !important;
  color: #687386 !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  line-height: 28px !important;
  white-space: nowrap !important;
}
button.bk-btn:hover:not(:disabled) {
  border-color: #b9c5d4 !important;
  background: #f7f8fa !important;
  color: #263244 !important;
}
button.bk-btn:focus-visible {
  outline: 2px solid rgba(15, 111, 189, 0.35) !important;
  outline-offset: 1px !important;
}
"""

HEADER_RUNTIME_BUTTON_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  color: #687386 !important;
}
button.bk-btn {
  box-sizing: border-box !important;
  min-height: 28px !important;
  height: 28px !important;
  max-height: 28px !important;
  padding: 0 9px !important;
  border: 0 !important;
  border-left: 1px solid #d8dee8 !important;
  border-radius: 0 999px 999px 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #687386 !important;
  font-size: 11px !important;
  font-weight: 650 !important;
  line-height: 28px !important;
  white-space: nowrap !important;
}
button.bk-btn:hover:not(:disabled) {
  background: #edf1f5 !important;
  color: #263244 !important;
}
button.bk-btn:focus-visible {
  outline: 2px solid rgba(15, 111, 189, 0.35) !important;
  outline-offset: -2px !important;
}
"""

HEADER_SELECT_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  color: #263244 !important;
}
.bk-input-group {
  box-sizing: border-box !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  margin: 0 !important;
}
select,
select.bk-input,
.bk-input {
  box-sizing: border-box !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  margin: 0 !important;
  padding: 0 30px 0 9px !important;
  border: 1px solid #d8dee8 !important;
  border-radius: 7px !important;
  background-color: #f7f8fa !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  font-weight: 550 !important;
  line-height: 28px !important;
  text-overflow: ellipsis !important;
}
select:hover:not(:disabled),
.bk-input:hover:not(:disabled) {
  border-color: #c8d0dc !important;
  background-color: #edf1f5 !important;
}
select:focus,
.bk-input:focus {
  border-color: #0f6fbd !important;
  background-color: #ffffff !important;
  box-shadow: 0 0 0 2px rgba(15, 111, 189, 0.14) !important;
  outline: none !important;
}
select:disabled,
.bk-input:disabled {
  border-color: #e2e7ee !important;
  background-color: #f7f8fa !important;
  color: #9aa3b2 !important;
  opacity: 1 !important;
}
option {
  background: #ffffff !important;
  color: #263244 !important;
}
"""

__all__ = [
    "APPLICATION_CHROME_CSS",
    "HEADER_BUTTON_STYLESHEET",
    "HEADER_PRIMARY_BUTTON_STYLESHEET",
    "HEADER_RUNTIME_BUTTON_STYLESHEET",
    "HEADER_SELECT_STYLESHEET",
    "HEADER_STATUS_BUTTON_STYLESHEET",
]
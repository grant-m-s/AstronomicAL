from __future__ import annotations

from astronomicAL.platform.application_chrome_styles import APPLICATION_CHROME_CSS

TOOLBAR_CSS = APPLICATION_CHROME_CSS + """
:root {
  --al-toolbar-v8: 1;
  --al-toolbar-v9: 1;
  --al-toolbar-v10: 1;
  --al-toolbar-v11: 1;
  --al-toolbar-connected: 1;
}

#content {
  height: calc(100vh - var(--al-application-header-height));
}

#main {
  box-sizing: border-box;
  padding-top: var(--al-application-toolbar-height) !important;
}

.al-application-toolbar-mount {
  position: relative !important;
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

.al-application-toolbar-shell {
  box-sizing: border-box;
  position: fixed !important;
  top: var(--al-application-header-height);
  left: 0;
  right: 0;
  z-index: 1200;
  isolation: isolate;
  width: auto !important;
  min-width: 0;
  height: var(--al-application-toolbar-height);
  min-height: var(--al-application-toolbar-height);
  padding: 0 8px 6px;
  border: 0;
  background: var(--al-chrome-page);
  overflow: visible !important;
}

.al-application-toolbar {
  box-sizing: border-box;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  height: 42px;
  min-height: 42px;
  padding: 5px 10px;
  border: 1px solid var(--al-chrome-border);
  border-top-color: var(--al-chrome-divider);
  border-radius: 0 0 var(--al-chrome-radius) var(--al-chrome-radius);
  background: var(--al-chrome-toolbar-tier);
  box-shadow: var(--al-chrome-shadow);
  overflow: visible !important;
}

.al-application-toolbar > div,
.al-application-toolbar .bk-Row {
  min-width: 0 !important;
  max-width: 100% !important;
  overflow: visible !important;
}

.al-toolbar-group {
  box-sizing: border-box;
  display: flex;
  align-items: center;
  flex-wrap: nowrap;
  gap: 3px;
  min-width: 0;
  max-width: 100%;
  height: 30px;
  min-height: 30px;
  max-height: 30px;
  overflow: visible;
}

.al-toolbar-navigation {
  flex: 1 1 390px !important;
  min-width: 0 !important;
}

.al-toolbar-scope-group {
  flex: 0 0 201px !important;
  width: 201px !important;
}

.al-toolbar-workspace-group {
  flex: 0 0 136px !important;
  width: 136px !important;
  margin-left: auto !important;
  padding-left: 5px;
  overflow: visible !important;
}

.al-toolbar-inner-divider {
  align-self: center;
  width: 1px;
  min-width: 1px;
  height: 20px;
  margin: 0 4px;
  background: var(--al-chrome-divider);
}

.al-toolbar-icon-only,
.al-toolbar-icon-only > div {
  flex: 0 0 30px !important;
  min-width: 30px !important;
  width: 30px !important;
  max-width: 30px !important;
  height: 30px !important;
  max-height: 30px !important;
}

.al-toolbar-search {
  flex: 1 1 240px !important;
  min-width: 110px !important;
  max-width: 260px !important;
  height: 30px !important;
  max-height: 30px !important;
  overflow: visible !important;
}

.al-toolbar-scope {
  flex: 0 0 112px !important;
  width: 112px !important;
  height: 30px !important;
  max-height: 30px !important;
}

.al-toolbar-selection-button {
  flex: 0 0 86px !important;
  width: 86px !important;
  height: 30px !important;
  max-height: 30px !important;
}

.al-toolbar-menu {
  height: 30px !important;
  max-height: 30px !important;
}

.al-toolbar-position {
  flex: 0 0 90px !important;
  width: 90px !important;
  min-width: 90px;
  height: 30px !important;
  max-height: 30px !important;
  color: var(--al-chrome-text);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  line-height: 30px;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
}

.al-toolbar-position .al-position-current {
  font-weight: 650;
}

.al-toolbar-position .al-position-separator,
.al-toolbar-position .al-position-total {
  color: var(--al-chrome-muted);
}

.al-toolbar-busy {
  cursor: progress;
}

.al-toolbar-menu-wrapper {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: visible !important;
}

.al-toolbar-popover {
  box-sizing: border-box;
  isolation: isolate;
  padding: 0 !important;
  border: 1px solid var(--al-chrome-border);
  border-radius: 10px;
  background: var(--al-chrome-card);
  box-shadow:
    0 10px 28px rgba(15, 23, 42, 0.12),
    0 2px 8px rgba(15, 23, 42, 0.08);
  overflow: hidden !important;
}

.al-toolbar-popover > div,
.al-toolbar-popover > .bk-Column,
.al-toolbar-popover > .bk-column {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.al-toolbar-popover::before,
.al-toolbar-popover::after,
.al-toolbar-menu-wrapper::before,
.al-toolbar-menu-wrapper::after {
  display: none !important;
  content: none !important;
  width: 0 !important;
  height: 0 !important;
  border: 0 !important;
  box-shadow: none !important;
}

.al-toolbar-popover-divider {
  box-sizing: border-box;
  height: 1px !important;
  min-height: 1px !important;
  max-height: 1px !important;
  background: var(--al-chrome-divider);
}

.al-toolbar-layout-popover {
  left: auto !important;
  right: 0 !important;
  min-width: 224px !important;
  width: 224px !important;
  max-width: calc(100vw - 24px) !important;
  transform: none !important;
}

.al-toolbar-overflow-popover {
  position: absolute !important;
  top: 34px !important;
  left: auto !important;
  right: 0 !important;
  inset-inline-start: auto !important;
  inset-inline-end: 0 !important;
  min-width: 188px !important;
  width: 188px !important;
  max-width: calc(100vw - 24px) !important;
  transform: none !important;
  overflow: hidden !important;
}

.al-toolbar-empty-state {
  opacity: 0.48 !important;
  filter: saturate(0.72);
  transition: opacity 120ms ease;
}

.al-toolbar-empty-state:hover,
.al-toolbar-empty-state:focus-within {
  opacity: 0.62 !important;
}

.al-toolbar-popover-heading {
  box-sizing: border-box;
  height: 22px !important;
  min-height: 22px !important;
  margin: 0 !important;
  padding: 5px 10px 0 !important;
  color: var(--al-chrome-muted);
  font-size: 9.5px;
  font-weight: 750;
  letter-spacing: 0.07em;
  line-height: 20px;
  text-transform: uppercase;
  white-space: nowrap;
}

@media (max-width: 920px) {
  .al-toolbar-scope-group {
    flex-basis: 112px !important;
    width: 112px !important;
  }

  .al-toolbar-selection-button {
    display: none !important;
  }
}

@media (max-width: 720px) {
  .al-application-toolbar-shell {
    padding-left: 6px;
    padding-right: 6px;
  }

  .al-application-toolbar {
    padding-left: 7px;
    padding-right: 7px;
  }

  .al-toolbar-scope-group {
    display: none !important;
  }

  .al-toolbar-search {
    min-width: 80px !important;
  }

  .al-toolbar-position {
    flex-basis: 72px !important;
    width: 72px !important;
    min-width: 72px !important;
  }
}
"""

_TOOLBAR_BUTTON_STYLESHEET = """
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

_TOOLBAR_ICON_BUTTON_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  width: 30px !important;
  min-width: 30px !important;
  max-width: 30px !important;
  height: 30px !important;
  color: #4d5a6d !important;
}
button.bk-btn {
  box-sizing: border-box !important;
  width: 30px !important;
  min-width: 30px !important;
  max-width: 30px !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  padding: 0 !important;
  border: 1px solid transparent !important;
  border-radius: 7px !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #4d5a6d !important;
  line-height: 28px !important;
}
button.bk-btn:hover:not(:disabled) {
  border-color: #d8dee8 !important;
  background: #edf1f5 !important;
  color: #263244 !important;
}
button.bk-btn:focus-visible {
  outline: 2px solid rgba(15, 111, 189, 0.35) !important;
  outline-offset: 1px !important;
}
button.bk-btn:disabled {
  border-color: transparent !important;
  background: transparent !important;
  color: #a2aab7 !important;
  opacity: 1 !important;
}
"""

_TOOLBAR_MENU_STYLESHEET = """
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
  font-weight: 600 !important;
  line-height: 28px !important;
  white-space: nowrap !important;
}
button.bk-btn:hover:not(:disabled) {
  border-color: #c8d0dc !important;
  background: #edf1f5 !important;
}
button.bk-btn:focus-visible {
  outline: 2px solid rgba(15, 111, 189, 0.35) !important;
  outline-offset: 1px !important;
}
button.bk-btn:disabled {
  color: #9aa3b2 !important;
  opacity: 1 !important;
}
"""

_TOOLBAR_SELECTION_STYLESHEET = """
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
  border-radius: 7px !important;
  background: #ffffff !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  line-height: 28px !important;
  white-space: nowrap !important;
}
button.bk-btn:hover:not(:disabled) {
  border-color: #a9bfd2 !important;
  background: #f3f8fc !important;
}
:host(.al-is-selected) button.bk-btn,
.al-is-selected button.bk-btn {
  border-color: #9dc6e5 !important;
  background: rgba(15, 111, 189, 0.10) !important;
  color: #0f5f9f !important;
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

_TOOLBAR_TEXT_INPUT_STYLESHEET = """
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
input,
input.bk-input,
.bk-input {
  box-sizing: border-box !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  margin: 0 !important;
  padding: 0 9px !important;
  border: 1px solid #d8dee8 !important;
  border-radius: 7px !important;
  background: #f7f8fa !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  line-height: 28px !important;
}
input::placeholder,
.bk-input::placeholder {
  color: #8993a3 !important;
  opacity: 1 !important;
}
input:hover:not(:disabled),
.bk-input:hover:not(:disabled) {
  border-color: #c8d0dc !important;
  background: #edf1f5 !important;
}
input:focus,
.bk-input:focus {
  border-color: #0f6fbd !important;
  background: #ffffff !important;
  box-shadow: 0 0 0 2px rgba(15, 111, 189, 0.14) !important;
  outline: none !important;
}
input:disabled,
.bk-input:disabled {
  border-color: #e2e7ee !important;
  background: #f7f8fa !important;
  color: #9aa3b2 !important;
  opacity: 1 !important;
}
"""

_TOOLBAR_SELECT_STYLESHEET = """
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
  padding: 0 28px 0 9px !important;
  border: 1px solid #d8dee8 !important;
  border-radius: 7px !important;
  background-color: #f7f8fa !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  font-weight: 550 !important;
  line-height: 28px !important;
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

_TOOLBAR_POPUP_ITEM_STYLESHEET = """
:host {
  box-sizing: border-box !important;
  color: #263244 !important;
}
button.bk-btn {
  box-sizing: border-box !important;
  width: 100% !important;
  min-height: 32px !important;
  height: 32px !important;
  max-height: 32px !important;
  padding: 0 9px !important;
  border: 0 !important;
  border-radius: 6px !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  font-weight: 450 !important;
  line-height: 30px !important;
  text-align: left !important;
  justify-content: flex-start !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
button.bk-btn:hover:not(:disabled) {
  background: #edf1f5 !important;
  color: #263244 !important;
}
button.bk-btn:focus-visible {
  outline: 2px solid rgba(15, 111, 189, 0.35) !important;
  outline-offset: -2px !important;
}
button.bk-btn:disabled {
  background: transparent !important;
  color: #9aa3b2 !important;
  opacity: 1 !important;
}
"""

__all__ = [
    "TOOLBAR_CSS",
    "_TOOLBAR_BUTTON_STYLESHEET",
    "_TOOLBAR_ICON_BUTTON_STYLESHEET",
    "_TOOLBAR_MENU_STYLESHEET",
    "_TOOLBAR_POPUP_ITEM_STYLESHEET",
    "_TOOLBAR_SELECTION_STYLESHEET",
    "_TOOLBAR_SELECT_STYLESHEET",
    "_TOOLBAR_TEXT_INPUT_STYLESHEET",
]
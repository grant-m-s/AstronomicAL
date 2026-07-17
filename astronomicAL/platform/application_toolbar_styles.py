from __future__ import annotations



TOOLBAR_CSS = """
:root {
  --al-toolbar-v8: 1;
  --al-toolbar-card: #ffffff;
  --al-toolbar-page: #f3f5f8;
  --al-toolbar-border: #d8dee8;
  --al-toolbar-divider: #e2e7ee;
  --al-toolbar-text: #263244;
  --al-toolbar-muted: #687386;
  --al-toolbar-accent: #0f6fbd;
  --al-toolbar-accent-soft: rgba(15, 111, 189, 0.10);
  --al-toolbar-control: #f7f8fa;
  --al-toolbar-control-hover: #edf1f5;
}

/*
 * The ReactTemplate header sits outside #main, while #main is the scrolling
 * element. Mount the toolbar as a zero-height header root and fix the toolbar
 * immediately below the one-row application header. This avoids relying on
 * position: sticky through Panel's responsive-grid wrappers.
 */
:root {
  --al-application-header-height: 64px;
  --al-application-toolbar-height: 62px;
}

#header {
  box-sizing: border-box;
  height: var(--al-application-header-height);
  min-height: var(--al-application-header-height);
  max-height: var(--al-application-header-height);
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
  padding: 8px 12px 10px;
  border: 0;
  background: var(--al-toolbar-page);
  overflow: visible !important;
}

.al-application-toolbar {
  box-sizing: border-box;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  height: 44px;
  min-height: 44px;
  padding: 6px 10px;
  border: 1px solid var(--al-toolbar-border);
  border-radius: 11px;
  background: var(--al-toolbar-card);
  box-shadow:
    0 1px 2px rgba(15, 23, 42, 0.05),
    0 3px 10px rgba(15, 23, 42, 0.045);
  overflow: visible !important;
}

.al-application-toolbar > div,
.al-application-toolbar .bk-Row {
  min-width: 0 !important;
  max-width: 100% !important;
  overflow: visible !important;
}

.al-toolbar-group {
  display: flex;
  align-items: center;
  gap: 3px;
  min-width: 0;
  max-width: 100%;
  height: 30px;
  min-height: 30px;
  max-height: 30px;
  box-sizing: border-box;
  flex-wrap: nowrap;
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
  background: var(--al-toolbar-divider);
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
  white-space: nowrap;
  color: var(--al-toolbar-text);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  line-height: 30px;
  text-align: center;
  overflow: hidden;
}

.al-toolbar-position .al-position-current {
  font-weight: 650;
}

.al-toolbar-position .al-position-separator,
.al-toolbar-position .al-position-total {
  color: var(--al-toolbar-muted);
}

.al-toolbar-busy {
  cursor: progress;
}

/*
 * Popup menus are ordinary Panel layouts rather than Bokeh MenuButton popups.
 * This avoids the native dropdown width/inheritance problems seen inside the
 * fixed ReactTemplate header root. Each popup is independently sized and
 * right-aligned to its trigger.
 */
.al-toolbar-menu-wrapper {
  position: relative !important;
  min-width: 0 !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  overflow: visible !important;
}

.al-toolbar-popover,
.al-toolbar-popover > div,
.al-toolbar-popover .bk-Column {
  box-sizing: border-box !important;
  overflow: visible !important;
}

.al-toolbar-popover {
  position: absolute !important;
  top: 34px !important;
  right: 0 !important;
  z-index: 2400 !important;
  width: 204px !important;
  min-width: 204px !important;
  max-width: min(260px, calc(100vw - 24px)) !important;
  height: auto !important;
  min-height: 0 !important;
  max-height: min(420px, calc(100vh - 150px)) !important;
  margin: 0 !important;
  padding: 5px !important;
  border: 1px solid var(--al-toolbar-border) !important;
  border-radius: 9px !important;
  background: #ffffff !important;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18) !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
}

.al-toolbar-overflow-popover {
  width: 188px !important;
  min-width: 188px !important;
}

.al-toolbar-popover-divider {
  width: calc(100% - 12px) !important;
  height: 1px !important;
  min-height: 1px !important;
  max-height: 1px !important;
  margin: 4px 6px !important;
  padding: 0 !important;
  background: var(--al-toolbar-divider) !important;
}

@media (max-width: 1120px) {
  .al-toolbar-selection-button { display: none !important; }
  .al-toolbar-scope-group {
    flex-basis: 112px !important;
    width: 112px !important;
  }
  .al-toolbar-position {
    flex-basis: 72px !important;
    min-width: 72px !important;
    width: 72px !important;
  }
}

@media (max-width: 920px) {
  .al-toolbar-scope-group { display: none !important; }
}

@media (max-width: 680px) {
  .al-application-toolbar-shell {
    padding-left: 8px;
    padding-right: 8px;
  }
  .al-toolbar-position { display: none !important; }
  .al-toolbar-search { min-width: 80px !important; }
  .al-toolbar-menu:not(.al-toolbar-icon-only) { display: none !important; }
  .al-toolbar-workspace-group {
    flex-basis: 35px !important;
    width: 35px !important;
  }
}
"""


_TOOLBAR_BUTTON_STYLESHEET = """
:host {
  box-sizing: border-box;
  height: 30px;
  min-height: 30px;
  max-height: 30px;
  overflow: visible;
}
button.bk-btn {
  box-sizing: border-box !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  border: 0 !important;
  border-radius: 6px !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  line-height: 28px !important;
}
button.bk-btn:hover:not(:disabled) {
  background: #edf1f5 !important;
}
button.bk-btn:focus-visible {
  outline: 2px solid rgba(15, 111, 189, 0.36) !important;
  outline-offset: -2px !important;
}
button.bk-btn:disabled {
  opacity: 0.42 !important;
}
"""


_TOOLBAR_ICON_BUTTON_STYLESHEET = _TOOLBAR_BUTTON_STYLESHEET + """
:host {
  width: 30px !important;
  min-width: 30px !important;
  max-width: 30px !important;
}
button.bk-btn {
  width: 30px !important;
  min-width: 30px !important;
  max-width: 30px !important;
  padding: 0 !important;
}
"""



_TOOLBAR_MENU_STYLESHEET = _TOOLBAR_BUTTON_STYLESHEET + """
button.bk-btn {
  padding: 0 9px !important;
  white-space: nowrap !important;
}
"""

_TOOLBAR_SELECTION_STYLESHEET = _TOOLBAR_BUTTON_STYLESHEET + """
button.bk-btn {
  border: 1px solid transparent !important;
  padding: 0 9px !important;
}
:host(.al-is-selected) button.bk-btn {
  border-color: rgba(15, 111, 189, 0.18) !important;
  color: #0f6fbd !important;
  background: rgba(15, 111, 189, 0.10) !important;
}
"""

_TOOLBAR_TEXT_INPUT_STYLESHEET = """
:host {
  box-sizing: border-box;
  height: 30px;
  min-height: 30px;
  max-height: 30px;
  overflow: visible;
}
input.bk-input {
  box-sizing: border-box !important;
  width: 100% !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  margin: 0 !important;
  padding: 0 9px !important;
  border: 1px solid #d8dee8 !important;
  border-radius: 6px !important;
  background: #f7f8fa !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  line-height: 28px !important;
}
input.bk-input:hover:not(:disabled) {
  border-color: #c6ced9 !important;
  background: #ffffff !important;
}
input.bk-input:focus-visible {
  outline: 2px solid rgba(15, 111, 189, 0.36) !important;
  outline-offset: -2px !important;
}
input.bk-input:disabled {
  opacity: 0.42 !important;
}
input.bk-input::placeholder {
  color: #8b95a5;
}
"""


_TOOLBAR_SELECT_STYLESHEET = """
:host {
  box-sizing: border-box;
  height: 30px;
  min-height: 30px;
  max-height: 30px;
  overflow: visible;
}
select.bk-input {
  box-sizing: border-box !important;
  width: 100% !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  margin: 0 !important;
  padding: 0 24px 0 9px !important;
  border: 1px solid #d8dee8 !important;
  border-radius: 6px !important;
  background-color: #f7f8fa !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  line-height: 28px !important;
}
select.bk-input:hover:not(:disabled) {
  border-color: #c6ced9 !important;
  background-color: #ffffff !important;
}
select.bk-input:focus-visible {
  outline: 2px solid rgba(15, 111, 189, 0.36) !important;
  outline-offset: -2px !important;
}
select.bk-input:disabled {
  opacity: 0.42 !important;
}
"""


_TOOLBAR_POPUP_ITEM_STYLESHEET = """
:host {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  height: 32px;
  min-height: 32px;
  max-height: 32px;
  margin: 0;
  overflow: visible;
}
button.bk-btn {
  box-sizing: border-box !important;
  width: 100% !important;
  min-width: 0 !important;
  height: 32px !important;
  min-height: 32px !important;
  max-height: 32px !important;
  margin: 0 !important;
  padding: 0 10px !important;
  border: 0 !important;
  border-radius: 6px !important;
  background: #ffffff !important;
  box-shadow: none !important;
  color: #263244 !important;
  font-size: 12px !important;
  font-weight: 400 !important;
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
  outline: 2px solid rgba(15, 111, 189, 0.36) !important;
  outline-offset: -2px !important;
}
button.bk-btn:disabled {
  color: #9aa3b2 !important;
  background: #ffffff !important;
  opacity: 1 !important;
}
"""

__all__ = [
    "TOOLBAR_CSS",
    "_TOOLBAR_BUTTON_STYLESHEET",
    "_TOOLBAR_ICON_BUTTON_STYLESHEET",
    "_TOOLBAR_MENU_STYLESHEET",
    "_TOOLBAR_SELECTION_STYLESHEET",
    "_TOOLBAR_TEXT_INPUT_STYLESHEET",
    "_TOOLBAR_SELECT_STYLESHEET",
    "_TOOLBAR_POPUP_ITEM_STYLESHEET",
]
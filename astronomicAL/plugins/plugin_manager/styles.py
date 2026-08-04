PLUGIN_MANAGER_CSS = r"""
:host,
.al-plugin-manager {
  --pm-ink: #152238;
  --pm-muted: #667085;
  --pm-border: #dfe5ee;
  --pm-surface: #ffffff;
  --pm-subtle: #f8fafc;
  --pm-navy: #101a2d;
  --pm-blue: #2878c7;
  --pm-cyan: #4cc9d9;
  --pm-success: #16845b;
  --pm-warning: #ad6508;
  --pm-danger: #bd3548;
  color: var(--pm-ink);
  background: transparent !important;
}

.al-plugin-manager {
  width: 100%;
  min-width: 0;
}

.al-pm-header {
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: stretch;
  flex-direction: column;
  gap: 9px;
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  padding: 13px 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 9px;
  color: #f8fafc;
  background:
    radial-gradient(circle at 100% -30%, rgba(76, 201, 217, 0.30), transparent 45%),
    linear-gradient(112deg, #101a2d 0%, #172845 70%, #1d3658 100%);
  box-shadow: 0 5px 16px rgba(14, 28, 50, 0.14);
}

.al-pm-header-main,
.al-pm-header-meta {
  position: relative;
  z-index: 1;
  min-width: 0;
}

.al-pm-eyebrow {
  margin-bottom: 3px;
  color: #8edce6;
  font-size: 9px;
  font-weight: 750;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.al-pm-title {
  margin: 0;
  font-size: 19px;
  font-weight: 700;
  line-height: 1.1;
}

.al-pm-subtitle {
  max-width: 38rem;
  margin-top: 5px;
  color: #c7d2e3;
  font-size: 11px;
  line-height: 1.35;
}

.al-pm-header-meta {
  display: flex;
  align-items: flex-start;
  flex-direction: column;
  gap: 5px;
}

.al-pm-health {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  font-size: 10px;
  font-weight: 700;
}

.al-pm-health-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #54d7b0;
  box-shadow: 0 0 0 3px rgba(84, 215, 176, 0.13);
}

.al-pm-health.warning .al-pm-health-dot {
  background: #f3b45b;
  box-shadow: 0 0 0 3px rgba(243, 180, 91, 0.13);
}

.al-pm-health.danger .al-pm-health-dot {
  background: #ff7f91;
  box-shadow: 0 0 0 3px rgba(255, 127, 145, 0.13);
}

.al-pm-updated {
  color: #aebed3;
  font-size: 9px;
}

.al-pm-card {
  min-width: 0;
  box-sizing: border-box;
  border: 1px solid var(--pm-border);
  border-radius: 9px;
  background: var(--pm-surface);
  box-shadow: 0 2px 8px rgba(27, 43, 65, 0.045);
}

.al-pm-summary-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
  margin: 0 0 10px;
}

.al-pm-metric {
  min-width: 0;
  padding: 10px 11px;
  border: 1px solid var(--pm-border);
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 2px 8px rgba(27, 43, 65, 0.035);
}

.al-pm-metric-label {
  color: var(--pm-muted);
  font-size: 9px;
  font-weight: 750;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.al-pm-metric-value {
  margin-top: 5px;
  color: var(--pm-ink);
  font-size: 18px;
  font-weight: 750;
  line-height: 1.05;
  overflow-wrap: anywhere;
}

.al-pm-metric-detail {
  margin-top: 5px;
  color: var(--pm-muted);
  font-size: 9px;
  line-height: 1.3;
}

.al-pm-metric-value.success { color: var(--pm-success); }
.al-pm-metric-value.warning { color: var(--pm-warning); }
.al-pm-metric-value.danger { color: var(--pm-danger); }

.al-pm-section-title {
  display: flex;
  align-items: flex-start;
  flex-direction: column;
  gap: 3px;
  margin-bottom: 9px;
}

.al-pm-section-title h3 {
  margin: 0;
  color: var(--pm-ink);
  font-size: 13px;
  font-weight: 700;
  line-height: 1.25;
}

.al-pm-section-title span {
  color: var(--pm-muted);
  font-size: 9px;
  line-height: 1.3;
}

.al-pm-banner {
  box-sizing: border-box;
  width: 100%;
  padding: 9px 10px;
  border: 1px solid var(--pm-border);
  border-left-width: 3px;
  border-radius: 7px;
  color: var(--pm-ink);
  background: #fff;
  font-size: 10px;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.al-pm-banner.info { border-left-color: var(--pm-blue); }
.al-pm-banner.success { border-left-color: var(--pm-success); }
.al-pm-banner.warning { border-left-color: var(--pm-warning); }
.al-pm-banner.danger { border-left-color: var(--pm-danger); }

.al-pm-empty {
  box-sizing: border-box;
  width: 100%;
  padding: 13px;
  border: 1px dashed #ccd5e2;
  border-radius: 7px;
  color: var(--pm-muted);
  background: #fff;
  font-size: 11px;
  line-height: 1.4;
  text-align: center;
}

.al-pm-plugin-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 9px;
}

.al-pm-plugin-name {
  margin: 0;
  color: var(--pm-ink);
  font-size: 17px;
  font-weight: 750;
  line-height: 1.15;
  overflow-wrap: anywhere;
}

.al-pm-plugin-version {
  margin-top: 3px;
  color: var(--pm-muted);
  font-size: 9px;
}

.al-pm-status-pill {
  display: inline-flex;
  align-items: center;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 9px;
  font-weight: 750;
  white-space: nowrap;
}

.al-pm-status-pill.success { color: #0e6b49; background: #e8f7f1; }
.al-pm-status-pill.info { color: #1d5f9f; background: #eaf3fb; }
.al-pm-status-pill.muted { color: #596579; background: #eef1f5; }
.al-pm-status-pill.danger { color: #a9273a; background: #fdecef; }

.al-pm-description {
  margin: 0 0 10px;
  color: #445064;
  font-size: 11px;
  line-height: 1.45;
}

.al-pm-mini-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 7px;
  margin: 0 0 11px;
}

.al-pm-mini {
  min-width: 0;
  padding: 8px 9px;
  border: 1px solid var(--pm-border);
  border-radius: 7px;
  background: var(--pm-subtle);
}

.al-pm-mini-label {
  color: var(--pm-muted);
  font-size: 8px;
  font-weight: 750;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.al-pm-mini-value {
  margin-top: 4px;
  color: var(--pm-ink);
  font-size: 12px;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.al-pm-subheading {
  margin: 13px 0 7px;
  color: var(--pm-ink);
  font-size: 11px;
  font-weight: 750;
}

.al-pm-chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.al-pm-chip {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  padding: 4px 7px;
  border: 1px solid #d9e1ec;
  border-radius: 999px;
  color: #32445d;
  background: #f7f9fc;
  font-size: 9px;
  line-height: 1.25;
  overflow-wrap: anywhere;
}

.al-pm-chip strong {
  margin-right: 4px;
  color: var(--pm-blue);
  font-weight: 750;
}

.al-pm-instance,
.al-pm-issue {
  box-sizing: border-box;
  width: 100%;
  padding: 8px 9px;
  border: 1px solid var(--pm-border);
  border-radius: 7px;
  background: #fff;
}

.al-pm-instance + .al-pm-instance,
.al-pm-issue + .al-pm-issue {
  margin-top: 6px;
}

.al-pm-instance-title,
.al-pm-issue-title {
  color: var(--pm-ink);
  font-size: 10px;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.al-pm-instance-meta,
.al-pm-issue-meta {
  margin-top: 3px;
  color: var(--pm-muted);
  font-size: 9px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.al-pm-issue {
  border-left: 3px solid var(--pm-danger);
}

.al-pm-readiness {
  margin: 9px 0;
}

.al-pm-readiness ul {
  margin: 5px 0 0 18px;
  padding: 0;
}

.al-pm-technical {
  margin-top: 12px;
  border-top: 1px solid var(--pm-border);
  padding-top: 9px;
}

.al-pm-technical summary {
  color: var(--pm-blue);
  cursor: pointer;
  font-size: 10px;
  font-weight: 700;
}

.al-pm-technical-grid {
  display: grid;
  grid-template-columns: minmax(88px, auto) 1fr;
  gap: 5px 9px;
  margin-top: 9px;
  font-size: 9px;
  line-height: 1.4;
}

.al-pm-technical-key {
  color: var(--pm-muted);
  font-weight: 700;
}

.al-pm-technical-value {
  min-width: 0;
  color: #344054;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.al-pm-table .tabulator {
  border: 0 !important;
  background: #fff !important;
  font-size: 10px !important;
}

.al-pm-table .tabulator-header {
  border-bottom: 1px solid var(--pm-border) !important;
  background: var(--pm-subtle) !important;
  color: #445064 !important;
  font-size: 9px !important;
  font-weight: 750 !important;
}

.al-pm-table .tabulator-row {
  min-height: 34px !important;
  border-bottom: 1px solid #edf1f5 !important;
  background: #fff !important;
}

.al-pm-table .tabulator-row:hover {
  background: #f5f8fc !important;
}

.al-pm-table .tabulator-row.tabulator-selected {
  background: #eaf3fb !important;
}

.al-pm-table .tabulator-cell {
  padding: 8px 7px !important;
  border-right: 0 !important;
  color: #344054 !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

@media (min-width: 480px) {
  .al-pm-summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .al-pm-header { flex-direction: row; justify-content: space-between; }
  .al-pm-header-meta { align-items: flex-end; }
  .al-pm-updated { text-align: right; }
}

@media (min-width: 760px) {
  .al-pm-summary-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
}
"""


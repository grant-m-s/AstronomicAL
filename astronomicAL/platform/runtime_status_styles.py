RUNTIME_STATUS_CSS = r"""
.al-runtime-status-box,
.al-runtime-status-modal-card,
.al-runtime-diagnostics {
  --rd-ink: #152238;
  --rd-muted: #667085;
  --rd-border: #dfe5ee;
  --rd-surface: #ffffff;
  --rd-subtle: #f8fafc;
  --rd-navy: #101a2d;
  --rd-blue: #2878c7;
  --rd-cyan: #4cc9d9;
  --rd-success: #16845b;
  --rd-success-soft: #e8f7f1;
  --rd-warning: #ad6508;
  --rd-warning-soft: #fff4df;
  --rd-danger: #bd3548;
  --rd-danger-soft: #fdecef;
  --rd-info: #2769a8;
  --rd-info-soft: #eaf3fb;
  box-sizing: border-box;
  color: var(--rd-ink);
}

.al-runtime-status-box,
.al-runtime-status-box * {
  box-sizing: border-box;
}

.al-runtime-status-pill {
  display: flex;
  align-items: center;
  gap: 7px;
  width: 100%;
  min-width: 0;
  height: 28px;
  padding: 0 10px;
  border: 0;
  background: transparent;
  color: var(--rd-ink);
  font-size: 11px;
  font-weight: 650;
  line-height: 28px;
  white-space: nowrap;
  overflow: hidden;
}

.al-runtime-status-pill .al-rd-status-dot {
  display: inline-block;
  flex: 0 0 auto;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--rd-success);
  box-shadow: 0 0 0 3px rgba(22, 132, 91, 0.11);
}

.al-runtime-status-pill .al-rd-status-label {
  flex: 0 0 auto;
  font-weight: 750;
}

.al-runtime-status-pill .al-rd-status-detail {
  min-width: 0;
  color: var(--rd-muted);
  font-weight: 550;
  overflow: hidden;
  text-overflow: ellipsis;
}

.al-runtime-status-pill.success .al-rd-status-dot {
  background: var(--rd-success);
  box-shadow: 0 0 0 3px rgba(22, 132, 91, 0.11);
}

.al-runtime-status-pill.info .al-rd-status-dot {
  background: var(--rd-blue);
  box-shadow: 0 0 0 3px rgba(40, 120, 199, 0.11);
}

.al-runtime-status-pill.warning .al-rd-status-dot {
  background: var(--rd-warning);
  box-shadow: 0 0 0 3px rgba(173, 101, 8, 0.13);
}

.al-runtime-status-pill.danger .al-rd-status-dot {
  background: var(--rd-danger);
  box-shadow: 0 0 0 3px rgba(189, 53, 72, 0.12);
}

.al-runtime-status-modal-card {
  width: min(1080px, calc(100vw - 48px)) !important;
  height: min(780px, calc(100vh - 70px)) !important;
  min-width: 0;
  min-height: 480px;
  padding: 0;
  overflow: hidden;
  border: 1px solid rgba(15, 23, 42, 0.18);
  border-radius: 14px;
  background: #eef2f6;
  box-shadow: 0 16px 48px rgba(15, 23, 42, 0.22);
}

.al-runtime-diagnostics {
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  background: #eef2f6;
}

.al-rd-header {
  position: relative;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  box-sizing: border-box;
  min-width: 0;
  padding: 18px 20px;
  color: #f8fafc;
  background:
    radial-gradient(circle at 100% -40%, rgba(76, 201, 217, 0.30), transparent 44%),
    linear-gradient(112deg, #101a2d 0%, #172845 68%, #1d3658 100%);
}

.al-rd-header-main,
.al-rd-header-meta {
  min-width: 0;
}

.al-rd-eyebrow {
  margin-bottom: 4px;
  color: #8edce6;
  font-size: 10px;
  font-weight: 750;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.al-rd-title {
  margin: 0;
  color: #ffffff;
  font-size: 21px;
  font-weight: 720;
  line-height: 1.15;
}

.al-rd-subtitle {
  max-width: 680px;
  margin-top: 6px;
  color: #c7d2e3;
  font-size: 11px;
  line-height: 1.45;
}

.al-rd-header-meta {
  display: flex;
  align-items: flex-end;
  flex-direction: column;
  gap: 6px;
  text-align: right;
}

.al-rd-live {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  color: #f8fafc;
  font-size: 10px;
  font-weight: 700;
}

.al-rd-live-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #54d7b0;
  box-shadow: 0 0 0 3px rgba(84, 215, 176, 0.13);
}

.al-rd-updated,
.al-rd-header-status {
  color: #aebed3;
  font-size: 10px;
}

.al-rd-header-status.warning { color: #ffd89d; }
.al-rd-header-status.danger { color: #ffd3d9; }

.al-rd-scroll {
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  min-width: 0;
  padding: 14px;
  overflow-x: hidden;
  overflow-y: auto;
  background: #eef2f6;
}

.al-rd-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}

.al-rd-metric,
.al-rd-card {
  box-sizing: border-box;
  min-width: 0;
  border: 1px solid var(--rd-border);
  border-radius: 10px;
  background: var(--rd-surface);
  box-shadow: 0 2px 8px rgba(27, 43, 65, 0.045);
}

.al-rd-metric {
  padding: 12px;
}

.al-rd-metric-label {
  color: var(--rd-muted);
  font-size: 9px;
  font-weight: 750;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.al-rd-metric-value {
  margin-top: 5px;
  color: var(--rd-ink);
  font-size: 19px;
  font-weight: 760;
  line-height: 1.1;
  overflow-wrap: anywhere;
}

.al-rd-metric-value.success { color: var(--rd-success); }
.al-rd-metric-value.info { color: var(--rd-info); }
.al-rd-metric-value.warning { color: var(--rd-warning); }
.al-rd-metric-value.danger { color: var(--rd-danger); }

.al-rd-metric-detail {
  margin-top: 5px;
  color: var(--rd-muted);
  font-size: 9px;
  line-height: 1.35;
}

.al-rd-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
  gap: 12px;
  align-items: start;
}

.al-rd-card {
  margin-bottom: 12px;
  padding: 13px;
}

.al-rd-card:last-child { margin-bottom: 0; }

.al-rd-section-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.al-rd-section-heading h3 {
  margin: 0;
  color: var(--rd-ink);
  font-size: 13px;
  font-weight: 720;
  line-height: 1.25;
}

.al-rd-section-heading p {
  margin: 3px 0 0;
  color: var(--rd-muted);
  font-size: 9px;
  line-height: 1.35;
}

.al-rd-section-count {
  flex: 0 0 auto;
  padding: 2px 7px;
  border-radius: 999px;
  color: var(--rd-muted);
  background: #f0f3f7;
  font-size: 9px;
  font-weight: 700;
}

.al-rd-stack {
  display: flex;
  flex-direction: column;
  gap: 7px;
  width: 100%;
  min-width: 0;
}

.al-rd-empty {
  box-sizing: border-box;
  width: 100%;
  padding: 14px;
  border: 1px dashed #ccd5e2;
  border-radius: 8px;
  color: var(--rd-muted);
  background: #fbfcfe;
  font-size: 10px;
  line-height: 1.4;
  text-align: center;
}

.al-rd-empty.success {
  border-color: #b9ddcf;
  color: #326e58;
  background: #f4fbf8;
}

.al-rd-signal,
.al-rd-row {
  box-sizing: border-box;
  min-width: 0;
  padding: 9px 10px;
  border: 1px solid var(--rd-border);
  border-radius: 8px;
  background: #ffffff;
}

.al-rd-signal {
  border-left-width: 3px;
}

.al-rd-signal.info { border-left-color: var(--rd-blue); }
.al-rd-signal.warning { border-left-color: var(--rd-warning); }
.al-rd-signal.danger { border-left-color: var(--rd-danger); }

.al-rd-row-top,
.al-rd-signal-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  min-width: 0;
}

.al-rd-row-title,
.al-rd-signal-title {
  min-width: 0;
  color: var(--rd-ink);
  font-size: 11px;
  font-weight: 700;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.al-rd-row-value {
  flex: 0 0 auto;
  color: var(--rd-ink);
  font-size: 10px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.al-rd-row-meta,
.al-rd-signal-meta,
.al-rd-row-detail {
  margin-top: 4px;
  color: var(--rd-muted);
  font-size: 9px;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.al-rd-row-detail {
  padding-top: 5px;
  border-top: 1px solid #edf0f5;
}

.al-rd-badge {
  flex: 0 0 auto;
  padding: 2px 6px;
  border-radius: 999px;
  font-size: 8px;
  font-weight: 750;
  text-transform: uppercase;
}

.al-rd-badge.success { color: #0e6847; background: var(--rd-success-soft); }
.al-rd-badge.info { color: #205b91; background: var(--rd-info-soft); }
.al-rd-badge.warning { color: #8a5005; background: var(--rd-warning-soft); }
.al-rd-badge.danger { color: #9f2638; background: var(--rd-danger-soft); }

.al-rd-progress {
  height: 5px;
  margin-top: 7px;
  overflow: hidden;
  border-radius: 999px;
  background: #e9eef6;
}

.al-rd-progress > span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--rd-blue), var(--rd-cyan));
}

.al-rd-code {
  color: #183c66;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 9px;
  font-weight: 650;
  overflow-wrap: anywhere;
}

.al-rd-trace {
  margin-top: 6px;
}

.al-rd-trace summary {
  color: var(--rd-blue);
  cursor: pointer;
  font-size: 9px;
  font-weight: 650;
}

.al-rd-trace pre {
  margin: 6px 0 0;
  padding: 8px;
  overflow: auto;
  border-radius: 6px;
  color: #d8e2f1;
  background: #111827;
  font-size: 9px;
  line-height: 1.35;
  white-space: pre-wrap;
}

.al-rd-footnote {
  margin-top: 8px;
  color: var(--rd-muted);
  font-size: 9px;
  line-height: 1.4;
}

.al-rd-footer {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  padding: 8px 12px;
  border-top: 1px solid #d8dee8;
  background: #f7f9fb;
}

.al-rd-footer-status {
  color: var(--rd-muted);
  font-size: 10px;
  line-height: 32px;
}

@media (max-width: 900px) {
  .al-runtime-status-modal-card {
    width: calc(100vw - 24px) !important;
    height: calc(100vh - 36px) !important;
  }

  .al-rd-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .al-rd-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 560px) {
  .al-rd-header {
    flex-direction: column;
    padding: 14px;
  }

  .al-rd-header-meta {
    align-items: flex-start;
    text-align: left;
  }

  .al-rd-scroll { padding: 9px; }
  .al-rd-metrics { grid-template-columns: 1fr; }
  .al-rd-section-heading { flex-direction: column; }
}
"""


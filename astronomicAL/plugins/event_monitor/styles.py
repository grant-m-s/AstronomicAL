EVENT_MONITOR_CSS = r"""
:host,
.al-event-monitor {
  --em-ink: #152238;
  --em-muted: #667085;
  --em-border: #dfe5ee;
  --em-surface: #ffffff;
  --em-subtle: #f8fafc;
  --em-navy: #101a2d;
  --em-blue: #2878c7;
  --em-cyan: #4cc9d9;
  --em-success: #16845b;
  --em-warning: #ad6508;
  --em-danger: #bd3548;
  color: var(--em-ink);
  background: transparent !important;
}

.al-event-monitor {
  width: 100%;
  min-width: 0;
}

.al-em-header {
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
    radial-gradient(circle at 100% -30%, rgba(76, 201, 217, 0.3), transparent 45%),
    linear-gradient(112deg, #101a2d 0%, #172845 70%, #1d3658 100%);
  box-shadow: 0 5px 16px rgba(14, 28, 50, 0.14);
}

.al-em-header-main,
.al-em-header-meta {
  position: relative;
  z-index: 1;
  min-width: 0;
}

.al-em-eyebrow {
  margin-bottom: 3px;
  color: #8edce6;
  font-size: 9px;
  font-weight: 750;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.al-em-title {
  margin: 0;
  font-size: 19px;
  font-weight: 700;
  line-height: 1.1;
}

.al-em-subtitle {
  max-width: 34rem;
  margin-top: 5px;
  color: #c7d2e3;
  font-size: 11px;
  line-height: 1.35;
}

.al-em-header-meta {
  display: flex;
  align-items: flex-start;
  flex-direction: column;
  gap: 5px;
}

.al-em-live {
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

.al-em-live-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #54d7b0;
  box-shadow: 0 0 0 3px rgba(84, 215, 176, 0.13);
}

.al-em-live.is-paused .al-em-live-dot {
  background: #f3b45b;
  box-shadow: 0 0 0 3px rgba(243, 180, 91, 0.13);
}

.al-em-updated,
.al-em-header-warning {
  color: #aebed3;
  font-size: 9px;
  text-align: left;
}

.al-em-header-warning { color: #ffd3d9; }

.al-em-card {
  min-width: 0;
  box-sizing: border-box;
  border: 1px solid var(--em-border);
  border-radius: 9px;
  background: var(--em-surface);
  box-shadow: 0 2px 8px rgba(27, 43, 65, 0.045);
}

.al-em-controls {
  gap: 8px !important;
}

.al-em-summary-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
  margin: 0 0 10px;
}

.al-em-metric {
  min-width: 0;
  padding: 10px 11px;
  border: 1px solid var(--em-border);
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 2px 8px rgba(27, 43, 65, 0.035);
}

.al-em-metric-label {
  color: var(--em-muted);
  font-size: 9px;
  font-weight: 750;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.al-em-metric-value {
  margin-top: 5px;
  color: var(--em-ink);
  font-size: 18px;
  font-weight: 750;
  line-height: 1.05;
  overflow-wrap: anywhere;
}

.al-em-metric-detail {
  margin-top: 5px;
  color: var(--em-muted);
  font-size: 9px;
  line-height: 1.3;
}

.al-em-metric-value.success { color: var(--em-success); }
.al-em-metric-value.warning { color: var(--em-warning); }
.al-em-metric-value.danger { color: var(--em-danger); }

.al-em-section-title {
  display: flex;
  align-items: flex-start;
  flex-direction: column;
  gap: 3px;
  margin-bottom: 9px;
}

.al-em-section-title h3 {
  margin: 0;
  color: var(--em-ink);
  font-size: 13px;
  font-weight: 700;
  line-height: 1.25;
}

.al-em-section-title span {
  max-width: none;
  color: var(--em-muted);
  font-size: 9px;
  line-height: 1.3;
  text-align: left;
}

.al-em-stack {
  display: flex;
  flex-direction: column;
  gap: 7px;
  width: 100%;
  min-width: 0;
}

.al-em-empty {
  box-sizing: border-box;
  width: 100%;
  padding: 12px;
  border: 1px dashed #ccd5e2;
  border-radius: 7px;
  color: var(--em-muted);
  background: #fff;
  font-size: 11px;
  line-height: 1.4;
  text-align: center;
}

.al-em-issue,
.al-em-event,
.al-em-subscription,
.al-em-topic-row {
  min-width: 0;
  box-sizing: border-box;
  padding: 9px 10px;
  border: 1px solid var(--em-border);
  border-radius: 7px;
  background: #fff;
}

.al-em-issue { border-left-width: 3px; }
.al-em-issue.danger { border-left-color: var(--em-danger); }
.al-em-issue.warning { border-left-color: var(--em-warning); }
.al-em-issue.info { border-left-color: var(--em-blue); }

.al-em-issue-title {
  color: var(--em-ink);
  font-size: 11px;
  font-weight: 700;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.al-em-issue-meta {
  margin-top: 3px;
  color: var(--em-muted);
  font-size: 9px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

details.al-em-trace { margin-top: 6px; }
details.al-em-trace summary {
  color: var(--em-blue);
  cursor: pointer;
  font-size: 9px;
  font-weight: 650;
}

details.al-em-trace pre {
  margin: 6px 0 0;
  padding: 8px;
  overflow: visible;
  border-radius: 6px;
  color: #d8e2f1;
  background: #111827;
  font-size: 9px;
  line-height: 1.35;
  white-space: pre-wrap;
}

.al-em-event-top,
.al-em-topic-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
}

code {
  min-width: 0;
  color: #183c66;
  background: transparent;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 10px;
  font-weight: 650;
  line-height: 1.35;
  overflow-wrap: anywhere;
  white-space: normal;
}

.al-em-badge {
  flex: 0 0 auto;
  padding: 2px 6px;
  border-radius: 999px;
  font-size: 8px;
  font-weight: 750;
  text-transform: uppercase;
}

.al-em-badge.success { color: #0e6847; background: #e8f7f1; }
.al-em-badge.warning { color: #8a5005; background: #fff4df; }
.al-em-badge.danger { color: #9f2638; background: #fdecef; }

.al-em-event-meta,
.al-em-event-payload,
.al-em-subscription-meta,
.al-em-subscription-module,
.al-em-topic-stat {
  margin-top: 4px;
  color: var(--em-muted);
  font-size: 9px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.al-em-event-payload {
  padding-top: 5px;
  border-top: 1px solid #edf0f5;
}

.al-em-subscription-owner {
  margin-top: 5px;
  color: var(--em-ink);
  font-size: 11px;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.al-em-subscription-module {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.al-em-topic-top span {
  flex: 0 0 auto;
  color: var(--em-muted);
  font-size: 9px;
}

.al-em-topic-bar {
  height: 5px;
  margin-top: 7px;
  overflow: hidden;
  border-radius: 999px;
  background: #e9eef6;
}

.al-em-topic-bar > span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--em-blue), var(--em-cyan));
}

@media (min-width: 380px) {
  .al-em-summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
"""
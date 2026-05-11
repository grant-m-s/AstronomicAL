from __future__ import annotations

PLUGIN_ID = "core.visualisation"
STATE_SERVICE_KEY = f"{PLUGIN_ID}.state"

PLOT_MIN_HEIGHT = 170

# Fixed settings height. Do not couple this to the outer panel height.
# The settings content itself should scroll if it overflows.
SETTINGS_HEIGHT = 156

DEFAULT_DATASHADE_THRESHOLD = 50_000
DEFAULT_INTERACTIVE_SAMPLE_LIMIT = 50_000
DEFAULT_MAX_SELECTION_IDS = 100_000

INTERNAL_X = "__x__"
INTERNAL_Y = "__y__"
INTERNAL_ROW_ID = "__row_id__"
INTERNAL_LABEL_RAW = "__label_raw__"
INTERNAL_LABEL_DISPLAY = "__label_display__"
INTERNAL_LABEL_COLOUR = "__label_colour__"
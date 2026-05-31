import param
from panel.custom import Children, ReactComponent

class DynamicReactGrid(ReactComponent):

    objects = Children()
    keys = param.List(default=[])
    titles = param.Dict(default={})
    layouts = param.Dict(default={})

    breakpoints = param.Dict(default={"lg": 1500, "md": 1050, "sm": 0})
    cols_by_breakpoint = param.Dict(default={"lg": 12, "md": 12, "sm": 12})

    close_key = param.String(default="")
    close_click_count = param.Integer(default=0)

    current_breakpoint = param.String(default="lg")

    current_layout = param.List(default=[])

    row_height = param.Integer(default=80)

    margin = param.List(default=[10, 10])

    compact_type = param.ObjectSelector(
        default=None,
        objects=[None, "vertical", "horizontal"],
    )

    prevent_collision = param.Boolean(default=False)
    resize_handles = param.List(
        default=["s", "w", "e", "n", "sw", "nw", "se", "ne"]
    )

    _stylesheets = [
        "https://unpkg.com/react-grid-layout/css/styles.css",
        "https://unpkg.com/react-resizable/css/styles.css",
        """
        .pn-dynamic-rgl { width: 100%; height: 100%; min-height: 600px; }
        .tile { border: 1px solid rgba(0,0,0,0.15); border-radius: 8px; overflow: hidden; height: 100%; display:flex; flex-direction:column; }
        .tile-header { padding: 6px 10px; font-size: 12px; user-select:none; cursor:grab; border-bottom:1px solid rgba(0,0,0,0.10); background:rgba(0,0,0,0.06); display:flex; align-items:center; justify-content:space-between; gap:8px; }
        .tile-title { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .tile-close { border:none; background:transparent; cursor:pointer; font-size:16px; line-height:1; padding:2px 6px; border-radius:6px; }
        .tile-close:hover { background: rgba(0,0,0,0.10); }
        .tile-body { padding: 8px; overflow:auto; flex:1; min-height:0; }
        """,
    ]

    _importmap = {
        "imports": {
            "react": "https://esm.sh/react@18.2.0?dev",
            "react-dom": "https://esm.sh/react-dom@18.2.0?dev&deps=react@18.2.0",
            "prop-types": "https://esm.sh/prop-types@15.8.1",
            "react-draggable": (
                "https://esm.sh/react-draggable@4.4.6"
                "?dev&deps=react@18.2.0,react-dom@18.2.0"
            ),
            "react-resizable": (
                "https://esm.sh/react-resizable@3.0.5"
                "?dev&deps=react@18.2.0,react-dom@18.2.0,react-draggable@4.4.6"
            ),
            "clsx": "https://unpkg.com/clsx@2.1.1/dist/clsx.mjs",
            "react-grid-layout": (
                "https://esm.sh/react-grid-layout@1.4.4"
                "?dev"
                "&deps=react@18.2.0,react-dom@18.2.0,"
                "react-draggable@4.4.6,react-resizable@3.0.5,prop-types@15.8.1"
                "&external=react,react-dom,clsx,prop-types,react-draggable,react-resizable"
            ),
        }
    }

    _esm = r"""
import * as RGLib from "react-grid-layout";

// React is already provided by Panel; DO NOT import it again.

// Robust CJS/ESM interop: try common locations.
const Root = RGLib?.default ?? RGLib;
const Responsive = RGLib?.Responsive ?? Root?.Responsive;
const WidthProvider = RGLib?.WidthProvider ?? Root?.WidthProvider;

if (!Responsive || !WidthProvider) {
  console.error("react-grid-layout module keys:", Object.keys(RGLib || {}));
  console.error("react-grid-layout default keys:", Object.keys(RGLib?.default || {}));
  throw new Error("Could not resolve Responsive/WidthProvider from react-grid-layout (CJS/ESM interop).");
}

const ResponsiveRGL = WidthProvider(Responsive);

function numberOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function integerOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.trunc(n) : fallback;
}

function safeString(value, fallback = "") {
  if (value === undefined || value === null) {
    return fallback;
  }
  return String(value);
}

function sameJSON(a, b) {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

function bottomY(layout) {
  let bottom = 0;

  for (const item of layout || []) {
    bottom = Math.max(
      bottom,
      integerOr(item?.y, 0) + integerOr(item?.h, 1)
    );
  }

  return bottom;
}

function sanitizeLayoutItem(item, keys, cols) {
  const rawId = item?.i;

  if (rawId === undefined || rawId === null) {
    return null;
  }

  const id = String(rawId);
  const keySet = new Set((keys || []).map((key) => String(key)));

  if (!keySet.has(id)) {
    return null;
  }

  const C = Math.max(1, integerOr(cols, 12));

  let w = integerOr(item?.w, 4);
  let h = integerOr(item?.h, 4);
  let x = integerOr(item?.x, 0);
  let y = integerOr(item?.y, 0);

  w = Math.max(1, Math.min(w, C));
  h = Math.max(1, h);

  const maxX = Math.max(0, C - w);
  x = Math.max(0, Math.min(x, maxX));
  y = Math.max(0, y);

  // Important: return only JSON-safe fields. Do not spread raw RGL items,
  // because raw items may contain undefined optional fields that Bokeh cannot
  // serialize.
  const cleaned = {
    i: id,
    x,
    y,
    w,
    h,
    static: !!item?.static,
  };

  const minW = numberOr(item?.minW, NaN);
  if (Number.isFinite(minW)) {
    cleaned.minW = Math.max(1, Math.trunc(minW));
  }

  const minH = numberOr(item?.minH, NaN);
  if (Number.isFinite(minH)) {
    cleaned.minH = Math.max(1, Math.trunc(minH));
  }

  const maxW = numberOr(item?.maxW, NaN);
  if (Number.isFinite(maxW)) {
    cleaned.maxW = Math.max(1, Math.trunc(maxW));
  }

  const maxH = numberOr(item?.maxH, NaN);
  if (Number.isFinite(maxH)) {
    cleaned.maxH = Math.max(1, Math.trunc(maxH));
  }

  return cleaned;
}

function sanitizeLayout(layout, keys, cols) {
  const out = [];

  for (const item of layout || []) {
    const cleaned = sanitizeLayoutItem(item, keys, cols);

    if (cleaned) {
      out.push(cleaned);
    }
  }

  return out;
}

function defaultLayoutItem(key, existingLayout, cols) {
  const C = Math.max(1, integerOr(cols, 12));
  const w = Math.max(1, Math.min(4, C));

  return {
    i: String(key),
    x: 0,
    y: bottomY(existingLayout || []),
    w,
    h: 4,
    static: false,
  };
}

function ensureLayoutsForKeys(layouts, keys, colsByBp) {
  const safeKeys = (keys || []).map((key) => String(key));
  const colsMap = colsByBp || {};
  const source = layouts || {};

  const breakpoints = new Set([
    ...Object.keys(colsMap),
    ...Object.keys(source),
  ]);

  if (breakpoints.size === 0) {
    breakpoints.add("lg");
    breakpoints.add("md");
    breakpoints.add("sm");
  }

  const out = {};

  for (const bp of breakpoints) {
    const cols = integerOr(colsMap[bp], 12);

    const existingClean = sanitizeLayout(
      source[bp] || [],
      safeKeys,
      cols
    );

    const existingByKey = new Map(
      existingClean.map((item) => [String(item.i), item])
    );

    const full = [];

    for (const key of safeKeys) {
      const hit = existingByKey.get(String(key));

      if (hit) {
        full.push(hit);
      } else {
        full.push(defaultLayoutItem(key, full, cols));
      }
    }

    out[bp] = sanitizeLayout(full, safeKeys, cols);
  }

  return out;
}

function mergeActiveBreakpointLayout(layouts, breakpoint, currentLayout, keys, colsByBp) {
  const bp = safeString(breakpoint, "lg") || "lg";
  const safeKeys = (keys || []).map((key) => String(key));
  const colsMap = colsByBp || {};
  const cols = integerOr(colsMap[bp], 12);

  const previous = ensureLayoutsForKeys(layouts || {}, safeKeys, colsMap);
  const currentClean = sanitizeLayout(currentLayout || [], safeKeys, cols);

  return ensureLayoutsForKeys(
    {
      ...previous,
      [bp]: currentClean,
    },
    safeKeys,
    colsMap
  );
}

function stopPanelChromeEvent(event) {
  event.preventDefault();
  event.stopPropagation();
}

export function render({ model }) {

  const [keys] = model.useState("keys");
  const [titles] = model.useState("titles");
  const [layouts, setLayouts] = model.useState("layouts");

  const [breakpoints] = model.useState("breakpoints");
  const [colsByBp] = model.useState("cols_by_breakpoint");
  const [rowHeight] = model.useState("row_height");
  const [margin] = model.useState("margin");
  const [compactType] = model.useState("compact_type");
  const [preventCollision] = model.useState("prevent_collision");
  const [resizeHandles] = model.useState("resize_handles");

  const [, setCloseKey] = model.useState("close_key");
  const [closeEventCount, setCloseEventCount] = model.useState("close_click_count");

  const [currentBp, setCurrentBp] = model.useState("current_breakpoint");
  const [, setCurrentLayout] = model.useState("current_layout");

  const currentBpRef = React.useRef(safeString(currentBp, "lg") || "lg");
  const closeEventCountRef = React.useRef(integerOr(closeEventCount, 0));

  React.useEffect(() => {
    currentBpRef.current = safeString(currentBp, "lg") || "lg";
  }, [currentBp]);

  React.useEffect(() => {
    closeEventCountRef.current = integerOr(closeEventCount, closeEventCountRef.current || 0);
  }, [closeEventCount]);

  const childrenArray = React.Children.toArray(model.get_child("objects"));

  // Stability trick to avoid mismatched title/content while Panel patches
  // `keys` and `objects`.
  const incomingKeys = (keys || []).map((key) => String(key));
  const stable = React.useRef({ keys: [], children: [] });

  if (incomingKeys.length === childrenArray.length) {
    stable.current = {
      keys: incomingKeys,
      children: childrenArray,
    };
  }

  const stableKeys = stable.current.keys;
  const stableChildren = stable.current.children;

  const keySignature = stableKeys.join("|");
  const previousKeySignatureRef = React.useRef(keySignature);
  const suppressProgrammaticLayoutWriteRef = React.useRef(false);

  if (previousKeySignatureRef.current !== keySignature) {
    previousKeySignatureRef.current = keySignature;
    suppressProgrammaticLayoutWriteRef.current = true;
  }

  const contentByKey = {};

  for (let i = 0; i < stableKeys.length; i++) {
    contentByKey[String(stableKeys[i])] = stableChildren[i];
  }

  const titleByKey = {};
  for (const key of stableKeys || []) {
    const keyString = String(key);
    const title = titles?.[keyString];
    titleByKey[keyString] =
      title === undefined || title === null || String(title).trim() === ""
        ? keyString
        : String(title);
  }

  const normalizedLayouts = React.useMemo(
    () => ensureLayoutsForKeys(layouts || {}, stableKeys, colsByBp || {}),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      JSON.stringify(layouts || {}),
      keySignature,
      JSON.stringify(colsByBp || {}),
    ]
  );

  React.useEffect(() => {
    if (!sameJSON(layouts || {}, normalizedLayouts || {})) {
      setLayouts(normalizedLayouts || {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    keySignature,
    JSON.stringify(normalizedLayouts || {}),
  ]);

  const resolvedCompactType =
    compactType === undefined || compactType === null || compactType === "none"
      ? null
      : compactType;

  const safeMargin = Array.isArray(margin)
    ? margin.map((value) => integerOr(value, 10))
    : [10, 10];

  const safeResizeHandles = Array.isArray(resizeHandles) && resizeHandles.length > 0
    ? resizeHandles.map((value) => String(value))
    : ["s", "w", "e", "n", "sw", "nw", "se", "ne"];

  return (
    <div className="pn-dynamic-rgl">
      <ResponsiveRGL
        layouts={normalizedLayouts || {}}
        breakpoints={breakpoints || {}}
        cols={colsByBp || {}}
        rowHeight={integerOr(rowHeight, 80)}
        margin={safeMargin}
        compactType={resolvedCompactType}
        preventCollision={!!preventCollision}
        draggableHandle=".tile-header"
        draggableCancel=".tile-close"
        resizeHandles={safeResizeHandles}
        onBreakpointChange={(bp) => {
          const nextBp = safeString(bp, "lg") || "lg";
          currentBpRef.current = nextBp;
          setCurrentBp(nextBp);
        }}
        onLayoutChange={(currentLayout) => {
          const bp = currentBpRef.current || "lg";

          const cleanCur = sanitizeLayout(
            currentLayout || [],
            stableKeys,
            integerOr((colsByBp || {})[bp], 12)
          );

          // Do not write ReactGridLayout's generated allLayouts back wholesale.
          // It can rewrite untouched breakpoints and reflow unrelated panels.
          const cleanAll = mergeActiveBreakpointLayout(
            layouts || {},
            bp,
            cleanCur,
            stableKeys,
            colsByBp || {}
          );

          if (suppressProgrammaticLayoutWriteRef.current) {
            suppressProgrammaticLayoutWriteRef.current = false;

            // ReactGridLayout can emit a generated/default layout during programmatic
            // key/object changes. Do not accept that temporary layout as user state.
            const expectedCurrent = sanitizeLayout(
              ((normalizedLayouts || layouts || {})[bp] || []),
              stableKeys,
              integerOr((colsByBp || {})[bp], 12)
            );

            if (expectedCurrent.length > 0) {
              setCurrentLayout(expectedCurrent);
            }

            return;
          }

          setCurrentLayout(cleanCur || []);

          if (!sameJSON(layouts || {}, cleanAll || {})) {
            setLayouts(cleanAll || {});
          }
        }}
      >
        {(stableKeys || []).map((k) => (
          <div key={String(k)} className="tile">
            <div className="tile-header">
              <div className="tile-title">{titleByKey[String(k)] || String(k)}</div>
              <button
                type="button"
                className="tile-close"
                title="Close"
                onPointerDownCapture={stopPanelChromeEvent}
                onMouseDownCapture={stopPanelChromeEvent}
                onTouchStartCapture={stopPanelChromeEvent}
                onClickCapture={(event) => {
                  event.preventDefault();
                  event.stopPropagation();

                  closeEventCountRef.current += 1;

                  setCloseKey(String(k));
                  setCloseEventCount(closeEventCountRef.current);
                }}
              >
                ×
              </button>
            </div>
            <div className="tile-body">
              {contentByKey[String(k)] ?? <div>Missing content for {String(k)}</div>}
            </div>
          </div>
        ))}
      </ResponsiveRGL>
    </div>
  );
}
"""
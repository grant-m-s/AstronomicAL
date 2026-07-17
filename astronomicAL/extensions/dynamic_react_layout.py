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
        .pn-dynamic-rgl {
            width: 100%;
            height: 100%;
            min-height: 600px;
            background-color: #f3f5f8;
        }

        .pn-dynamic-rgl-bottom-spacer {
            height: 260px;
            min-height: 260px;
            pointer-events: none;
            background-color: #f3f5f8;
        }

        .tile {
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow: hidden;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            background-color: #ffffff;
        }

        .pn-dynamic-rgl .tile > .tile-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 6px 10px;
            border-bottom: 1px solid #cbd5e1;
            background: #e2e8f0 !important;
            color: #263244;
            font-size: 12px;
            cursor: grab;
            user-select: none;
        }

        .tile-title {
            flex: 1;
            overflow: hidden;
            color: #263244;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .tile-close {
            padding: 2px 6px;
            border: none;
            border-radius: 6px;
            background: transparent;
            color: #263244;
            font-size: 16px;
            line-height: 1;
            cursor: pointer;
        }

        .tile-close:hover {
            background-color: rgba(38, 50, 68, 0.08);
        }

        .tile-body {
            flex: 1;
            min-height: 0;
            padding: 8px;
            overflow: auto;
            background-color: #ffffff;
        }
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

const DEBUG_RGL = false;

function layoutSummary(layout) {
  return (layout || []).map((item) => ({
    i: String(item?.i),
    x: item?.x,
    y: item?.y,
    w: item?.w,
    h: item?.h,
  }));
}

function layoutsSummary(layouts) {
  const out = {};

  for (const [bp, layout] of Object.entries(layouts || {})) {
    out[bp] = layoutSummary(layout);
  }

  return out;
}

function debugRGL(label, payload = {}) {
  if (!DEBUG_RGL) {
    return;
  }

  console.log(
    `[DynamicReactGrid] ${label}`,
    JSON.stringify(payload, null, 2)
  );
}

function forceInitialPageScrollTop(root) {
  if (window.__astronomicalDidInitialScrollTop) {
    return;
  }

  window.__astronomicalDidInitialScrollTop = true;

  try {
    if ("scrollRestoration" in window.history) {
      window.history.scrollRestoration = "manual";
    }
  } catch (_error) {
    // Ignore unsupported browsers.
  }

  const scrollTop = () => {
    try {
      window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    } catch (_error) {
      window.scrollTo(0, 0);
    }

    const scrollingElement = document.scrollingElement || document.documentElement;

    if (scrollingElement) {
      scrollingElement.scrollTop = 0;
      scrollingElement.scrollLeft = 0;
    }

    if (document.documentElement) {
      document.documentElement.scrollTop = 0;
      document.documentElement.scrollLeft = 0;
    }

    if (document.body) {
      document.body.scrollTop = 0;
      document.body.scrollLeft = 0;
    }

    let node = root?.parentElement || null;

    while (node && node !== document.body && node !== document.documentElement) {
      const style = window.getComputedStyle(node);
      const overflowY = style?.overflowY || "";

      if (
        (overflowY === "auto" || overflowY === "scroll") &&
        node.scrollHeight > node.clientHeight
      ) {
        node.scrollTop = 0;
      }

      node = node.parentElement;
    }
  };

  scrollTop();
  window.requestAnimationFrame(scrollTop);
  window.setTimeout(scrollTop, 50);
  window.setTimeout(scrollTop, 250);
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

  const rootRef = React.useRef(null);
  const currentBpRef = React.useRef(safeString(currentBp, "lg") || "lg");
  const closeEventCountRef = React.useRef(integerOr(closeEventCount, 0));

  React.useEffect(() => {
    forceInitialPageScrollTop(rootRef.current);
  }, []);

  React.useEffect(() => {
    currentBpRef.current = safeString(currentBp, "lg") || "lg";
  }, [currentBp]);

  React.useEffect(() => {
    closeEventCountRef.current = integerOr(closeEventCount, closeEventCountRef.current || 0);
  }, [closeEventCount]);

  const childrenArray = React.Children.toArray(model.get_child("objects"));

  // Stability trick to avoid mismatched title/content while Panel patches
  // `keys` and `objects`. While a patch is unstable, keep rendering the last
  // complete key/object pair and do not write normalized layouts back to Python.
  const incomingKeys = (keys || []).map((key) => String(key));
  const patchIsStable = incomingKeys.length === childrenArray.length;
  const stable = React.useRef({ keys: [], children: [] });

  if (patchIsStable) {
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

  const layoutSignature = JSON.stringify(layouts || {});
  const previousLayoutSignatureRef = React.useRef(layoutSignature);
  const lastClientLayoutSignatureRef = React.useRef(layoutSignature);

  if (previousKeySignatureRef.current !== keySignature) {
      debugRGL("keySignature.changed", {
      previous: previousKeySignatureRef.current,
      next: keySignature,
    });
    previousKeySignatureRef.current = keySignature;
    suppressProgrammaticLayoutWriteRef.current = true;
  }

  if (previousLayoutSignatureRef.current !== layoutSignature) {

    const cameFromThisComponent =
      lastClientLayoutSignatureRef.current === layoutSignature;

        debugRGL("layoutSignature.changed", {
        cameFromThisComponent,
        previous: previousLayoutSignatureRef.current,
        next: layoutSignature,
        layouts: layoutsSummary(layouts || {}),
      });

    previousLayoutSignatureRef.current = layoutSignature;

    if (!cameFromThisComponent) {
      suppressProgrammaticLayoutWriteRef.current = true;
    }
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

debugRGL("render.state", {
  stableKeys,
  keySignature,
  patchIsStable,
  incomingKeys,
  childCount: childrenArray.length,
  currentBp,
  currentBpRef: currentBpRef.current,
  suppressProgrammaticLayoutWrite: suppressProgrammaticLayoutWriteRef.current,
  layouts: layoutsSummary(layouts || {}),
  normalizedLayouts: layoutsSummary(normalizedLayouts || {}),
});

  React.useEffect(() => {
    if (!patchIsStable) {
      debugRGL("normalize.skip.unstable_patch", {
        incomingKeys,
        childCount: childrenArray.length,
        layouts: layoutsSummary(layouts || {}),
        normalizedLayouts: layoutsSummary(normalizedLayouts || {}),
      });
      return;
    }

    if (!sameJSON(layouts || {}, normalizedLayouts || {})) {
      lastClientLayoutSignatureRef.current = JSON.stringify(normalizedLayouts || {});
      setLayouts(normalizedLayouts || {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    patchIsStable,
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
    <div ref={rootRef} className="pn-dynamic-rgl">
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
        onLayoutChange={(currentLayout, allLayouts) => {
          const bp = currentBpRef.current || "lg";

          debugRGL("onLayoutChange.enter", {
            bp,
            stableKeys,
            patchIsStable,
            suppressProgrammaticLayoutWrite: suppressProgrammaticLayoutWriteRef.current,
            rawCurrentLayout: layoutSummary(currentLayout || []),
            rawAllLayouts: layoutsSummary(allLayouts || {}),
            incomingLayouts: layoutsSummary(layouts || {}),
            normalizedLayouts: layoutsSummary(normalizedLayouts || {}),
          });

          if (!patchIsStable) {
            debugRGL("onLayoutChange.skip.unstable_patch", {
              bp,
              incomingKeys,
              childCount: childrenArray.length,
              rawCurrentLayout: layoutSummary(currentLayout || []),
              rawAllLayouts: layoutsSummary(allLayouts || {}),
            });
            return;
          }

          const sourceLayouts =
            allLayouts && Object.keys(allLayouts || {}).length > 0
              ? allLayouts
              : {
                  ...(layouts || {}),
                  [bp]: currentLayout || [],
                };

          const cleanAll = ensureLayoutsForKeys(
            sourceLayouts || {},
            stableKeys,
            colsByBp || {}
          );

          const cleanCur = sanitizeLayout(
            ((cleanAll || {})[bp] || currentLayout || []),
            stableKeys,
            integerOr((colsByBp || {})[bp], 12)
          );

          debugRGL("onLayoutChange.cleaned", {
            bp,
            cleanCur: layoutSummary(cleanCur || []),
            cleanAll: layoutsSummary(cleanAll || {}),
            layoutsChanged: !sameJSON(layouts || {}, cleanAll || {}),
          });

          if (suppressProgrammaticLayoutWriteRef.current) {
            suppressProgrammaticLayoutWriteRef.current = false;

            const expectedCurrent = sanitizeLayout(
              ((normalizedLayouts || layouts || {})[bp] || []),
              stableKeys,
              integerOr((colsByBp || {})[bp], 12)
            );

            debugRGL("onLayoutChange.suppressed", {
              bp,
              rawCurrentLayout: layoutSummary(currentLayout || []),
              cleanCur: layoutSummary(cleanCur || []),
              expectedCurrent: layoutSummary(expectedCurrent || []),
              layoutsBeforeReturn: layoutsSummary(layouts || {}),
              normalizedLayoutsBeforeReturn: layoutsSummary(normalizedLayouts || {}),
            });

            if (expectedCurrent.length > 0) {
              setCurrentLayout(expectedCurrent);
            }

            return;
          }

          debugRGL("onLayoutChange.acceptedCurrentLayout", {
            bp,
            cleanCur: layoutSummary(cleanCur || []),
          });

          setCurrentLayout(cleanCur || []);

          if (!sameJSON(layouts || {}, cleanAll || {})) {
            debugRGL("onLayoutChange.setLayouts", {
              bp,
              previousLayouts: layoutsSummary(layouts || {}),
              nextLayouts: layoutsSummary(cleanAll || {}),
            });

            lastClientLayoutSignatureRef.current = JSON.stringify(cleanAll || {});
            setLayouts(cleanAll || {});
          } else {
            debugRGL("onLayoutChange.noLayoutsChange", {
              bp,
            });
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

      <div className="pn-dynamic-rgl-bottom-spacer" aria-hidden="true" />
    </div>
  );
}
"""

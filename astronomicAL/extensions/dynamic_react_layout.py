import param
from panel.custom import Children, ReactComponent

class DynamicReactGrid(ReactComponent):
    objects = Children()
    keys = param.List(default=[])
    layouts = param.Dict(default={})

    breakpoints = param.Dict(default={"lg": 1500, "md": 1050, "sm": 0})
    cols_by_breakpoint = param.Dict(default={"lg": 12, "md": 12, "sm": 12})

    close_key = param.String(default="")
    current_breakpoint = param.String(default="lg")
    current_layout = param.List(default=[])

    row_height = param.Integer(default=80)
    margin = param.List(default=[10, 10])
    compact_type = param.ObjectSelector(default="vertical", objects=[None, "vertical", "horizontal"])
    resize_handles = param.List(default=["se"])

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
            "react": "https://esm.sh/react@18.2.0",
            "react-dom": "https://esm.sh/react-dom@18.2.0",

            # clsx ESM with default function export
            "clsx": "https://unpkg.com/clsx@2.1.1/dist/clsx.mjs",

            # IMPORTANT: externalize clsx so the above mapping is used
            "react-grid-layout": "https://esm.sh/react-grid-layout@1.4.4?external=react,react-dom,clsx",
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

    function sanitizeLayout(arr, keys, cols) {
      const keySet = new Set(keys || []);
      const C = Number.isFinite(cols) ? cols : 12;

      return (arr || [])
        .filter((it) => it && keySet.has(String(it.i)))
        .map((it) => {
          let w = Number.isFinite(it.w) ? it.w : 1;
          let h = Number.isFinite(it.h) ? it.h : 1;
          let x = Number.isFinite(it.x) ? it.x : 0;
          let y = Number.isFinite(it.y) ? it.y : 0;

          w = Math.max(1, Math.min(w, C));
          const maxX = Math.max(0, C - w);
          x = Math.max(0, Math.min(x, maxX));

          return { i: String(it.i), x, y, w, h, static: !!it.static };
        });
    }

    function ensureLayoutsForKeys(layouts, keys, colsByBp) {
      const out = { ...(layouts || {}) };
      const ks = keys || [];
      const colsMap = colsByBp || {};

      for (const bp of Object.keys(colsMap)) {
        const cols = colsMap[bp] ?? 12;
        const existing = new Map((out[bp] || []).map((it) => [String(it.i), it]));
        const full = ks.map((k, i) => {
          const hit = existing.get(String(k));
          if (hit) return hit;
          return { i: String(k), x: (i * 4) % cols, y: 1000000, w: 4, h: 4 };
        });
        out[bp] = sanitizeLayout(full, ks);
      }

      for (const bp of Object.keys(out)) {
        out[bp] = sanitizeLayout(out[bp], ks);
      }

      return out;
    }

    export function render({ model }) {
      const [keys] = model.useState("keys");

      const [layouts, setLayouts] = model.useState("layouts");
      const [breakpoints] = model.useState("breakpoints");
      const [colsByBp] = model.useState("cols_by_breakpoint");

      const [rowHeight] = model.useState("row_height");
      const [margin] = model.useState("margin");
      const [compactType] = model.useState("compact_type");
      const [resizeHandles] = model.useState("resize_handles");
      const [, setCloseKey] = model.useState("close_key");

      const [currentBp, setCurrentBp] = model.useState("current_breakpoint");
      const [, setCurrentLayout] = model.useState("current_layout");


      const bp = currentBp || "lg";

      const childrenArray = React.Children.toArray(model.get_child("objects"));

      // Stability trick to avoid mismatched title/content during patching.
      const stable = React.useRef({ keys: [], children: [] });
      if ((keys || []).length === childrenArray.length) {
        stable.current = { keys: [...(keys || [])], children: childrenArray };
      }
      const stableKeys = stable.current.keys;
      const stableChildren = stable.current.children;

      const contentByKey = {};
      for (let i = 0; i < stableKeys.length; i++) {
        contentByKey[stableKeys[i]] = stableChildren[i];
      }

      const normalizedLayouts = React.useMemo(
        () => ensureLayoutsForKeys(layouts, stableKeys, colsByBp),
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [JSON.stringify(layouts || {}), stableKeys.join("|"), JSON.stringify(colsByBp || {})]
      );

      React.useEffect(() => {
        const clean = sanitizeLayouts(normalizedLayouts, stableKeys, colsByBp);
        const a = JSON.stringify(layouts || {});
        const b = JSON.stringify(clean || {});
        if (a !== b) setLayouts(clean);
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, [stableKeys.join("|")]);

      return (
        <div className="pn-dynamic-rgl">
          <ResponsiveRGL
            layouts={normalizedLayouts || {}}
            breakpoints={breakpoints || {}}
            cols={colsByBp || {}}
            rowHeight={rowHeight}
            margin={margin}
            compactType={compactType === null ? null : compactType}
            draggableHandle=".tile-header"
            resizeHandles={resizeHandles}
            onBreakpointChange={(bp) => setCurrentBp(bp)}

            onLayoutChange={(currentLayout, allLayouts) => {
            const cleanAll = sanitizeLayouts(allLayouts, stableKeys, colsByBp);
            const bp = currentBp || "lg";
            const cleanCur = sanitizeLayout(currentLayout, stableKeys, (colsByBp && colsByBp[bp]) || 12);

            setLayouts(cleanAll);
            setCurrentLayout(cleanCur);
            }}
          >
            {(stableKeys || []).map((k) => (
              <div
                key={k}
                className="tile"
              >
                <div className="tile-header">
                  <div className="tile-title">{k}</div>
                  <button
                    className="tile-close"
                    title="Close"
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => { e.stopPropagation(); setCloseKey(k); }}
                  >
                    ×
                  </button>
                </div>
                <div className="tile-body">
                  {contentByKey[k] ?? <div>Missing content for {k}</div>}
                </div>
              </div>
            ))}
          </ResponsiveRGL>
        </div>
      );
    }

    function sanitizeLayouts(allLayouts, keys, colsByBp) {
      const out = {};
      const src = allLayouts || {};
      const colsMap = colsByBp || {};
      for (const bp of Object.keys(src)) {
        out[bp] = sanitizeLayout(src[bp], keys, colsMap[bp] ?? 12);
      }
      return out;
    }
    """
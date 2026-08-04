import pandas as pd
import numpy as np
import pandas.api.types as pdt
from astropy.io import fits
import gc
import os
import psutil
import time

proc = psutil.Process(os.getpid())

def rss_gib():
    return proc.memory_info().rss / (1024**3)

def df_mem_gib(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024**3)

class PrintProgress:
    def __init__(self, every=1.0):
        """
        every: minimum seconds between prints (rate limit)
        """
        self.t0 = time.time()
        self.last = 0.0
        self.every = float(every)

    def _now(self):
        return time.time()

    def _fmt(self, sec):
        sec = max(0, int(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def log(self, msg):
        elapsed = self._fmt(self._now() - self.t0)
        print(f"[{elapsed}] {msg}", flush=True)

    def prog(self, msg, frac):
        """
        frac: 0..1
        """
        now = self._now()
        if (now - self.last) < self.every and frac < 1.0:
            return
        self.last = now

        frac = max(0.0, min(1.0, float(frac)))
        elapsed_s = now - self.t0
        if frac > 0:
            eta_s = elapsed_s * (1 - frac) / frac
            eta = self._fmt(eta_s)
        else:
            eta = "?"
        if frac > 0.025:
            print(f"[{self._fmt(elapsed_s)}] {msg} — {int(frac*100)}% — ETA {eta}", flush=True)

def _fits_to_df_streaming(filename, hdu=1, p=None, cast_float32=False):
    p = p or PrintProgress(every=2.0)

    p.log(f"Opening FITS (memmap=True): {filename}")
    with fits.open(filename, memmap=True) as hdul:
        data = hdul[hdu].data
        arr = np.asarray(data)

        names = [n for n in arr.dtype.names if arr.dtype[n].shape == ()]
        p.log(f"Keeping {len(names)} 1D columns")

        cols = {}
        n = len(names)

        # float32 range limits (avoid overflow warning)
        f32max = np.finfo(np.float32).max
        f32min = -f32max

        p.log("Building DataFrame column-by-column…")
        for i, name in enumerate(names, 1):
            col = arr[name]

            if col.dtype.kind == "S":  # fixed-length bytes
                col = np.char.decode(col, "utf-8", errors="ignore")
            
            # endian fix per column if needed
            if col.dtype.byteorder == ">":
                col = col.byteswap().newbyteorder()

            # optional early downcast to cut peak memory
            if cast_float32 and col.dtype == np.float64:
                # cheap-ish safety check; still scans the column
                mn = np.nanmin(col)
                mx = np.nanmax(col)
                if mn >= f32min and mx <= f32max:
                    col = col.astype(np.float32, copy=False)

            cols[name] = col
            p.prog(f"Columns loaded ({name})", i / n)

        df = pd.DataFrame(cols, copy=False)

    p.log(f"DF built: rows={len(df):,}, cols={df.shape[1]:,}, mem≈{df_mem_gib(df):.2f} GiB, RSS≈{rss_gib():.2f} GiB")
    return df

def can_fit_float32(x: np.ndarray) -> bool:
    f32max = np.finfo(np.float32).max
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return (mn >= -f32max) and (mx <= f32max)


def _fits_to_df_fast(filename, hdu=1, p: PrintProgress | None = None):
    p = p or PrintProgress(every=2.0)

    p.log(f"Opening FITS (memmap=True): {filename}")
    with fits.open(filename, memmap=True) as hdul:
        p.log(f"Selecting HDU {hdu}")
        data = hdul[hdu].data  # FITS_rec

        p.log("Converting FITS_rec -> numpy array view")
        arr = np.asarray(data)

        names = [n for n in arr.dtype.names if arr.dtype[n].shape == ()]
        p.log(f"Keeping {len(names)} 1D columns (skipping vector columns)")

        needs_swap = (arr.dtype.byteorder == ">" or any(arr.dtype[n].byteorder == ">" for n in names))
        if needs_swap:
            p.log("Byteorder is big-endian -> byteswapping to native")
            arr = arr.byteswap().newbyteorder()
        else:
            p.log("Byteorder already native")

        p.log("Building DataFrame (this step can look stuck on very large tables)…")
        df = pd.DataFrame.from_records(arr[names])
        p.log(f"DF built: rows={len(df):,}, cols={df.shape[1]:,}, mem≈{df_mem_gib(df):.2f} GiB, RSS≈{rss_gib():.2f} GiB")

    p.log(f"DataFrame built: rows={len(df):,}, cols={df.shape[1]:,}")
    return df

def _decode_bytes_columns(df: pd.DataFrame, p: PrintProgress | None = None) -> pd.DataFrame:
    p = p or PrintProgress()
    obj_cols = list(df.select_dtypes(include=["object"]).columns)
    n = len(obj_cols)
    if n == 0:
        p.log("Decode: no object columns")
        return df

    p.log(f"Decode: scanning {n} object columns")
    for i, c in enumerate(obj_cols, 1):
        s = df[c]
        if len(s) and isinstance(s.iloc[0], (bytes, bytearray, np.bytes_)):
            df[c] = s.str.decode("utf-8", errors="ignore")
        p.prog(f"Decoding bytes ({c})", i / n)

    return df

def optimise_streaming(df: pd.DataFrame, p=None, log_every=10) -> pd.DataFrame:
    """
    Optimise without big peak memory:
    - pop one column at a time from df (releasing blocks gradually)
    - convert to smaller dtype
    - store into new dict
    - build new df at end
    """
    p = p or PrintProgress(every=2.0)

    cols = {}
    n = df.shape[1]
    p.log(f"Optimise(stream): starting, cols={n}, rows={len(df):,}")

    f32max = np.finfo(np.float32).max
    f32min = -f32max

    for i, c in enumerate(list(df.columns), 1):
        s = df.pop(c)  # IMPORTANT: remove from old df to free memory progressively
        arr = s.to_numpy(copy=False)

        dt = arr.dtype

        # floats -> float32 ONLY if safe
        if dt == np.float64:
            # nanmin/nanmax scan but avoids overflow->inf
            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
            if mn >= f32min and mx <= f32max:
                arr = arr.astype(np.float32, copy=True)

        # ints -> smallest fitting
        elif dt == np.int64:
            mn = int(arr.min())
            mx = int(arr.max())
            if mn >= np.iinfo(np.int8).min and mx <= np.iinfo(np.int8).max:
                arr = arr.astype(np.int8, copy=True)
            elif mn >= np.iinfo(np.int16).min and mx <= np.iinfo(np.int16).max:
                arr = arr.astype(np.int16, copy=True)
            elif mn >= np.iinfo(np.int32).min and mx <= np.iinfo(np.int32).max:
                arr = arr.astype(np.int32, copy=True)

        # objects -> category heuristic (if any)
        elif dt == object:
            # Use original Series for nunique/categorical conversion
            k = s.nunique(dropna=False)
            if k / max(len(s), 1) < 0.5:
                arr = s.astype("category")

        cols[c] = arr

        if p:
            p.prog(f"Optimising: {c}", i / n)

        # Periodic GC can help drop freed blocks sooner
        if (i % log_every) == 0:
            gc.collect()
            try:
                p.log(f"…optimised {i}/{n} cols, RSS≈{rss_gib():.2f} GiB")
            except Exception:
                pass

    # Old df now has no columns; encourage release
    del df
    gc.collect()

    out = pd.DataFrame(cols, copy=False)
    p.log(f"Optimise(stream): done, mem≈{df_mem_gib(out):.2f} GiB")
    return out



########### Maybe this is not the best place where to define this function but here we are also dealing with column type

def matches_type(dtype, type_list):
    """Check if a dtype matches any keyword in type_list"""
    for t in type_list:
        if t == "bool" and pdt.is_bool_dtype(dtype):
            return True
        if t == "float" and (pdt.is_float_dtype(dtype)):
            return True
        if t == "int" and (pdt.is_integer_dtype(dtype)):
            return True
        if t == "number" and pdt.is_numeric_dtype(dtype) and not pdt.is_bool_dtype(dtype):
            return True
        if t == "object" and (pdt.is_object_dtype(dtype)):
            return True
    return False


def get_series_type(s: pd.Series):
    """Returns the dtype of the passed series. 'mixed' is for Series with both numbers and strings """
    s = s.dropna()
    if s.empty:
        return "empty" 
    if pdt.is_bool_dtype(s):
        return "bool"
    if pdt.is_integer_dtype(s):
        return "int"
    if pdt.is_float_dtype(s):
        return "float"
    if pdt.is_string_dtype(s):
        return "string"
    if pdt.is_object_dtype(s):
        types = set(s.map(type))
        if types == {str}:
            return "string"
        if types <= {int, bool}:
            return "int"
        if types <= {int, float, bool, np.number}:
            return "float"
        return "mixed"

    return "mixed"






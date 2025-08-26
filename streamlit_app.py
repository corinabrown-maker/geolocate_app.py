# DXF/DWG Geolocator – Web App (Streamlit Cloud Deployable)

This is a **pure Streamlit** version (no Gradio) designed to deploy cleanly on **Streamlit Community Cloud**. It geolocates a CAD **DXF** (DWG after conversion) using a best‑fit **2D similarity transform** from user control points. Outputs a transformed **DXF** and optional **GeoJSON**.

---

## streamlit\_app.py

```python
import io
import tempfile
from typing import List, Tuple

import streamlit as st
import numpy as np
import ezdxf
import geojson

st.set_page_config(page_title="DXF/DWG Geolocator", page_icon="📐", layout="wide")
st.title("📐 DXF/DWG Geolocator")
st.caption("Upload DXF → enter control points → transform to target coords → download geolocated DXF/GeoJSON")

with st.sidebar:
    st.header("Settings")
    default_epsg = 2193  # NZTM2000 (for reference only; transform uses your control points)
    epsg = st.number_input("Target CRS (EPSG)", value=default_epsg, step=1, min_value=1)
    allow_geojson = st.checkbox("Export GeoJSON preview", value=True)

st.markdown(
    """
**Instructions**
1) Convert DWG → DXF if needed (ODA File Converter).
2) Upload a DXF.
3) Enter ≥2 control points: **src_x/src_y** (CAD local) → **dst_x/dst_y** (real‑world in your CRS units).
4) Click **Transform** and download outputs.
    """
)

uploaded = st.file_uploader("Upload DXF", type=["dxf"], accept_multiple_files=False)

# Editable control point table
cp_df = st.data_editor(
    {
        "src_x": [0.0, 10.0],
        "src_y": [0.0, 0.0],
        "dst_x": [0.0, 10.0],
        "dst_y": [0.0, 0.0],
    },
    num_rows="dynamic",
    use_container_width=True,
    key="cp_table",
)

# --- math helpers (pure NumPy) ---

def similarity_transform_from_points(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float, float]:
    """Best‑fit 2D similarity (scale+rotation+translation) via Umeyama.
    Returns (a, b, tx, ty) mapping [x, y] → [a*x − b*y + tx, b*x + a*y + ty]."""
    assert src.shape == dst.shape and src.shape[1] == 2
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = (src_c ** 2).sum() / n
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / var_src
    a = scale * R[0, 0]
    b = scale * R[1, 0]
    tx, ty = (mu_dst - (np.array([[a, -b], [b, a]]) @ mu_src)).tolist()
    return a, b, tx, ty


def rmse(src: np.ndarray, dst: np.ndarray, params: Tuple[float, float, float, float]) -> float:
    a, b, tx, ty = params
    M = np.array([[a, -b], [b, a]])
    pred = (src @ M.T) + np.array([tx, ty])
    e = dst - pred
    return float(np.sqrt((e ** 2).sum(axis=1).mean()))


def apply_affine_to_xy(x: float, y: float, a: float, b: float, tx: float, ty: float):
    return a * x - b * y + tx, b * x + a * y + ty


def apply_affine_to_coords(coords, a, b, tx, ty):
    return [apply_affine_to_xy(float(x), float(y), a, b, tx, ty) for (x, y) in coords]

# --- lightweight geometry helpers (no Shapely) ---
SUPPORTED = {"LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "POINT", "TEXT", "MTEXT"}


def entity_to_geom(e):
    dt = e.dxftype()
    try:
        if dt == "LINE":
            p1 = (float(e.dxf.start[0]), float(e.dxf.start[1]))
            p2 = (float(e.dxf.end[0]), float(e.dxf.end[1]))
            return {"type": "LineString", "coordinates": [p1, p2]}
        if dt == "LWPOLYLINE":
            pts = [(float(p[0]), float(p[1])) for p in e]
            return {"type": "LineString", "coordinates": pts}
        if dt == "POLYLINE":
            pts = [(float(v.dxf.location[0]), float(v.dxf.location[1])) for v in e.vertices]
            return {"type": "LineString", "coordinates": pts}
        if dt == "POINT":
            loc = e.dxf.location
            return {"type": "Point", "coordinates": (float(loc[0]), float(loc[1]))}
        if dt == "CIRCLE":
            cx, cy = float(e.dxf.center[0]), float(e.dxf.center[1])
            r = float(e.dxf.radius)
            theta = np.linspace(0, 2*np.pi, 64)
            pts = [(cx + r*np.cos(t), cy + r*np.sin(t)) for t in theta]
            return {"type": "LineString", "coordinates": pts}
        if dt == "ARC":
            cx, cy = float(e.dxf.center[0]), float(e.dxf.center[1])
            r = float(e.dxf.radius)
            start = np.deg2rad(float(e.dxf.start_angle))
            end = np.deg2rad(float(e.dxf.end_angle))
            theta = np.linspace(start, end, 64)
            pts = [(cx + r*np.cos(t), cy + r*np.sin(t)) for t in theta]
            return {"type": "LineString", "coordinates": pts}
        if dt in {"TEXT", "MTEXT"}:
            if hasattr(e.dxf, "insert"):
                ins = e.dxf.insert
                return {"type": "Point", "coordinates": (float(ins[0]), float(ins[1]))}
            return None
    except Exception:
        return None
    return None


def write_geom_to_dxf(modelspace, geom):
    t = geom.get("type")
    if t == "Point":
        x, y = geom["coordinates"]
        modelspace.add_point((x, y))
    elif t == "LineString":
        coords = geom["coordinates"]
        if len(coords) == 2:
            modelspace.add_line(coords[0], coords[1])
        elif len(coords) > 2:
            modelspace.add_lwpolyline(coords)


# --- main workflow ---
if st.button("Transform", type="primary", disabled=(uploaded is None)):
    if uploaded is None:
        st.error("Please upload a DXF first.")
        st.stop()

    # Read control points
    try:
        src = cp_df[["src_x", "src_y"]].to_numpy(dtype=float)
        dst = cp_df[["dst_x", "dst_y"]].to_numpy(dtype=float)
    except Exception:
        st.error("Control points must be numeric.")
        st.stop()

    mask = ~np.isnan(src).any(axis=1) & ~np.isnan(dst).any(axis=1)
    src = src[mask]
    dst = dst[mask]

    if src.shape[0] < 2:
        st.error("Please provide at least two valid control point pairs.")
        st.stop()

    params = similarity_transform_from_points(src, dst)
    a, b, tx, ty = params
    st.success(f"Estimated transform: a={a:.6f}, b={b:.6f}, tx={tx:.3f}, ty={ty:.3f}")
    if src.shape[0] >= 3:
        st.info(f"Control‑point RMSE: {rmse(src, dst, params):.3f} (target units)")

    # Persist upload to a temp file for ezdxf
    try:
        data = uploaded.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        dxf = ezdxf.readfile(tmp_path)
    except Exception as e:
        st.error(f"Failed to read DXF: {e}")
        st.stop()

    msp = dxf.modelspace()

    # Extract → transform geometries
    geoms = []
    for e in msp:
        if e.dxftype() in SUPPORTED:
            g = entity_to_geom(e)
            if g:
                geoms.append(g)

    if not geoms:
        st.warning("No supported entities found to transform.")

    geoms_t = []
    for g in geoms:
        if g["type"] == "Point":
            x, y = g["coordinates"]
            geoms_t.append({"type": "Point", "coordinates": apply_affine_to_xy(x, y, a, b, tx, ty)})
        elif g["type"] == "LineString":
            coords_t = apply_affine_to_coords(g["coordinates"], a, b, tx, ty)
            geoms_t.append({"type": "LineString", "coordinates": coords_t})

    # Create output DXF in‑memory
    out_doc = ezdxf.new(setup=True)
    out_msp = out_doc.modelspace()
    for g in geoms_t:
        write_geom_to_dxf(out_msp, g)

    out_buf = io.BytesIO()
    out_doc.write(out_buf)
    out_bytes = out_buf.getvalue()

    st.download_button(
        label="Download transformed DXF",
        data=out_bytes,
        file_name="geolocated.dxf",
        mime="image/vnd.dxf",
    )

    if allow_geojson and geoms_t:
        features = []
        for g in geoms_t:
            try:
                if g["type"] == "Point":
                    geom = {"type": "Point", "coordinates": list(g["coordinates"]) }
                elif g["type"] == "LineString":
                    geom = {"type": "LineString", "coordinates": [list(p) for p in g["coordinates"]] }
                else:
                    continue
                features.append(geojson.Feature(geometry=geom, properties={}))
            except Exception:
                pass
        fc = geojson.FeatureCollection(features)
        st.download_button(
            label="Download GeoJSON preview",
            data=geojson.dumps(fc).encode("utf-8"),
            file_name="preview.geojson",
            mime="application/geo+json",
        )

st.divider()
st.caption("Tip: NZTM2000 = EPSG:2193 • WGS84 = EPSG:4326 • Use ≥3 points for a better fit (shows RMSE)")
```

---

## requirements.txt

```txt
streamlit==1.37.1
ezdxf==1.3.0
numpy==1.26.4
geojson==3.1.0
```

> Pure‑Python wheels only (no Shapely/PROJ) → reliable on Streamlit Cloud.

---

## Deployment on Streamlit Community Cloud

1. Create a **GitHub repo** and add `streamlit_app.py` and `requirements.txt`.
2. Go to **streamlit.io/cloud** → **New app** → select your repo.
3. Set **Main file path** to `streamlit_app.py` → **Deploy**.

**Tips**

* If your input is **DWG**, convert to **DXF** (ASCII DXF R2018 recommended).
* For best accuracy, use **≥3 control points** (RMSE shown).
* If geometry is missing after transform, re‑save DXF as ASCII and retry.

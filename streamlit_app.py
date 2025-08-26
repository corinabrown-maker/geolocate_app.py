import io
import tempfile
from typing import Tuple

import streamlit as st
import numpy as np
import ezdxf
import geojson

st.set_page_config(page_title="DXF/DWG Geolocator", page_icon=":triangular_ruler:", layout="wide")
st.title("DXF/DWG Geolocator")
st.caption("Upload DXF -> enter control points -> transform -> download geolocated DXF or GeoJSON")

with st.sidebar:
    st.header("Settings")
    epsg = st.number_input("Target CRS (EPSG, for reference only)", value=2193, step=1, min_value=1)
    allow_geojson = st.checkbox("Export GeoJSON preview", value=True)

st.markdown(
    "Instructions\n"
    "1) Convert DWG to DXF if needed (e.g., ODA File Converter).\n"
    "2) Upload a DXF.\n"
    "3) Enter 2 or more control points: src_x/src_y (CAD local) -> dst_x/dst_y (target coords).\n"
    "4) Click Transform and download outputs."
)

uploaded = st.file_uploader("Upload DXF", type=["dxf"], accept_multiple_files=False)

# Initial control-point table
cp_df = st.data_editor(
    {"src_x": [0.0, 10.0], "src_y": [0.0, 0.0], "dst_x": [0.0, 10.0], "dst_y": [0.0, 0.0]},
    num_rows="dynamic",
    use_container_width=True,
    key="cp_table",
)

def similarity_transform_from_points(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float, float]:
    """Best-fit 2D similarity transform (scale, rotation, translation). Returns (a, b, tx, ty)
    mapping [x, y] to [a*x - b*y + tx, b*x + a*y + ty].
    """
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
    a = float(scale * R[0, 0])
    b = float(scale * R[1, 0])
    tx, ty = (mu_dst - (np.array([[a, -b], [b, a]]) @ mu_src)).tolist()
    return a, b, float(tx), float(ty)

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

if st.button("Transform", type="primary", disabled=(uploaded is None)):
    if uploaded is None:
        st.error("Please upload a DXF first.")
        st.stop()

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
        st.info(f"Control-point RMSE: {rmse(src, dst, params):.3f} (target units)")

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
                    geom = {"type": "Point", "coordinates": list(g["coordinates"])}
                elif g["type"] == "LineString":
                    geom = {"type": "LineString", "coordinates": [list(p) for p in g["coordinates"]]}
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
st.caption("Tip: NZTM2000 = EPSG:2193; WGS84 = EPSG:4326; Use 3+ points for a better fit.")

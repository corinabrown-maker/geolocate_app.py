import io
import json
from typing import List, Tuple

import streamlit as st
import numpy as np
import ezdxf
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.affinity import affine_transform as shapely_affine
from pyproj import CRS, Transformer
import geojson

st.set_page_config(page_title="DXF/DWG Geolocator", page_icon="📐", layout="wide")
st.title("📐 DXF/DWG Geolocator")
st.caption("Upload DXF → enter control points → transform to target CRS → download geolocated DXF/GeoJSON")

with st.sidebar:
    st.header("Settings")
    default_epsg = 2193  # NZTM2000
    epsg = st.number_input("Target CRS (EPSG)", value=default_epsg, step=1, min_value=1)
    allow_geojson = st.checkbox("Export GeoJSON preview", value=True)

st.markdown(
    """
**Instructions**
1) Convert DWG → DXF if needed (ODA File Converter).
2) Upload a DXF.
3) Enter ≥2 control points: **src_x/src_y** (CAD local) → **dst_x/dst_y** (real-world in selected EPSG).
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

# --- math helpers ---
def similarity_transform_from_points(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float, float]:
    """Estimate best-fit 2D similarity transform using Umeyama (scale, rotation, translation).
    Returns (a, b, tx, ty) that maps [x y 1] via matrix [[a, -b, tx],[b, a, ty],[0,0,1]].
    """
    assert src.shape == dst.shape and src.shape[1] == 2

    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    var_src = (src_centered ** 2).sum() / n
    cov = (dst_centered.T @ src_centered) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / var_src

    # rotation matrix R = [[r11,r12],[r21,r22]]
    a = scale * R[0, 0]
    b = scale * R[1, 0]

    tx, ty = (mu_dst - (np.array([[a, -b], [b, a]]) @ mu_src)).tolist()
    return a, b, tx, ty


def affine_params(a: float, b: float, tx: float, ty: float) -> List[float]:
    """Return shapely_affine params [a, -b, b, a, tx, ty]."""
    return [a, -b, b, a, tx, ty]


def rmse(src: np.ndarray, dst: np.ndarray, params: Tuple[float, float, float, float]) -> float:
    a, b, tx, ty = params
    M = np.array([[a, -b], [b, a]])
    pred = (src @ M.T) + np.array([tx, ty])
    e = dst - pred
    return float(np.sqrt((e ** 2).sum(axis=1).mean()))


# --- DXF helpers ---
SUPPORTED = {"LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "POINT", "SPLINE", "TEXT", "MTEXT"}


def entity_to_geometry(e):
    try:
        if e.dxftype() == "LINE":
            return LineString([e.dxf.start, e.dxf.end])
        if e.dxftype() == "LWPOLYLINE":
            pts = [tuple(p[0:2]) for p in e]
            return LineString(pts)
        if e.dxftype() == "POLYLINE":
            pts = [tuple(v.dxf.location[0:2]) for v in e.vertices]
            return LineString(pts)
        if e.dxftype() == "POINT":
            return Point(e.dxf.location[0:2])
        if e.dxftype() == "CIRCLE":
            center = e.dxf.center[0:2]
            r = e.dxf.radius
            # approximate circle with 64-pt linestring
            theta = np.linspace(0, 2*np.pi, 64)
            pts = [(center[0] + r*np.cos(t), center[1] + r*np.sin(t)) for t in theta]
            return LineString(pts)
        if e.dxftype() == "ARC":
            c = e.dxf.center[0:2]
            r = e.dxf.radius
            start = np.deg2rad(e.dxf.start_angle)
            end = np.deg2rad(e.dxf.end_angle)
            theta = np.linspace(start, end, 64)
            pts = [(c[0] + r*np.cos(t), c[1] + r*np.sin(t)) for t in theta]
            return LineString(pts)
        if e.dxftype() in {"TEXT", "MTEXT"}:
            # represent as point for preview purposes only
            if hasattr(e.dxf, "insert"):
                ins = e.dxf.insert
                return Point(ins[0:2])
            return None
    except Exception:
        return None
    return None


def geometry_to_dxf(modelspace, geom):
    # Writes basic geometries back as entities (line/polylines/points). Circles/arcs are output as polylines.
    if isinstance(geom, Point):
        modelspace.add_point((geom.x, geom.y))
    elif isinstance(geom, LineString):
        coords = list(geom.coords)
        if len(coords) == 2:
            modelspace.add_line(coords[0], coords[1])
        else:
            modelspace.add_lwpolyline(coords)
    elif isinstance(geom, Polygon):
        modelspace.add_lwpolyline(list(geom.exterior.coords))


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

    # Filter valid rows
    mask = ~np.isnan(src).any(axis=1) & ~np.isnan(dst).any(axis=1)
    src = src[mask]
    dst = dst[mask]

    if src.shape[0] < 2:
        st.error("Please provide at least two valid control point pairs.")
        st.stop()

    # Compute similarity transform in target CRS space directly
    params = similarity_transform_from_points(src, dst)
    a, b, tx, ty = params
    st.success(f"Estimated transform: a={a:.6f}, b={b:.6f}, tx={tx:.3f}, ty={ty:.3f}")
    if src.shape[0] >= 3:
        err = rmse(src, dst, params)
        st.info(f"Control-point RMSE: {err:.3f} units (in target CRS units)")

    # Load DXF
    try:
        dxf = ezdxf.readfile(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Failed to read DXF: {e}")
        st.stop()

    msp = dxf.modelspace()

    # Collect and transform geometries
    geoms = []
    for e in msp:
        if e.dxftype() in SUPPORTED:
            g = entity_to_geometry(e)
            if g is not None:
                geoms.append(g)

    if not geoms:
        st.warning("No supported entities found to transform.")

    # Apply affine
    A = affine_params(a, b, tx, ty)
    geoms_t = [shapely_affine(g, A) for g in geoms]

    # Create output DXF in-memory
    out_doc = ezdxf.new(setup=True)
    out_msp = out_doc.modelspace()

    for g in geoms_t:
        geometry_to_dxf(out_msp, g)

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
                gj = mapping(g)
                features.append(geojson.Feature(geometry=gj, properties={}))
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

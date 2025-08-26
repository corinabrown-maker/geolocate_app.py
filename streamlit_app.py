import gradio as gr
import numpy as np
import ezdxf
import tempfile
import os

SUPPORTED = {"LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "POINT", "SPLINE", "TEXT", "MTEXT"}

def similarity_transform(src, dst):
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

def apply_affine(coords, a, b, tx, ty):
    return [(a*x - b*y + tx, b*x + a*y + ty) for x, y in coords]

def entity_to_geom(e):
    dt = e.dxftype()
    try:
        if dt == "LINE":
            return [(float(e.dxf.start[0]), float(e.dxf.start[1])),
                    (float(e.dxf.end[0]), float(e.dxf.end[1]))]
        if dt == "LWPOLYLINE":
            return [(float(p[0]), float(p[1])) for p in e]
        if dt == "POLYLINE":
            return [(float(v.dxf.location[0]), float(v.dxf.location[1])) for v in e.vertices]
        if dt == "POINT":
            return [(float(e.dxf.location[0]), float(e.dxf.location[1]))]
        if dt == "CIRCLE":
            cx, cy = float(e.dxf.center[0]), float(e.dxf.center[1])
            r = float(e.dxf.radius)
            theta = np.linspace(0, 2*np.pi, 64)
            return [(cx + r*np.cos(t), cy + r*np.sin(t)) for t in theta]
        if dt == "ARC":
            cx, cy = float(e.dxf.center[0]), float(e.dxf.center[1])
            r = float(e.dxf.radius)
            start = np.deg2rad(float(e.dxf.start_angle))
            end = np.deg2rad(float(e.dxf.end_angle))
            theta = np.linspace(start, end, 64)
            return [(cx + r*np.cos(t), cy + r*np.sin(t)) for t in theta]
    except Exception:
        return None
    return None

def process(file_path, src_points, dst_points):
    # Parse control points: you can enter e.g. [(0,0),(10,0)] in both boxes.
    src = np.array(eval(src_points), dtype=float)
    dst = np.array(eval(dst_points), dtype=float)
    if src.shape[0] < 2:
        return None, "Need at least two points"

    a, b, tx, ty = similarity_transform(src, dst)

    if not (isinstance(file_path, str) and os.path.exists(file_path)):
        return None, "DXF path invalid — please re-upload"

    # Read DXF from path
    try:
        dxf = ezdxf.readfile(file_path)
    except Exception as e:
        return None, f"Failed to read DXF: {e}"

    msp = dxf.modelspace()
    transformed_geoms = []
    for e in msp:
        if e.dxftype() in SUPPORTED:
            coords = entity_to_geom(e)
            if coords:
                transformed_geoms.append(apply_affine(coords, a, b, tx, ty))

    # Write transformed DXF to a temp file and return its path
    out_doc = ezdxf.new(setup=True)
    out_msp = out_doc.modelspace()
    for coords in transformed_geoms:
        if len(coords) == 1:
            out_msp.add_point(coords[0])
        elif len(coords) == 2:
            out_msp.add_line(coords[0], coords[1])
        else:
            out_msp.add_lwpolyline(coords)

    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf")
    out_doc.saveas(out_tmp.name)
    return out_tmp.name, "Transform complete"

demo = gr.Interface(
    fn=process,
    inputs=[
        gr.File(file_types=[".dxf"], label="Upload DXF", type="filepath"),
        gr.Textbox(label="Source Points [(x1,y1),(x2,y2),...]", value="[(0,0),(10,0)]"),
        gr.Textbox(label="Destination Points [(x1,y1),(x2,y2),...]", value="[(0,0),(10,0)]"),
    ],
    outputs=[gr.File(label="Download Transformed DXF"), gr.Textbox(label="Status")],
    title="DXF/DWG Geolocator"
)

demo.launch()

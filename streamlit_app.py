import io


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

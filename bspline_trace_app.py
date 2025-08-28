
# bspline_trace_app.py
# Streamlit app: load PNG, extract contour(s), fit B-spline, visualize, and export SVG/JSON.
# Run:  streamlit run bspline_trace_app.py
import io
import json
import base64
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# skimage / scipy for image ops + splines
from skimage import filters, measure, morphology, color, util
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import disk, binary_opening, binary_closing, remove_small_holes, remove_small_objects
from scipy.interpolate import splprep, splev

st.set_page_config(page_title="B‑spline Tracer", layout="wide")

# ---------- Helpers ----------

def to_grayscale(img_pil):
    if img_pil.mode != "L":
        return ImageOps.grayscale(img_pil)
    return img_pil

def binarize(np_gray, method="otsu", invert=False, manual_t=128, block_size=35, offset=0):
    img = np_gray.astype(np.float32) / 255.0
    if method == "otsu":
        t = threshold_otsu(img)
        mask = img > t
    elif method == "adaptive":
        # threshold_local expects 2D float, block_size odd
        bs = max(3, block_size | 1)  # ensure odd
        local_t = threshold_local(img, bs, offset=offset)
        mask = img > local_t
    else:
        mask = img > (manual_t / 255.0)
    if invert:
        mask = ~mask
    return mask

def postprocess_mask(mask_bool, close_r=0, open_r=0, min_hole=0, min_obj=0):
    m = mask_bool.copy()
    if close_r > 0:
        m = binary_closing(m, footprint=disk(close_r))
    if open_r > 0:
        m = binary_opening(m, footprint=disk(open_r))
    if min_hole > 0:
        m = remove_small_holes(m, area_threshold=int(min_hole))
    if min_obj > 0:
        m = remove_small_objects(m, min_size=int(min_obj))
    return m

def find_contours_sorted(mask_bool, level=0.5):
    # skimage returns contours as arrays of [row(y), col(x)]
    contours = measure.find_contours(mask_bool.astype(float), level=level)
    # Sort by absolute signed area (descending)
    def poly_area_xy(poly):
        if len(poly) < 3:
            return 0.0
        x = poly[:,1]; y = poly[:,0]
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    contours = sorted(contours, key=poly_area_xy, reverse=True)
    return contours

def simplify_polyline(poly, tol=1.0):
    # poly: Nx2 in (y,x). We'll keep coordinate order but use skimage's approx.
    if tol <= 0 or len(poly) < 5:
        return poly
    approx = measure.approximate_polygon(poly, tolerance=tol)
    # ensure closed
    if np.linalg.norm(approx[0] - approx[-1]) > 1e-9:
        approx = np.vstack([approx, approx[0]])
    return approx

def bspline_fit_closed(poly_yx, smooth=2.0, samples=400, degree=3):
    """Fit a periodic B-spline to a closed polyline (in y,x order).
       Returns (tck, xs, ys) where xs,ys are sampled points (len=samples)."""
    x = poly_yx[:,1].astype(np.float64)  # columns
    y = poly_yx[:,0].astype(np.float64)  # rows
    # Ensure closed
    if np.hypot(x[0]-x[-1], y[0]-y[-1]) > 1e-9:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    # Fit periodic spline
    try:
        tck, u = splprep([x, y], s=float(smooth), per=True, k=int(degree))
    except Exception as e:
        # If per=True fails (e.g., too few points for degree), fall back to open
        tck, u = splprep([x, y], s=float(smooth), per=False, k=min(int(degree), 3))
    unew = np.linspace(0, 1, int(samples), endpoint=False)
    xs, ys = splev(unew, tck)
    return tck, np.asarray(xs), np.asarray(ys)

def catmull_rom_beziers(points_xy, tension=0.0, closed=True):
    """Convert a polyline to cubic Bezier segments via Cardinal (Catmull-Rom with tension).
       tension in [0,1], 0 = standard Catmull-Rom, 1 = straight lines.
       Returns list of segments: [(B0,B1,B2,B3), ...] with B* as (x,y)."""
    pts = np.asarray(points_xy, dtype=float)
    if closed:
        pts = np.vstack([pts[-2], pts, pts[1]])  # pad for P0,P1,...,Pn with wrap
    else:
        if len(pts) < 4:
            return []
        pts = pts.copy()

    segs = []
    c = float(tension)
    n = len(pts)
    # iterate segments from P1->P2 for i = 1..n-3
    for i in range(1, n-2):
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]
        b0 = p1
        b3 = p2
        # derivative estimates
        d1 = (1.0 - c) * 0.5 * (p2 - p0)
        d2 = (1.0 - c) * 0.5 * (p3 - p1)
        b1 = b0 + d1 / 3.0
        b2 = b3 - d2 / 3.0
        segs.append((tuple(b0), tuple(b1), tuple(b2), tuple(b3)))
    return segs

def svg_path_from_beziers(list_of_segments, close_path=True):
    """Build SVG path 'd' string from cubic Bezier segments."""
    if not list_of_segments:
        return ""
    # Start at B0 of first segment
    d = []
    b0 = list_of_segments[0][0]
    d.append(f"M {b0[0]:.3f},{b0[1]:.3f}")
    for (B0,B1,B2,B3) in list_of_segments:
        d.append(f"C {B1[0]:.3f},{B1[1]:.3f} {B2[0]:.3f},{B2[1]:.3f} {B3[0]:.3f},{B3[1]:.3f}")
    if close_path:
        d.append("Z")
    return " ".join(d)

def svg_path_from_polyline(points_xy, close_path=True):
    if len(points_xy) == 0:
        return ""
    d = [f"M {points_xy[0,0]:.3f},{points_xy[0,1]:.3f}"]
    for p in points_xy[1:]:
        d.append(f"L {p[0]:.3f},{p[1]:.3f}")
    if close_path:
        d.append("Z")
    return " ".join(d)

def make_svg(width, height, path_list, fill="#000000", stroke="#000000", stroke_width=1, fill_rule="evenodd"):
    """path_list: list of dicts {d:<path_d>, fill:<color or None>}"""
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    out = [svg_header]
    for p in path_list:
        fill_val = p.get("fill", fill)
        if fill_val is None:
            fill_val = "none"
        out.append(f'<path d="{p["d"]}" fill="{fill_val}" stroke="{stroke}" stroke-width="{stroke_width}" fill-rule="{fill_rule}" />')
    out.append('</svg>')
    return "\n".join(out)

def download_button_bytes(label, data_bytes, file_name, mime):
    st.download_button(label, data=data_bytes, file_name=file_name, mime=mime)

# ---------- Sidebar UI ----------
st.sidebar.title("⚙️ Налаштування")

uploaded = st.sidebar.file_uploader("Завантажте PNG (ч/б або кольорове)", type=["png","jpg","jpeg"])
downscale = st.sidebar.slider("Даунскел зображення (для швидкості)", 0.1, 1.0, 1.0, 0.05)
invert = st.sidebar.checkbox("Інвертувати", value=False)

st.sidebar.subheader("Порогова бінаризація")
bin_method = st.sidebar.selectbox("Метод", ["otsu", "adaptive", "manual"])
manual_t = st.sidebar.slider("Manual threshold", 0, 255, 128)
block_size = st.sidebar.slider("Adaptive block size", 3, 99, 35, step=2)
adapt_offs = st.sidebar.slider("Adaptive offset", -0.5, 0.5, 0.0, 0.05)

st.sidebar.subheader("Морфологія")
close_r = st.sidebar.slider("Closing (радіус)", 0, 15, 1)
open_r  = st.sidebar.slider("Opening (радіус)", 0, 15, 0)
min_hole = st.sidebar.slider("Min hole area (залити)", 0, 5000, 64, step=16)
min_obj  = st.sidebar.slider("Min object area (видалити)", 0, 5000, 0, step=16)

st.sidebar.subheader("Контури → B‑spline")
keep_only_largest = st.sidebar.checkbox("Лише найбільший контур", True)
simplify_tol = st.sidebar.slider("Спрощення ламаної (пікс.)", 0.0, 10.0, 1.5, 0.1)
smooth = st.sidebar.slider("Згладжування сплайну s", 0.0, 20.0, 2.0, 0.5)
degree = st.sidebar.slider("Ступінь сплайну", 2, 5, 3, 1)
samples = st.sidebar.slider("Точок на кривій (візуалізація)", 100, 2000, 600, 100)

st.sidebar.subheader("Експорт Bezier (Catmull‑Rom)")
tension = st.sidebar.slider("Натяг (tension) 0..1", 0.0, 1.0, 0.0, 0.05)

# ---------- Main ----------
st.title("🧭 B‑spline Tracer")

col_in, col_mid, col_out = st.columns([1,0.05,1])

default_img = None
if uploaded is None:
    st.info("Завантажте зображення зліва, або використовуйте демо силует нижче.")
    # Create a simple demo image
    demo = Image.new("L", (512, 384), 255)
    import math
    arr = np.array(demo)
    cy, cx, r = 192, 256, 120
    Y, X = np.ogrid[:arr.shape[0], :arr.shape[1]]
    mask = (X-cx)**2 + (Y-cy)**2 <= r*r
    arr[mask] = 0
    default_img = Image.fromarray(arr)
    img_pil = default_img
else:
    img_pil = Image.open(uploaded).convert("L")

if downscale < 1.0:
    new_size = (int(img_pil.width*downscale), int(img_pil.height*downscale))
    img_pil = img_pil.resize(new_size, Image.LANCZOS)

with col_in:
    st.subheader("Вхідне зображення")
    st.image(img_pil, width='stretch', clamp=True)

# Binarize
gray_np = np.array(img_pil)
mask0 = binarize(gray_np, method=bin_method, invert=invert, manual_t=manual_t,
                 block_size=block_size, offset=adapt_offs)
mask = postprocess_mask(mask0, close_r=close_r, open_r=open_r, min_hole=min_hole, min_obj=min_obj)

# Find contours
contours = find_contours_sorted(mask, level=0.5)
if keep_only_largest and len(contours) > 0:
    contours = contours[:1]

# Simplify each contour and fit B-spline
results = []
for poly in contours:
    simp = simplify_polyline(poly, tol=simplify_tol)
    if len(simp) < 4:
        continue
    tck, xs, ys = bspline_fit_closed(simp, smooth=smooth, samples=samples, degree=degree)
    results.append({"poly": poly, "simp": simp, "tck": tck, "samp": np.vstack([xs, ys]).T})

# Visualize
with col_out:
    st.subheader("Контур та сплайн")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img_pil, cmap="gray")
    # overlay mask edge
    for r in results:
        poly = r["poly"]
        simp = r["simp"]
        samp = r["samp"]
        ax.plot(poly[:,1], poly[:,0], linewidth=1)       # raw contour
        ax.plot(simp[:,1], simp[:,0], linewidth=1, linestyle="--")  # simplified
        ax.plot(samp[:,0], samp[:,1], linewidth=2)       # B-spline sampling
    ax.set_title(f"Знайдено контурів: {len(results)}")
    ax.set_axis_off()
    st.pyplot(fig, width='stretch')

# ---------- Exports ----------
st.subheader("Експорт")

if len(results) == 0:
    st.warning("Контури не знайдені. Спробуйте змінити поріг/морфологію.")
else:
    # Export: B-spline JSON (tck)
    tck_all = []
    for r in results:
        (t, c, k) = r["tck"]
        tck_all.append({
            "t": list(map(float, t)),
            "cx": list(map(float, c[0])),
            "cy": list(map(float, c[1])),
            "k": int(k)
        })
    meta = {"width": img_pil.width, "height": img_pil.height, "countours": len(results)}
    payload = {"meta": meta, "tck_list": tck_all}
    json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    download_button_bytes("⬇️ Завантажити B‑spline JSON (tck)", json_bytes, "bspline_tck.json", "application/json")

    # Export: SVG (polyline path built from B-spline sampling)
    path_list_poly = []
    for r in results:
        samp = r["samp"]
        # Build path from sampled points (x,y); our helper wants [x,y], but SVG uses x,y in the same order.
        # We'll provide (x,y) pairs in the same image coordinate space.
        pts = np.column_stack([samp[:,0], samp[:,1]])
        d_poly = svg_path_from_polyline(pts, close_path=True)
        path_list_poly.append({"d": d_poly, "fill": None})
    svg_poly = make_svg(img_pil.width, img_pil.height, path_list_poly, fill="none", stroke="#00f", stroke_width=2)
    download_button_bytes("⬇️ SVG (шлях з ліній через дискретизацію сплайну)", svg_poly.encode("utf-8"), "bspline_poly.svg", "image/svg+xml")

    # Export: SVG Bezier via Catmull-Rom on sampled points (optionally smoother for editors)
    path_list_bez = []
    for r in results:
        samp = r["samp"]
        pts = np.column_stack([samp[:,0], samp[:,1]])
        segs = catmull_rom_beziers(pts, tension=tension, closed=True)
        d_bez = svg_path_from_beziers(segs, close_path=True)
        path_list_bez.append({"d": d_bez, "fill": None})
    svg_bez = make_svg(img_pil.width, img_pil.height, path_list_bez, fill="none", stroke="#f00", stroke_width=2)
    download_button_bytes("⬇️ SVG (кубічні Безьє через Catmull‑Rom)", svg_bez.encode("utf-8"), "bezier_path.svg", "image/svg+xml")

    # Export: sampled points CSV (for debugging)
    import csv
    csv_buf = io.StringIO()
    w = csv.writer(csv_buf)
    w.writerow(["contour_id","x","y"])
    for cid, r in enumerate(results):
        for x,y in r["samp"]:
            w.writerow([cid, f"{x:.6f}", f"{y:.6f}"])
    download_button_bytes("⬇️ CSV вибірки сплайну", csv_buf.getvalue().encode("utf-8"), "spline_sample.csv", "text/csv")

st.markdown("---")
with st.expander("ℹ️ Пояснення методу"):
    st.write("""
    1) Бінаризація (Otsu/Adaptive/Manual) → 2) Морфологічне чистіння → 3) Пошук контурів (marching squares) →
    4) Спрощення ламаної → 5) Підігнання періодичного B‑spline → 6) Візуалізація/Експорт.

    **Експорт**:
    * JSON — вузли/коефіцієнти B‑spline (`tck`) для кожного контуру.
    * SVG (poly) — шлях `path` зі шматків `L` через дискретизований сплайн.
    * SVG (Bezier) — перетворення вибірки на кубічні Безьє через Cardinal (Catmull‑Rom) з обраною *tension*.

    """)

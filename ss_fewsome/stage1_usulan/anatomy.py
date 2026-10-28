"""
Lokalisasi anatomi dan penyejajaran (Ide 3).

Langkah per gambar (lihat Bab 4 dokumen panduan):
  1. Profil per baris di separuh kiri dan kanan -> lembah = garis sendi -> kemiringan.
  2. Putar supaya garis sendi mendatar.
  3. Profil per kolom di pita sekitar garis sendi -> tepi kiri dan kanan -> lebar dan titik tengah.
  4. Fibula -> sisi lateral; gambar dibalik supaya lateral selalu di kanan.
  5. Satu transformasi gabungan (putar + skala + geser) -> titik tengah sendi ke CENTER,
     lebar lutut menjadi KNEE_WIDTH_REF.
Peta wilayah (tepi, celah, subkondral) didefinisikan sekali dalam koordinat acuan.
"""
import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import uniform_filter1d

from .config import IMG, CENTER, KNEE_WIDTH_REF
from .data_split import load_gray


# ------------------------------------------------------------------ deteksi
def _valley(profile, lo, hi):
    """Baris terkecil di area pencarian + kedalaman lembahnya."""
    seg = profile[lo:hi]
    y = lo + int(np.argmin(seg))
    around = np.r_[profile[max(lo, y - 30):max(lo, y - 8)], profile[y + 8:min(hi, y + 30)]]
    depth = float(np.median(around) - profile[y]) if around.size else 0.0
    return y, depth


def _sampler(img, theta, src_center, scale, ref_center):
    """Peta koordinat untuk satu transformasi gabungan (dipakai cv2.remap, interpolasi bilinear)."""
    c, s = np.cos(theta), np.sin(theta)
    yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    vx, vy = (xx - ref_center[0]) / scale, (yy - ref_center[1]) / scale
    map_x = (c * vx - s * vy + src_center[0]).astype(np.float32)
    map_y = (s * vx + c * vy + src_center[1]).astype(np.float32)
    return cv2.remap(img.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _fibula_side(rot, y0, le, re):
    """Cari tulang kecil terang di bawah ujung tibia. Kembalikan 'left', 'right', atau None."""
    r0, r1 = int(min(IMG - 2, y0 + 35)), int(min(IMG - 1, y0 + 85))
    if r1 - r0 < 10:
        return None, 0.0
    q = uniform_filter1d(rot[r0:r1].mean(0), 3)
    w = re - le
    scores = {}
    for side in ("left", "right"):
        if side == "left":
            a, b = int(max(0, le - 5)), int(min(IMG, le + 0.4 * w))
            seg = q[a:b]
        else:
            a, b = int(max(0, re - 0.4 * w)), int(min(IMG, re + 5))
            seg = q[a:b][::-1]
        if seg.size < 12:
            scores[side] = 0.0
            continue
        # dari luar ke dalam: struktur terang pertama (fibula), lalu celah gelap, lalu batang tibia
        bg = np.percentile(q, 10)
        bright = seg > bg + 20
        scores[side] = 0.0
        if bright.any():
            s0 = int(np.argmax(bright))
            s1 = s0
            while s1 < seg.size and bright[s1]:
                s1 += 1
            if s1 < seg.size and bright[s1:].any():          # masih ada struktur terang lain di dalam (batang tibia)
                nxt = s1 + int(np.argmax(bright[s1:]))
                dip = seg[s1:nxt].min() if nxt > s1 else seg[s1]
                if s1 - s0 <= 0.25 * w:                        # fibula sempit
                    scores[side] = float(seg[s0:s1].max() - dip)
    side = max(scores, key=scores.get)
    other = "right" if side == "left" else "left"
    conf = scores[side] - scores[other]
    return (side if scores[side] > 12 and conf > 6 else None), conf


def detect(img: np.ndarray, cfg, fname: str = "") -> dict:
    """Hitung parameter penyejajaran satu gambar. Tidak mengubah gambar."""
    im = img.astype(np.float32)
    lo, hi = cfg.row_search
    pL = uniform_filter1d(im[:, 20:112].mean(1), cfg.smooth_rows)
    pR = uniform_filter1d(im[:, 112:204].mean(1), cfg.smooth_rows)
    yL, dL = _valley(pL, lo, hi)
    yR, dR = _valley(pR, lo, hi)
    theta = float(np.arctan2(yR - yL, 112.0))
    y0 = (yL + yR) / 2.0

    rot = _sampler(im, theta, (112.0, y0), 1.0, (112.0, y0))
    yi = int(round(y0))
    rows = np.r_[max(0, yi - cfg.band_outer):max(0, yi - cfg.band_inner), min(IMG, yi + cfg.band_inner):min(IMG, yi + cfg.band_outer)]
    cp = uniform_filter1d(rot[rows].mean(0), cfg.smooth_cols)
    dif = np.diff(cp)
    le = float(np.argmax(dif[3:112]) + 3.5)
    re = float(np.argmin(dif[112:IMG - 4]) + 112.5)
    width = re - le

    side, conf = _fibula_side(rot, y0, le, re)
    if cfg.side_source == "none":
        flip = False
    elif cfg.side_source == "filename" or side is None:
        # Asumsi tampilan radiologi standar: fibula lutut kanan tampak di kiri gambar.
        # PERIKSA asumsi ini pada beberapa gambar asli sebelum dipakai.
        stem = os.path.splitext(os.path.basename(fname))[0].upper()
        flip = stem.endswith("R") if cfg.side_source != "none" and stem[-1:] in ("L", "R") else False
    else:
        flip = side == "left"

    reasons = []
    if min(dL, dR) < cfg.min_valley_depth:
        reasons.append("lembah_dangkal")
    if abs(np.degrees(theta)) > cfg.max_tilt_deg:
        reasons.append("terlalu_miring")
    if not (cfg.width_range[0] <= width <= cfg.width_range[1]):
        reasons.append("lebar_tidak_wajar")

    c, s = np.cos(theta), np.sin(theta)
    vx = (le + re) / 2.0 - 112.0
    return dict(yL=yL, yR=yR, depth=min(dL, dR), angle=float(np.degrees(theta)), le=le, re=re, width=width,
                scale=KNEE_WIDTH_REF / width if width > 0 else 1.0,
                cx_src=float(c * vx + 112.0), cy_src=float(s * vx + y0), theta=theta,
                fibula=side or "", fibula_conf=conf, flip=bool(flip),
                ok=len(reasons) == 0, reason=";".join(reasons))


def align(img: np.ndarray, p: dict, use_alignment: bool = True) -> np.ndarray:
    """Terapkan penyejajaran (satu kali interpolasi). Gambar gagal tidak diubah posisinya."""
    if use_alignment and p["ok"]:
        out = _sampler(img, p["theta"], (p["cx_src"], p["cy_src"]), p["scale"], CENTER)
    else:
        out = img.astype(np.float32)
    if use_alignment and p["flip"]:
        out = out[:, ::-1]
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ peta wilayah
def region_masks() -> dict:
    """Wilayah anatomi dalam koordinat acuan (setelah sejajar, medial di kiri, lateral di kanan)."""
    jx, jy = CENTER
    xl, xr = jx - KNEE_WIDTH_REF / 2, jx + KNEE_WIDTH_REF / 2
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    box = lambda x0, x1, y0, y1: (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    m = dict(
        tepi_med=box(xl - 20, xl + 20, jy - 30, jy + 30),
        tepi_lat=box(xr - 20, xr + 20, jy - 30, jy + 30),
        celah_med=box(xl + 20, jx, jy - 8, jy + 8),
        celah_lat=box(jx, xr - 20, jy - 8, jy + 8),
        subk=box(xl + 20, xr - 20, jy - 24, jy - 8) | box(xl + 20, xr - 20, jy + 8, jy + 24),
    )
    m["edge"] = m["tepi_med"] | m["tepi_lat"]
    m["joint"] = m["edge"] | m["celah_med"] | m["celah_lat"] | m["subk"]
    return m


def region_grid(mask: np.ndarray, g: int, min_frac: float = 0.15) -> np.ndarray:
    """Sel grid g x g yang dianggap masuk wilayah (cakupan > min_frac)."""
    frac = cv2.resize(mask.astype(np.float32), (g, g), interpolation=cv2.INTER_AREA)
    return frac > min_frac


# ------------------------------------------------------------------ cache
def precompute(df: pd.DataFrame, cfg, name: str):
    """Hitung parameter anatomi + simpan gambar yang sudah sejajar (uint8) untuk satu split."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    tag = "al" if cfg.use_alignment else "raw"
    f_csv = os.path.join(cfg.cache_dir, f"anatomi_{name}.csv")
    f_npy = os.path.join(cfg.cache_dir, f"img_{name}_{tag}.npy")
    if os.path.exists(f_csv) and os.path.exists(f_npy):
        params = pd.read_csv(f_csv)
        if len(params) == len(df) and list(params["id"]) == list(df["id"]):
            return params, np.load(f_npy, mmap_mode="r")
    rows, imgs = [], np.zeros((len(df), IMG, IMG), np.uint8)
    for k, pid in enumerate(df["id"]):
        img = load_gray(pid)
        p = detect(img, cfg, pid)
        imgs[k] = align(img, p, cfg.use_alignment)
        rows.append(dict(id=pid, **p))
    params = pd.DataFrame(rows)
    params.to_csv(f_csv, index=False)
    np.save(f_npy, imgs)
    ok = params["ok"].mean() * 100
    print(f"[anatomi] {name}: {len(df)} gambar, berhasil {ok:.1f}%, sudut rata-rata {params['angle'].abs().mean():.2f} derajat")
    return params, np.load(f_npy, mmap_mode="r")

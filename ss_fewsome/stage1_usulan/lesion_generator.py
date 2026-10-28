"""
Generator lesi sintetis untuk DRA-OA (langkah 1-15 Fase 1 di dokumen panduan).

Masukan: lutut normal yang sudah disejajarkan. Keluaran: gambar berlesi, masker lesi,
dan label keparahan. Label patch lunak dihitung oleh model sesuai ukuran grid fiturnya.

Mode:
  cutmix  anomali semu DRA asli (potongan gambar dipindah + diubah kecerahannya)
  spur    hanya taji dari bank taji (tingkat 1-2)
  oa      taji + penyempitan celah + sklerosis (tingkat 1-4)
"""
import numpy as np
import cv2
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from .config import CENTER, KNEE_WIDTH_REF
from .spur_bank import contour_extent, ROWS, XL, XR

JX, JY = CENTER

# tingkat keparahan -> resep lesi dan label (lihat tabel di Bab 7 dokumen)
LEVELS = {
    1: dict(label=0.25, n_spur=(1, 1), scale=(0.5, 0.7), alpha=(0.5, 0.8), jsn=(0, 0), scl=(0, 0)),
    2: dict(label=0.50, n_spur=(1, 1), scale=(0.8, 1.1), alpha=(1, 1), jsn=(0, 3), scl=(0, 0)),
    3: dict(label=0.75, n_spur=(1, 2), scale=(0.9, 1.2), alpha=(1, 1), jsn=(5, 8), scl=(0, 15)),
    4: dict(label=1.00, n_spur=(2, 3), scale=(1.0, 1.3), alpha=(1, 1), jsn=(8, 12), scl=(25, 40)),
}


def _uniform(rng, lo_hi):
    lo, hi = lo_hi
    return lo if lo == hi else rng.uniform(lo, hi)


def paste_spur(img: np.ndarray, item: dict, rng, scale: float, alpha: float, side=None, bone=None):
    """Tempel satu taji dari bank ke tepi tulang yang sesuai, dengan blending halus."""
    side = side or item["side"]
    bone = bone or item["bone"]
    crop, mask = item["crop"], item["mask"].astype(np.float32)
    if side != item["side"]:  # cerminkan kalau dipindah ke sisi lain
        crop, mask = crop[:, ::-1], mask[:, ::-1]
    h, w = crop.shape
    nh, nw = max(3, int(round(h * scale))), max(3, int(round(w * scale)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR) > 0.5
    if mask.sum() < 3:
        return img, np.zeros_like(img, bool)

    ext, xs, thr = contour_extent(img, side, bone)
    rows = ROWS[bone]
    y_mid = int(np.clip(JY + item["y_rel"], rows[0] + nh // 2, rows[-1] - nh // 2))
    x_edge = float(xs[int(np.clip(y_mid - rows[0], 0, len(rows) - 1))])
    y0 = y_mid - nh // 2
    x0 = int(round(x_edge - nw + 1)) if side == "med" else int(round(x_edge))
    y0, x0 = int(np.clip(y0, 0, 224 - nh)), int(np.clip(x0, 0, 224 - nw))

    out = img.astype(np.float32).copy()
    region = out[y0:y0 + nh, x0:x0 + nw]
    soft = region < thr                                   # hanya menimpa jaringan lunak
    place = mask & soft
    if place.sum() < 3:
        return img, np.zeros_like(img, bool)
    bone_level = np.median(out[max(0, y0 - 4):y0 + nh + 4, int(XL + 10):int(XL + 25)] if side == "med"
                           else out[max(0, y0 - 4):y0 + nh + 4, int(XR - 25):int(XR - 10)])
    tex = crop - crop[mask].mean() + bone_level           # samakan kecerahan dengan tulang setempat
    region[place] = (1 - alpha) * region[place] + alpha * tex[place]
    full = np.zeros(img.shape, bool)
    full[y0:y0 + nh, x0:x0 + nw] = place
    ring = binary_dilation(full, iterations=2) & ~binary_erosion(full, iterations=1)
    blurred = cv2.GaussianBlur(out, (5, 5), 1.2)
    out[ring] = blurred[ring]                             # tepi luar dihaluskan
    return np.clip(out, 0, 255), full


def narrow_joint(img: np.ndarray, amount: float, medial: bool):
    """Penyempitan celah: permukaan tibia di satu kompartemen ditarik ke atas (warping lokal)."""
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    lo, hi = (XL + 4, JX) if medial else (JX, XR - 4)
    t = np.clip((xx - lo) / max(hi - lo, 1), 0, 1)
    wx = np.clip(np.sin(np.pi * t), 0, None) ** 0.6 * ((xx >= lo) & (xx <= hi))  # clip: sin(pi) float32 bisa -1e-8 -> NaN
    gy = np.exp(-((yy - (JY + 7)) / 9.0) ** 2)
    shift = (amount * wx * gy).astype(np.float32)
    out = cv2.remap(img.astype(np.float32), xx, yy + shift, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, shift


def sclerosis(img: np.ndarray, amount: float, medial: bool, jsn_amount: float):
    """Pita tipis di bawah permukaan tibia (satu kompartemen) dibuat lebih terang."""
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    lo, hi = (XL + 6, JX - 6) if medial else (JX + 6, XR - 6)
    surf = JY + 7 - jsn_amount * 0.8
    band = ((xx > lo) & (xx < hi) & (yy > surf) & (yy < surf + 9)).astype(np.float32)
    band = gaussian_filter(band, 1.5)
    return img + amount * band


def cutmix(img: np.ndarray, rng):
    """Anomali semu gaya DRA asli: potongan persegi dipindah, kecerahan diubah."""
    h, w = img.shape
    ph, pw = int(rng.uniform(0.15, 0.35) * h), int(rng.uniform(0.15, 0.35) * w)
    sy, sx = rng.integers(0, h - ph), rng.integers(0, w - pw)
    ty, tx = rng.integers(0, h - ph), rng.integers(0, w - pw)
    patch = img[sy:sy + ph, sx:sx + pw].astype(np.float32) * rng.uniform(0.7, 1.3)
    out = img.astype(np.float32).copy()
    out[ty:ty + ph, tx:tx + pw] = patch
    return np.clip(out, 0, 255)


def make_lesion(base: np.ndarray, bank: list, cfg, rng, level: int = None):
    """Satu contoh latih sintetis: (gambar uint8, masker lesi bool, label keparahan)."""
    base_f = base.astype(np.float32)
    if cfg.pseudo_gen == "cutmix" or not bank:
        out = cutmix(base, rng)
        return out.astype(np.uint8), np.abs(out - base_f) > 8, 1.0
    if level is None:
        level = int(rng.integers(1, 3)) if cfg.pseudo_gen == "spur" else int(rng.integers(1, 5))
    rec = LEVELS[level]
    out = base_f.copy()
    n = int(rng.integers(rec["n_spur"][0], rec["n_spur"][1] + 1))
    medial = rng.random() < cfg.jsn_medial_prob
    for _ in range(n):
        item = bank[int(rng.integers(len(bank)))]
        side = ("med" if medial else "lat") if rng.random() < 0.7 else None
        out, _ = paste_spur(out, item, rng, _uniform(rng, rec["scale"]), _uniform(rng, rec["alpha"]), side=side)
    if cfg.pseudo_gen == "oa":
        j = _uniform(rng, rec["jsn"])
        if j > 0:
            out, _ = narrow_joint(out, j, medial)
        s = _uniform(rng, rec["scl"])
        if s > 0:
            out = sclerosis(out, s, medial, j)
    out = np.clip(out, 0, 255)
    mask = np.abs(out - base_f) > 8
    return out.astype(np.uint8), mask, rec["label"]


def soft_patch_labels(mask: np.ndarray, severity: float, grid: int, gain: float) -> np.ndarray:
    """label patch = min(1, cakupan x gain) x tingkat keparahan."""
    cov = cv2.resize(mask.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    return np.clip(cov * gain, 0, 1) * severity

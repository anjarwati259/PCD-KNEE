"""
Bank taji asli (jembatan antara memory bank dan DRA-OA).

1. Kontur tepi: untuk setiap baris di wilayah tepi (medial/lateral x femur/tibia), cari
   kolom tempat tulang mulai dari arah luar. Disimpan sebagai "jarak keluar" dari tepi acuan.
2. Pita normal: rata-rata +- k SD jarak keluar di setiap baris, dari lutut normal yang sejajar.
3. Lutut OA: baris yang keluar dari pita lebih dari spur_min_depth piksel, minimal
   spur_min_rows baris berturut-turut, dianggap taji.
   Mode "guided": hanya taji yang berada di sel dengan skor memory bank positif.
4. Setiap taji disimpan: potongan piksel, masker, sisi, tulang, posisi baris, kedalaman.
"""
import pickle
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d

from .config import CENTER, KNEE_WIDTH_REF

JX, JY = CENTER
XL, XR = JX - KNEE_WIDTH_REF / 2, JX + KNEE_WIDTH_REF / 2
ROWS = {"fem": np.arange(int(JY - 30), int(JY - 3)), "tib": np.arange(int(JY + 3), int(JY + 31))}
SIDES = ("med", "lat")      # medial = kiri, lateral = kanan (setelah penyejajaran)
SCAN = 34                   # lebar pencarian kontur di sekitar tepi acuan


def _threshold(img, side):
    """Ambang tulang vs jaringan lunak untuk satu sisi."""
    rows = slice(int(JY - 30), int(JY + 30))
    if side == "med":
        soft = np.median(img[rows, int(XL - SCAN):int(XL - SCAN + 8)])
        bone = np.median(img[rows, int(XL + 12):int(XL + 30)])
    else:
        soft = np.median(img[rows, int(XR + SCAN - 8):int(XR + SCAN)])
        bone = np.median(img[rows, int(XR - 30):int(XR - 12)])
    return (soft + bone) / 2.0, soft, bone


def contour_extent(img: np.ndarray, side: str, bone: str):
    """Jarak keluar tepi tulang (piksel, positif = ke luar) untuk setiap baris wilayah."""
    im = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
    thr, _, _ = _threshold(im, side)
    ext = np.zeros(len(ROWS[bone]), np.float32)
    xs = np.zeros(len(ROWS[bone]), np.float32)
    for k, y in enumerate(ROWS[bone]):
        if side == "med":
            line = im[y, int(XL - SCAN):int(XL + SCAN)]
            hit = np.nonzero(line >= thr)[0]
            x = (XL - SCAN + hit[0]) if hit.size else XL
            ext[k], xs[k] = XL - x, x
        else:
            line = im[y, int(XR - SCAN):int(XR + SCAN)][::-1]
            hit = np.nonzero(line >= thr)[0]
            x = (XR + SCAN - 1 - hit[0]) if hit.size else XR
            ext[k], xs[k] = x - XR, x
    return ext, xs, thr


def normal_band(normal_imgs, k_sd: float):
    """Pita normal per (sisi, tulang, baris)."""
    band = {}
    for side in SIDES:
        for bone in ROWS:
            E = np.stack([contour_extent(im, side, bone)[0] for im in normal_imgs])
            band[(side, bone)] = (E.mean(0), E.std(0) + 0.5, E.mean(0) + k_sd * (E.std(0) + 0.5))
    return band


def _cell_positive(score_map, x, y):
    if score_map is None:
        return True
    g = score_map.shape[0]
    i, j = int(np.clip(y / 224 * g, 0, g - 1)), int(np.clip(x / 224 * g, 0, g - 1))
    return score_map[i, j] > 0


def extract_spurs(img, band, cfg, score_map=None, src_id=""):
    """Semua taji yang memenuhi aturan pada satu lutut OA."""
    found = []
    for side in SIDES:
        for bone, rows in ROWS.items():
            ext, xs, thr = contour_extent(img, side, bone)
            limit = band[(side, bone)][2]
            excess = uniform_filter1d(ext - limit, 3)
            out = excess > cfg.spur_min_depth
            k = 0
            while k < len(out):
                if not out[k]:
                    k += 1
                    continue
                s = k
                while k < len(out) and out[k]:
                    k += 1
                e = k  # baris s..e-1
                if e - s < cfg.spur_min_rows:
                    continue
                y_mid = int(rows[(s + e) // 2])
                x_tip = float(xs[s:e].min()) if side == "med" else float(xs[s:e].max())
                if cfg.spur_bank_mode == "guided" and not _cell_positive(score_map, x_tip, y_mid):
                    continue
                y0, y1 = int(rows[s]) - 2, int(rows[e - 1]) + 3
                if side == "med":
                    base = XL - limit[s:e].min()            # batas dalam pita normal (kolom)
                    x0, x1 = int(np.floor(x_tip)) - 2, int(np.ceil(base)) + 2
                else:
                    base = XR + limit[s:e].min()
                    x0, x1 = int(np.floor(base)) - 2, int(np.ceil(x_tip)) + 3
                if x1 - x0 < 3 or y1 - y0 < 3:
                    continue
                crop = img[y0:y1, x0:x1].astype(np.float32)
                mask = crop >= thr - 10
                # hanya bagian di luar pita normal yang dihitung sebagai taji
                for r in range(y1 - y0):
                    yy = y0 + r
                    kk = int(np.clip(yy - rows[0], 0, len(limit) - 1))
                    if side == "med":
                        edge = int(round(XL - limit[kk])) - x0
                        mask[r, max(edge, 0):] = False
                    else:
                        edge = int(round(XR + limit[kk])) - x0
                        mask[r, :max(edge + 1, 0)] = False
                if mask.sum() < 4:
                    continue
                found.append(dict(crop=crop, mask=mask, side=side, bone=bone, y_rel=y_mid - JY,
                                  depth=float(excess[s:e].max()), rows=int(e - s), src=src_id,
                                  base_col=(x1 - x0 - 1) if side == "med" else 0))
    return found


def build(normal_imgs, oa_imgs, oa_ids, cfg, oa_score_maps=None):
    band = normal_band(normal_imgs, cfg.spur_band_sd)
    bank = []
    for k, im in enumerate(oa_imgs):
        sm = oa_score_maps[k] if (oa_score_maps is not None and cfg.spur_bank_mode == "guided") else None
        bank += extract_spurs(np.asarray(im), band, cfg, sm, oa_ids[k])
    return bank, band


def save(bank, path):
    with open(path, "wb") as f:
        pickle.dump(bank, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

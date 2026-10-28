"""
Dataset lutut SINTETIS kecil dengan struktur folder sama seperti dataset Chen (2018),
hanya untuk menguji bahwa kode berjalan. Hasil metriknya tidak bermakna secara klinis.

  python -m stage1_usulan.tools.make_synthetic_dataset --out ./data_sintetis/ --meta ./meta_sintetis/
"""
import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter, map_coordinates

BELTON_SEEDS = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
H = W = 224
YY, XX = np.mgrid[0:H, 0:W].astype(float)
PTS = np.stack([XX.ravel(), YY.ravel()], 1)


def tri(p):
    return Path(p).contains_points(PTS).reshape(H, W)


def knee(rng, grade):
    d = rng.normal(0, 1.5, 4)
    img = np.full((H, W), 62.0) - 14 * np.abs(XX - 112) / 112
    cb = 100 + d[0] + (7 + d[1]) * np.abs(np.sin((XX - 40) / 146 * np.pi * 2))
    img[(XX > 40) & (XX < 186) & (YY < cb)] = 188
    jsn = [0, 0, 3, 7, 11][grade] + rng.uniform(0, 2)
    top = np.full(W, 120.0 + d[3])
    m = (XX[0] > 40) & (XX[0] < 112)
    top[m] -= jsn * np.clip((112 - XX[0][m]) / 40, 0, 1)
    img[((XX > 36) & (XX < 190) & (YY > top[None]) & (YY < 150)) | ((XX > 62) & (XX < 164) & (YY >= 150))] = 184
    img[(XX > 172) & (XX < 186) & (YY > 158)] = 150            # fibula di kanan (lateral)
    spur = [0, 0.35, 0.8, 1.1, 1.4][grade] * rng.uniform(0.8, 1.2)
    if spur > 0:
        L = 15 * spur
        img[tri([(37, 124), (37 - L, 131), (37, 140)])] = 182
        img[tri([(41, 91), (41 - 0.9 * L, 98), (41, 104)])] = 186
    if grade >= 3:
        img[(XX > 40) & (XX < 112) & (YY > top[None]) & (YY < top[None] + 10)] += [0, 0, 0, 15, 35][grade]
    # posisi foto acak
    ang, s, tx, ty = np.deg2rad(rng.uniform(-6, 6)), rng.uniform(0.9, 1.06), rng.uniform(-8, 8), rng.uniform(-6, 6)
    c, si = np.cos(ang), np.sin(ang)
    vx, vy = XX - 112 - tx, YY - 112 - ty
    img = map_coordinates(img, [(-si * vx + c * vy) / s + 112, (c * vx + si * vy) / s + 112], order=1, mode="nearest")
    img = gaussian_filter(img, 1.1) + rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data_sintetis/")
    ap.add_argument("--meta", default="./meta_sintetis/")
    ap.add_argument("--n_train", type=int, nargs=5, default=[160, 50, 60, 50, 20])
    ap.add_argument("--n_val", type=int, nargs=5, default=[20, 10, 10, 8, 4])
    ap.add_argument("--n_test", type=int, nargs=5, default=[40, 20, 25, 20, 8])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    pid = 9000000
    for split, counts in (("train", a.n_train), ("val", a.n_val), ("test", a.n_test)):
        for grade, n in enumerate(counts):
            d = os.path.join(a.out, split, str(grade))
            os.makedirs(d, exist_ok=True)
            for _ in range(n):
                pid += 1
                side = "L" if rng.random() < 0.5 else "R"
                img = knee(rng, grade)
                if side == "R":
                    img = img[:, ::-1]                              # lutut kanan: fibula tampak di kiri
                Image.fromarray(np.ascontiguousarray(img)).save(os.path.join(d, f"{pid}{side}.png"))
    os.makedirs(a.meta, exist_ok=True)
    normals = sorted(os.listdir(os.path.join(a.out, "train", "0")))
    union = set()
    for seed in BELTON_SEEDS:
        r = np.random.default_rng(seed)
        pick = ["0/" + normals[i] for i in r.choice(len(normals), 30, replace=False)]
        union |= set(pick)
        pd.DataFrame({"ind": range(30), "id": pick}).to_csv(os.path.join(a.meta, f"train_seed_{seed}.csv"))
    pd.DataFrame({"id": sorted(union)}).to_csv(os.path.join(a.meta, "train_ids.csv"))
    rows = []
    for grade in range(5):
        for f in sorted(os.listdir(os.path.join(a.out, "train", str(grade)))):
            rows.append(dict(id=f"{grade}/{f}", sim=float(rng.normal(21, 1))))
    sim = pd.DataFrame(rows)
    sim.insert(0, "Unnamed: 0", range(len(sim)))
    sim.to_csv(os.path.join(a.meta, "sim_scores.csv"))
    print(f"Dataset sintetis dibuat di {a.out}, meta di {a.meta}")


if __name__ == "__main__":
    main()

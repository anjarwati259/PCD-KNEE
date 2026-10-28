"""
Pembagian data untuk Stage 1 usulan.

- Lutut normal per seed: dibaca dari meta/train_seed_X.csv milik Belton (30 gambar KL-0).
- Lutut OA per seed: dipilih dari train/2 dan train/3, disimpan sebagai
  meta/train_oa_seed_X.csv dengan format yang sama seperti file Belton.
- ID gambar ditulis persis seperti di kode Belton (data_path + 'train/' + folder + '/' + nama),
  supaya file skor bisa dibaca get_pseudo_labels dan print_ensemble_results.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image


def load_gray(path: str) -> np.ndarray:
    """Baca PNG grayscale sebagai uint8 224x224 (sama seperti Belton: Image.open -> np.asarray)."""
    img = np.asarray(Image.open(path))
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.uint8)


def list_split(data_path: str, split: str) -> pd.DataFrame:
    """Semua gambar di data_path/split/{0..4}/, dengan id berformat Belton."""
    rows = []
    base = os.path.join(data_path, split)
    if not os.path.isdir(base):
        return pd.DataFrame(columns=["id", "label", "rel"])
    for folder in sorted(os.listdir(base)):
        d = os.path.join(base, folder)
        if not os.path.isdir(d) or not folder.isdigit():
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append(dict(id=data_path + split + "/" + folder + "/" + f, label=int(folder), rel=folder + "/" + f))
    return pd.DataFrame(rows)


def belton_normals(meta_path: str, data_path: str, seed: int) -> list:
    """30 lutut normal milik seed ini (kolom ke-3 file Belton berisi id seperti '0/9935439L.png')."""
    ids = pd.read_csv(os.path.join(meta_path, f"train_seed_{seed}.csv")).iloc[:, 2].tolist()
    return [data_path + "train/" + i for i in ids]


def oa_meta_file(meta_path: str, seed: int) -> str:
    return os.path.join(meta_path, f"train_oa_seed_{seed}.csv")


def make_oa_split(meta_path: str, data_path: str, seed: int, n_kl2: int, n_kl3: int, overwrite: bool = False) -> list:
    """Pilih lutut OA (KL-2 dan KL-3) untuk seed ini. Hasil disimpan supaya bisa diulang persis."""
    f = oa_meta_file(meta_path, seed)
    if os.path.exists(f) and not overwrite:
        ids = pd.read_csv(f).iloc[:, 2].tolist()
        return [data_path + "train/" + i for i in ids]
    rng = np.random.default_rng(seed)
    chosen = []
    for grade, n in ((2, n_kl2), (3, n_kl3)):
        d = os.path.join(data_path, "train", str(grade))
        files = sorted(x for x in os.listdir(d) if x.lower().endswith((".png", ".jpg", ".jpeg")))
        if len(files) < n:
            raise ValueError(f"train/{grade} hanya berisi {len(files)} gambar, butuh {n}")
        pick = rng.choice(len(files), size=n, replace=False)
        chosen += [f"{grade}/{files[i]}" for i in sorted(pick)]
    df = pd.DataFrame({"ind": range(len(chosen)), "id": chosen})
    df.to_csv(f)  # kolom: (index), ind, id  -> sama seperti train_seed_X.csv
    return [data_path + "train/" + i for i in chosen]


class SeedData:
    """Semua daftar gambar yang dibutuhkan satu seed."""

    def __init__(self, cfg, seed: int, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame):
        self.seed = seed
        self.normals = belton_normals(cfg.meta_path, cfg.data_path, seed)[: cfg.n_normal]
        self.oa = make_oa_split(cfg.meta_path, cfg.data_path, seed, cfg.n_oa_kl2, cfg.n_oa_kl3)
        self.oa_grade = [int(p.split("/")[-2]) for p in self.oa]
        self.train_df, self.test_df, self.val_df = train_df, test_df, val_df
        missing = [p for p in self.normals + self.oa if p not in set(train_df["id"])]
        if missing:
            raise FileNotFoundError(f"{len(missing)} gambar berlabel tidak ditemukan di data train, contoh: {missing[:2]}")

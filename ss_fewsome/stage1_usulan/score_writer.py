"""
Menulis skor Stage 1 dengan format yang sama persis seperti Stage 1 Belton, supaya
Stage 2 dan 3 (main.py --train_ss 0) bisa langsung membacanya.

Belton menulis (utils.write_results):
  outputs/dfs/ss/ss_training_<model_name>_bs_..._N_30_seed_<seed>_epoch_400
  outputs/dfs/ss/ss_training_<model_name>_bs_..._N_30_seed_<seed>_on_test_set_epoch_400
dengan kolom pertama indeks, lalu 'id', 'label', lalu skor. Stage 2 membaca kolom 'centre_mean'.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

EPOCH = 400  # main.py Belton mencari file Stage 1 dengan akhiran _epoch_400 (TRAIN_PLATEAU_EPOCH)


def robust_z(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826 + 1e-8
    return (x - med) / mad


def to_unit(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Skala ke 0..1 berdasarkan data train (monoton, jadi AUC dan Spearman tidak berubah)."""
    lo, hi = np.min(ref), np.max(ref)
    return (x - lo) / (hi - lo + 1e-8)


def metrics(score, label) -> dict:
    """Metrik yang sama dengan utils.get_metrics Belton."""
    score, label = np.asarray(score, float), np.asarray(label)
    out = dict(spearman=float(stats.spearmanr(score, label)[0]))
    for name, pos in (("auc", label > 0), ("auc_mid", label > 1), ("auc_mid2", label > 2), ("auc_sev", label == 4)):
        out[name] = float(roc_auc_score(pos.astype(int), score)) if 0 < pos.sum() < len(pos) else float("nan")
    return out


def res_name(cfg, seed: int) -> str:
    n = cfg.n_normal
    return f"ss/ss_training_{cfg.model_name}bs_1_task_test_N_{n}_seed_{seed}"


def write_seed(cfg, seed: int, train_df: pd.DataFrame, test_df: pd.DataFrame,
               train_score: np.ndarray, test_score: np.ndarray, extra_train: dict, extra_test: dict):
    base = res_name(cfg, seed)
    for sub in ("dfs/ss", "results/ss"):
        os.makedirs(os.path.join(cfg.out_root, sub), exist_ok=True)
    ref = train_score
    for df, sc, extra, suffix in ((train_df, train_score, extra_train, ""), (test_df, test_score, extra_test, "_on_test_set")):
        out = pd.DataFrame({"id": df["id"].values, "label": df["label"].values,
                            "centre_mean": to_unit(sc, ref)})
        for k, v in extra.items():
            out[k] = v
        out = out.sort_values(by="centre_mean", ascending=False).reset_index(drop=True)
        out.to_csv(os.path.join(cfg.out_root, "dfs", base + suffix + f"_epoch_{EPOCH}"))
        m = metrics(sc, df["label"].values)
        pd.DataFrame([m], index=["centre_mean"]).to_csv(
            os.path.join(cfg.out_root, "results", base + suffix + f"_epoch_{EPOCH}"))
    return metrics(test_score, test_df["label"].values)

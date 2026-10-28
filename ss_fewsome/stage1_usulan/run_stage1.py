"""
Menjalankan Stage 1 usulan untuk semua seed, lalu menulis skor dengan format Belton.

Contoh (dari folder ss_fewsome/):
  python -m stage1_usulan.run_stage1 --preset A9 --data_path ./data/
  python main.py --train_ss 0 --model_name usulan_A9_ --data_path ./data/   # Stage 2-3 Belton

Urutan kerja per seed:
  1. 30 lutut normal (Belton) + 30 lutut OA (KL-2/3) untuk seed ini
  2. memory bank anatomi -> skor + peta (leave-one-out untuk gambar di bank)
  3. bank taji dari 30 lutut OA (dipandu peta memory bank)
  4. latih DRA-OA -> skor + peta
  5. fusi skor -> tulis file outputs/dfs/ss/..._epoch_400
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch

from .config import make_config
from .data_split import list_split, SeedData
from . import anatomy, features, spur_bank
from .memory_bank import MemoryBank
from .train_dra import train_one_seed, score_images
from .score_writer import write_seed, robust_z, metrics
from .features import get_device


def parse():
    p = argparse.ArgumentParser(description="Stage 1 usulan (lokalisasi anatomi + memory bank + bank taji + DRA-OA)")
    p.add_argument("--preset", default="A9", help="A1, A2, A3, A3g, A4, A5, A6, A7, A8, A9")
    p.add_argument("--data_path", default=None)
    p.add_argument("--meta_path", default=None)
    p.add_argument("--out_root", default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--k1_backbone", default=None, choices=["alexnet", "wrn50"])
    p.add_argument("--spur_bank_mode", default=None, choices=["contour", "guided"])
    p.add_argument("--dra_feat_layer", default=None, choices=["layer3", "layer4"])
    p.add_argument("--fuse_w_k1", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--no_pretrained", action="store_true", help="hanya untuk uji tanpa internet")
    p.add_argument("--quick", action="store_true", help="uji cepat: training sangat pendek")
    return p.parse_args()


def main():
    a = parse()
    over = {k: getattr(a, k) for k in ("data_path", "meta_path", "out_root", "cache_dir", "model_name", "device", "seeds",
                                        "k1_backbone", "spur_bank_mode", "dra_feat_layer", "fuse_w_k1", "epochs")}
    if a.no_pretrained:
        over["pretrained"] = False
    if a.quick:
        over.update(quick=True, epochs=a.epochs or 2, steps_per_epoch=3, batch_size=12)
    cfg = make_config(a.preset, **over)
    os.makedirs(cfg.out_root, exist_ok=True)
    print(f"Preset {a.preset} | model_name '{cfg.model_name}' | memory bank: {cfg.k1_mode} | DRA-OA: {cfg.use_dra} | fusi: {cfg.fuse}")

    # ---------------- data + anatomi (sekali untuk semua seed) ----------------
    tr, te, va = list_split(cfg.data_path, "train"), list_split(cfg.data_path, "test"), list_split(cfg.data_path, "val")
    print(f"Data: train {len(tr)}, val {len(va)}, test {len(te)}")
    _, img_tr = anatomy.precompute(tr, cfg, "train")
    _, img_te = anatomy.precompute(te, cfg, "test")
    img_va = anatomy.precompute(va, cfg, "val")[1] if len(va) else None
    regions = anatomy.region_masks()
    row_of = {p: k for k, p in enumerate(tr["id"])}

    use_k1 = cfg.k1_mode != "none"
    if use_k1:
        f_tr = features.extract(img_tr, cfg, "train")
        f_te = features.extract(img_te, cfg, "test")
        f_va = features.extract(img_va, cfg, "val") if img_va is not None else None
        g = f_tr.shape[1]
        joint_grid = anatomy.region_grid(regions["joint"], g)

    summary = []
    for seed in cfg.seeds:
        t0 = time.time()
        sd = SeedData(cfg, seed, tr, te, va)
        n_rows = [row_of[p] for p in sd.normals]
        o_rows = [row_of[p] for p in sd.oa]
        res = {"train": {}, "test": {}, "val": {}}

        # ---------------- memory bank anatomi ----------------
        oa_maps = None
        if use_k1:
            mb = MemoryBank(cfg, f_tr[n_rows], f_tr[o_rows], sd.oa_grade, joint_grid, get_device(cfg.device))
            s_tr, m_tr = mb.score_all(f_tr, list(tr["id"]), sd.normals, sd.oa)
            s_te, _ = mb.score_all(f_te, list(te["id"]), [], [])
            res["train"]["k1"], res["test"]["k1"] = s_tr, s_te
            if f_va is not None:
                res["val"]["k1"] = mb.score_all(f_va, list(va["id"]), [], [])[0]
            oa_maps = m_tr[o_rows]
            kept = int(mb.keep.sum()) if mb.keep is not None else 0
            print(f"[seed {seed}] memory bank: batas saringan {mb.T:.3f}, patch OA disimpan {kept}")

        # ---------------- bank taji + DRA-OA ----------------
        if cfg.use_dra:
            normals = np.stack([img_tr[r] for r in n_rows])
            oa = np.stack([img_tr[r] for r in o_rows])
            bank = []
            if cfg.pseudo_gen in ("spur", "oa"):
                bank, _ = spur_bank.build(normals, oa, sd.oa, cfg, oa_maps)
                os.makedirs(cfg.cache_dir, exist_ok=True)
                spur_bank.save(bank, os.path.join(cfg.cache_dir, f"bank_taji_{cfg.model_name}seed_{seed}.pkl"))
                print(f"[seed {seed}] bank taji: {len(bank)} taji dari {len(oa)} lutut OA (mode {cfg.spur_bank_mode})")
                if not bank:
                    print("  peringatan: bank taji kosong, generator memakai cutmix untuk seed ini")
            model = train_one_seed(cfg, normals, oa, bank, regions["edge"], seed)
            for split, imgs in (("train", img_tr), ("test", img_te), ("val", img_va)):
                if imgs is None:
                    continue
                sc, maps = score_images(model, imgs)
                res[split]["dra"], res[split]["sev"] = sc["anomaly"], sc["sev"]
                if split == "train" and cfg.save_maps:
                    np.save(os.path.join(cfg.cache_dir, f"peta_dra_{cfg.model_name}seed_{seed}_train.npy"), maps)
            torch.save(model.state_dict(), os.path.join(cfg.cache_dir, f"dra_oa_{cfg.model_name}seed_{seed}.pt"))

        # ---------------- fusi skor ----------------
        def final(split):
            r = res[split]
            if cfg.use_dra and use_k1 and cfg.fuse:
                zd = robust_z(r["dra"], res["train"]["dra"])
                zk = robust_z(r["k1"], res["train"]["k1"])
                return (1 - cfg.fuse_w_k1) * zd + cfg.fuse_w_k1 * zk
            return r["dra"] if cfg.use_dra else r["k1"]

        extra = lambda split: {k: v for k, v in res[split].items()}
        m_test = write_seed(cfg, seed, tr, te, final("train"), final("test"), extra("train"), extra("test"))
        m_val = metrics(final("val"), va["label"].values) if len(va) else {}
        summary.append(dict(seed=seed, **{f"test_{k}": v for k, v in m_test.items()}, **{f"val_{k}": v for k, v in m_val.items()}))
        print(f"[seed {seed}] selesai {time.time() - t0:.0f} dtk | test AUC_KL {m_test['auc_mid']:.3f}  SRC {m_test['spearman']:.3f}"
              + (f" | val AUC_KL {m_val['auc_mid']:.3f}" if m_val else ""))

    df = pd.DataFrame(summary)
    f = os.path.join(cfg.out_root, f"ringkasan_stage1_{cfg.model_name}.csv")
    df.to_csv(f, index=False)
    print("\nRata-rata antar seed (test):")
    print(df[[c for c in df.columns if c.startswith("test_")]].agg(["mean", "std"]).round(3).to_string())
    with open(os.path.join(cfg.out_root, f"config_{cfg.model_name}.json"), "w") as fh:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()}, fh, indent=1)
    print(f"\nRingkasan: {f}")
    print(f"Lanjut ke Stage 2-3 Belton:\n  python main.py --train_ss 0 --model_name {cfg.model_name} --data_path {cfg.data_path}")


if __name__ == "__main__":
    main()

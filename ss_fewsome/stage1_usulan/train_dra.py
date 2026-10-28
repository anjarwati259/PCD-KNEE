"""
Training dan inferensi DRA-OA untuk satu seed.

Setiap langkah mengambil satu batch campuran (normal, OA asli, lesi sintetis), lalu
setiap kepala hanya belajar dari jenis contoh yang menjadi tugasnya:

  kepala      normal      OA asli      sintetis
  normal      positif     negatif      negatif     (deviation loss, positif = normal)
  OA asli     0           1 (top-K tepi)   -
  sintetis    0           -            1  (+ BCE label patch lunak)
  residual    0           1            1
  keparahan   target 0    -            target tingkat (+ ranking loss)
"""
import numpy as np
import torch
import torch.nn.functional as F

from .dra_oa import DRAOA, deviation_loss
from .lesion_generator import make_lesion, soft_patch_labels
from .anatomy import region_grid
from .features import get_device


def _augment(img: np.ndarray, rng) -> np.ndarray:
    """Variasi ringan yang tetap dianggap normal: kecerahan, kontras, geser kecil."""
    x = img.astype(np.float32)
    x = (x - x.mean()) * rng.uniform(0.9, 1.1) + x.mean() * rng.uniform(0.92, 1.08)
    dy, dx = rng.integers(-3, 4, size=2)
    x = np.roll(x, (int(dy), int(dx)), axis=(0, 1))
    return np.clip(x, 0, 255).astype(np.uint8)


def train_one_seed(cfg, normals: np.ndarray, oa: np.ndarray, bank: list, edge_mask: np.ndarray, seed: int, log=print):
    dev = get_device(cfg.device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DRAOA(cfg).to(dev)
    g = model.grid_size()
    edge_grid = torch.from_numpy(region_grid(edge_mask, g)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    nb = cfg.batch_size
    n_norm = int(round(nb * cfg.batch_mix[0]))
    n_oa = int(round(nb * cfg.batch_mix[1]))
    n_syn = nb - n_norm - n_oa
    steps = cfg.epochs * cfg.steps_per_epoch

    for step in range(steps):
        model.train()
        imgs, kind, sev_t, patch_t = [], [], [], []
        for _ in range(n_norm):
            imgs.append(_augment(normals[rng.integers(len(normals))], rng)); kind.append(0); sev_t.append(0.0)
            patch_t.append(np.zeros((g, g), np.float32))
        for _ in range(n_oa):
            imgs.append(_augment(oa[rng.integers(len(oa))], rng)); kind.append(1); sev_t.append(-1.0)
            patch_t.append(np.zeros((g, g), np.float32))
        for _ in range(n_syn):
            les, mask, sv = make_lesion(normals[rng.integers(len(normals))], bank, cfg, rng)
            imgs.append(_augment(les, rng)); kind.append(2); sev_t.append(sv)
            patch_t.append(soft_patch_labels(mask, sv, g, cfg.soft_label_gain).astype(np.float32) if cfg.soft_patch_labels
                           else np.zeros((g, g), np.float32))
        x = torch.from_numpy(np.stack(imgs)).to(dev)
        kind = torch.tensor(kind, device=dev)
        sev_t = torch.tensor(sev_t, device=dev, dtype=torch.float32)
        patch_t = torch.from_numpy(np.stack(patch_t)).to(dev)

        # peta fitur acuan untuk kepala residual: rata-rata beberapa lutut normal (tanpa augmentasi)
        ref_idx = rng.choice(len(normals), size=min(cfg.n_ref_residual, len(normals)), replace=False)
        with torch.no_grad():
            ref = model.features(torch.from_numpy(np.stack([normals[i] for i in ref_idx])).to(dev)).mean(0)

        out = model(x, ref, edge_grid)
        is_n, is_o, is_s = kind == 0, kind == 1, kind == 2
        m = cfg.dev_margin
        loss_norm = deviation_loss(out["s_norm"], is_n.float(), m)
        sel = is_n | is_o
        loss_seen = deviation_loss(out["s_seen"][sel], is_o[sel].float(), m)
        sel = is_n | is_s
        loss_pseudo = deviation_loss(out["s_pseudo"][sel], is_s[sel].float(), m)
        if cfg.soft_patch_labels:
            loss_pseudo = loss_pseudo + F.binary_cross_entropy_with_logits(out["pseudo_map"][sel], patch_t[sel])
        loss_res = deviation_loss(out["s_res"], (~is_n).float(), m)
        loss = loss_norm + loss_seen + loss_pseudo + loss_res
        if cfg.severity_head:
            sel = is_n | is_s
            loss_sev = F.mse_loss(out["sev"][sel], sev_t[sel])
            if cfg.ranking_loss and is_s.sum() > 1:
                sv, tv = out["sev"][is_s], sev_t[is_s]
                i, j = torch.triu_indices(len(sv), len(sv), 1, device=dev)
                diff = torch.sign(tv[i] - tv[j])
                keep = diff != 0
                if keep.any():
                    loss_sev = loss_sev + F.margin_ranking_loss(sv[i][keep], sv[j][keep], diff[keep], margin=0.05)
            loss = loss + loss_sev
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg.steps_per_epoch) == 0 or step == steps - 1:
            log(f"[DRA-OA seed {seed}] langkah {step + 1}/{steps}  loss {loss.item():.3f} "
                f"(normal {loss_norm.item():.2f}, OA {loss_seen.item():.2f}, sintetis {loss_pseudo.item():.2f}, residual {loss_res.item():.2f})")

    # acuan tetap untuk inferensi: rata-rata peta fitur semua lutut normal
    model.eval()
    with torch.no_grad():
        feats = [model.features(torch.from_numpy(np.array(normals[i:i + 16])).to(dev)) for i in range(0, len(normals), 16)]
        model.ref_feat = torch.cat(feats).mean(0)
    model.edge_grid = edge_grid
    return model


@torch.no_grad()
def score_images(model, images: np.ndarray, batch: int = 32):
    """Skor semua gambar dengan bobot terkunci."""
    dev = next(model.parameters()).device
    model.eval()
    keys = ("anomaly", "s_seen", "s_pseudo", "s_res", "s_norm", "sev")
    res = {k: [] for k in keys}
    maps = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.array(images[i:i + batch])).to(dev)
        o = model(x, model.ref_feat, model.edge_grid)
        for k in keys:
            res[k].append(o[k].cpu().numpy())
        maps.append((o["seen_map"] + o["pseudo_map"]).cpu().numpy().astype(np.float16))
    return {k: np.concatenate(v) for k, v in res.items()}, np.concatenate(maps)

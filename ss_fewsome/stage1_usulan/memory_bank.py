"""
Memory bank berbasis anatomi (eks-Kontribusi 1).

Mode:
  global       PatchCore asli: satu bank berisi semua patch normal, lokasi diabaikan.
  perloc       bank per lokasi: patch hanya dibandingkan dengan patch normal di lokasi yang sama
               (+ tetangga sejauh k1_neigh sel).
  dual         + bank OA per lokasi (disaring). skor patch = d_N - d_OA.
  dual_graded  + bank OA dipisah KL-2 dan KL-3: skor += alpha * (d_OA2 - d_OA3).

Gambar yang ada di dalam bank dinilai secara leave-one-out (patch miliknya sendiri dikeluarkan),
supaya batas di Stage 2 Belton tidak rusak.
"""
import numpy as np
import torch


def _agg(score_maps: torch.Tensor, how: str) -> torch.Tensor:
    flat = score_maps.flatten(1)
    if how == "max":
        return flat.max(1).values
    k = int(how.replace("top", "")) if how.startswith("top") else 3
    return flat.topk(min(k, flat.shape[1]), dim=1).values.mean(1)


class MemoryBank:
    def __init__(self, cfg, f_normal: np.ndarray, f_oa: np.ndarray, oa_grades, joint_grid: np.ndarray, device):
        self.cfg, self.dev = cfg, device
        self.mode = cfg.k1_mode
        self.N = torch.from_numpy(np.asarray(f_normal, np.float32)).to(device)    # [Nn,G,G,D]
        self.O = torch.from_numpy(np.asarray(f_oa, np.float32)).to(device)        # [No,G,G,D]
        self.G = self.N.shape[1]
        self.r = cfg.k1_neigh
        self.oa_grades = np.asarray(oa_grades)
        self.joint = torch.from_numpy(joint_grid.astype(bool)).to(device)
        # batas saringan dari jarak leave-one-out antar lutut normal
        loo = torch.cat([self._d_bank(self.N[k:k + 1], self.N, exclude=[k]).flatten() for k in range(len(self.N))])
        loo = loo[torch.isfinite(loo)]
        self.T = float(torch.quantile(loo, cfg.k1_filter_pct / 100.0)) if loo.numel() else 0.0
        # saringan bank OA: jauh dari normal (+ hanya wilayah sendi)
        if self.mode in ("dual", "dual_graded"):
            d_oa_to_n = torch.cat([self._d_bank(self.O[k:k + 1], self.N) for k in range(len(self.O))])  # [No,G,G]
            keep = d_oa_to_n > self.T
            if cfg.k1_region_only:
                keep &= self.joint[None]
            self.keep = keep
            self.d_oa_to_n = d_oa_to_n
        else:
            self.keep = None

    # ---------------------------------------------------------------- jarak
    def _d_bank(self, q: torch.Tensor, bank: torch.Tensor, exclude=None, keep=None) -> torch.Tensor:
        """Jarak tiap patch q [B,G,G,D] ke patch terdekat di bank [N,G,G,D] pada lokasi yang sama (+tetangga)."""
        B, G = q.shape[0], self.G
        if self.mode == "global":
            flat = bank.reshape(-1, bank.shape[-1])                                   # [N*G*G, D]
            d = torch.cdist(q.reshape(B, G * G, -1), flat[None].expand(B, -1, -1))    # [B,G*G,N*G*G]
            if exclude is not None:
                for b, k in enumerate(exclude):
                    if k is not None and k >= 0:
                        d[b][:, k * G * G:(k + 1) * G * G] = float("inf")
            return d.min(-1).values.view(B, G, G)
        best = torch.full((B, G, G), float("inf"), device=q.device)
        for di in range(-self.r, self.r + 1):
            for dj in range(-self.r, self.r + 1):
                sh = torch.roll(bank, shifts=(-di, -dj), dims=(1, 2))                # sh[:,i,j] = bank[:,i+di,j+dj]
                ii = torch.arange(G, device=q.device)
                valid = ((ii + di >= 0) & (ii + di < G))[:, None] & ((ii + dj >= 0) & (ii + dj < G))[None, :]
                d = torch.sqrt(((q[:, None] - sh[None]) ** 2).sum(-1) + 1e-12)       # [B,N,G,G]
                d = d.masked_fill(~valid[None, None], float("inf"))
                if keep is not None:
                    kk = torch.roll(keep, shifts=(-di, -dj), dims=(1, 2))
                    d = d.masked_fill(~kk[None], float("inf"))
                if exclude is not None:
                    for b, k in enumerate(exclude):
                        if k is not None and k >= 0:
                            d[b, k] = float("inf")
                best = torch.minimum(best, d.min(1).values)
        return best

    # ---------------------------------------------------------------- skor
    @torch.no_grad()
    def score(self, q: np.ndarray, excl_n=None, excl_o=None):
        """q: [B,G,G,D]. excl_n/excl_o: indeks gambar di bank normal/OA yang harus dikeluarkan (atau None)."""
        q = torch.from_numpy(np.asarray(q, np.float32)).to(self.dev)
        B = q.shape[0]
        excl_n = excl_n or [None] * B
        excl_o = excl_o or [None] * B
        dN = self._d_bank(q, self.N, exclude=excl_n)
        if self.mode in ("global", "perloc"):
            smap, dOA = dN, torch.full_like(dN, float("nan"))
        else:
            dOA = self._d_bank(q, self.O, exclude=excl_o, keep=self.keep)
            has = torch.isfinite(dOA)
            smap = torch.where(has, dN - dOA, torch.zeros_like(dN))
            if self.mode == "dual_graded":
                g2 = torch.from_numpy(self.oa_grades == 2).to(self.dev)
                k2 = self.keep & g2[:, None, None]
                k3 = self.keep & ~g2[:, None, None]
                d2 = self._d_bank(q, self.O, exclude=excl_o, keep=k2)
                d3 = self._d_bank(q, self.O, exclude=excl_o, keep=k3)
                ok = torch.isfinite(d2) & torch.isfinite(d3)
                smap = smap + self.cfg.k1_graded_alpha * torch.where(ok, d2 - d3, torch.zeros_like(d2))
            dOA = torch.where(has, dOA, torch.full_like(dOA, float("nan")))
        img = _agg(smap, self.cfg.k1_agg)
        return img.cpu().numpy(), smap.cpu().numpy(), dN.cpu().numpy(), dOA.cpu().numpy()

    def score_all(self, feats: np.ndarray, ids, normal_ids, oa_ids, batch: int = 8):
        """Skor semua gambar satu split; gambar yang ada di bank dinilai leave-one-out."""
        n_idx = {p: k for k, p in enumerate(normal_ids)}
        o_idx = {p: k for k, p in enumerate(oa_ids)}
        out_img, out_map = [], []
        for i in range(0, len(ids), batch):
            sub = ids[i:i + batch]
            s, m, _, _ = self.score(feats[i:i + batch], [n_idx.get(p) for p in sub], [o_idx.get(p) for p in sub])
            out_img.append(s)
            out_map.append(m)
        return np.concatenate(out_img), np.concatenate(out_map)

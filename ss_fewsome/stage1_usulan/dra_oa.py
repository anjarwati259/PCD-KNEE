"""
Model DRA-OA (eks-Kontribusi 2).

Backbone ResNet-18 (seperti DRA asli) -> peta fitur [C, h, w], lalu lima kepala:
  normal     : rata-rata seluruh patch -> 1 skor        (dilatih: positif = normal)
  OA asli    : skor per patch -> top-K, hanya di wilayah tepi (mil_region="edge")
  sintetis   : skor per patch -> top-K (+ label patch lunak dari generator)
  residual   : selisih dengan peta fitur acuan lutut normal -> skor per patch -> top-K
  keparahan  : rata-rata patch -> MLP -> sigmoid (0..1)       (kepala baru)
Skor anomali = OA asli + sintetis + residual - normal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def deviation_loss(score: torch.Tensor, y: torch.Tensor, margin: float = 5.0) -> torch.Tensor:
    """Deviation loss (Pang dkk. 2019, dipakai DRA). y=1 -> didorong >= margin deviasi."""
    ref = torch.randn(5000, device=score.device)
    dev = (score - ref.mean()) / ref.std()
    return ((1 - y) * dev.abs() + y * F.relu(margin - dev)).mean()


def topk_mean(maps: torch.Tensor, frac: float, allowed: torch.Tensor = None) -> torch.Tensor:
    """maps: [B,h,w]; rata-rata K patch tertinggi (K = frac x jumlah patch yang diizinkan)."""
    B = maps.shape[0]
    flat = maps.reshape(B, -1)
    if allowed is not None:
        a = allowed.reshape(-1)
        flat = flat[:, a]
    k = max(1, int(round(frac * flat.shape[1])))
    return flat.topk(k, dim=1).values.mean(1)


class DRAOA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if cfg.pretrained else None)
        layers = [net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3]
        if cfg.dra_feat_layer == "layer4":
            layers.append(net.layer4)
        self.backbone = nn.Sequential(*layers)
        C = 512 if cfg.dra_feat_layer == "layer4" else 256
        self.C = C
        self.h_normal = nn.Linear(C, 1)
        self.h_seen = nn.Conv2d(C, 1, 1)
        self.h_pseudo = nn.Conv2d(C, 1, 1)
        self.h_res = nn.Conv2d(C, 1, 1)
        self.h_sev = nn.Sequential(nn.Linear(C, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("ref_feat", torch.zeros(1))      # diisi setelah training
        self.register_buffer("edge_grid", torch.zeros(1, dtype=torch.bool))

    # ----------------------------------------------------------------
    def features(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
        x = (x - _MEAN.to(x.device)) / _STD.to(x.device)
        return self.backbone(x)                                # [B,C,h,w]

    def forward(self, x_uint8: torch.Tensor, ref_feat: torch.Tensor, edge_grid: torch.Tensor):
        f = self.features(x_uint8)
        g = f.mean((2, 3))
        seen_map = self.h_seen(f)[:, 0]
        pseudo_map = self.h_pseudo(f)[:, 0]
        res_map = self.h_res(ref_feat[None] - f)[:, 0]
        allowed = edge_grid if self.cfg.mil_region == "edge" else None
        out = dict(
            s_norm=self.h_normal(g)[:, 0],
            s_seen=topk_mean(seen_map, self.cfg.topk_frac, allowed),
            s_pseudo=topk_mean(pseudo_map, self.cfg.topk_frac),
            s_res=topk_mean(res_map, self.cfg.topk_frac),
            sev=torch.sigmoid(self.h_sev(g)[:, 0]),
            seen_map=seen_map, pseudo_map=pseudo_map, res_map=res_map,
        )
        out["anomaly"] = out["s_seen"] + out["s_pseudo"] + out["s_res"] - out["s_norm"]
        return out

    def grid_size(self) -> int:
        return 7 if self.cfg.dra_feat_layer == "layer4" else 14

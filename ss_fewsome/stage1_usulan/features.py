"""
Ekstraksi "sidik jari" patch untuk memory bank.

- "alexnet": sama seperti Stage 1 Belton (AlexNet .features -> 6x6 -> patch 3x3 stride 1
  -> rata-rata) sehingga grid 4x4 x 256. Dipakai untuk perbandingan yang setara dengan Belton.
- "wrn50": gaya PatchCore (WideResNet-50, layer2 + layer3, agregasi tetangga 3x3),
  diringkas ke grid G x G dan D dimensi.
Fitur tidak bergantung pada seed, jadi dihitung sekali per split lalu disimpan.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _weights(pretrained, enum):
    return enum.DEFAULT if pretrained else None


class PatchFeatures(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg.k1_backbone
        if self.kind == "alexnet":
            self.net = models.alexnet(weights=_weights(cfg.pretrained, models.AlexNet_Weights)).features
            self.grid, self.dim = 4, 256
        elif self.kind == "wrn50":
            net = models.wide_resnet50_2(weights=_weights(cfg.pretrained, models.Wide_ResNet50_2_Weights))
            self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
            self.layer2, self.layer3 = net.layer2, net.layer3
            self.grid, self.dim = cfg.k1_grid, cfg.k1_dim
        else:
            raise ValueError(f"backbone tidak dikenal: {self.kind}")
        self.eval()

    @torch.no_grad()
    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """x: [B,224,224] uint8 -> [B, G, G, D] float32."""
        x = x_uint8.float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
        if self.kind == "alexnet":
            f = self.net(x)                                            # [B,256,6,6]
            f = F.unfold(f, kernel_size=3, stride=1)                   # [B,256*9,16]
            B = f.shape[0]
            f = f.view(B, 256, 9, 16).mean(2)                          # rata-rata 3x3 (sama dengan Belton)
            return f.permute(0, 2, 1).reshape(B, 4, 4, 256)
        x = (x - _MEAN.to(x.device)) / _STD.to(x.device)
        h = self.stem(x)
        f2 = self.layer2(h)                                            # [B,512,28,28]
        f3 = self.layer3(f2)                                           # [B,1024,14,14]
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)
        f = F.avg_pool2d(f, 3, stride=1, padding=1)                    # agregasi tetangga ala PatchCore
        f = F.adaptive_avg_pool2d(f, (self.grid, self.grid))           # [B,1536,G,G]
        B, C, G, _ = f.shape
        f = f.permute(0, 2, 3, 1).reshape(B * G * G, 1, C)
        f = F.adaptive_avg_pool1d(f, self.dim).reshape(B, G, G, self.dim)  # ringkas dimensi
        return f


def extract(images: np.ndarray, cfg, name: str, batch: int = 32) -> np.ndarray:
    """Fitur semua gambar satu split, dengan cache di disk (float16)."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    tag = "al" if cfg.use_alignment else "raw"
    g = 4 if cfg.k1_backbone == "alexnet" else cfg.k1_grid
    d = 256 if cfg.k1_backbone == "alexnet" else cfg.k1_dim
    pre = "pt" if cfg.pretrained else "acak"
    f = os.path.join(cfg.cache_dir, f"fitur_{cfg.k1_backbone}_{pre}_g{g}_d{d}_{name}_{tag}.npy")
    if os.path.exists(f):
        arr = np.load(f, mmap_mode="r")
        if arr.shape[0] == len(images):
            return arr
    dev = get_device(cfg.device)
    torch.manual_seed(0)  # supaya bobot acak (mode uji tanpa pretrained) sama di setiap pemanggilan
    model = PatchFeatures(cfg).to(dev)
    out = np.zeros((len(images), g, g, d), np.float16)
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.array(images[i:i + batch])).to(dev)
        out[i:i + batch] = model(x).cpu().numpy().astype(np.float16)
    np.save(f, out)
    return np.load(f, mmap_mode="r")

"""
Pengaturan Stage 1 usulan.

Semua angka yang perlu diubah saat eksperimen dikumpulkan di sini, supaya modul lain
cukup membaca satu objek `Config`. Preset ablation (A1 sampai A9) didefinisikan di
bagian bawah file dan hanya menyalakan atau mematikan komponen.
"""
from dataclasses import dataclass, field, replace
from typing import List, Tuple

# Seed yang sama persis dengan main.py Belton (ss_training)
BELTON_SEEDS = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]

IMG = 224                      # ukuran gambar dataset Chen
CENTER = (112.0, 112.0)        # titik tengah sendi setelah penyejajaran (x, y)
KNEE_WIDTH_REF = 150.0         # lebar lutut setelah penyejajaran (piksel)


@dataclass
class Config:
    # ---------------- lokasi data dan output ----------------
    data_path: str = "./data/"            # berisi train/, val/, test/ dengan subfolder 0..4
    meta_path: str = "../meta/"           # berisi train_seed_X.csv milik Belton
    out_root: str = "./outputs/"          # sama dengan main.py Belton
    cache_dir: str = "./outputs/stage1_usulan_cache/"
    model_name: str = "usulan"            # dipakai sebagai --model_name di main.py Belton
    seeds: List[int] = field(default_factory=lambda: list(BELTON_SEEDS))
    device: str = "cuda:0"

    # ---------------- data berlabel per seed ----------------
    n_normal: int = 30                    # dari meta/train_seed_X.csv (Belton)
    n_oa_kl2: int = 15                    # diambil dari train/2
    n_oa_kl3: int = 15                    # diambil dari train/3

    # ---------------- lokalisasi anatomi ----------------
    use_alignment: bool = True
    row_search: Tuple[int, int] = (60, 175)   # area pencarian garis sendi (baris)
    smooth_rows: int = 9                       # jendela penghalusan profil baris
    smooth_cols: int = 5                       # jendela penghalusan profil kolom
    band_inner: int = 7                        # pita tepi: jarak dari garis sendi
    band_outer: int = 16
    max_tilt_deg: float = 15.0
    min_valley_depth: float = 8.0              # selisih kecerahan minimal lembah vs sekitarnya
    width_range: Tuple[float, float] = (80.0, 215.0)
    side_source: str = "fibula"                # "fibula", "filename", atau "none"

    # ---------------- memory bank anatomi (eks-Kontribusi 1) ----------------
    k1_mode: str = "dual"            # "none", "global" (PatchCore asli), "perloc", "dual", "dual_graded"
    k1_backbone: str = "alexnet"     # "alexnet" (setara Belton) atau "wrn50" (ala PatchCore)
    k1_grid: int = 14                # ukuran grid untuk wrn50 (alexnet selalu 4x4 seperti Belton)
    k1_dim: int = 384                # dimensi fitur wrn50 setelah diringkas
    k1_neigh: int = 1                # toleransi lokasi: 0 = sel yang sama, 1 = sel tetangga 3x3
    k1_filter_pct: float = 95.0      # batas saringan bank OA (persentil jarak normal-normal)
    k1_region_only: bool = True      # bank OA hanya dari wilayah sendi
    k1_agg: str = "top3"             # "max" atau "topK" (contoh "top3")
    k1_graded_alpha: float = 0.5     # bobot pembeda KL-2 vs KL-3 untuk mode dual_graded
    pretrained: bool = True

    # ---------------- bank taji ----------------
    spur_bank_mode: str = "guided"   # "contour" (kontur saja) atau "guided" (dipandu memory bank)
    spur_min_rows: int = 3           # tonjolan minimal sekian baris berturut-turut
    spur_min_depth: float = 2.0      # minimal sekian piksel keluar dari pita normal
    spur_band_sd: float = 2.0        # lebar pita normal: rata-rata +- sekian SD

    # ---------------- DRA-OA (eks-Kontribusi 2) ----------------
    use_dra: bool = True
    dra_feat_layer: str = "layer3"   # "layer3" (14x14) atau "layer4" (7x7)
    mil_region: str = "edge"         # "edge" (dibatasi wilayah tepi) atau "all" (DRA asli)
    topk_frac: float = 0.1           # K = 10% patch, seperti DRA
    pseudo_gen: str = "oa"           # "cutmix" (DRA asli), "spur", atau "oa" (taji + JSN + sklerosis)
    soft_patch_labels: bool = True
    severity_head: bool = True
    ranking_loss: bool = True
    n_ref_residual: int = 5          # jumlah lutut normal acuan untuk kepala residual
    epochs: int = 30                 # pengaturan DRA: 30 epoch x 20 iterasi, batch 48
    steps_per_epoch: int = 20
    batch_size: int = 48
    batch_mix: Tuple[float, float, float] = (0.5, 0.25, 0.25)  # normal, OA asli, sintetis
    lr: float = 2e-4
    weight_decay: float = 1e-2
    dev_margin: float = 5.0          # margin deviation loss
    soft_label_gain: float = 6.0     # label patch = min(1, cakupan x gain) x tingkat
    jsn_medial_prob: float = 0.75    # proporsi penyempitan di sisi medial

    # ---------------- fusi skor ----------------
    fuse: bool = True
    fuse_w_k1: float = 0.5           # skor akhir = (1-w) z(DRA) + w z(K1), dipilih di data val

    # ---------------- lain-lain ----------------
    num_workers: int = 0
    save_maps: bool = True           # simpan peta skor patch untuk analisis
    quick: bool = False              # mode uji cepat


# Preset ablation, sesuai tabel A0-A9 di rencana eksperimen.
# A0 adalah Stage 1 Belton asli (jalankan main.py Belton seperti biasa).
PRESETS = {
    "A1": dict(use_alignment=False, k1_mode="global", use_dra=False, fuse=False),
    "A2": dict(use_alignment=True, k1_mode="perloc", use_dra=False, fuse=False),
    "A3": dict(use_alignment=True, k1_mode="dual", use_dra=False, fuse=False),
    "A3g": dict(use_alignment=True, k1_mode="dual_graded", use_dra=False, fuse=False),
    "A4": dict(use_alignment=True, k1_mode="none", use_dra=True, mil_region="all", pseudo_gen="cutmix",
               soft_patch_labels=False, severity_head=False, ranking_loss=False, fuse=False),
    "A5": dict(use_alignment=True, k1_mode="none", use_dra=True, mil_region="edge", pseudo_gen="cutmix",
               soft_patch_labels=False, severity_head=False, ranking_loss=False, fuse=False),
    "A6": dict(use_alignment=True, k1_mode="dual", use_dra=True, mil_region="edge", pseudo_gen="spur",
               soft_patch_labels=False, severity_head=False, ranking_loss=False, fuse=False),
    "A7": dict(use_alignment=True, k1_mode="dual", use_dra=True, mil_region="edge", pseudo_gen="oa",
               soft_patch_labels=False, severity_head=False, ranking_loss=False, fuse=False),
    "A8": dict(use_alignment=True, k1_mode="dual", use_dra=True, mil_region="edge", pseudo_gen="oa",
               soft_patch_labels=True, severity_head=True, ranking_loss=True, fuse=False),
    "A9": dict(use_alignment=True, k1_mode="dual", use_dra=True, mil_region="edge", pseudo_gen="oa",
               soft_patch_labels=True, severity_head=True, ranking_loss=True, fuse=True),
}


def make_config(preset: str = "A9", **overrides) -> Config:
    cfg = Config()
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"Preset tidak dikenal: {preset}. Pilihan: {list(PRESETS)}")
        cfg = replace(cfg, **PRESETS[preset])
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    if cfg.model_name == "usulan":
        # garis bawah di akhir supaya "usulan_A3_" tidak ikut cocok dengan "usulan_A3g_"
        # (main.py Belton menyaring file dengan `model_name in nama_file`)
        cfg = replace(cfg, model_name=f"usulan_{preset}_")
    return cfg

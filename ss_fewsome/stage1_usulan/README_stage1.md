# Stage 1 Usulan untuk SS-FewSOME (Belton dkk., 2025)

Paket ini **mengganti Stage 1** SS-FewSOME. Stage 2 (pseudo-label + CLIP) dan Stage 3 (DCRL)
tetap memakai kode Belton (`main.py`, `utils.py`) tanpa diubah.

```
lokalisasi anatomi -> memory bank anatomi (bank normal + bank OA)
                   -> bank taji -> DRA-OA (5 kepala) -> fusi skor -> file skor format Belton
```

## 1. Letak folder

Taruh folder `stage1_usulan/` di dalam folder `ss_fewsome/` milik Belton, sejajar dengan `main.py`:

```
SS-FewSOME_.../
├── meta/                      <- train_seed_X.csv Belton (+ train_oa_seed_X.csv dibuat otomatis)
└── ss_fewsome/
    ├── main.py, utils.py ...  <- kode Belton (tidak diubah)
    ├── data/                  <- dataset Chen: train/ val/ test/, masing-masing subfolder 0..4
    └── stage1_usulan/         <- paket ini
```

Semua perintah dijalankan dari folder `ss_fewsome/`.

## 2. Peta file

| File | Isi | Bagian metode |
|---|---|---|
| `config.py` | Semua pengaturan + preset ablation A1–A9 | – |
| `data_split.py` | Baca split train/val/test, 30 KL-0 Belton, pilih 15 KL-2 + 15 KL-3 per seed | Data |
| `anatomy.py` | Garis sendi, kemiringan, tepi lutut, fibula, penyejajaran, peta wilayah | Lokalisasi anatomi |
| `features.py` | Fitur patch (AlexNet setara Belton, atau WRN-50 ala PatchCore) | Memory bank |
| `memory_bank.py` | Bank normal + bank OA (global / perloc / dual / dual_graded), skor leave-one-out | Memory bank anatomi |
| `spur_bank.py` | Ekstraksi taji dari 30 lutut OA (kontur vs pita normal, dipandu peta skor) | Bank taji |
| `lesion_generator.py` | Tempel taji, penyempitan celah, sklerosis, label patch lunak | Lesi sintetis |
| `dra_oa.py` | Model DRA-OA: kepala normal, OA asli (MIL top-K tepi), sintetis, residual, keparahan | DRA-OA |
| `train_dra.py` | Training per seed + inferensi | DRA-OA |
| `score_writer.py` | Fusi skor, metrik (sama dengan Belton), tulis file `outputs/dfs/ss/...` | Output |
| `run_stage1.py` | Program utama (CLI) | – |
| `tools/check_belton_compat.py` | Cek output terbaca `get_pseudo_labels` Belton | Uji |
| `tools/make_synthetic_dataset.py` | Dataset lutut sintetis kecil untuk uji kode | Uji |

## 3. Persiapan

```bash
pip install torch torchvision opencv-python scipy pandas scikit-learn
```

Bobot pretrained (AlexNet, ResNet-18, WRN-50) diunduh otomatis oleh torchvision pada pemakaian
pertama, jadi perlu internet sekali.

## 4. Menjalankan

### Metode lengkap (A9), 10 seed Belton

```bash
python -m stage1_usulan.run_stage1 --preset A9 --data_path ./data/ --meta_path ../meta/ --device cuda:0
```

Output:
- `outputs/dfs/ss/ss_training_usulan_A9_bs_1_task_test_N_30_seed_<seed>_epoch_400` (train) dan
  `..._on_test_set_epoch_400` (test). Kolom: `id, label, centre_mean` (+ kolom tambahan `k1, dra, sev`).
- `outputs/ringkasan_stage1_usulan_A9_.csv`: metrik per seed (val dan test) + rata-rata ± SD.
- `outputs/config_usulan_A9_.json`: pengaturan yang dipakai.
- `meta/train_oa_seed_<seed>.csv`: 30 lutut OA berlabel (15 KL-2 + 15 KL-3) per seed. Dibuat sekali
  lalu dipakai ulang, jadi semua preset memakai OA yang sama.

### Lanjut ke Stage 2–3 Belton

```bash
python main.py --train_ss 0 --model_name usulan_A9_ --data_path ./data/
```

`--train_ss 0` membuat Belton melewati Stage 1 miliknya dan langsung membaca file skor dari paket ini.

### Ablation (tabel ablation di tesis)

| Preset | Isi |
|---|---|
| A1 | PatchCore global, tanpa penyejajaran |
| A2 | + penyejajaran, memory bank per lokasi |
| A3 | + bank OA (dual), tanpa training |
| A3g | dual bertingkat (bobot KL-2/KL-3) |
| A4 | DRA + penyejajaran, MIL semua patch, CutMix |
| A5 | + MIL hanya di tepi |
| A6 | + generator taji dari bank taji |
| A7 | + penyempitan celah dan sklerosis |
| A8 | + label patch lunak, kepala keparahan, ranking loss |
| A9 | + fusi dengan skor memory bank = **metode lengkap** |

```bash
for p in A1 A2 A3 A3g A4 A5 A6 A7 A8 A9; do
  python -m stage1_usulan.run_stage1 --preset $p --data_path ./data/ --meta_path ../meta/
done
```

Variasi lain lewat argumen: `--k1_backbone wrn50`, `--spur_bank_mode contour`,
`--dra_feat_layer layer4`, `--fuse_w_k1 0.3`, `--epochs 50`, `--seeds 1001 34`.

### Cek kompatibilitas dengan Stage 2 Belton

Setelah 10 seed selesai:

```bash
python -m stage1_usulan.tools.check_belton_compat --model_name usulan_A9_ --data_path ./data/ --meta_path ../meta/
```

### Uji kode tanpa dataset asli

```bash
python -m stage1_usulan.tools.make_synthetic_dataset --out ./data_sintetis/ --meta ./meta_sintetis/
python -m stage1_usulan.run_stage1 --preset A9 --data_path ./data_sintetis/ --meta_path ./meta_sintetis/ \
       --device cpu --no_pretrained --quick --seeds 1001
```

Angka dari data sintetis **tidak bermakna**. Data ini hanya untuk memastikan kode berjalan.

## 5. Catatan penting sebelum eksperimen asli

1. **Sisi lateral / flip.** `anatomy.py` mencari fibula dari gambar. Kalau tidak yakin, kode memakai
   nama file (akhiran `L`/`R`) dengan asumsi fibula lutut kanan tampak di kiri gambar. Periksa
   `outputs/stage1_usulan_cache/anatomi_train.csv` (kolom `fibula`, `flip`) dan lihat 10–20 gambar
   hasil penyejajaran sebelum menjalankan semua preset. Kalau asumsinya terbalik, ubah `detect()`.
2. **Gambar gagal.** Kolom `ok`/`reason` di file anatomi. Gambar gagal tidak digeser (hanya di-flip)
   dan tetap diberi skor.
3. **Tuning hanya di val.** `fuse_w_k1`, `lr`, `epochs`, `topk_frac`, `soft_label_gain` dipilih dari
   AUC_KL di val (lihat ringkasan). Test hanya dipakai untuk laporan akhir.
4. **Data berlabel per seed:** 30 KL-0 (sama dengan Belton) + 30 KL-2/3. Ini harus ditulis jelas
   di tesis karena Belton hanya memakai 30 KL-0.
5. **Cache.** Gambar sejajar dan fitur disimpan di `outputs/stage1_usulan_cache/`. Hapus folder itu
   kalau mengubah pengaturan anatomi atau backbone.
6. Skor `centre_mean` disusun naik (semakin besar semakin OA) dan dinormalisasi min-max pada train,
   sehingga pencarian margin `get_pseudo_labels` Belton tetap bekerja.

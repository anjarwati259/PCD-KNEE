# Dokumentasi code `anatomy` — Metode Usulan Stage 1 sampai Stage 3

## 1. Gambaran umum alur

Alur default `Run_all.py` adalah:

1. **Meta dan pool OA** — menyalin metadata Belton ke folder kerja lalu membangun pool gambar OA.
2. **Lokalisasi anatomi** — mendeteksi landmark sendi untuk train, val, dan test.
3. **Penyejajaran** — menyelaraskan gambar berdasarkan landmark agar posisi anatomi lebih konsisten.
4. **Memory bank anatomi (K1)** — menjalankan varian `k1c` untuk 10 seed, memilih konfigurasi berdasarkan validation, lalu menghitung skor train yang diperlukan Belton.
5. **Bank taji (K2)** — mengekstrak kandidat osteofit dari ROI anatomi.
6. **Sintesis OA** — membentuk contoh OA sintetis dari taji dan penyempitan celah sendi.
7. **DRA-OA** — melatih dan mengevaluasi preset `utama` per seed.
8. **Ekspor Stage 1** — menggabungkan komponen terpilih (`fusi_utama+k1c`) ke format yang dibaca Belton.
9. **Stage 2–3 Belton** — menjalankan `main.py` dari repositori Belton eksternal.
10. **Evaluasi** — menjalankan `evaluasi_pipeline.py` untuk menggabungkan hasil Stage 1–3.
11. **Laporan** — membuat `LAPORAN.txt` dari ringkasan yang tersedia.


## 2. Fungsi setiap file

### File utama dan komponen pipeline

| File | Fungsi | Dipanggil otomatis oleh `Run_all.py`? |
|---|---|---|
| `Run_all.py` | Orkestrator pipeline. Memvalidasi folder data, menetapkan konfigurasi, memanggil setiap tahap, menyimpan log dan `LAPORAN.txt`. | **Ya — jalankan file ini untuk full run.** |
| `make_oa_pool.py` | Menyiapkan pool gambar OA dan metadata seed OA berdasarkan data train dan metadata Belton. | Ya, pada tahap `meta`. |
| `localize_joint.py` | Mendeteksi landmark anatomi lutut, garis sendi, ROI medial/lateral, dan informasi geometri; dapat menyimpan overlay untuk QA. | Ya, pada tahap `lokalisasi`. |
| `align_joint.py` | Menggunakan landmark untuk menyejajarkan gambar dan menyimpan landmark dalam koordinat gambar yang telah disejajarkan. | Ya, pada tahap `penyejajaran`. |
| `run_seeds.py` | Mengatur eksperimen memory bank untuk daftar seed/varian, memilih konfigurasi berdasarkan validation, merangkum skor, dan menghasilkan ensemble. | Ya, pada tahap `memory_bank`. |
| `patchcore_k1.py` | Implementasi ekstraksi fitur patch, memory bank, perhitungan skor anomali dan peta skor patch. Dipakai untuk varian global, location, dan dual-bank. | Ya, dipanggil melalui `run_seeds.py` dan pemanggilan lanjutan untuk skor train. Bukan langkah mandiri terpisah di `Run_all.py`. |
| `extract_osteophyte_bank.py` | Menyeleksi kandidat taji/osteofit dari ROI tepi sendi menggunakan landmark dan skor patch; membuat bank kandidat serta keluaran QA. | Ya, tahap `bank_taji`. |
| `make_synthetic_oa.py` | Membuat contoh OA sintetis dengan menambahkan taji dan menyimulasikan penyempitan celah sendi; melakukan validasi/preview hasil sintesis. | Ya, tahap `sintesis`. |
| `train_dra_oa.py` | Melatih preset DRA-OA menggunakan bank taji dan data sintetis, menilai skor pada train/val/test, lalu menentukan konfigurasi berdasarkan validation. | Ya, tahap `dra_oa`. |
| `export_belton.py` | Menggabungkan skor komponen yang dipilih, menghitung ringkasan Stage 1, dan mengekspor hasil ke struktur/nama yang dapat dibaca pipeline Belton. | Ya, tahap `stage1`. |
| `eval_scores.py` | Fungsi dan CLI untuk menghitung metrik dari peta skor patch, membandingkan konfigurasi region/agregasi, menghitung bootstrap CI, dan uji berpasangan jika diminta. | Diimpor oleh beberapa skrip; **tidak dipanggil sebagai tahap terpisah oleh `Run_all.py`**. |

### File eksperimen tambahan, laporan, dan pemeriksaan

| File | Fungsi | Status dalam full run default |
|---|---|---|
| `ablation_dual_global.py` | Pembanding/ablation dual memory bank tanpa prior anatomi; membantu menguji apakah peningkatan berasal dari anatomi atau sekadar tambahan bank OA. | Tidak dipanggil otomatis oleh `Run_all.py`. |
| `Cek_belton.py` | Memeriksa apakah hasil ekspor dapat dibaca oleh utilitas Belton dan menampilkan ringkasan/pseudo-label untuk beberapa margin. | Tidak dipanggil otomatis; alat pemeriksaan integrasi. Jalankan dengan working directory repositori Belton. |


## 3. Di mana baseline SS-FewSOME/Belton?

Baseline dari paper Belton ada di folder ss_fewsome
## 4. Struktur folder yang diharapkan

```text
PCD-knee/
├── anatomy/
│   ├── README.md
│   ├── Run_all.py
│   ├── make_oa_pool.py
│   ├── localize_joint.py
│   ├── align_joint.py
│   ├── run_seeds.py
│   ├── patchcore_k1.py
│   ├── extract_osteophyte_bank.py
│   ├── make_synthetic_oa.py
│   ├── train_dra_oa.py
│   ├── export_belton.py
│   ├── eval_scores.py
│   └── ... file pendukung lainnya
├── data/
│   ├── train/0 ... train/4/
│   ├── val/0 ... val/4/
│   └── test/0 ... test/4/
├── meta_input/
│   ├── train_seed_*.csv
│   ├── train_ids.csv
│   ├── sim_scores.csv
│   └── metadata Belton lain yang diperlukan
├── ss_fewsome/                 # folder baseline dari paper belton
└── hasil_stage1/               # dibuat oleh pipeline
    └── LAPORAN.txt
```

## 5. Cara menjalankan full pipeline
python run_all.py --data ../ss_fewsome/data --meta ../meta --belton_root ../ss_fewsome


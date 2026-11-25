"""
siapkan_C1C2.py - Siapkan data untuk percobaan C1 + C2 (anatomi Stage 1 dibawa ke DCRL), TANPA training.

C1 (potong ke daerah sendi): SEMUA gambar (train/val/test) diproses dengan cara yang SAMA:
     gambar asli -> penyejajaran (parameter dari landmarks Stage 1: flip medial ke kiri, rotasi, garis sendi
     ke baris 133, pembagi ke kolom 112) -> potong pita sendi baris 133 +- PITA -> ubah ukuran ke TINGGI x LEBAR.
     Hasil: <out>/data_crop/{train,val,test}/{0..4}/<file>.png  (dipakai sebagai --data_path main.py Belton)
C2 (contoh OA tambahan untuk DCRL iterasi 1), per seed (anggaran label per seed tetap):
     + 30 OA berlabel seed itu (meta/oa_seed_<seed>.csv), dipotong seperti C1
     + 30 OA sintetis (10 per grade 2, 3, 4) dibuat dari 30 normal seed itu dengan generator Stage 1
       (make_synthetic_oa.OAGenerator: taji asli + penyempitan celah), lalu disejajarkan dan dipotong seperti C1
     Hasil: <out>/tambahan_oa/seed_<seed>.csv (daftar path) + <out>/sintetis/seed_<seed>/*.png

Pemeriksaan otomatis:
  * penyejajaran ulang dibandingkan dengan gambar di hasil_stage1/aligned (selisih piksel rata-rata)
  * jumlah gambar data_crop = jumlah gambar dataset asli
  * lembar_cek.png: contoh potongan asli, OA berlabel, dan sintetis per grade

  python siapkan_C1C2.py            (dari folder anatomy; path bawaan mengikuti susunan D:\\PCD\\PCD-KNEE)
"""

import argparse
import inspect
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
SEEDS = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
SPLITS = ['train', 'val', 'test']


def nid(x):
    x = str(x).replace('\\', '/')
    return x.split('/')[-2] + '/' + x.split('/')[-1]


def ids_csv(path):
    d = pd.read_csv(path)
    for c in d.columns:
        if d[c].astype(str).str.contains(r'\.png', case=False).any():
            return d[c].map(nid).tolist()
    raise SystemExit('Kolom id (.png) tidak ditemukan di ' + path)


# ----------------------------------------------------------------------------- penyejajaran (sama dengan align_joint)
def _rot_params_bawaan(row, W):
    flip = row['medial_on'] == 'right'
    xl, xr = float(row['x_left']), float(row['x_right'])
    yl, yr = float(row['y_joint_left']), float(row['y_joint_right'])
    xc = float(row['x_center'])
    if flip:
        xl, xr = W - 1 - xr, W - 1 - xl
        yl, yr = yr, yl
        xc = W - 1 - xc
    return flip, np.arctan2(yr - yl, xr - xl), (xl + xr) / 2.0, (yl + yr) / 2.0, xc


def _warp_bawaan(img, prm, joint_y, center_x, size):
    H, W = img.shape
    flip, ang, cx, cy, xc_f = prm
    c, s = np.cos(-ang), np.sin(-ang)
    xc_r = c * (xc_f - cx) + cx
    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    xr = np.where(u <= center_x, u * xc_r / center_x, xc_r + (u - center_x) * (W - 1 - xc_r) / (size - 1 - center_x))
    yr = v - joint_y + cy
    c2, s2 = np.cos(ang), np.sin(ang)
    x = c2 * (xr - cx) - s2 * (yr - cy) + cx
    y = s2 * (xr - cx) + c2 * (yr - cy) + cy
    if flip:
        x = W - 1 - x
    return cv2.remap(img, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def fungsi_penyejajaran():
    """Pakai align_joint.py milik folder anatomy bila ada (sama persis dengan Stage 1); kalau tidak, versi bawaan."""
    sys.path.insert(0, HERE)
    try:
        import align_joint as AJ
        if hasattr(AJ, '_rot_params') and hasattr(AJ, 'warp_image'):
            return AJ._rot_params, (lambda im, p, jy, cxx, sz: AJ.warp_image(im, p, jy, cxx, sz)), 'align_joint.py (anatomy)'
    except Exception as e:
        print('[!] align_joint.py tidak bisa dipakai ({}), memakai versi bawaan'.format(str(e)[:80]))
    return _rot_params_bawaan, _warp_bawaan, 'bawaan (rekonstruksi align_joint)'


def main():
    ap = argparse.ArgumentParser(description='Siapkan data C1 (potong sendi) + C2 (OA berlabel + sintetis) tanpa training')
    ap.add_argument('--data', default=os.path.join(ROOT, 'ss_fewsome', 'data'), help='dataset asli (train/val/test/0..4)')
    ap.add_argument('--stage1', default=os.path.join(ROOT, 'hasil_stage1'), help='berisi anatomy_out/, aligned/, meta/, k2_out/')
    ap.add_argument('--out', default=os.path.join(ROOT, 'hasil_C1C2'))
    ap.add_argument('--pita', type=int, default=56, help='setengah tinggi pita sendi (px, di gambar tersejajar 224)')
    ap.add_argument('--tinggi', type=int, default=160)
    ap.add_argument('--lebar', type=int, default=320)
    ap.add_argument('--n_per_grade', type=int, default=10, help='OA sintetis per grade per seed')
    ap.add_argument('--grades', default='2,3,4')
    ap.add_argument('--joint_y', type=float, default=133)
    ap.add_argument('--center_x', type=float, default=112)
    a = ap.parse_args()

    rot_params, warp, sumber_warp = fungsi_penyejajaran()
    meta = os.path.join(a.stage1, 'meta')
    crop_root = os.path.join(a.out, 'data_crop')
    y0, y1 = int(round(a.joint_y - a.pita)), int(round(a.joint_y + a.pita))
    print('Penyejajaran: {} | pita sendi baris {}..{} -> {} x {} (tinggi x lebar)'.format(sumber_warp, y0, y1, a.tinggi, a.lebar))

    def potong(img, row):
        al = warp(img, rot_params(row, img.shape[1]), a.joint_y, a.center_x, 224)
        return al, cv2.resize(al[y0:y1], (a.lebar, a.tinggi), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ C1: semua gambar
    lm = {}
    for sp in SPLITS:
        f = os.path.join(a.stage1, 'anatomy_out', sp, 'landmarks.csv')
        if not os.path.exists(f):
            raise SystemExit('Tidak ada ' + f)
        d = pd.read_csv(f, keep_default_na=False)
        d['key'] = d['id'].map(nid)
        lm[sp] = d.set_index('key')
    selisih, hilang = [], []
    for sp in SPLITS:
        n_tulis = 0
        for g in '01234':
            src_dir = os.path.join(a.data, sp, g)
            for fn in sorted(os.listdir(src_dir)):
                if not fn.lower().endswith('.png'):
                    continue
                key = g + '/' + fn
                dst = os.path.join(crop_root, sp, g, fn)
                if os.path.exists(dst):
                    n_tulis += 1
                    continue
                if key not in lm[sp].index:
                    hilang.append(sp + '/' + key)
                    continue
                img = cv2.imread(os.path.join(src_dir, fn), cv2.IMREAD_GRAYSCALE)
                al, cr = potong(img, lm[sp].loc[key])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                cv2.imwrite(dst, cr)
                n_tulis += 1
                if sp == 'train' and len(selisih) < 60:
                    ref = cv2.imread(os.path.join(a.stage1, 'aligned', sp, g, fn), cv2.IMREAD_GRAYSCALE)
                    if ref is not None and ref.shape == al.shape:
                        selisih.append(float(np.abs(ref.astype(float) - al.astype(float)).mean()))
        print('  {:<5}: {} gambar dipotong'.format(sp, n_tulis))
    if hilang:
        raise SystemExit('[GAGAL] {} gambar tidak punya landmarks, contoh: {}'.format(len(hilang), hilang[:3]))
    if selisih:
        m = float(np.mean(selisih))
        print('Cek penyejajaran ulang vs hasil_stage1/aligned: selisih piksel rata-rata {:.2f} (dari {} gambar) {}'.format(
            m, len(selisih), 'OK' if m < 2 else '[!] berbeda, laporkan angka ini'))
    for sp in SPLITS:
        asli = [len([f for f in os.listdir(os.path.join(a.data, sp, g)) if f.lower().endswith('.png')]) for g in '01234']
        crop = [len(os.listdir(os.path.join(crop_root, sp, g))) for g in '01234']
        print('  cek jumlah {:<5} asli {} | crop {} {}'.format(sp, asli, crop, 'OK' if asli == crop else '[!] BEDA'))
        if asli != crop:
            raise SystemExit('[GAGAL] jumlah gambar data_crop tidak sama')

    # ------------------------------------------------------------------ C2: OA berlabel + sintetis per seed
    sys.path.insert(0, HERE)
    import make_synthetic_oa as SG
    sig = inspect.signature(SG.OAGenerator.__init__).parameters
    kw = dict(meta_path=meta, sumber='bank', komponen=['taji', 'jsn'], taji_ids=None)
    kw = {k: v for k, v in kw.items() if k in sig}
    gen = SG.OAGenerator(os.path.join(a.stage1, 'k2_out', 'bank_bersih'), a.data,
                         os.path.join(a.stage1, 'anatomy_out', 'train', 'landmarks.csv'), **kw)
    grades = [int(g) for g in a.grades.split(',')]
    os.makedirs(os.path.join(a.out, 'tambahan_oa'), exist_ok=True)
    contoh = []
    for s in SEEDS:
        normals = ids_csv(os.path.join(meta, 'train_seed_{}.csv'.format(s)))
        oas = ids_csv(os.path.join(meta, 'oa_seed_{}.csv'.format(s)))
        target = [i for i in normals if gen.valid_target(i)]
        if not target:
            raise SystemExit('Seed {}: tidak ada normal yang valid untuk sintesis'.format(s))
        rng = np.random.default_rng(s)
        sdir = os.path.join(a.out, 'sintetis', 'seed_{}'.format(s))
        os.makedirs(sdir, exist_ok=True)
        rows = [dict(path=os.path.join(crop_root, 'train', *i.split('/')).replace('\\', '/'), jenis='OA berlabel',
                     grade=int(i.split('/')[0])) for i in oas]
        k = 0
        for g in grades:
            for n in range(a.n_per_grade):
                j = target[k % len(target)]
                k += 1
                r = gen.synthesize(gen.read(j), j, g, rng)
                im = np.clip(np.asarray(r['img'], dtype=np.float64), 0, 255).astype(np.uint8)
                _, cr = potong(im, lm['train'].loc[j])
                p = os.path.join(sdir, 'g{}_{:02d}_{}'.format(g, n, j.split('/')[-1]))
                cv2.imwrite(p, cr)
                rows.append(dict(path=p.replace('\\', '/'), jenis='sintetis', grade=g))
                if s == SEEDS[0] and n == 0:
                    contoh.append(('sintetis KL{}'.format(g), cr))
        pd.DataFrame(rows).to_csv(os.path.join(a.out, 'tambahan_oa', 'seed_{}.csv'.format(s)), index=False)
        print('  seed {:<7}: {} OA berlabel + {} sintetis'.format(s, len(oas), len(rows) - len(oas)))
        if s == SEEDS[0]:
            contoh = [('normal ' + normals[0], cv2.imread(os.path.join(crop_root, 'train', *normals[0].split('/')), 0)),
                      ('OA berlabel ' + oas[0], cv2.imread(os.path.join(crop_root, 'train', *oas[0].split('/')), 0))] + contoh

    # ------------------------------------------------------------------ lembar cek visual
    pad = 26
    lembar = np.full(((a.tinggi + pad) * len(contoh), a.lebar), 255, np.uint8)
    for k, (judul, im) in enumerate(contoh):
        y = k * (a.tinggi + pad)
        cv2.putText(lembar, judul, (4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
        lembar[y + pad:y + pad + a.tinggi] = im
    cv2.imwrite(os.path.join(a.out, 'lembar_cek.png'), lembar)
    print('\nSelesai. --data_path untuk main.py: {}/'.format(crop_root.replace('\\', '/')))
    print('Daftar contoh OA tambahan: {}'.format(os.path.join(a.out, 'tambahan_oa')))
    print('Cek visual: {}'.format(os.path.join(a.out, 'lembar_cek.png')))


if __name__ == '__main__':
    main()

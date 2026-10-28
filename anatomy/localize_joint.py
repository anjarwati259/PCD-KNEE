import argparse
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# ----------------------------------------------------------------------------- parameter
# Semua angka dalam piksel untuk gambar 224x224; otomatis diskalakan untuk ukuran lain.
REF_SIZE = 224
P = dict(
    clahe_clip=2.0, clahe_tile=8,
    band=(0.30, 0.78),          # rentang baris (fraksi tinggi) tempat garis sendi dicari
    sigmas=(1.5, 2.5, 3.5),     # skala filter lembah (lebar celah sendi bervariasi)
    max_step=1,                 # perubahan baris maksimum per kolom pada jalur DP
    step_penalty=0.08,          # penalti untuk tiap perubahan baris (kehalusan kurva)
    contour_penalty=0.15,       # penalti kehalusan kontur femur / tibia
    edge_search=14,             # jarak cari kontur femur/tibia dari garis sendi
    bone_band=10,               # tinggi pita di sekitar garis sendi untuk tepi tulang
    side_frac=0.30,             # tepi tulang dicari di 30% kiri / kanan gambar
    edge_min_grad=0.012,        # gradien minimum agar dianggap tepi tulang
    center_frac=(0.30, 0.70),   # rentang pencarian intercondylar notch
    roi_w=40, roi_h=28,         # ukuran ROI area tepi (per femur / per tibia)
    roi_out=14,                 # berapa piksel ROI keluar dari tepi tulang
)


# ----------------------------------------------------------------------------- set training
BELTON_SEEDS = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _norm_id(x):
    """'/path/ke/train/0/9935439L.png' atau '0\\9935439L.png' -> '0/9935439L.png'"""
    return '/'.join(str(x).replace('\\', '/').split('/')[-2:])


def select_training_files(input_root, meta_path, train_seed='1001', n_oa=30, oa_grades=(2, 3)):
    """Daftar gambar training sesuai rencana awal: 30 normal (KL0) + 30 OA (KL2/3).

    * Normal: daftar Belton meta/train_seed_<seed>.csv (tidak dipilih ulang).
    * OA    : meta/oa_seed_<seed>.csv, dibuat oleh make_oa_pool.py (30 OA diambil dari pool OA,
              pola yang sama dengan daftar normal Belton).
    train_seed: satu seed Belton (mis. '1001') atau 'all' (gabungan 10 seed).
    Return: list path file, DataFrame daftar (id, kl, role, seed).
    """
    seeds = BELTON_SEEDS if str(train_seed) == 'all' else [int(train_seed)]
    rows = []
    for sd in seeds:
        f_norm = os.path.join(meta_path, 'train_seed_{}.csv'.format(sd))
        if not os.path.exists(f_norm):
            raise SystemExit('Tidak ditemukan: ' + f_norm)
        for i in pd.read_csv(f_norm)['id']:
            rows.append(dict(id=_norm_id(i), role='normal', seed=sd))

        f_oa = os.path.join(meta_path, 'oa_seed_{}.csv'.format(sd))
        if not os.path.exists(f_oa):
            raise SystemExit('Tidak ada {}. Buat dulu pool OA & daftar per seed:\n'
                             '  python make_oa_pool.py'.format(f_oa))
        oa = pd.read_csv(f_oa)
        for i in oa['id']:
            rows.append(dict(id=_norm_id(i), role='oa', seed=sd))

    lst = pd.DataFrame(rows)
    lst['kl'] = lst['id'].str.split('/').str[0].astype(int)
    files, missing = [], []
    for i in lst['id'].drop_duplicates():
        f = os.path.join(input_root, *i.split('/'))
        (files if os.path.exists(f) else missing).append(f)
    if missing:
        print('  [!] {} gambar di daftar tidak ditemukan, contoh: {}'.format(len(missing), missing[:3]))
    u = lst.drop_duplicates('id')
    print('  Set training (seed {}): {} normal unik + {} OA unik = {} gambar unik'.format(
        train_seed, (u.role == 'normal').sum(), (u.role == 'oa').sum(), len(files)))
    return sorted(files), lst


def _s(v, scale):
    return max(1, int(round(v * scale)))


def parse_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'([LR])$', stem, flags=re.I)
    side = m.group(1).upper() if m else None
    kl = None
    m2 = re.search(r'(\d)$', os.path.basename(os.path.dirname(path)))
    if m2:
        kl = int(m2.group(1))
    return stem, side, kl


def preprocess(img):
    clahe = cv2.createCLAHE(clipLimit=P['clahe_clip'], tileGridSize=(P['clahe_tile'],) * 2)
    eq = clahe.apply(img)
    return eq.astype(np.float32) / 255.0


def valley_response(I, scale):
    """Respons tinggi pada garis gelap horizontal (turunan kedua Gaussian arah y, ternormalisasi skala)."""
    R = np.zeros_like(I)
    for s in P['sigmas']:
        s = s * scale
        d2 = gaussian_filter(I, sigma=(s, s * 1.5), order=(2, 0))
        R = np.maximum(R, (s ** 2) * d2)
    return R


def normalize_band(R, y0, y1):
    """Skala respons ke ~[0,1] berdasarkan persentil 99 di pita pencarian."""
    hi = np.percentile(R[y0:y1], 99)
    return np.clip(R / (hi + 1e-6), -1, 1.5)


def dp_path(R, y0, y1, x0, x1, max_step, penalty):
    """Jalur kiri->kanan (kolom x0..x1) dengan jumlah respons maksimum di baris y0..y1."""
    sub = R[y0:y1, x0:x1]
    H, W = sub.shape
    cost = np.full((H, W), -np.inf, dtype=np.float32)
    back = np.zeros((H, W), dtype=np.int16)
    cost[:, 0] = sub[:, 0]
    offs = range(-max_step, max_step + 1)
    for x in range(1, W):
        best = np.full(H, -np.inf, dtype=np.float32)
        arg = np.zeros(H, dtype=np.int16)
        for d in offs:
            shifted = np.full(H, -np.inf, dtype=np.float32)
            if d >= 0:
                shifted[d:] = cost[:H - d, x - 1] if d > 0 else cost[:, x - 1]
            else:
                shifted[:d] = cost[-d:, x - 1]
            shifted = shifted - penalty * abs(d)
            better = shifted > best
            best[better] = shifted[better]
            arg[better] = d
        cost[:, x] = best + sub[:, x]
        back[:, x] = arg
    y = int(np.argmax(cost[:, -1]))
    path = np.zeros(W, dtype=np.int32)
    for x in range(W - 1, -1, -1):
        path[x] = y
        y = y - back[y, x]
    return path + y0, float(np.max(cost[:, -1]) / W)


def _outer_edge(g, lo, hi, from_left, scale):
    """Tepi terluar: gradien kuat pertama dari arah batas gambar, lalu disempurnakan ke puncak lokal."""
    seg = g[lo:hi] if from_left else -g[lo:hi][::-1]
    thr = max(P['edge_min_grad'], 0.5 * float(seg.max()))
    idx = np.where(seg >= thr)[0]
    if len(idx) == 0:
        return None
    i0 = int(idx[0])
    i = i0 + int(np.argmax(seg[i0:i0 + _s(6, scale)]))
    return lo + i if from_left else hi - 1 - i


def bone_edges(I, y_line, scale):
    """Tepi kiri & kanan femur (pita di atas garis sendi) dan tibia (pita di bawahnya).
    Kalau tepi tidak terlihat (tulang menyentuh batas gambar), dipakai batas gambar + flag."""
    H, W = I.shape
    yc = int(np.median(y_line))
    a, b = _s(6, scale), _s(30, scale)
    sw = int(W * P['side_frac'])
    m = _s(2, scale)
    out, missing = {}, []
    for bone, (r0, r1) in {'fem': (yc - b, yc - a), 'tib': (yc + a, yc + b)}.items():
        prof = gaussian_filter1d(I[max(0, r0):min(H, r1)].mean(axis=0), 2 * scale)
        g = np.gradient(prof)
        xl = _outer_edge(g, m, sw, True, scale)
        xr = _outer_edge(g, W - sw, W - m, False, scale)
        if xl is None:
            xl = m; missing.append(bone + '_left')
        if xr is None:
            xr = W - 1 - m; missing.append(bone + '_right')
        out[bone] = (int(xl), int(xr))
    return out, missing


def _dp_offsets(M, max_step, penalty):
    """DP pada matriks (offset, kolom): jalur halus dengan jumlah nilai maksimum."""
    K, W = M.shape
    cost = M[:, 0].copy()
    back = np.zeros((K, W), dtype=np.int16)
    for x in range(1, W):
        best = np.full(K, -np.inf, dtype=np.float32)
        arg = np.zeros(K, dtype=np.int16)
        for d in range(-max_step, max_step + 1):
            sh = np.full(K, -np.inf, dtype=np.float32)
            if d > 0:
                sh[d:] = cost[:K - d]
            elif d < 0:
                sh[:d] = cost[-d:]
            else:
                sh[:] = cost
            sh -= penalty * abs(d)
            b = sh > best
            best[b] = sh[b]
            arg[b] = d
        cost = best + M[:, x]
        back[:, x] = arg
    k = int(np.argmax(cost))
    out = np.zeros(W, dtype=np.int32)
    for x in range(W - 1, -1, -1):
        out[x] = k
        k -= back[k, x]
    return out


def contours_along(I, path, x0, scale):
    """Kontur femur (tepi bawah femur, di atas celah) dan tibia (tepi atas tibia, di bawah celah).
    Dicari dengan DP pada peta gradien vertikal supaya kontur halus & kontinu."""
    gy = gaussian_filter(I, sigma=(1.2 * scale, 2.0 * scale), order=(1, 0))  # >0: makin terang ke bawah
    H = I.shape[0]
    k = _s(P['edge_search'], scale)
    W = len(path)
    xs = x0 + np.arange(W)
    up = np.zeros((k + 1, W), np.float32)    # offset j -> baris y - j
    dn = np.zeros((k + 1, W), np.float32)    # offset j -> baris y + j
    for j in range(k + 1):
        up[j] = -gy[np.clip(path - j, 0, H - 1), xs]
        dn[j] = gy[np.clip(path + j, 0, H - 1), xs]
    norm = lambda M: M / (np.percentile(np.abs(M), 99) + 1e-6)
    st = _s(1, scale)
    fem = path - _dp_offsets(norm(up), st, P['contour_penalty'])
    tib = path + _dp_offsets(norm(dn), st, P['contour_penalty'])
    return fem, tib


def localize(img, side=None):
    H, W = img.shape
    scale = H / REF_SIZE
    I = preprocess(img)
    R = valley_response(I, scale)
    y0, y1 = int(P['band'][0] * H), int(P['band'][1] * H)
    R = normalize_band(R, y0, y1)

    # 1) jalur awal di seluruh lebar -> perkiraan baris garis sendi untuk cari tepi tulang
    m = _s(4, scale)
    path0, _ = dp_path(R, y0, y1, m, W - m, _s(P['max_step'], scale), P['step_penalty'])
    edges, missing = bone_edges(I, path0, scale)
    xl = min(edges['fem'][0], edges['tib'][0])
    xr = max(edges['fem'][1], edges['tib'][1])

    # 2) jalur final hanya di antara tepi tulang
    path, strength = dp_path(R, y0, y1, xl, xr + 1, _s(P['max_step'], scale), P['step_penalty'])
    xs = np.arange(xl, xr + 1)
    fem, tib = contours_along(I, path, xl, scale)
    jsw = (tib - fem).astype(np.float32)

    # 3) pembagi medial / lateral = puncak "punuk" garis sendi di tengah (tibial spines di bawah
    #    intercondylar notch). Diambil titik tengah dari kolom-kolom yang paling tinggi (y kecil).
    c0 = xl + int(P['center_frac'][0] * (xr - xl))
    c1 = xl + int(P['center_frac'][1] * (xr - xl))
    seg = gaussian_filter1d(path.astype(np.float32), 4 * scale)[c0 - xl:c1 - xl + 1]
    top = np.where(seg <= seg.min() + _s(2, scale))[0]
    xc = c0 + int(round(np.median(top)))
    hump = float(np.median(np.r_[path[:len(path) // 4], path[-len(path) // 4:]]) - seg.min())

    # 4) sisi medial di gambar (dari L/R; dataset tidak di-flip)
    medial_on = {'R': 'right', 'L': 'left'}.get(side)

    # 5) JSW per kompartemen: pakai bagian tengah tiap kompartemen (hindari tepi & spine)
    def comp_stats(a, b):
        a, b = int(a), int(b)
        seg = jsw[a - xl:b - xl + 1]
        if len(seg) < 3:
            return np.nan, np.nan
        return float(np.percentile(seg, 10)), float(np.mean(seg))
    wl, wr = xc - xl, xr - xc
    left_min, left_mean = comp_stats(xl + 0.2 * wl, xl + 0.8 * wl)
    right_min, right_mean = comp_stats(xc + 0.2 * wr, xc + 0.8 * wr)

    # 6) ROI area tepi sendi (4 kotak): melebar keluar dari tepi tulang, di atas & bawah garis sendi
    rw, rh, ro = _s(P['roi_w'], scale), _s(P['roi_h'], scale), _s(P['roi_out'], scale)
    yL, yR = int(path[0]), int(path[-1])
    yL = int(np.median(path[:max(3, len(path) // 8)]))
    yR = int(np.median(path[-max(3, len(path) // 8):]))

    def box(x_start, y_start):
        x_start = int(np.clip(x_start, 0, W - rw)); y_start = int(np.clip(y_start, 0, H - rh))
        return (x_start, y_start, rw, rh)
    rois = {
        'left_fem': box(edges['fem'][0] - ro, yL - rh),
        'left_tib': box(edges['tib'][0] - ro, yL),
        'right_fem': box(edges['fem'][1] + ro - rw, yR - rh),
        'right_tib': box(edges['tib'][1] + ro - rw, yR),
    }

    angle = float(np.degrees(np.arctan2(yR - yL, max(1, xr - xl))))

    # 7) flag kualitas
    flags = []
    if xr - xl < 0.55 * W:
        flags.append('narrow_bone')
    if abs(angle) > 12:
        flags.append('steep_angle')
    if strength < 0.004:
        flags.append('weak_joint')
    if not (c0 < xc < c1):
        flags.append('center_at_limit')
    if hump < _s(4, scale):
        flags.append('no_hump')
    if medial_on is None:
        flags.append('unknown_side')
    if missing:
        flags.append('edge_not_found:' + ','.join(missing))

    res = dict(
        x_left=int(xl), x_right=int(xr), x_center=int(xc),
        fem_x_left=edges['fem'][0], fem_x_right=edges['fem'][1],
        tib_x_left=edges['tib'][0], tib_x_right=edges['tib'][1],
        y_joint_left=yL, y_joint_right=yR, y_joint_center=int(path[xc - xl]),
        angle_deg=round(angle, 2), joint_strength=round(strength, 5),
        medial_on=medial_on,
        jsw_left_min=round(left_min, 2), jsw_left_mean=round(left_mean, 2),
        jsw_right_min=round(right_min, 2), jsw_right_mean=round(right_mean, 2),
        flags=';'.join(flags),
    )
    # nama kompartemen anatomis
    if medial_on == 'left':
        res.update(jsw_medial_min=res['jsw_left_min'], jsw_lateral_min=res['jsw_right_min'])
    elif medial_on == 'right':
        res.update(jsw_medial_min=res['jsw_right_min'], jsw_lateral_min=res['jsw_left_min'])
    for k, (x, y, w, h) in rois.items():
        side_name = k.split('_')[0]
        anat = ('medial' if side_name == medial_on else 'lateral') if medial_on else side_name
        name = anat + '_' + k.split('_')[1]
        res.update({name + '_x': x, name + '_y': y, name + '_w': w, name + '_h': h})
    curves = dict(xs=xs, joint=path, femur=fem, tibia=tib)
    return res, curves


def draw_overlay(img, res, curves, title=''):
    big = 3
    clahe = cv2.createCLAHE(clipLimit=P['clahe_clip'], tileGridSize=(P['clahe_tile'],) * 2)
    vis = cv2.cvtColor(clahe.apply(img), cv2.COLOR_GRAY2BGR)   # latar CLAHE agar mudah dinilai
    vis = cv2.resize(vis, None, fx=big, fy=big, interpolation=cv2.INTER_CUBIC)
    pts = lambda ys: np.stack([curves['xs'] * big, ys * big], 1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts(curves['femur'])], False, (255, 200, 0), 1, cv2.LINE_AA)     # biru muda
    cv2.polylines(vis, [pts(curves['tibia'])], False, (0, 200, 255), 1, cv2.LINE_AA)     # oranye
    cv2.polylines(vis, [pts(curves['joint'])], False, (0, 255, 0), 1, cv2.LINE_AA)       # hijau
    H = vis.shape[0]
    cv2.line(vis, (res['x_center'] * big, 0), (res['x_center'] * big, H), (200, 200, 200), 1)
    yj = res['y_joint_center'] * big
    for xk, (ya, yb) in {'fem_x_left': (yj - 90, yj), 'fem_x_right': (yj - 90, yj),
                         'tib_x_left': (yj, yj + 90), 'tib_x_right': (yj, yj + 90)}.items():
        cv2.line(vis, (res[xk] * big, ya), (res[xk] * big, yb), (255, 0, 255), 2)   # tepi tulang
    for k in [c[:-2] for c in res if c.endswith('_x') and c.split('_')[0] in ('medial', 'lateral', 'left', 'right')]:
        if k + '_w' not in res:
            continue
        x, y, w, h = (res[k + s] for s in ('_x', '_y', '_w', '_h'))
        col = (0, 0, 255) if k.startswith('medial') else (255, 0, 0)
        cv2.rectangle(vis, (x * big, y * big), ((x + w) * big, (y + h) * big), col, 1)
    txt = '{}  med:{}  JSWmed:{} lat:{}  {}'.format(
        title, res['medial_on'], res.get('jsw_medial_min', '-'), res.get('jsw_lateral_min', '-'), res['flags'])
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(vis, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def make_qa_sheet(df, overlays, per_grade, seed, out_path, tile=280, ncol=5):
    """Lembar cek visual: sampel acak per grade KL (satu blok per grade) dalam satu gambar.
    Isi tabel cek di qa_check.csv: tulis 1 (benar) / 0 (salah) di kolom joint_ok, center_ok, roi_ok."""
    groups = df.groupby(df['kl'].fillna(-1))
    blocks, picked = [], []
    for kl, g in groups:
        g = g.sample(n=min(per_grade, len(g)), random_state=seed)
        ims = []
        for rel in g['id']:
            im = cv2.imread(overlays[rel])
            ims.append(cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA))
            picked.append(dict(id=rel, kl=kl, joint_ok='', center_ok='', roi_ok='', note=''))
        while len(ims) % ncol:
            ims.append(np.zeros_like(ims[0]))
        grid = np.vstack([np.hstack(ims[i:i + ncol]) for i in range(0, len(ims), ncol)])
        bar = np.zeros((26, grid.shape[1], 3), np.uint8)
        cv2.putText(bar, 'KL{}'.format(int(kl)) if kl >= 0 else 'KL ?', (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        blocks.append(np.vstack([bar, grid]))
    cv2.imwrite(out_path, np.vstack(blocks))
    pd.DataFrame(picked).to_csv(os.path.splitext(out_path)[0].replace('qa_sheet', 'qa_check') + '.csv', index=False)
    print('Lembar cek visual: {} ({} gambar) + qa_check.csv'.format(out_path, len(picked)))


def main():
    ap = argparse.ArgumentParser(description='Lokalisasi garis sendi, kontur tulang & area tepi sendi lutut')
    ap.add_argument('--input', required=True, help='folder gambar (dicari rekursif)')
    ap.add_argument('--out', default='./anatomy_out')
    ap.add_argument('--overlay', type=int, default=1, help='1 = simpan gambar overlay untuk cek visual')
    ap.add_argument('--qa_per_grade', type=int, default=10,
                    help='jumlah gambar acak per grade di lembar cek qa_sheet.png (0 = tidak dibuat)')
    ap.add_argument('--seed', type=int, default=0, help='seed untuk pemilihan acak lembar cek')
    ap.add_argument('--train_only', type=int, default=0,
                    help='1 = hanya gambar training (30 normal Belton + 30 OA KL2/3); 0 = semua gambar di --input')
    ap.add_argument('--meta_path', default='../meta/', help='folder meta/ (train_seed_*.csv, oa_seed_*.csv)')
    ap.add_argument('--train_seed', default='1001', help="seed Belton untuk set training, atau 'all' (10 seed)")
    ap.add_argument('--n_oa', type=int, default=30, help='(tidak dipakai lagi; daftar OA dibuat make_oa_pool.py)')
    args = ap.parse_args()

    if args.train_only:
        files, train_list = select_training_files(args.input, args.meta_path, args.train_seed, args.n_oa)
    else:
        files = sorted(f for f in glob.glob(os.path.join(args.input, '**', '*'), recursive=True)
                       if f.lower().endswith(IMG_EXT))
        train_list = None
    if not files:
        raise SystemExit('Tidak ada gambar di ' + args.input)
    os.makedirs(os.path.join(args.out, 'curves'), exist_ok=True)

    rows, overlays = [], {}
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('  [!] gagal dibaca:', f)
            continue
        stem, side, kl = parse_name(f)
        res, curves = localize(img, side)
        rel = os.path.relpath(f, args.input).replace('\\', '/')
        rows.append(dict(id=rel, file=stem, kl=kl, knee=side, **res))
        key = rel.replace('/', '__').rsplit('.', 1)[0]
        np.savez_compressed(os.path.join(args.out, 'curves', key + '.npz'), **curves)
        if args.overlay:
            d = os.path.join(args.out, 'overlay', 'KL{}'.format(kl) if kl is not None else 'all')
            os.makedirs(d, exist_ok=True)
            op = os.path.join(d, stem + '.png')
            cv2.imwrite(op, draw_overlay(img, res, curves, stem))
            overlays[rel] = op
        if (i + 1) % 500 == 0:
            print('  {}/{} gambar'.format(i + 1, len(files)))

    df = pd.DataFrame(rows)
    if train_list is not None:   # tandai peran tiap gambar: normal / oa
        roles = train_list.drop_duplicates('id').set_index('id')['role']
        df.insert(3, 'role', df['id'].map(roles))
    df.to_csv(os.path.join(args.out, 'landmarks.csv'), index=False)
    if args.overlay and args.qa_per_grade > 0:
        make_qa_sheet(df, overlays, args.qa_per_grade, args.seed, os.path.join(args.out, 'qa_sheet.png'))
    print('\nSelesai: {} gambar -> {}'.format(len(df), os.path.join(args.out, 'landmarks.csv')))
    flagged = df[df['flags'] != '']
    print('Perlu dicek manual (ada flag): {} gambar ({:.1f}%)'.format(len(flagged), 100 * len(flagged) / len(df)))
    if df['kl'].notna().any() and 'jsw_medial_min' in df:
        print('\nRata-rata JSW (piksel) per grade KL - harus cenderung turun saat KL naik:')
        print(df.groupby('kl')[['jsw_medial_min', 'jsw_lateral_min']].mean().round(2).to_string())


if __name__ == '__main__':
    main()
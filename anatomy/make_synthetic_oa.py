import argparse
import os

import cv2
import numpy as np
import pandas as pd

ROIS = ['medial_fem', 'medial_tib', 'lateral_fem', 'lateral_tib']
BELTON_SEEDS = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]

# (n_taji min, max), skala taji, kekuatan tempel (alpha), penyempitan celah (fraksi), peluang JSN lateral
GRADE_CFG = {
    1: dict(n=(1, 1), scale=(0.7, 0.9), alpha=(0.45, 0.65), jsn=(0.0, 0.0), p_lat=0.0),
    2: dict(n=(1, 2), scale=(0.9, 1.1), alpha=(0.8, 1.0), jsn=(0.0, 0.15), p_lat=0.15),
    3: dict(n=(2, 3), scale=(1.0, 1.25), alpha=(0.9, 1.0), jsn=(0.25, 0.45), p_lat=0.2),
    4: dict(n=(3, 4), scale=(1.15, 1.45), alpha=(1.0, 1.0), jsn=(0.55, 0.80), p_lat=0.2),
}


def norm_id(x):
    return '/'.join(str(x).replace('\\', '/').split('/')[-2:])


def key_of(i):
    return i.replace('/', '__').rsplit('.', 1)[0]


# ----------------------------------------------------------------------------- geometri
def roi_side(row, roi):
    """'left'/'right' di gambar (tidak di-flip) untuk ROI medial/lateral."""
    med = row.get('medial_on', '')
    if med not in ('left', 'right'):
        return None
    return med if roi.startswith('medial') else ('right' if med == 'left' else 'left')


def edge_point(row, roi):
    """(x tepi tulang, y garis sendi, arah keluar +1/-1) untuk ROI; None bila tidak valid."""
    side = roi_side(row, roi)
    if side is None:
        return None
    bone = roi.split('_')[1]
    flags = str(row.get('flags', ''))
    if 'edge_not_found' in flags and '{}_{}'.format(bone, side) in flags:
        return None
    try:
        x = float(row['{}_x_{}'.format(bone, side)])
        y = float(row['y_joint_{}'.format(side)])
    except (KeyError, ValueError, TypeError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y, (-1 if side == 'left' else 1)


def ellipse_mask(h, w, margin=2):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (max(1, w // 2 - margin), max(1, h // 2 - margin)), 0, 0, 360, 255, -1)
    return m


# ----------------------------------------------------------------------------- tanda OA (a): taji
def paste_patch(img, patch, cx, cy, alpha, blend='poisson'):
    """Tempel patch (grayscale) berpusat di (cx, cy). Mengembalikan (gambar baru, masker float 0..1).
    blend: 'poisson' (Poisson biasa: tekstur taji sumber, intensitas menyesuaikan tujuan),
           'mixed' (Poisson campuran: gradien terkuat dipertahankan, hasil lebih samar),
           'alpha' (campuran lembut dengan penyesuaian intensitas)."""
    H, W = img.shape
    ph, pw = patch.shape
    cx = int(np.clip(round(cx), pw // 2 + 2, W - pw // 2 - 3))
    cy = int(np.clip(round(cy), ph // 2 + 2, H - ph // 2 - 3))
    m8 = ellipse_mask(ph, pw)
    x0, y0 = cx - pw // 2, cy - ph // 2
    # masker lembut: elips lebih kecil lalu di-blur, supaya bernilai ~0 di tepi kotak (tanpa "kotak" terlihat)
    soft = cv2.GaussianBlur(ellipse_mask(ph, pw, margin=max(3, pw // 5)).astype(np.float32) / 255, (0, 0), max(1.0, pw / 10))
    full = np.zeros((H, W), np.float32)
    full[y0:y0 + ph, x0:x0 + pw] = soft
    if blend in ('poisson', 'mixed'):
        src3 = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        dst3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            cl = cv2.cvtColor(cv2.seamlessClone(src3, dst3, m8, (cx, cy),
                                                   cv2.MIXED_CLONE if blend == 'mixed' else cv2.NORMAL_CLONE), cv2.COLOR_BGR2GRAY)
        except cv2.error:
            blend = 'alpha'
    if blend == 'alpha':
        cl = img.astype(np.float32).copy()
        reg = cl[y0:y0 + ph, x0:x0 + pw]
        p = patch.astype(np.float32)
        # samakan rata-rata intensitas tepi patch dengan tujuan supaya tidak ada "kotak"
        ring = (m8 == 0)
        if ring.any():
            p = p - p[ring].mean() + reg[ring].mean()
        cl[y0:y0 + ph, x0:x0 + pw] = p
    # campuran halus: masker elips lembut x kekuatan alpha
    out = img.astype(np.float32) * (1 - alpha * full) + cl.astype(np.float32) * (alpha * full)
    return np.clip(out, 0, 255).astype(np.uint8), full


# ----------------------------------------------------------------------------- tanda OA (b): penyempitan
def narrow_joint(img, row, curves, compartment, frac, taper=0.15):
    """Persempit celah sendi kompartemen ('medial'/'lateral') sebesar `frac` (0..0.9) dengan warp vertikal
    monoton & halus. Mengembalikan (gambar, masker float, penyempitan rata-rata px)."""
    H, W = img.shape
    if frac <= 0 or curves is None:
        return img, np.zeros((H, W), np.float32), 0.0
    side = roi_side(row, compartment + '_fem')
    if side is None:
        return img, np.zeros((H, W), np.float32), 0.0
    xs = np.asarray(curves['xs']).astype(int)
    fem = np.asarray(curves['femur']).astype(np.float32)
    tib = np.asarray(curves['tibia']).astype(np.float32)
    xl, xr, xc = int(row['x_left']), int(row['x_right']), int(row['x_center'])
    a0, a1 = (xl, xc) if side == 'left' else (xc, xr)
    if a1 - a0 < 10:
        return img, np.zeros((H, W), np.float32), 0.0
    # bobot horizontal: 1 di bagian tengah kompartemen, turun halus ke 0 di ujung (hindari spine & tepi)
    u = (np.arange(W) - a0) / float(a1 - a0)
    wx = np.clip(np.minimum(u, 1 - u) / taper, 0, 1)
    wx = 0.5 - 0.5 * np.cos(np.pi * wx)
    gap = np.zeros(W, np.float32)
    mid = np.zeros(W, np.float32)
    ok = (xs >= 0) & (xs < W)
    gap[xs[ok]] = np.maximum(tib[ok] - fem[ok], 0)
    mid[xs[ok]] = (tib[ok] + fem[ok]) / 2
    wx[gap <= 1] = 0
    a = frac * gap / 2 * wx                     # tiap tulang bergeser a px ke arah garis sendi
    R = gap / 2 + 14                            # jangkauan warp (px)
    tau = 1.5
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = yy - mid[None, :]
    g = np.exp(-(np.maximum(np.abs(t) - gap[None, :] / 2, 0) / R[None, :]) ** 2)
    disp = a[None, :] * np.tanh(t / tau) * g
    map_y = (yy + disp).astype(np.float32)
    out = cv2.remap(img, xx, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = ((np.abs(t) <= gap[None, :] / 2 + 3) & (wx[None, :] > 0.3)).astype(np.float32)
    used = wx > 0.99
    return out, mask, float((2 * a[used]).mean()) if used.any() else 0.0


# ----------------------------------------------------------------------------- generator
class OAGenerator:
    def __init__(self, bank_dir, img_root='../data', landmarks='../anatomy_out/train/landmarks.csv',
                 curves_dir=None, meta_path='../meta/', patch=32, sumber='bank', komponen=('taji', 'jsn'),
                 blend='poisson', grade_cfg=None, taji_ids=None, max_try=6, max_darken=6.0):
        self.img_root, self.patch, self.sumber, self.blend = img_root, patch, sumber, blend
        self.komponen = set(komponen)
        self.max_try, self.max_darken = max_try, max_darken
        self.cfg = grade_cfg or GRADE_CFG
        self.curves_dir = curves_dir or os.path.join(os.path.dirname(landmarks), 'curves')
        lm = pd.read_csv(landmarks, keep_default_na=False)
        lm['id'] = lm['id'].map(norm_id)
        self.lm = lm.drop_duplicates('id').set_index('id')
        f = os.path.join(bank_dir, 'osteophyte_bank.csv')
        if not os.path.exists(f):
            raise SystemExit('Tidak ada {} (jalankan extract_osteophyte_bank.py dulu)'.format(f))
        b = pd.read_csv(f)
        b = b[b['pakai_auto'] == 1].copy()
        b['id'] = b['id'].map(norm_id)
        if taji_ids is not None:                     # mis. hanya taji dari daftar OA seed tertentu
            b = b[b['id'].isin(set(map(norm_id, taji_ids)))]
        b = b[b['id'].isin(self.lm.index)]
        if b.empty:
            raise SystemExit('Bank taji kosong setelah disaring')
        # posisi taji relatif terhadap tepi tulang & garis sendi gambar sumbernya
        rel = []
        for _, r in b.iterrows():
            e = edge_point(self.lm.loc[r['id']], r['roi'])
            rel.append((np.nan, np.nan, 0) if e is None else ((r['cx'] - e[0]) * e[2], r['cy'] - e[1], e[2]))
        b['dx_out'], b['dy'], b['out_dir'] = zip(*rel)
        self.bank = b[np.isfinite(b['dx_out'])].reset_index(drop=True)
        self.by_roi = {r: self.bank[self.bank.roi == r] for r in ROIS}
        n = np.array([len(self.by_roi[r]) for r in ROIS], float)
        self.roi_p = dict(zip(ROIS, n / n.sum()))     # medial lebih sering, mengikuti data asli
        self._img_cache, self._curve_cache = {}, {}
        self.normal_pool = None
        if sumber == 'acak':
            self.normal_pool = pool_normals(meta_path)

    # -- I/O dengan cache kecil
    def read(self, i):
        if i not in self._img_cache:
            im = cv2.imread(os.path.join(self.img_root, 'train', *i.split('/')), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise SystemExit('Gagal membaca gambar ' + i)
            if len(self._img_cache) > 512:
                self._img_cache.clear()
            self._img_cache[i] = im
        return self._img_cache[i]

    def curves(self, i):
        if i not in self._curve_cache:
            f = os.path.join(self.curves_dir, key_of(i) + '.npz')
            self._curve_cache[i] = dict(np.load(f)) if os.path.exists(f) else None
        return self._curve_cache[i]

    def valid_target(self, i):
        if i not in self.lm.index:
            return False
        row = self.lm.loc[i]
        return any(edge_point(row, r) is not None for r in ROIS)

    # -- ambil potongan taji
    def _get_patch(self, roi, out_dir_t, grade, scale, rng):
        cand = self.by_roi[roi]
        if cand.empty:
            return None
        w = np.where(cand['kl'].values >= 3, 2.0, 1.0) if grade >= 3 else np.ones(len(cand))
        r = cand.iloc[rng.choice(len(cand), p=w / w.sum())]
        P = self.patch
        if self.sumber == 'acak':                      # KONTROL: potongan acak dari gambar normal lain
            j = self.normal_pool[rng.integers(len(self.normal_pool))]
            src = self.read(j)
            cx, cy = rng.integers(P, src.shape[1] - P), rng.integers(P, src.shape[0] - P)
        else:
            src = self.read(r['id'])
            cx, cy = r['cx'], r['cy']
        h = P // 2
        pad = cv2.copyMakeBorder(src, h, h, h, h, cv2.BORDER_REFLECT)
        x, y = int(round(cx)) + h, int(round(cy)) + h
        patch = pad[y - h:y + h, x - h:x + h]
        if r['out_dir'] != out_dir_t:                  # sisi sumber & tujuan beda -> cermin horizontal
            patch = patch[:, ::-1]
        if abs(scale - 1) > 1e-3:
            s = max(8, int(round(P * scale)) // 2 * 2)
            patch = cv2.resize(patch, (s, s), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(patch), r

    def synthesize(self, img, target_id, grade, rng, return_info=False):
        """img: uint8 HxW lutut NORMAL (gambar asli, tidak di-flip). Mengembalikan dict img/mask/grade/..."""
        cfg = self.cfg[int(grade)]
        row = self.lm.loc[norm_id(target_id)]
        out = img.copy()
        mask = np.zeros(img.shape, np.float32)
        info = dict(grade=int(grade), n_taji=0, jsn_frac=0.0, jsn_px=0.0, taji=[], ditolak_bingkai=0, ditolak_gelap=0)
        # (b) penyempitan celah dulu (supaya taji ditempel pada geometri akhir)
        if 'jsn' in self.komponen and cfg['jsn'][1] > 0:
            frac = float(rng.uniform(*cfg['jsn']))
            comp = 'lateral' if rng.random() < cfg['p_lat'] else 'medial'
            out, m, px = narrow_joint(out, row, self.curves(norm_id(target_id)), comp, frac)
            if px > 0:
                mask = np.maximum(mask, m)
                info.update(jsn_frac=round(frac, 3), jsn_px=round(px, 2), jsn_comp=comp)
        # (a) taji di ROI yang tersedia (tanpa mengulang ROI yang sama). Setiap taji dicoba ulang (sampai
        #     max_try kali, boleh ganti taji/ROI) bila posisinya jatuh di luar bingkai atau hasil tempelnya
        #     MENGGELAPKAN tulang (osteofit = tambahan tulang, tidak boleh membuat area lebih gelap).
        if 'taji' in self.komponen:
            avail = [r for r in ROIS if edge_point(row, r) is not None and len(self.by_roi[r])]
            k = int(min(len(avail), rng.integers(cfg['n'][0], cfg['n'][1] + 1)))
            used, tries = set(), 0
            while len(info['taji']) < k and tries < self.max_try * max(k, 1):
                tries += 1
                left = [r for r in avail if r not in used]
                if not left:
                    break
                p = np.array([self.roi_p[r] for r in left]); p = p / p.sum()
                roi = left[rng.choice(len(left), p=p)]
                ex, ey, od = edge_point(row, roi)
                scale = float(rng.uniform(*cfg['scale']))
                got = self._get_patch(roi, od, grade, scale, rng)
                if got is None:
                    info['ditolak_bingkai'] += 1
                    continue
                patch, src = got
                # posisi relatif yang sama terhadap tepi tulang & garis sendi (tidak ikut diskalakan)
                cx = ex + od * src['dx_out'] + rng.normal(0, 1.0)
                cy = ey + src['dy'] + rng.normal(0, 1.0)
                hw = patch.shape[1] / 2
                if not (hw / 2 <= cx <= img.shape[1] - hw / 2):     # sebagian besar di luar bingkai
                    info['ditolak_bingkai'] += 1
                    continue
                new, m = paste_patch(out, patch, cx, cy, float(rng.uniform(*cfg['alpha'])), self.blend)
                core = m > 0.5
                if core.any() and (new[core].astype(np.float32) - out[core]).mean() < -self.max_darken:
                    info['ditolak_gelap'] += 1
                    continue
                out = new
                mask = np.maximum(mask, m)
                used.add(roi)
                info['taji'].append(dict(roi=roi, src=src['id'], src_kl=int(src['kl']), cx=round(cx, 1),
                                         cy=round(cy, 1), scale=round(scale, 2)))
            info['n_taji'] = len(info['taji'])
        res = dict(img=out, mask=np.clip(mask, 0, 1), grade=int(grade), n_taji=info['n_taji'],
                   jsn_frac=info['jsn_frac'])
        if return_info:
            res['info'] = info
        return res


def pool_normals(meta_path):
    ids = []
    for s in BELTON_SEEDS:
        f = os.path.join(meta_path, 'train_seed_{}.csv'.format(s))
        if os.path.exists(f):
            ids += [norm_id(i) for i in pd.read_csv(f)['id']]
    if not ids:
        raise SystemExit('Tidak ada meta/train_seed_*.csv')
    return sorted(set(ids))


# ----------------------------------------------------------------------------- skrip: cek visual + validasi
def main():
    ap = argparse.ArgumentParser(description='K2 tahap 2: generator OA sintetis (cek visual & validasi)')
    ap.add_argument('--bank', default='../k2_out/bank_bersih', help='folder output extract_osteophyte_bank.py')
    ap.add_argument('--out', default=None, help='default: ../k2_out/sintetis_<nama bank>')
    ap.add_argument('--img_root', default='../data')
    ap.add_argument('--meta_path', default='../meta/')
    ap.add_argument('--landmarks', default='../anatomy_out/train/landmarks.csv')
    ap.add_argument('--sumber', default='bank', choices=['bank', 'acak'], help="'acak' = kontrol ablation")
    ap.add_argument('--komponen', default='taji,jsn', help='taji,jsn | taji | jsn')
    ap.add_argument('--blend', default='poisson', choices=['poisson', 'mixed', 'alpha'],
                    help="poisson (default) | mixed (lebih samar) | alpha; pilihan akhir di val (tahap 3)")
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--max_darken', type=float, default=6.0,
                    help='tolak taji bila rata-rata intensitas area tempel turun lebih dari ini (gray level)')
    ap.add_argument('--n_stat', type=int, default=300, help='jumlah sintesis untuk statistik jumlah taji')
    ap.add_argument('--grades', default='1,2,3,4')
    ap.add_argument('--n_preview', type=int, default=6, help='jumlah lutut normal di preview_grid.png')
    ap.add_argument('--n_valid', type=int, default=60, help='jumlah lutut normal untuk validasi JSN (0 = lewati)')
    ap.add_argument('--n_save', type=int, default=40, help='jumlah gambar sintetis contoh yang disimpan')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out = args.out or os.path.join('../k2_out', 'sintetis_' + os.path.basename(os.path.normpath(args.bank)))
    os.makedirs(os.path.join(out, 'contoh'), exist_ok=True)
    grades = [int(g) for g in args.grades.split(',')]
    gen = OAGenerator(args.bank, args.img_root, args.landmarks, meta_path=args.meta_path, patch=args.patch,
                      sumber=args.sumber, komponen=[k.strip() for k in args.komponen.split(',')], blend=args.blend,
                      max_darken=args.max_darken)
    targets = [i for i in pool_normals(args.meta_path) if gen.valid_target(i)]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(targets)
    print('  Bank: {} taji ({}) | target normal valid: {}'.format(
        len(gen.bank), ', '.join('{} {}'.format(r, len(gen.by_roi[r])) for r in ROIS), len(targets)))

    # --- preview: baris = lutut normal; kolom = asli | grade 1..4 | masker grade tertinggi
    T = 160
    lab = lambda im, t: cv2.putText(cv2.cvtColor(cv2.resize(im, (T, T)), cv2.COLOR_GRAY2BGR), t, (4, 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    rows, log = [], []
    for n, i in enumerate(targets[:args.n_preview]):
        img = gen.read(i)
        tiles = [lab(img, 'normal ' + i.split('/')[1][:8])]
        last = None
        for g in grades:
            r = gen.synthesize(img, i, g, np.random.default_rng(args.seed * 1000 + n * 10 + g), return_info=True)
            tiles.append(lab(r['img'], 'g{} taji{} jsn{:.0f}%'.format(g, r['n_taji'], 100 * r['jsn_frac'])))
            last = r
        ov = cv2.cvtColor(cv2.resize(img, (T, T)), cv2.COLOR_GRAY2BGR)
        mk = cv2.resize(last['mask'], (T, T))
        ov[..., 2] = np.clip(ov[..., 2] + 150 * mk, 0, 255).astype(np.uint8)
        tiles.append(cv2.putText(ov, 'masker g{}'.format(grades[-1]), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1))
        rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(os.path.join(out, 'preview_grid.png'), np.vstack(rows))

    # --- contoh gambar sintetis (untuk dilihat / gambar paper)
    for n in range(min(args.n_save, len(targets))):
        i = targets[n % len(targets)]
        g = grades[n % len(grades)]
        r = gen.synthesize(gen.read(i), i, g, np.random.default_rng(10 ** 6 + n), return_info=True)
        cv2.imwrite(os.path.join(out, 'contoh', 'g{}__{}.png'.format(g, key_of(i))), r['img'])
        log.append(dict(id=i, grade=g, n_taji=r['n_taji'], jsn_frac=r['jsn_frac'],
                        taji=';'.join('{}<-{}'.format(t['roi'], t['src']) for t in r['info']['taji'])))
    pd.DataFrame(log).to_csv(os.path.join(out, 'contoh_log.csv'), index=False)

    # --- statistik: apakah jumlah taji sesuai target grade (taji gagal = sampel "OA" tanpa tanda OA)
    st = []
    for n in range(args.n_stat if 'taji' in gen.komponen else 0):
        i = targets[n % len(targets)]
        g = grades[n % len(grades)]
        r = gen.synthesize(gen.read(i), i, g, np.random.default_rng(3 * 10 ** 6 + n), return_info=True)
        st.append(dict(grade=g, n_taji=r['n_taji'], target_min=gen.cfg[g]['n'][0],
                       bingkai=r['info']['ditolak_bingkai'], gelap=r['info']['ditolak_gelap']))
    stat_lines = []
    if st:
        st = pd.DataFrame(st)
        stat_lines = ['JUMLAH TAJI TERTEMPEL ({} sintesis):'.format(len(st)),
                      '{:>6} {:>10} {:>12} {:>16} {:>14} {:>12}'.format(
                          'grade', 'target', 'rata-rata', '< target min (%)', 'tolak bingkai', 'tolak gelap')]
        for g, d in st.groupby('grade'):
            c = gen.cfg[g]['n']
            stat_lines.append('{:>6} {:>10} {:>12.2f} {:>16.1f} {:>14.2f} {:>12.2f}'.format(
                g, '{}-{}'.format(*c), d['n_taji'].mean(), 100 * (d['n_taji'] < d['target_min']).mean(),
                d['bingkai'].mean(), d['gelap'].mean()))
        stat_lines.append('-> "< target min" idealnya ~0%; tolak = rata-rata percobaan yang ditolak per sintesis.')
        stat_lines.append('')

    # --- validasi: JSW medial/lateral diukur ULANG oleh localize_joint (bukti penyempitan benar terjadi)
    lines = []
    if args.n_valid > 0 and 'jsn' in gen.komponen:
        import localize_joint as LJ
        vrows = []
        for n, i in enumerate(targets[:args.n_valid]):
            img = gen.read(i)
            side = i.split('/')[1].rsplit('.', 1)[0][-1:].upper()
            base, _ = LJ.localize(img, side)
            for g in grades:
                r = gen.synthesize(img, i, g, np.random.default_rng(2 * 10 ** 6 + n * 10 + g), return_info=True)
                after, _ = LJ.localize(r['img'], side)
                comp = r['info'].get('jsn_comp', 'medial')
                k = 'jsw_{}_min'.format(comp)
                vrows.append(dict(id=i, grade=g, kompartemen=comp, jsn_frac=r['jsn_frac'], n_taji=r['n_taji'],
                                  jsw_sebelum=base.get(k), jsw_sesudah=after.get(k)))
            print('  validasi {}/{}'.format(n + 1, min(args.n_valid, len(targets))), end='\r')
        print()
        v = pd.DataFrame(vrows)
        v['turun_pct'] = 100 * (v['jsw_sebelum'] - v['jsw_sesudah']) / v['jsw_sebelum'].where(v['jsw_sebelum'] > 0)
        v.to_csv(os.path.join(out, 'validasi_jsn.csv'), index=False)
        lines.append('VALIDASI PENYEMPITAN (JSW min diukur ulang dengan localize_joint, {} lutut normal):'.format(
            v['id'].nunique()))
        lines.append('{:>6} {:>12} {:>14} {:>14} {:>12}'.format('grade', 'target JSN', 'JSW sebelum', 'JSW sesudah', 'turun (%)'))
        for g, d in v.groupby('grade'):
            lines.append('{:>6} {:>11.0f}% {:>14.2f} {:>14.2f} {:>12.1f}'.format(
                g, 100 * d['jsn_frac'].mean(), d['jsw_sebelum'].mean(), d['jsw_sesudah'].mean(), d['turun_pct'].median()))
        lines.append('-> "turun (%)" harus naik mengikuti grade dan kira-kira sesuai target JSN.')

    head = ['K2 TAHAP 2 - GENERATOR OA SINTETIS',
            'bank: {} | sumber: {} | komponen: {} | blend: {}'.format(args.bank, args.sumber, args.komponen, args.blend),
            'taji terpakai: {} dari {} gambar OA | per ROI: {}'.format(
                len(gen.bank), gen.bank['id'].nunique(), ', '.join('{} {}'.format(r, len(gen.by_roi[r])) for r in ROIS)),
            'target normal valid: {}'.format(len(targets)), '']
    txt = '\n'.join(head + stat_lines + lines)
    open(os.path.join(out, 'ringkasan.txt'), 'w', encoding='utf-8').write(txt + '\n')
    print(txt)
    print('\nCek visual: {}'.format(os.path.join(out, 'preview_grid.png')))


if __name__ == '__main__':
    main()
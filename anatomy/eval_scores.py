import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

CELL = 8      # piksel per sel peta (224 / 28)


def norm_id(x):
    return '/'.join(str(x).replace('\\', '/').split('/')[-2:])


def metrics(score, label):
    score, label = np.asarray(score, float), np.asarray(label)
    auc = lambda y: roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan
    return dict(SRC=stats.spearmanr(score, label)[0], AUC_KL=auc(label > 1),
                AUC_KLgt0=auc(label > 0), AUC_KLge3=auc(label > 2), AUC_KL4=auc(label == 4))


def region_masks(lm, H, W, band, pad):
    """Mask boolean (H x W sel) per gambar untuk region joint_band dan margins."""
    yc = (np.arange(H) * CELL + CELL / 2)[:, None]          # pusat sel (piksel)
    xc = (np.arange(W) * CELL + CELL / 2)[None, :]
    aligned = 'joint_y' in lm.columns
    jb, mg = [], []
    for _, r in lm.iterrows():
        if aligned:
            yj = np.full((1, W), float(r['joint_y']))
        else:   # garis sendi miring: interpolasi linear antar ujung kiri-kanan
            xl, xr = float(r['x_left']), float(r['x_right'])
            t = np.clip((xc - xl) / max(xr - xl, 1), 0, 1)
            yj = float(r['y_joint_left']) + t * (float(r['y_joint_right']) - float(r['y_joint_left']))
        jb.append(np.abs(yc - yj) <= band)
        m = np.zeros((H, W), bool)
        for anat in ('medial', 'lateral'):
            for bone in ('fem', 'tib'):
                k = '{}_{}'.format(anat, bone)
                if k + '_x' not in r or pd.isna(r[k + '_x']) or r[k + '_x'] == '':
                    continue
                x0, y0 = float(r[k + '_x']) - pad, float(r[k + '_y']) - pad
                x1, y1 = x0 + float(r[k + '_w']) + 2 * pad, y0 + float(r[k + '_h']) + 2 * pad
                m |= (yc >= y0) & (yc <= y1) & (xc >= x0) & (xc <= x1)
        mg.append(m)
    return np.stack(jb), np.stack(mg)


def aggregate(maps, masks, how):
    out = []
    for m, k in zip(maps, masks):
        v = m[k] if k is not None and k.any() else m.ravel()
        v = np.sort(v)[::-1]
        out.append(v[0] if how == 'max' else v[:3].mean() if how == 'top3' else v.mean())
    return np.array(out)


# ----------------------------------------------------------------------------- bootstrap
def _rank_auc(y, s_rank):
    """AUC dari peringkat (Mann-Whitney); y biner, s_rank = rankdata(skor)."""
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (s_rank[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _fast_metrics(score, label):
    """AUC_KL (KL>=2) dan SRC, cepat untuk bootstrap."""
    rs = stats.rankdata(score)
    rl = stats.rankdata(label)
    src = np.corrcoef(rs, rl)[0, 1]
    return _rank_auc(label > 1, rs), src


def bootstrap(score_a, label, score_b=None, n_boot=1000, seed=0):
    """CI 95% AUC_KL & SRC untuk A; bila B diberikan: CI selisih (A-B) + p-value dua sisi (berpasangan)."""
    rng = np.random.default_rng(seed)
    n = len(label)
    A, D = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a = _fast_metrics(score_a[i], label[i])
        A.append(a)
        if score_b is not None:
            b = _fast_metrics(score_b[i], label[i])
            D.append((a[0] - b[0], a[1] - b[1]))
    A = np.array(A)
    out = dict(AUC_KL_lo=np.nanpercentile(A[:, 0], 2.5), AUC_KL_hi=np.nanpercentile(A[:, 0], 97.5),
               SRC_lo=np.nanpercentile(A[:, 1], 2.5), SRC_hi=np.nanpercentile(A[:, 1], 97.5))
    if D:
        D = np.array(D)
        for k, j in (('AUC_KL', 0), ('SRC', 1)):
            d = D[:, j][~np.isnan(D[:, j])]
            out['d_' + k] = float(np.mean(d))
            out['d_' + k + '_lo'], out['d_' + k + '_hi'] = np.percentile(d, [2.5, 97.5])
            out['p_' + k] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return out


def all_configs(maps_path, landmarks_path, band, pad):
    """{(region, agregasi): skor per gambar}, label, id."""
    d = np.load(maps_path)
    ids, labels, maps = [norm_id(i) for i in d['ids']], np.asarray(d['labels']), d['maps'].astype(np.float32)
    lm = pd.read_csv(landmarks_path, keep_default_na=False)
    lm['id'] = lm['id'].map(norm_id)
    missing = set(ids) - set(lm['id'])
    if missing:
        raise SystemExit('{} id di maps tidak ada di landmarks ({}). Pastikan split & jenis landmark cocok.'.format(
            len(missing), sorted(missing)[:2]))
    lm = lm.drop_duplicates('id').set_index('id').reindex(ids).reset_index()
    H, W = maps.shape[1:]
    jb, mg = region_masks(lm, H, W, band, pad)
    cfg = {}
    for region, masks in (('all', [None] * len(maps)), ('joint_band', jb), ('margins', mg)):
        for how in ('max', 'top3', 'mean'):
            cfg[(region, how)] = aggregate(maps, masks, how)
    return cfg, labels, ids


def fmt_p(p, n_boot):
    return '<{:.3f}'.format(1.0 / n_boot) if p == 0 else '{:.3f}'.format(p)


REGIONS, AGGS = ('all', 'joint_band', 'margins'), ('max', 'top3', 'mean')


def parse_cfg(txt):
    """'joint_band:mean' -> ('joint_band', 'mean'); pesan jelas bila format salah."""
    parts = [p.strip() for p in str(txt).split(':')]
    if len(parts) != 2 or parts[0] not in REGIONS or parts[1] not in AGGS:
        raise SystemExit("Kombinasi '{}' tidak valid. Format: region:agregasi, region = {}, agregasi = {}. "
                         "Contoh: --config joint_band:mean".format(txt, '/'.join(REGIONS), '/'.join(AGGS)))
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser(description='Evaluasi region & agregasi skor dari peta PatchCore (+ bootstrap)')
    ap.add_argument('--maps', required=True, help='maps_<split>.npz dari patchcore_k1.py')
    ap.add_argument('--landmarks', required=True, help='aligned_landmarks.csv (tersejajar) atau landmarks.csv (asli)')
    ap.add_argument('--band', type=float, default=32, help='setengah tinggi pita garis sendi (px)')
    ap.add_argument('--pad', type=float, default=8, help='perluasan ROI area tepi (px)')
    ap.add_argument('--n_boot', type=int, default=1000, help='jumlah ulangan bootstrap (0 = tanpa CI)')
    ap.add_argument('--ref_maps', default=None, help='peta metode PEMBANDING (mis. PatchCore global) untuk uji beda')
    ap.add_argument('--ref_landmarks', default=None, help='landmark untuk --ref_maps')
    ap.add_argument('--config', default=None, help="satu kombinasi metode ini, mis. 'joint_band:mean'")
    ap.add_argument('--ref_config', default=None,
                    help="kombinasi pembanding, mis. 'all:max'. Tanpa --ref_maps = dibandingkan dengan peta yang sama")
    ap.add_argument('--out', default=None, help='csv hasil (default: di sebelah file maps)')
    args = ap.parse_args()

    cfg, labels, ids = all_configs(args.maps, args.landmarks, args.band, args.pad)
    ref_cfg = None
    if args.ref_maps:
        ref_cfg, ref_labels, ref_ids = all_configs(args.ref_maps, args.ref_landmarks or args.landmarks, args.band, args.pad)
        if ref_ids != ids:   # samakan urutan gambar untuk uji berpasangan
            pos = {i: k for k, i in enumerate(ref_ids)}
            if set(pos) != set(ids):
                raise SystemExit('Gambar di --maps dan --ref_maps berbeda; harus split yang sama.')
            order = [pos[i] for i in ids]
            ref_cfg = {k: v[order] for k, v in ref_cfg.items()}

    # ---- mode 1: satu perbandingan spesifik
    if args.config or args.ref_config:
        a_key = parse_cfg(args.config) if args.config else ('all', 'max')
        b_key = parse_cfg(args.ref_config) if args.ref_config else a_key
        a = cfg[a_key]
        b = (ref_cfg or cfg)[b_key]
        r = metrics(a, labels); rb = metrics(b, labels)
        bs = bootstrap(a, labels, b, n_boot=max(args.n_boot, 1))
        nm_a = '{} [{}:{}]'.format(args.maps, *a_key)
        nm_b = '{} [{}:{}]'.format(args.ref_maps or args.maps, *b_key)
        print('\nA = ' + nm_a + '\nB = ' + nm_b + '   ({} gambar, {} bootstrap)\n'.format(len(labels), args.n_boot))
        print('            AUC_KL                     SRC')
        print('A        {:.4f}                     {:.4f}'.format(r['AUC_KL'], r['SRC']))
        print('B        {:.4f}                     {:.4f}'.format(rb['AUC_KL'], rb['SRC']))
        print('A - B   {:+.4f} [{:+.4f}, {:+.4f}]  {:+.4f} [{:+.4f}, {:+.4f}]'.format(
            bs['d_AUC_KL'], bs['d_AUC_KL_lo'], bs['d_AUC_KL_hi'], bs['d_SRC'], bs['d_SRC_lo'], bs['d_SRC_hi']))
        print('p-value  {:<26} {}'.format(fmt_p(bs['p_AUC_KL'], args.n_boot), fmt_p(bs['p_SRC'], args.n_boot)))
        for k in ('AUC_KL', 'SRC'):
            sig = bs['p_' + k] < 0.05
            print('  {}: {}'.format(k, ('A LEBIH BAIK secara signifikan' if bs['d_' + k] > 0 else
                                         'A LEBIH BURUK secara signifikan') if sig else
                                        'TIDAK berbeda signifikan (p >= 0.05)'))
        return

    # ---- mode 2: tabel semua kombinasi + CI (+ uji beda vs pembanding pada kombinasi yang sama)
    rows = []
    for (region, how), sc in cfg.items():
        r = metrics(sc, labels)
        row = dict(region=region, agregasi=how, **{k: round(v, 4) for k, v in r.items()})
        if args.n_boot:
            bs = bootstrap(sc, labels, ref_cfg[(region, how)] if ref_cfg else None, n_boot=args.n_boot)
            row.update({k: round(float(v), 4) for k, v in bs.items()})
        row['pilih'] = round((r['AUC_KL'] + r['SRC']) / 2, 4)
        rows.append(row)
    res = pd.DataFrame(rows)

    print('\nPeta: {}  ({} gambar{})\n'.format(args.maps, len(labels),
                                            ', {} bootstrap'.format(args.n_boot) if args.n_boot else ''))
    show = res.copy()
    if args.n_boot:
        show['AUC_KL [CI95]'] = show.apply(lambda x: '{:.4f} [{:.3f}-{:.3f}]'.format(x.AUC_KL, x.AUC_KL_lo, x.AUC_KL_hi), axis=1)
        show['SRC [CI95]'] = show.apply(lambda x: '{:.4f} [{:.3f}-{:.3f}]'.format(x.SRC, x.SRC_lo, x.SRC_hi), axis=1)
        cols = ['region', 'agregasi', 'AUC_KL [CI95]', 'SRC [CI95]', 'AUC_KL4', 'pilih']
        if ref_cfg:
            show['beda AUC (p)'] = show.apply(lambda x: '{:+.4f} ({})'.format(x.d_AUC_KL, fmt_p(x.p_AUC_KL, args.n_boot)), axis=1)
            show['beda SRC (p)'] = show.apply(lambda x: '{:+.4f} ({})'.format(x.d_SRC, fmt_p(x.p_SRC, args.n_boot)), axis=1)
            cols += ['beda AUC (p)', 'beda SRC (p)']
    else:
        cols = ['region', 'agregasi', 'SRC', 'AUC_KL', 'AUC_KLgt0', 'AUC_KLge3', 'AUC_KL4', 'pilih']
    print(show[cols].to_string(index=False))
    if ref_cfg:
        print('\nbeda = metode ini - pembanding ({}) pada kombinasi yang sama; p < 0.05 = signifikan'.format(args.ref_maps))

    best = res.sort_values('pilih', ascending=False).iloc[0]
    print('\nAturan pilih: skor "pilih" = (AUC_KL + SRC) / 2 tertinggi')
    print('Terpilih: region={} agregasi={} (AUC_KL {:.4f}, SRC {:.4f})'.format(
        best['region'], best['agregasi'], best['AUC_KL'], best['SRC']))
    print('Ingat: pilih di VAL, laporkan TEST untuk pilihan yang sama.')
    out = args.out or os.path.splitext(args.maps)[0] + '_eval.csv'
    res.to_csv(out, index=False)
    print('Tersimpan:', out)


if __name__ == '__main__':
    main()
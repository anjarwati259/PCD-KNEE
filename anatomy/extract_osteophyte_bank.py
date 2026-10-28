import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch

import align_joint as AJ
import eval_scores as ES
import patchcore_k1 as PC

BELTON_SEEDS = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
ROIS = ['medial_fem', 'medial_tib', 'lateral_fem', 'lateral_tib']
CELL = 8


# ----------------------------------------------------------------------------- bank dengan sumber patch
def coreset_with_idx(X, ratio, device, proj_dim=128, seed=0):
    """Greedy coreset (sama dengan PatchCore) yang juga mengembalikan indeks baris terpilih."""
    n = X.shape[0]
    k = max(1, int(n * ratio))
    if k >= n:
        return X, torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(X.shape[1], proj_dim, generator=g) / proj_dim ** 0.5
    Z = (X @ P).to(device)
    idx = [int(torch.randint(n, (1,), generator=g))]
    mind = torch.cdist(Z, Z[idx[-1]:idx[-1] + 1]).squeeze(1)
    for _ in range(k - 1):
        j = int(torch.argmax(mind))
        idx.append(j)
        mind = torch.minimum(mind, torch.cdist(Z, Z[j:j + 1]).squeeze(1))
    idx = torch.tensor(idx)
    return X[idx], idx


def build_bank(feats, ratio, device):
    """feats: N x H x W x D -> (bank M x D, sumber_gambar M)."""
    N, H, W, D = feats.shape
    flat = feats.reshape(-1, D)
    src = torch.arange(N).repeat_interleave(H * W)
    bank, idx = coreset_with_idx(flat, ratio, device)
    return bank, src[idx]


def nn_dist(feat, bank, src=None, exclude=None, device='cpu', chunk=8192):
    """Jarak tiap patch (H x W) ke patch bank terdekat; patch bank dari gambar di `exclude`
    (satu indeks atau daftar indeks gambar) diabaikan."""
    if exclude is not None and not torch.is_tensor(exclude):
        exclude = torch.as_tensor(exclude if isinstance(exclude, (list, tuple)) else [exclude])
    if exclude is not None and exclude.numel() == 0:
        exclude = None
    H, W, D = feat.shape
    q = feat.reshape(-1, D).to(device)
    qq = (q * q).sum(1, keepdim=True)
    best = torch.full((q.shape[0],), float('inf'), device=device)
    for i in range(0, bank.shape[0], chunk):
        b = bank[i:i + chunk].to(device)
        d2 = qq + (b * b).sum(1)[None] - 2 * q @ b.T
        if exclude is not None:
            keep = (~torch.isin(src[i:i + chunk], exclude)).to(device)
            d2 = d2.masked_fill(~keep[None], float('inf'))
        best = torch.minimum(best, d2.min(1).values)
    return best.clamp_min(0).sqrt().reshape(H, W).cpu()


def patient_of(i):
    """'2/9011420L.png' -> '9011420' (lutut kiri & kanan satu pasien)."""
    stem = os.path.splitext(i.split('/')[-1])[0]
    return stem[:-1] if stem[-1:].upper() in ('L', 'R') else stem


# ----------------------------------------------------------------------------- geometri
def roi_cells(row, roi, pad, H=28, W=28):
    """Mask sel (H x W) yang pusatnya di dalam ROI, koordinat gambar asli.
    Perluasan `pad` hanya ke arah horizontal, dan batas vertikal setengah-terbuka, supaya ROI femur
    (di atas garis sendi) dan tibia (di bawahnya) tidak pernah berbagi sel."""
    if row.get(roi + '_x', '') in ('', None) or pd.isna(row.get(roi + '_x')):
        return None
    x0, y0 = float(row[roi + '_x']) - pad, float(row[roi + '_y'])
    x1, y1 = x0 + float(row[roi + '_w']) + 2 * pad, y0 + float(row[roi + '_h'])
    yc = (np.arange(H) * CELL + CELL / 2)[:, None]
    xc = (np.arange(W) * CELL + CELL / 2)[None, :]
    return (yc >= y0) & (yc < y1) & (xc >= x0) & (xc <= x1)


def roi_side(row, roi):
    """'left' / 'right' di gambar asli untuk ROI medial/lateral."""
    anat = roi.split('_')[0]
    med = row.get('medial_on', '')
    if med not in ('left', 'right'):
        return None
    return med if anat == 'medial' else ('right' if med == 'left' else 'left')


def protrusion(img, row, roi, band=12):
    """Seberapa jauh (px) tulang terang di pita garis sendi melewati tepi tulang hasil lokalisasi.
    Positif = menonjol keluar (khas osteofit marginal). NaN bila tepi tidak terdeteksi."""
    side = roi_side(row, roi)
    bone = roi.split('_')[1]
    if side is None:
        return np.nan
    flags = str(row.get('flags', ''))
    if 'edge_not_found' in flags and '{}_{}'.format(bone, side) in flags:
        return np.nan
    edge = float(row['{}_x_{}'.format(bone, side)])
    yj = float(row['y_joint_{}'.format(side)])
    y0, y1 = (int(yj - band), int(yj)) if bone == 'fem' else (int(yj), int(yj + band))
    H, W = img.shape
    y0, y1 = max(0, y0), min(H, y1)
    if y1 - y0 < 3:
        return np.nan
    strip = cv2.createCLAHE(2.0, (8, 8)).apply(img)[y0:y1]
    prof = cv2.GaussianBlur(strip.mean(0).astype(np.float32).reshape(1, -1), (0, 0), 1.5).ravel()
    thr = (np.percentile(prof, 90) + np.percentile(prof, 10)) / 2
    bonecols = np.where(prof > thr)[0]
    if len(bonecols) == 0:
        return np.nan
    outer = bonecols.min() if side == 'left' else bonecols.max()
    return float(edge - outer) if side == 'left' else float(outer - edge)


def crop_at(img, cx, cy, size):
    h = size // 2
    pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_REPLICATE)
    x, y = int(round(cx)) + h, int(round(cy)) + h
    return pad[y - h:y - h + size, x - h:x - h + size]


# ----------------------------------------------------------------------------- utama
def main():
    ap = argparse.ArgumentParser(description='K2 tahap 1: penambangan taji asli dari pool OA')
    ap.add_argument('--img_root', default='../data', help='gambar ASLI (mesin dual global C)')
    ap.add_argument('--meta_path', default='../meta/')
    ap.add_argument('--landmarks', default='../anatomy_out/train/landmarks.csv')
    ap.add_argument('--out', default='../k2_out/osteophyte_bank')
    ap.add_argument('--coreset', type=float, default=0.1)
    ap.add_argument('--sev_temp', type=float, default=0.1)
    ap.add_argument('--pad', type=float, default=8, help='perluasan ROI (px)')
    ap.add_argument('--pct', type=float, default=95, help='ambang: persentil skor ROI gambar normal')
    ap.add_argument('--crop', type=int, default=32, help='ukuran potongan taji (px)')
    ap.add_argument('--auto_rule', default='kandidat_tonjol', choices=['kandidat_tonjol', 'kandidat'],
                    help='aturan pemilihan otomatis taji yang dipakai tahap 2')
    ap.add_argument('--n_audit', type=int, default=50, help='jumlah taji acak untuk evaluasi kualitas')
    ap.add_argument('--tonjol_ref', default='median', choices=['median', 'p90'],
                    help="syarat tonjolan (aturan kandidat_tonjol): lebih menonjol dari 'median' lutut normal "
                         "(default) atau dari persentil 90 ('p90', lebih ketat)")
    ap.add_argument('--joint_y', type=float, default=133)
    ap.add_argument('--center_x', type=float, default=112)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--pretrained', type=int, default=1)
    args = ap.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    os.makedirs(os.path.join(args.out, 'crops'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'context'), exist_ok=True)
    torch.manual_seed(0)

    # --- daftar gambar: pool normal (gabungan 10 daftar Belton) + pool OA
    nid = lambda f: [PC.norm_id(i) for i in pd.read_csv(os.path.join(args.meta_path, f))['id']]
    normals = sorted(set(sum([nid('train_seed_{}.csv'.format(s)) for s in BELTON_SEEDS
                              if os.path.exists(os.path.join(args.meta_path, 'train_seed_{}.csv'.format(s)))], [])))
    if not os.path.exists(os.path.join(args.meta_path, 'oa_pool.csv')):
        raise SystemExit('Tidak ada meta/oa_pool.csv. Jalankan dulu: python make_oa_pool.py')
    oa = nid('oa_pool.csv')
    by_g = PC.split_by_grade(oa)
    path = lambda i: os.path.join(args.img_root, 'train', *i.split('/'))
    miss = [i for i in normals + oa if not os.path.exists(path(i))]
    if miss:
        raise SystemExit('{} gambar tidak ada, mis. {}'.format(len(miss), path(miss[0])))
    lm = pd.read_csv(args.landmarks, keep_default_na=False)
    lm['id'] = lm['id'].map(PC.norm_id)
    lm = lm.drop_duplicates('id').set_index('id')
    miss = [i for i in normals + oa if i not in lm.index]
    if miss:
        raise SystemExit('{} gambar tidak ada di {} (jalankan localize_joint.py untuk train)'.format(len(miss), args.landmarks))
    print('  Pool normal: {} gambar | pool OA: {} ({})'.format(
        len(normals), len(oa), ', '.join('KL{}={}'.format(g, len(v)) for g, v in by_g.items())))

    # --- fitur & bank (dual global C, dengan sumber patch untuk leave-one-out)
    model = PC.FeatureExtractor(pretrained=bool(args.pretrained)).to(device)
    f_n = PC.extract(model, [path(i) for i in normals], device, args.batch)
    f_a = PC.extract(model, [path(i) for i in oa], device, args.batch)
    bank_n, src_n = build_bank(f_n, args.coreset, device)
    banks = {g: build_bank(f_a[idx], args.coreset, device) + (idx,) for g, idx in by_g.items()}
    print('  Bank normal {} patch | '.format(bank_n.shape[0]) +
          ' | '.join('bank KL{} {} patch'.format(g, b[0].shape[0]) for g, b in banks.items()))

    # leave-one-out PER PASIEN: semua gambar pasien yang sama (lutut kiri/kanan) dikeluarkan dari SEMUA bank
    pat_n = [patient_of(i) for i in normals]
    pat_g = {g: [patient_of(oa[j]) for j in idx] for g, (_, _, idx) in banks.items()}

    def loo_map(feat, patient):
        dn = nn_dist(feat, bank_n, src_n, [k for k, p in enumerate(pat_n) if p == patient], device)
        dg = [(g, nn_dist(feat, b, src, [k for k, p in enumerate(pat_g[g]) if p == patient], device))
              for g, (b, src, idx) in banks.items()]
        return PC.grade_score(dn, dg, 'expected', temp=args.sev_temp)

    # --- 1) ambang dari gambar normal (leave-one-out pada bank normal)
    print('  Menghitung ambang dari {} gambar normal ...'.format(len(normals)))
    roi_scores = {r: [] for r in ROIS}
    prot_n = {r: [] for r in ROIS}
    for k, i in enumerate(normals):
        m = loo_map(f_n[k], patient_of(i)).numpy()
        row = lm.loc[i]
        img = cv2.imread(path(i), cv2.IMREAD_GRAYSCALE)
        for r in ROIS:
            mask = roi_cells(row, r, args.pad)
            if mask is not None and mask.any():
                roi_scores[r].append(float(m[mask].max()))
            pv = protrusion(img, row, r)
            if np.isfinite(pv):
                prot_n[r].append(pv)
        print('  {}/{} gambar normal'.format(k + 1, len(normals)), end='\r')
    print()
    thr = {r: float(np.percentile(v, args.pct)) if v else np.inf for r, v in roi_scores.items()}
    # tonjolan dibandingkan lutut NORMAL di ROI yang sama: median & persentil 90
    prot_ref = {r: (float(np.median(v)), float(np.percentile(v, 90))) if v else (np.nan, np.nan)
                for r, v in prot_n.items()}
    print('  Ambang skor (persentil {:g} normal): '.format(args.pct) +
          ', '.join('{} {:.3f}'.format(r, t) for r, t in thr.items()))
    print('  Tonjolan normal (median / p90, px): ' +
          ', '.join('{} {:.1f}/{:.1f}'.format(r, *prot_ref[r]) for r in ROIS))

    # --- 2) kandidat taji dari pool OA
    rows, maps = [], []
    for k, i in enumerate(oa):
        m = loo_map(f_a[k], patient_of(i))
        maps.append(m.numpy().astype(np.float16))
        m = m.numpy()
        row = lm.loc[i]
        img = cv2.imread(path(i), cv2.IMREAD_GRAYSCALE)
        W = img.shape[1]
        prm = AJ._rot_params(row, W) if row.get('medial_on', '') in ('left', 'right') else None
        for r in ROIS:
            mask = roi_cells(row, r, args.pad)
            if mask is None or not mask.any():
                continue
            mm = np.where(mask, m, -np.inf)
            cy_c, cx_c = np.unravel_index(int(np.argmax(mm)), mm.shape)
            sc = float(m[cy_c, cx_c])
            cx, cy = cx_c * CELL + CELL / 2, cy_c * CELL + CELL / 2
            if prm is not None:
                ax, ay = AJ.forward_points([cx], [cy], prm, W, args.joint_y, args.center_x)
                ax, ay = float(ax[0]), float(ay[0])
            else:
                ax = ay = np.nan
            key = '{}__{}'.format(i.replace('/', '__').rsplit('.', 1)[0], r)
            cand = sc > thr[r]
            rec = dict(id=i, kl=int(i.split('/')[0]), roi=r, score=round(sc, 4), thr=round(thr[r], 4),
                       kandidat=int(cand), cx=cx, cy=cy, cx_aligned=round(ax, 1), cy_aligned=round(ay, 1),
                       )
            pv = protrusion(img, row, r)
            med, p90 = prot_ref[r]
            rec['protrusion_rel_px'] = round(pv - med, 1) if np.isfinite(pv) else np.nan   # >0: lebih menonjol dari normal
            rec['tonjol_di_atas_p90_normal'] = int(np.isfinite(pv) and pv > p90)
            rec['tonjol_lolos'] = int(np.isfinite(pv) and pv > (med if args.tonjol_ref == 'median' else p90))
            rec['pakai_auto'] = int(cand and (args.auto_rule == 'kandidat' or rec['tonjol_lolos'] == 1))
            if cand:
                cv2.imwrite(os.path.join(args.out, 'crops', key + '.png'), crop_at(img, cx, cy, args.crop))
                ctx = cv2.cvtColor(crop_at(cv2.createCLAHE(2.0, (8, 8)).apply(img), cx, cy, 96), cv2.COLOR_GRAY2BGR)
                h = args.crop // 2
                cv2.rectangle(ctx, (48 - h, 48 - h), (48 + h, 48 + h), (0, 0, 255), 1)
                cv2.imwrite(os.path.join(args.out, 'context', key + '.png'), ctx)
                rec['crop'] = 'crops/{}.png'.format(key)
                rec['context'] = 'context/{}.png'.format(key)
            rows.append(rec)
        print('  {}/{} gambar OA'.format(k + 1, len(oa)), end='\r')
    print()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'osteophyte_bank.csv'), index=False)
    np.savez_compressed(os.path.join(args.out, 'maps_pool_loo.npz'), ids=np.array(oa), maps=np.stack(maps))

    # --- 3) lembar cek visual per ROI (urut skor tertinggi)
    for r in ROIS:
        d = df[(df.roi == r) & (df.pakai_auto == 1)].sort_values('score', ascending=False)
        tiles = []
        for _, x in d.iterrows():
            t = cv2.resize(cv2.imread(os.path.join(args.out, x['context'])), (144, 144), interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, 'KL{} {:.2f}'.format(x['kl'], x['score']), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            cv2.putText(t, x['id'].split('/')[1][:8], (3, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            tiles.append(t)
        if not tiles:
            continue
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[j:j + 8]) for j in range(0, len(tiles), 8)])
        cv2.imwrite(os.path.join(args.out, 'qa_taji_{}.png'.format(r)), sheet)

    # --- evaluasi kualitas: sampel acak dari taji terpakai (TIDAK untuk menyaring)
    used = df[df.pakai_auto == 1]
    aud = used.sample(n=min(args.n_audit, len(used)), random_state=0) if len(used) else used
    if len(aud):
        aud[['id', 'kl', 'roi', 'score', 'protrusion_rel_px', 'context']].assign(osteofit='').to_csv(
            os.path.join(args.out, 'audit_kualitas.csv'), index=False)
        tiles = []
        for n, (_, x) in enumerate(aud.iterrows()):
            t = cv2.resize(cv2.imread(os.path.join(args.out, x['context'])), (144, 144), interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, '#{} {}'.format(n + 1, x['roi']), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
            tiles.append(t)
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(args.out, 'qa_audit_acak.png'),
                    np.vstack([np.hstack(tiles[j:j + 8]) for j in range(0, len(tiles), 8)]))

    # --- ringkasan
    lines = ['K2 TAHAP 1 - BANK TAJI (penambangan otomatis)',
             'mesin: dual global C (expected), leave-one-out per pasien | ambang skor: persentil {:g} ROI normal'.format(args.pct),
             'aturan pemilihan otomatis: {}{}'.format(
                 args.auto_rule, ' (tonjolan > {} lutut normal)'.format(args.tonjol_ref)
                 if args.auto_rule == 'kandidat_tonjol' else ''),
             'pool: {} normal, {} OA'.format(len(normals), len(oa)), '']
    lines.append('{:<12} {:>9} {:>14} {:>11} {:>6} {:>6}'.format(
        'ROI', 'kandidat', 'tonjol>' + args.tonjol_ref, 'TERPAKAI', 'KL2', 'KL3'))
    for r in ROIS:
        d = df[df.roi == r]
        u = d[d.pakai_auto == 1]
        lines.append('{:<12} {:>9} {:>14} {:>11} {:>6} {:>6}'.format(
            r, int(d.kandidat.sum()), int(((d.kandidat == 1) & (d.tonjol_lolos == 1)).sum()),
            len(u), int((u.kl == 2).sum()), int((u.kl == 3).sum())))
    lines += ['', 'total taji terpakai: {} (dari {} gambar OA, {} ROI)'.format(
        len(used), used['id'].nunique() if len(used) else 0, len(df))]
    if len(used) < 40:
        lines.append('[!] taji terpakai sedikit (<40). Pertimbangkan --auto_rule kandidat atau --pct 90.')
    lines += ['', 'EVALUASI KUALITAS (opsional, untuk paper; tidak menyaring bank):',
              '  buka qa_audit_acak.png, isi kolom osteofit (1/0) di audit_kualitas.csv sesuai nomor #',
              '  -> laporkan: persentase taji yang terkonfirmasi osteofit']
    txt = '\n'.join(lines)
    print('\n' + txt)
    with open(os.path.join(args.out, 'ringkasan.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
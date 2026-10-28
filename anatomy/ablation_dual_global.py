import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import patchcore_k1 as PC


def score_global_dual(feat, bank_n, bank_a, device, src_n=None, src_a=None, ex_n=None, ex_a=None):
    """Skor per patch = jarak ke patch normal terdekat - jarak ke patch OA terdekat (keduanya global)."""
    return (PC.score_global(feat, bank_n, device, src=src_n, exclude=ex_n)
            - PC.score_global(feat, bank_a, device, src=src_a, exclude=ex_a))


def masked_patches(feats, masks):
    """feats: N x H x W x D, masks: N x H x W bool -> M x D patch yang lolos masker."""
    return feats[masks]


def masked_src(masks):
    """Indeks gambar asal untuk tiap patch hasil masked_patches (urutan sama)."""
    n = masks.shape[0]
    return torch.arange(n)[:, None, None].expand_as(masks)[masks]


def coreset_src(patches, src, ratio, device):
    """Coreset + gambar asal tiap patch terpilih."""
    bank, idx = PC.greedy_coreset(patches, ratio, device, return_idx=True)
    return bank, src[idx]


def main():
    ap = argparse.ArgumentParser(description='Ablation: dual bank global tanpa anatomi')
    ap.add_argument('--img_root', default='../data', help='gambar ASLI (tanpa penyejajaran)')
    ap.add_argument('--meta_path', default='../meta/')
    ap.add_argument('--train_seed', type=int, default=1001)
    ap.add_argument('--eval_split', default='test', choices=['test', 'val', 'train'],
                    help='train = skor semua gambar train (ekspor Stage 2 Belton), leave-one-out per pasien')
    ap.add_argument('--out', required=True)
    ap.add_argument('--oa_region', default='all', choices=['all', 'joint_band', 'margins'],
                    help="bagian gambar OA yang masuk bank OA. 'all' = tanpa anatomi sama sekali (default)")
    ap.add_argument('--landmarks_train', default='../anatomy_out/train/landmarks.csv',
                    help='landmark gambar asli train (hanya untuk --oa_region joint_band/margins)')
    ap.add_argument('--band', type=float, default=32)
    ap.add_argument('--pad', type=float, default=8)
    ap.add_argument('--coreset', type=float, default=0.1, help='rasio coreset kedua bank (PatchCore asli: 0.1)')
    ap.add_argument('--oa_split', default='none', choices=['none', 'kl'],
                    help='kl = K1C versi global: bank OA dipisah per grade (KL2 & KL3)')
    ap.add_argument('--sev_score', default='expected', choices=['expected', 'diff2'])
    ap.add_argument('--sev_lambda', type=float, default=0.5)
    ap.add_argument('--sev_temp', type=float, default=0.1)
    ap.add_argument('--dim', type=int, default=1024)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--pretrained', type=int, default=1)
    ap.add_argument('--summary', default='../k1_out/ringkasan_hasil.csv')
    args = ap.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    if device != args.device:
        print('  CUDA tidak tersedia -> pakai CPU (lebih lambat)')
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    model = PC.FeatureExtractor(pretrained=bool(args.pretrained), dim=args.dim).to(device)

    def paths_of(ids):
        p = [os.path.join(args.img_root, 'train', *i.split('/')) for i in ids]
        miss = [x for x in p if not os.path.exists(x)]
        if miss:
            raise SystemExit('{} gambar tidak ada, mis. {}'.format(len(miss), miss[0]))
        return p

    # 1) bank normal: semua patch 30 gambar KL0 (global, coreset) - identik dengan PatchCore global
    n_ids = PC.normal_train_ids(args.meta_path, args.train_seed)
    f_n = PC.extract(model, paths_of(n_ids), device, args.batch)
    all_n = torch.ones(f_n.shape[:3], dtype=torch.bool)
    bank_n, src_n = coreset_src(masked_patches(f_n, all_n), masked_src(all_n), args.coreset, device)

    # 2) bank OA: patch 30 gambar KL2/3 yang SAMA dengan K1B (global, coreset)
    f_oa = os.path.join(args.meta_path, 'oa_seed_{}.csv'.format(args.train_seed))
    if not os.path.exists(f_oa):
        raise SystemExit('Tidak ada {}. Jalankan dulu: localize_joint.py --train_only 1'.format(f_oa))
    a_ids = [PC.norm_id(i) for i in pd.read_csv(f_oa)['id']]
    f_a = PC.extract(model, paths_of(a_ids), device, args.batch)
    if args.oa_region == 'all':
        mask_a = torch.ones(f_a.shape[:3], dtype=torch.bool)
    else:
        import eval_scores as ES
        lm = pd.read_csv(args.landmarks_train, keep_default_na=False)
        lm['id'] = lm['id'].map(PC.norm_id)
        miss = set(a_ids) - set(lm['id'])
        if miss:
            raise SystemExit('{} gambar OA tidak ada di {}'.format(len(miss), args.landmarks_train))
        lm = lm.drop_duplicates('id').set_index('id').reindex(a_ids).reset_index()
        jb, mg = ES.region_masks(lm, f_a.shape[1], f_a.shape[2], args.band, args.pad)
        mask_a = torch.from_numpy(jb if args.oa_region == 'joint_band' else mg)
    pa = masked_patches(f_a, mask_a)
    bank_a, src_a = coreset_src(pa, masked_src(mask_a), args.coreset, device)
    sub = None
    if args.oa_split == 'kl':   # K1C global: satu bank OA per grade (patch dari region yang sama)
        sub = []
        for g, idx in PC.split_by_grade(a_ids).items():
            mg_ = mask_a[idx]
            b, sl = coreset_src(masked_patches(f_a[idx], mg_), masked_src(mg_), args.coreset, device)
            sub.append((g, b, torch.as_tensor(idx)[sl]))     # src -> indeks di daftar OA (a_ids)
        print('  K1C global: ' + ', '.join('bank KL{} {} patch'.format(g, b.shape[0]) for g, b, _ in sub))
    print('  Bank normal (global): {} patch -> coreset {}'.format(f_n.shape[0] * f_n.shape[1] * f_n.shape[2], bank_n.shape[0]))
    print('  Bank OA (global, region {}): {} patch -> coreset {}'.format(args.oa_region, pa.shape[0], bank_a.shape[0]))

    # 3) skor split evaluasi
    files, ids, labels = PC.split_files(args.img_root, args.eval_split)
    smax, stop, allmaps = [], [], []
    for i in range(0, len(files), args.batch):
        feats = model(PC.load_batch(files[i:i + args.batch])[0].to(device)).cpu()
        for k, f in enumerate(feats):
            qid = ids[i + k]
            tr = args.eval_split == 'train'
            ex_n = PC.same_patient_idx(qid, n_ids) if tr else []
            ex_a = PC.same_patient_idx(qid, a_ids) if tr else []
            if sub is not None:
                m = PC.grade_score(PC.score_global(f, bank_n, device, src=src_n, exclude=ex_n),
                                   [(g, PC.score_global(f, b, device, src=sl, exclude=ex_a)) for g, b, sl in sub],
                                   args.sev_score, args.sev_lambda, args.sev_temp)
            else:
                m = score_global_dual(f, bank_n, bank_a, device, src_n, src_a, ex_n, ex_a)
            allmaps.append(m.numpy().astype(np.float16))
            v = m.flatten()
            smax.append(float(v.max()))
            stop.append(float(v.topk(max(1, len(v) // 100)).values.mean()))
        print('  {}/{} gambar'.format(min(i + args.batch, len(files)), len(files)), end='\r')
    print()

    df = pd.DataFrame(dict(id=ids, label=labels, score_max=smax, score_top1pct=stop))
    df.to_csv(os.path.join(args.out, 'scores_{}.csv'.format(args.eval_split)), index=False)
    np.savez_compressed(os.path.join(args.out, 'maps_{}.npz'.format(args.eval_split)),
                        ids=np.array(ids), labels=np.array(labels), maps=np.stack(allmaps))

    res = PC.belton_metrics(df['score_max'], df['label'])
    print('\n----- HASIL DUAL GLOBAL / tanpa anatomi ({}, {} gambar, bank OA: {}) -----'.format(
        args.eval_split, len(df), args.oa_region))
    print('Spearman rank correlation (SRC)  : {:.4f}'.format(res['spearman']))
    print('OA AUC (KL>=2, = AUC_KL)         : {:.4f}'.format(res['auc_kl_ge2']))
    print('AUC KL>0 | KL>=3 | KL=4          : {:.4f} | {:.4f} | {:.4f}'.format(
        res['auc_kl_gt0'], res['auc_kl_ge3'], res['auc_kl_eq4']))
    print('Rata-rata skor per grade KL      : ' + '  '.join(
        'KL{}={:.3f}'.format(g, v) for g, v in df.groupby('label')['score_max'].mean().items()))
    print('Waktu: {:.1f} menit'.format((time.time() - t0) / 60))

    row = dict(waktu=time.strftime('%Y-%m-%d %H:%M'), mode='dual_global', img_root=args.img_root,
               split=args.eval_split, seed=args.train_seed, radius='', oa_region=args.oa_region,
               dual_score='diff', oa_split=args.oa_split,
               sev_score=args.sev_score if args.oa_split == 'kl' else '', coreset=args.coreset, clahe=0, pretrained=args.pretrained, n=len(df),
               **{k: round(v, 4) for k, v in res.items()}, out=args.out)
    new = pd.DataFrame([row])
    if os.path.exists(args.summary):
        try:
            new = pd.concat([pd.read_csv(args.summary), new], ignore_index=True, sort=False)
        except Exception:
            os.replace(args.summary, args.summary + '.bak')
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    cols = list(row.keys()) + [c for c in new.columns if c not in row]
    new[cols].to_csv(args.summary, index=False)
    print('Ringkasan ditambahkan ke', args.summary)


if __name__ == '__main__':
    main()
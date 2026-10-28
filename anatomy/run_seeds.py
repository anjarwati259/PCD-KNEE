"""
Cara pakai (dari folder anatomy/, satu folder dengan patchcore_k1.py, ablation_dual_global.py, eval_scores.py):
    python run_seeds.py
    python run_seeds.py --device cpu            # tanpa GPU (jauh lebih lama)
    python run_seeds.py --seeds 1001,34,44      # sebagian seed saja
Run yang sudah selesai dilewati otomatis (aman dihentikan lalu dilanjutkan).

Output di ../k1_out/seeds/:
    <varian>/seed_<s>_<split>/        hasil tiap run (maps_<split>.npz, scores_<split>.csv)
    detail_per_seed.csv               metrik setiap varian x seed x split pada kombinasi terpilih
    ensemble_<varian>_<split>.csv     skor ensemble per gambar (dipakai nanti untuk Stage 2 Belton)
    laporan_seeds.txt                 ringkasan keputusan & hasil test
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import eval_scores as ES

BELTON_SEEDS = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
# {data}, {aligned}, {anatomy_out} diisi dari argumen (lihat make_variants)
VARIANTS_T = {
    # ablation: PatchCore asli (tanpa anatomi, tanpa bank OA) dan bank normal per lokasi (anatomi, tanpa bank OA)
    'global': dict(script='patchcore_k1.py', args=['--mode', 'global', '--img_root', '{data}'],
                   landmarks='{anatomy_out}/{split}/landmarks.csv'),
    'location': dict(script='patchcore_k1.py', args=['--mode', 'location', '--img_root', '{aligned}'],
                     landmarks='{aligned}/{split}/aligned_landmarks.csv'),
    'k1b': dict(script='patchcore_k1.py', args=['--mode', 'dual', '--img_root', '{aligned}'],
                landmarks='{aligned}/{split}/aligned_landmarks.csv'),
    'dual_global': dict(script='ablation_dual_global.py',
                        args=['--img_root', '{data}', '--landmarks_train', '{anatomy_out}/train/landmarks.csv'],
                        landmarks='{anatomy_out}/{split}/landmarks.csv'),
    # K1C: bank OA dipisah KL2 / KL3 (rumus skor dari --sev_score)
    'k1c': dict(script='patchcore_k1.py', args=['--mode', 'dual', '--oa_split', 'kl', '--img_root', '{aligned}'],
                landmarks='{aligned}/{split}/aligned_landmarks.csv', sev=True),
    'dual_global_c': dict(script='ablation_dual_global.py',
                          args=['--oa_split', 'kl', '--img_root', '{data}',
                                '--landmarks_train', '{anatomy_out}/train/landmarks.csv'],
                          landmarks='{anatomy_out}/{split}/landmarks.csv', sev=True),
}


def make_variants(data='../data', aligned='../aligned', anatomy_out='../anatomy_out'):
    """Isi folder data/aligned/anatomy_out ke template VARIANTS_T (landmarks masih berisi {split})."""
    f = lambda t: t.replace('{data}', data).replace('{aligned}', aligned).replace('{anatomy_out}', anatomy_out)
    out = {}
    for k, v in VARIANTS_T.items():
        out[k] = dict(v, args=[f(a) for a in v['args']], landmarks=f(v['landmarks']))
    return out


VARIANTS = make_variants()


def vdir_of(v, sev_score):
    return v + ('_' + sev_score if VARIANTS_T[v].get('sev') else '')


def log(lines, txt=''):
    print(txt)
    lines.append(txt)


def main():
    ap = argparse.ArgumentParser(description='K1B vs dual global untuk 10 seed + ensemble + keputusan berbasis val')
    ap.add_argument('--seeds', default='all', help="'all' (10 seed Belton) atau daftar dipisah koma")
    ap.add_argument('--variants', default='k1b,dual_global',
                    help='pilihan: global, location, k1b, dual_global, k1c, dual_global_c (dipisah koma)')
    ap.add_argument('--sev_score', default='expected', choices=['expected', 'diff2'],
                    help='rumus skor K1C (pilih dulu di val seed 1001)')
    ap.add_argument('--out', default='../k1_out/seeds')
    ap.add_argument('--meta_path', default='../meta/')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--pretrained', type=int, default=1)
    ap.add_argument('--band', type=float, default=32)
    ap.add_argument('--pad', type=float, default=8)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--rerun', type=int, default=0, help='1 = jalankan ulang walau hasil sudah ada')
    ap.add_argument('--data', default='../data', help='folder gambar asli (berisi train/ val/ test/)')
    ap.add_argument('--aligned', default='../aligned', help='folder gambar tersejajar (align_joint.py)')
    ap.add_argument('--anatomy_out', default='../anatomy_out', help='folder landmark gambar asli (localize_joint.py)')
    ap.add_argument('--hanya_val', type=int, default=0,
                    help='1 = hanya tuning di VAL (pilih region:agregasi); skor test disimpan untuk fusi tetapi '
                         'memory bank tidak dievaluasi sendiri di test')
    args = ap.parse_args()
    global VARIANTS
    VARIANTS = make_variants(args.data, args.aligned, args.anatomy_out)

    seeds = BELTON_SEEDS if args.seeds == 'all' else [int(s) for s in args.seeds.split(',')]
    variants = [v.strip() for v in args.variants.split(',')]
    splits = ['val', 'test']
    os.makedirs(args.out, exist_ok=True)
    lines = []

    # --- cek persiapan
    miss = [s for s in seeds for f in ('train_seed_{}.csv', 'oa_seed_{}.csv')
            if not os.path.exists(os.path.join(args.meta_path, f.format(s)))]
    if miss:
        raise SystemExit('Daftar training belum lengkap untuk seed {}. Jalankan dulu:\n  python make_oa_pool.py'
                         .format(sorted(set(miss))))

    # --- 1) jalankan semua run
    total = len(variants) * len(seeds) * len(splits)
    k = 0
    for v in variants:
        for s in seeds:
            for sp in splits:
                k += 1
                vdir = v + ('_' + args.sev_score if VARIANTS[v].get('sev') else '')
                od = os.path.join(args.out, vdir, 'seed_{}_{}'.format(s, sp))
                if os.path.exists(os.path.join(od, 'maps_{}.npz'.format(sp))) and not args.rerun:
                    print('[{}/{}] {} seed {} {}: sudah ada, dilewati'.format(k, total, v, s, sp))
                    continue
                print('[{}/{}] {} seed {} {} ...'.format(k, total, v, s, sp))
                extra = ['--sev_score', args.sev_score] if VARIANTS[v].get('sev') else []
                cmd = [sys.executable, VARIANTS[v]['script']] + VARIANTS[v]['args'] + extra + [
                    '--train_seed', str(s), '--eval_split', sp, '--out', od, '--device', args.device,
                    '--pretrained', str(args.pretrained), '--meta_path', args.meta_path,
                    '--summary', os.path.join(args.out, 'ringkasan_runs.csv')]
                if args.hanya_val and sp != 'val':
                    cmd += ['--metrik', '0']
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stdout[-2000:], r.stderr[-3000:])
                    raise SystemExit('Run gagal: ' + ' '.join(cmd))

    # --- 2) kumpulkan skor semua kombinasi
    data = {}   # (v, split, seed) -> (cfg, labels, ids)
    for v in variants:
        for sp in splits:
            lm = VARIANTS[v]['landmarks'].format(split=sp)
            for s in seeds:
                vdir = v + ('_' + args.sev_score if VARIANTS[v].get('sev') else '')
                mp = os.path.join(args.out, vdir, 'seed_{}_{}'.format(s, sp), 'maps_{}.npz'.format(sp))
                data[(v, sp, s)] = ES.all_configs(mp, lm, args.band, args.pad)
    ref_ids = {sp: data[(variants[0], sp, seeds[0])][2] for sp in splits}
    for key, (_, _, ids) in data.items():
        if ids != ref_ids[key[1]]:
            raise SystemExit('Urutan gambar berbeda antar run ({}); hapus folder run itu lalu jalankan ulang.'.format(key))
    labels = {sp: np.asarray(data[(variants[0], sp, seeds[0])][1]) for sp in splits}
    configs = list(data[(variants[0], 'val', seeds[0])][0].keys())
    pilih = lambda m: (m['AUC_KL'] + m['SRC']) / 2

    log(lines, '=' * 78)
    log(lines, '{} - {} seed: {}'.format(' vs '.join(variants), len(seeds), ', '.join(map(str, seeds))))
    if any(VARIANTS[v].get('sev') for v in variants):
        log(lines, 'Rumus skor K1C: {}'.format(args.sev_score))
    rep = os.path.join(args.out, 'laporan_seeds_{}.txt'.format('_'.join(
        v + ('_' + args.sev_score if VARIANTS[v].get('sev') else '') for v in variants)))
    log(lines, '=' * 78)
    nid = lambda f: {ES.norm_id(i) for i in pd.read_csv(os.path.join(args.meta_path, f))['id']}
    n_norm = set().union(*[nid('train_seed_{}.csv'.format(s)) for s in seeds])
    n_oa = set().union(*[nid('oa_seed_{}.csv'.format(s)) for s in seeds])
    pool_f = os.path.join(args.meta_path, 'oa_pool.csv')
    log(lines, 'Label unik dipakai (gabungan seed): {} normal + {} OA = {}{}'.format(
        len(n_norm), len(n_oa), len(n_norm) + len(n_oa),
        '  (pool OA: {} gambar)'.format(len(pd.read_csv(pool_f))) if os.path.exists(pool_f) else
        '  [!] tanpa oa_pool.csv: daftar OA tidak dari pool'))

    # --- 3) pilih kombinasi per varian (rata-rata 'pilih' di VAL atas seed)
    chosen, detail, ens, tuning = {}, [], {}, {}
    rep_splits = ['val'] if args.hanya_val else splits
    for v in variants:
        vm = {c: [ES.metrics(data[(v, 'val', s)][0][c], labels['val']) for s in seeds] for c in configs}
        means = {c: np.mean([pilih(m) for m in ms]) for c, ms in vm.items()}
        chosen[v] = max(means, key=means.get)
        tuning[v] = {'{}:{}'.format(*c): dict(pilih=float(means[c]), AUC_KL=float(np.mean([m['AUC_KL'] for m in vm[c]])),
                                              SRC=float(np.mean([m['SRC'] for m in vm[c]]))) for c in configs}
        log(lines, '\n[{}] TUNING di VAL: region x agregasi (rata-rata {} seed), pilih = (AUC_KL + SRC) / 2'.format(v, len(seeds)))
        for c in sorted(configs, key=lambda c: -means[c]):
            t = tuning[v]['{}:{}'.format(*c)]
            log(lines, '  {:<20} AUC_KL {:.4f}  SRC {:.4f}  pilih {:.4f}{}'.format(
                '{}:{}'.format(*c), t['AUC_KL'], t['SRC'], t['pilih'], '   <- terpilih' if c == chosen[v] else ''))
        # per seed + ensemble (z-score dengan statistik VAL tiap seed)
        z = {sp: [] for sp in splits}
        for s in seeds:
            sv = data[(v, 'val', s)][0][chosen[v]]
            mu, sd = float(np.mean(sv)), float(np.std(sv)) + 1e-12
            for sp in splits:
                sc = data[(v, sp, s)][0][chosen[v]]
                if sp in rep_splits:
                    m = ES.metrics(sc, labels[sp])
                    detail.append(dict(varian=v, seed=s, split=sp, region=chosen[v][0], agregasi=chosen[v][1],
                                       AUC_KL=round(m['AUC_KL'], 4), SRC=round(m['SRC'], 4), AUC_KL4=round(m['AUC_KL4'], 4)))
                z[sp].append((sc - mu) / sd)
        for sp in splits:
            ens[(v, sp)] = np.mean(z[sp], axis=0)
            vname = v + ('_' + args.sev_score if VARIANTS[v].get('sev') else '')
            pd.DataFrame(dict(id=ref_ids[sp], label=labels[sp], score=ens[(v, sp)])).to_csv(
                os.path.join(args.out, 'ensemble_{}_{}.csv'.format(vname, sp)), index=False)
    det = pd.DataFrame(detail)
    det.to_csv(os.path.join(args.out, 'detail_per_seed.csv'), index=False)

    # --- 4) tabel per seed (mean +- SD) dan ensemble
    log(lines, '\n{:<12} {:<5} {:>22} {:>22} {:>16}'.format('varian', 'split', 'AUC_KL mean+-SD', 'SRC mean+-SD', 'ENSEMBLE AUC/SRC'))
    ens_val_pilih = {}
    for v in variants:
        for sp in rep_splits:
            d = det[(det.varian == v) & (det.split == sp)]
            me = ES.metrics(ens[(v, sp)], labels[sp])
            if sp == 'val':
                ens_val_pilih[v] = pilih(me)
            log(lines, '{:<12} {:<5} {:>14.4f} +- {:.4f} {:>14.4f} +- {:.4f}   {:.4f} / {:.4f}'.format(
                v, sp, d.AUC_KL.mean(), d.AUC_KL.std(), d.SRC.mean(), d.SRC.std(), me['AUC_KL'], me['SRC']))

    # --- 5) keputusan berbasis VAL + laporan TEST
    win = max(ens_val_pilih, key=ens_val_pilih.get)
    if len(variants) > 1:
        log(lines, '\nKEPUTUSAN (berdasarkan VAL, ensemble): pemenang = {}  (pilih val: {})'.format(
            win, ', '.join('{} {:.4f}'.format(v, p) for v, p in ens_val_pilih.items())))
    if not args.hanya_val:
        log(lines, '\nTEST - ensemble {} seed, CI 95% bootstrap ({} ulangan):'.format(len(seeds), args.n_boot))
    for v in (variants if not args.hanya_val else []):
        b = ES.bootstrap(ens[(v, 'test')], labels['test'], n_boot=args.n_boot)
        m = ES.metrics(ens[(v, 'test')], labels['test'])
        log(lines, '  {:<12} AUC_KL {:.4f} [{:.3f}-{:.3f}]   SRC {:.4f} [{:.3f}-{:.3f}]   AUC_KL4 {:.4f}{}'.format(
            v, m['AUC_KL'], b['AUC_KL_lo'], b['AUC_KL_hi'], m['SRC'], b['SRC_lo'], b['SRC_hi'], m['AUC_KL4'],
            '   <- terpilih' if v == win else ''))
    for other in ([v for v in variants if v != win] if not args.hanya_val else []):
        b = ES.bootstrap(ens[(win, 'test')], labels['test'], ens[(other, 'test')], n_boot=args.n_boot)
        log(lines, '\n  {} - {} (berpasangan):'.format(win, other))
        log(lines, '    AUC_KL {:+.4f} [{:+.4f}, {:+.4f}] p={}'.format(
            b['d_AUC_KL'], b['d_AUC_KL_lo'], b['d_AUC_KL_hi'], ES.fmt_p(b['p_AUC_KL'], args.n_boot)))
        log(lines, '    SRC    {:+.4f} [{:+.4f}, {:+.4f}] p={}'.format(
            b['d_SRC'], b['d_SRC_lo'], b['d_SRC_hi'], ES.fmt_p(b['p_SRC'], args.n_boot)))
        sig = b['p_AUC_KL'] < 0.05 or b['p_SRC'] < 0.05
        log(lines, '    -> ' + ('perbedaan SIGNIFIKAN di test' if sig else 'TIDAK berbeda signifikan di test: setara'))
    if args.hanya_val:
        log(lines, '\n(skor test memory bank disimpan untuk fusi, tetapi tidak dievaluasi sendiri: --hanya_val 1)')
    else:
        log(lines, '\nPembanding: Belton Stage 1 (paper) 72.9 / 0.432 ; PatchCore (paper Belton) 64.6 / 0.270')

    # keputusan dalam bentuk mesin-baca (dipakai run_all.py / export_belton.py)
    kep = dict(pemenang=win, sev_score=args.sev_score, seeds=seeds,
               varian={v: dict(region=chosen[v][0], agregasi=chosen[v][1], folder=vdir_of(v, args.sev_score),
                               landmarks=VARIANTS[v]['landmarks'], script=VARIANTS[v]['script'],
                               args=VARIANTS[v]['args'] + (['--sev_score', args.sev_score] if VARIANTS[v].get('sev') else []),
                               ensemble={sp: os.path.join(args.out, 'ensemble_{}_{}.csv'.format(vdir_of(v, args.sev_score), sp))
                                         for sp in splits},
                               val_pilih_ensemble=float(ens_val_pilih[v]), tuning_val=tuning[v])
                       for v in variants})
    with open(os.path.join(args.out, 'keputusan_k1.json'), 'w', encoding='utf-8') as f:
        json.dump(kep, f, indent=1)

    for path in (os.path.join(args.out, 'laporan_seeds.txt'), rep):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    print('\nTersimpan: {} (dan {})'.format(os.path.join(args.out, 'laporan_seeds.txt'), rep))


if __name__ == '__main__':
    main()
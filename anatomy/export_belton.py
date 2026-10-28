"""
Cara pakai (biasanya dipanggil run_all.py):
  python export_belton.py --work ../hasil_stage1 --data ../data --belton_root ../ss_fewsome --model_name usulan_utuh_
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import eval_scores as ES

EPOCH = 400            # main.py Belton mencari file Stage 1 berakhiran _epoch_400 (TRAIN_PLATEAU_EPOCH)
N_NORMAL = 30


def belton_metrics(score, label):
    """Sama dengan utils.get_metrics Belton: SRC, AUC KL>0, KL>=2 (AUC_KL), KL>=3, KL=4 (AUC_KL>3)."""
    score, label = np.asarray(score, float), np.asarray(label)
    out = dict(spearman=float(stats.spearmanr(score, label)[0]))
    for name, pos in (('auc', label > 0), ('auc_mid', label > 1), ('auc_mid2', label > 2), ('auc_sev', label == 4)):
        out[name] = float(roc_auc_score(pos.astype(int), score)) if 0 < pos.sum() < len(pos) else float('nan')
    return out


def parse_winner(win, k1):
    """Nama pemenang report_dra_oa -> daftar komponen [('dra', varian) | ('k1', varian)]."""
    k1_names = set(k1['varian'])
    ref_folder = {v: k1['varian'][v]['folder'] for v in k1_names}
    def as_k1(name):
        for v in k1_names:
            if name in (v, ref_folder[v]):
                return v
        return None
    if win.startswith('fusi_') and '+' in win:
        d, r = win[len('fusi_'):].rsplit('+', 1)
        return [('dra', d), ('k1', as_k1(r) or r)]
    if as_k1(win):
        return [('k1', as_k1(win))]
    return [('dra', win)]


class Component:
    """Skor mentah satu komponen untuk (split, seed), dengan id berformat 'grade/file'."""

    def __init__(self, kind, name, work, k1, band, pad):
        self.kind, self.name, self.work, self.k1, self.band, self.pad = kind, name, work, k1, band, pad
        if kind == 'dra':
            f = os.path.join(work, 'k2_out', 'dra', name, 'keputusan.json')
            if not os.path.exists(f):
                raise SystemExit('Tidak ada {} (jalankan train_dra_oa.py --preset {})'.format(f, name))
            self.kep = json.load(open(f, encoding='utf-8'))
        else:
            if name not in k1['varian']:
                raise SystemExit('Varian K1 {} tidak ada di keputusan_k1.json'.format(name))
            self.kep = k1['varian'][name]

    def label(self):
        if self.kind == 'dra':
            return 'DRA-OA[{}:{}]'.format(self.name, self.kep['terpilih'])
        return 'K1[{}:{}:{}]'.format(self.name, self.kep['region'], self.kep['agregasi'])

    def raw(self, split, seed):
        if self.kind == 'k1':
            mp = os.path.join(self.work, 'k1_out', 'seeds', self.kep['folder'], 'seed_{}_{}'.format(seed, split),
                              'maps_{}.npz'.format(split))
            if not os.path.exists(mp):
                raise SystemExit('Tidak ada {} (run_all.py langkah skor_train)'.format(mp))
            cfg, labels, ids = ES.all_configs(mp, self.kep['landmarks'].format(split=split), self.band, self.pad)
            return pd.Series(cfg[(self.kep['region'], self.kep['agregasi'])], index=ids), pd.Series(labels, index=ids)
        f = os.path.join(self.work, 'k2_out', 'dra', self.name, 'scores_{}_seed_{}.csv'.format(split, seed))
        if not os.path.exists(f):
            raise SystemExit('Tidak ada {} (train_dra_oa.py --preset {} --score_train 1)'.format(f, self.name))
        d = pd.read_csv(f)
        d['id'] = d['id'].map(ES.norm_id)
        d = d.drop_duplicates('id').set_index('id')
        c = self.kep['terpilih']
        if c == 'gabungan':        # sama dengan train_dra_oa.cand_scores: rata-rata z kepala (statistik VAL)
            v = pd.read_csv(os.path.join(self.work, 'k2_out', 'dra', self.name, 'scores_val_seed_{}.csv'.format(seed)))
            z = [(d[h] - v[h].mean()) / (v[h].std() + 1e-12) for h in self.kep['kepala']]
            s = sum(z) / len(z)
        else:
            s = d[c]
        return s.astype(float), d['label']

    def z(self, split, seed):
        s, lab = self.raw(split, seed)
        v, _ = self.raw('val', seed)
        return (s - v.mean()) / (v.std(ddof=0) + 1e-12), lab


def main():
    ap = argparse.ArgumentParser(description='Ekspor skor Stage 1 pemenang ke format Belton')
    ap.add_argument('--work', required=True, help='folder hasil run_all.py (berisi k1_out/, k2_out/)')
    ap.add_argument('--belton_root', default=None, help='folder ss_fewsome Belton (berisi main.py); file ditulis ke '
                                                        '<belton_root>/outputs/dfs/ss/. Kosong = <work>/belton_outputs')
    ap.add_argument('--belton_data_path', default='./data/', help='nilai --data_path saat menjalankan main.py')
    ap.add_argument('--model_name', default='usulan_utuh_', help='diakhiri garis bawah (Belton menyaring dengan "in")')
    ap.add_argument('--winner', default=None, help='paksa komponen, mis. "fusi_utama+dual_global_c" (default: keputusan VAL)')
    ap.add_argument('--hasil_dir', default=None, help='folder ringkasan Stage 1 (default <work>/hasil_akhir)')
    ap.add_argument('--band', type=float, default=32)
    ap.add_argument('--pad', type=float, default=8)
    ap.add_argument('--n_boot', type=int, default=1000)
    args = ap.parse_args()

    k1 = json.load(open(os.path.join(args.work, 'k1_out', 'seeds', 'keputusan_k1.json'), encoding='utf-8'))
    f_akhir = os.path.join(args.work, 'k2_out', 'dra', 'keputusan_akhir.json')
    if args.winner:
        win = args.winner
    elif os.path.exists(f_akhir):
        win = json.load(open(f_akhir, encoding='utf-8'))['pemenang']
    else:
        win = k1['pemenang']
    comps = [Component(kind, name, args.work, k1, args.band, args.pad) for kind, name in parse_winner(win, k1)]
    seeds = k1['seeds']
    print('Stage 1: {}  ->  {}'.format(win, ' + '.join(c.label() for c in comps)))

    root = args.belton_root or os.path.join(args.work, 'belton_outputs')
    d_dfs = os.path.join(root, 'outputs', 'dfs', 'ss')
    d_res = os.path.join(root, 'outputs', 'results', 'ss')
    os.makedirs(d_dfs, exist_ok=True)
    os.makedirs(d_res, exist_ok=True)
    hasil = args.hasil_dir or os.path.join(args.work, 'hasil_akhir')
    os.makedirs(hasil, exist_ok=True)

    rows, ens = [], {sp: [] for sp in ('train', 'val', 'test')}
    ids_ref, lab_ref = {}, {}
    for s in seeds:
        final, parts = {}, {}
        for sp in ('train', 'val', 'test'):
            zs, lab = [], None
            for c in comps:
                z, l = c.z(sp, s)
                if lab is None:
                    lab = l
                zs.append(z.rename(c.label()))
            df = pd.concat(zs, axis=1, join='inner')
            if len(df) != len(zs[0]):
                raise SystemExit('Komponen punya gambar {} berbeda (seed {})'.format(sp, s))
            df = df.loc[sorted(df.index)]
            final[sp] = df.mean(axis=1)
            parts[sp] = df
            if sp not in ids_ref:
                ids_ref[sp], lab_ref[sp] = list(final[sp].index), lab.reindex(final[sp].index).astype(int).values
            elif list(final[sp].index) != ids_ref[sp]:
                raise SystemExit('Urutan gambar {} berbeda antar seed'.format(sp))
            ens[sp].append(final[sp].values)

        # --- tulis format Belton (train = "unlabelled", test = "_on_test_set")
        lo, hi = final['train'].min(), final['train'].max()
        base = 'ss_training_{}bs_1_task_test_N_{}_seed_{}'.format(args.model_name, N_NORMAL, s)
        for sp, suffix in (('train', ''), ('test', '_on_test_set')):
            sc = final[sp]
            out = pd.DataFrame({'id': [args.belton_data_path + sp + '/' + i for i in sc.index],
                                'label': lab_ref[sp], 'centre_mean': ((sc - lo) / (hi - lo + 1e-12)).values})
            out = out.sort_values('centre_mean', ascending=False).reset_index(drop=True)
            out.to_csv(os.path.join(d_dfs, base + suffix + '_epoch_{}'.format(EPOCH)))
            pd.DataFrame([belton_metrics(sc.values, lab_ref[sp])], index=['centre_mean']).to_csv(
                os.path.join(d_res, base + suffix + '_epoch_{}'.format(EPOCH)))
        r = dict(seed=s)
        for sp in ('val', 'test'):
            r.update({'{}_{}'.format(sp, k): v for k, v in belton_metrics(final[sp].values, lab_ref[sp]).items()})
        rows.append(r)
        print('  seed {:<7} test AUC_KL {:.4f}  SRC {:.4f}  AUC_KL>3 {:.4f} | val AUC_KL {:.4f}'.format(
            s, r['test_auc_mid'], r['test_spearman'], r['test_auc_sev'], r['val_auc_mid']))

    det = pd.DataFrame(rows)
    det.to_csv(os.path.join(hasil, 'per_seed.csv'), index=False)
    judul = ('STAGE 1 - METODE {} (komponen tetap; pilihan di dalam komponen dari VAL)'.format(
        args.model_name.rstrip('_')) if args.winner else 'STAGE 1 - PEMENANG DIPILIH DI VAL')
    lines = ['=' * 78, judul,
             'spesifikasi: {}'.format(win), 'komponen   : {}'.format(' + '.join(c.label() for c in comps)),
             'seed     : {} ({} seed)'.format(', '.join(map(str, seeds)), len(seeds)), '=' * 78,
             '{:<6} {:>20} {:>20} {:>20} {:>20}'.format('split', 'AUC_KL (KL>=2)', 'SRC', 'AUC_KL>3 (KL=4)', 'AUC KL>0')]
    for sp in ('val', 'test'):
        f = lambda k: '{:.4f} +- {:.4f}'.format(det['{}_{}'.format(sp, k)].mean(), det['{}_{}'.format(sp, k)].std())
        lines.append('{:<6} {:>20} {:>20} {:>20} {:>20}'.format(sp, f('auc_mid'), f('spearman'), f('auc_sev'), f('auc')))
    lines.append('(rata-rata +- SD antar seed, seperti tabel Belton)')
    for sp in ('val', 'test'):
        e = np.mean(ens[sp], axis=0)
        m = belton_metrics(e, lab_ref[sp])
        line = 'ENSEMBLE {} seed, {:<4}: AUC_KL {:.4f}  SRC {:.4f}  AUC_KL>3 {:.4f}'.format(
            len(seeds), sp, m['auc_mid'], m['spearman'], m['auc_sev'])
        if sp == 'test' and args.n_boot:
            b = ES.bootstrap(e, lab_ref[sp], n_boot=args.n_boot)
            line += '   CI95 AUC_KL [{:.3f}-{:.3f}] SRC [{:.3f}-{:.3f}]'.format(
                b['AUC_KL_lo'], b['AUC_KL_hi'], b['SRC_lo'], b['SRC_hi'])
        lines.append(line)
        pd.DataFrame(dict(id=ids_ref[sp], label=lab_ref[sp], score=e)).to_csv(
            os.path.join(hasil, 'ensemble_{}.csv'.format(sp)), index=False)
    # catatan TUNING (hanya VAL): pengaturan di dalam tiap komponen, bukan skor komponen di test
    lines += ['', 'PENGATURAN METODE (tuning di VAL, pilih = (AUC_KL + SRC) / 2, rata-rata {} seed)'.format(len(seeds))]
    for c in comps:
        if c.kind == 'k1':
            tv = c.kep.get('tuning_val', {})
            lines.append('  memory bank {}: region:agregasi terpilih = {}:{}'.format(c.name, c.kep['region'], c.kep['agregasi']))
            for k, t in sorted(tv.items(), key=lambda kv: -kv[1]['pilih']):
                lines.append('    {:<20} AUC_KL {:.4f}  SRC {:.4f}  pilih {:.4f}'.format(k, t['AUC_KL'], t['SRC'], t['pilih']))
        else:
            lines.append('  DRA-OA {}: kandidat skor terpilih = {} (kepala {})'.format(
                c.name, c.kep['terpilih'], ','.join(c.kep.get('kepala', []))))
            for k, v in sorted(c.kep.get('val_pilih', {}).items(), key=lambda kv: -kv[1]):
                lines.append('    {:<20} pilih {:.4f}'.format(k, v))
    lines.append('  fusi: rata-rata z-score komponen (mean/SD skor VAL per seed), bobot sama')
    lines += ['', 'File Belton : {}  ({} seed x 2 file)'.format(d_dfs, len(seeds)),
              'Lanjut      : (dari folder {}) python <ss_fewsome>/main.py --train_ss 0 --model_name {} --data_path {}'.format(
                  root, args.model_name, args.belton_data_path)]
    if len(seeds) != 10:
        lines.append('[!] get_pseudo_labels Belton butuh 10 seed (assert len(files) == 10); sekarang {}.'.format(len(seeds)))
    txt = '\n'.join(lines)
    print('\n' + txt)
    with open(os.path.join(hasil, 'ringkasan_stage1.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    json.dump(dict(pemenang=win, komponen=[c.label() for c in comps], model_name=args.model_name, dfs=d_dfs),
              open(os.path.join(hasil, 'ekspor.json'), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
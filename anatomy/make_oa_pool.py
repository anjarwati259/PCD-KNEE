import argparse
import os
import shutil

import numpy as np
import pandas as pd

BELTON_SEEDS = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
KEEP_SEED = 1001


def norm_id(x):
    return '/'.join(str(x).replace('\\', '/').split('/')[-2:])


def split_even(n, k):
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def main():
    ap = argparse.ArgumentParser(description='Pool OA + daftar OA per seed (pola Belton)')
    ap.add_argument('--train_root', default='../data/train')
    ap.add_argument('--meta_path', default='../meta/')
    ap.add_argument('--pool_size', type=int, default=150)
    ap.add_argument('--n_per_seed', type=int, default=30)
    ap.add_argument('--grades', default='2,3')
    ap.add_argument('--pool_seed', type=int, default=0, help='seed acak untuk mengisi pool')
    ap.add_argument('--overwrite', type=int, default=0, help='1 = buat ulang pool & SEMUA daftar, termasuk seed 1001')
    args = ap.parse_args()

    grades = [int(g) for g in args.grades.split(',')]
    m = args.meta_path
    f_pool = os.path.join(m, 'oa_pool.csv')
    pool_per_g = dict(zip(grades, split_even(args.pool_size, len(grades))))
    seed_per_g = dict(zip(grades, split_even(args.n_per_seed, len(grades))))

    # --- daftar seed 1001 yang sudah ada (dipertahankan kecuali --overwrite)
    keep = []
    f_keep = os.path.join(m, 'oa_seed_{}.csv'.format(KEEP_SEED))
    if os.path.exists(f_keep) and not args.overwrite:
        keep = [norm_id(i) for i in pd.read_csv(f_keep)['id']]
        print('  Daftar OA seed {} yang sudah ada dipertahankan ({} gambar)'.format(KEEP_SEED, len(keep)))

    # --- 1) pool OA
    if os.path.exists(f_pool) and not args.overwrite:
        pool = pd.read_csv(f_pool)
        pool['id'] = pool['id'].map(norm_id)
        print('  Pool OA sudah ada: {} ({} gambar) -> dipakai ulang'.format(f_pool, len(pool)))
    else:
        rng = np.random.default_rng(args.pool_seed)
        ids = []
        for g in grades:
            gdir = os.path.join(args.train_root, str(g))
            if not os.path.isdir(gdir):
                raise SystemExit('Folder tidak ada: ' + gdir)
            avail = sorted('{}/{}'.format(g, f) for f in os.listdir(gdir) if f.lower().endswith(IMG_EXT))
            mine = [i for i in keep if i.split('/')[0] == str(g)]
            need = pool_per_g[g] - len(mine)
            if need < 0:
                raise SystemExit('Daftar seed {} punya {} gambar KL{}, melebihi kuota pool {}'.format(
                    KEEP_SEED, len(mine), g, pool_per_g[g]))
            rest = [i for i in avail if i not in set(mine)]
            if len(rest) < need:
                raise SystemExit('Gambar KL{} di train hanya {} (butuh {})'.format(g, len(rest), need))
            ids += mine + list(rng.choice(rest, size=need, replace=False))
        pool = pd.DataFrame({'id': sorted(ids)})
        pool['kl'] = pool['id'].str.split('/').str[0].astype(int)
        os.makedirs(m, exist_ok=True)
        pool.to_csv(f_pool, index=False)
        print('  Pool OA dibuat: {} ({} gambar: {})'.format(
            f_pool, len(pool), ', '.join('KL{}={}'.format(g, n) for g, n in pool['kl'].value_counts().sort_index().items())))
    pool['kl'] = pool['id'].str.split('/').str[0].astype(int)
    missing = [i for i in keep if i not in set(pool['id'])]
    if missing:
        raise SystemExit('Daftar seed {} berisi gambar di luar pool ({}). Jalankan dengan --overwrite 1.'.format(
            KEEP_SEED, missing[:3]))

    # --- 2) daftar OA per seed, diambil dari pool
    bak = os.path.join(m, 'backup_oa_lama')
    lists = {}
    for s in BELTON_SEEDS:
        f = os.path.join(m, 'oa_seed_{}.csv'.format(s))
        if s == KEEP_SEED and keep:
            lists[s] = keep
            continue
        if os.path.exists(f):
            old = sorted(norm_id(i) for i in pd.read_csv(f)['id'])
            if not set(old) <= set(pool['id']) or args.overwrite:
                os.makedirs(bak, exist_ok=True)
                shutil.move(f, os.path.join(bak, os.path.basename(f)))
                print('  Daftar lama {} dipindah ke {}'.format(os.path.basename(f), bak))
            else:
                lists[s] = old       # sudah dari pool -> pakai ulang
                continue
        rng = np.random.default_rng(s)
        ids = []
        for g in grades:
            cand = sorted(pool.loc[pool['kl'] == g, 'id'])
            ids += list(rng.choice(cand, size=seed_per_g[g], replace=False))
        lists[s] = sorted(ids)
        pd.DataFrame({'id': lists[s]}).to_csv(f, index=False)

    # --- 3) ringkasan anggaran label
    normals = set()
    for s in BELTON_SEEDS:
        fn = os.path.join(m, 'train_seed_{}.csv'.format(s))
        if os.path.exists(fn):
            normals |= {norm_id(i) for i in pd.read_csv(fn)['id']}
    used_oa = set().union(*map(set, lists.values()))
    n_train = sum(len([f for f in os.listdir(os.path.join(args.train_root, d)) if f.lower().endswith(IMG_EXT)])
                  for d in os.listdir(args.train_root) if os.path.isdir(os.path.join(args.train_root, d)))
    lines = [
        'ANGGARAN LABEL (gambar unik, gabungan 10 seed)',
        '  normal (daftar Belton)      : {}'.format(len(normals)),
        '  OA dipakai (dari pool {:>3})   : {}  ({})'.format(
            len(pool), len(used_oa), ', '.join('KL{}={}'.format(g, sum(i.startswith('{}/'.format(g)) for i in used_oa))
                                              for g in grades)),
        '  total                       : {}  = {:.1f}% dari {} gambar train'.format(
            len(normals) + len(used_oa), 100 * (len(normals) + len(used_oa)) / max(n_train, 1), n_train),
        '  Belton (hanya normal)       : {}  = {:.1f}%'.format(len(normals), 100 * len(normals) / max(n_train, 1)),
        '  per seed                    : 30 normal + {} OA'.format(args.n_per_seed),
        '  tumpang tindih OA antar seed: rata-rata {:.1f} gambar sama per pasangan seed'.format(
            np.mean([len(set(lists[a]) & set(lists[b])) for i, a in enumerate(BELTON_SEEDS) for b in BELTON_SEEDS[i + 1:]])),
    ]
    txt = '\n'.join(lines)
    print('\n' + txt)
    with open(os.path.join(m, 'label_budget.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('\nLangkah berikutnya: cek visual semua gambar OA yang dipakai:\n'
          '  python localize_joint.py --input ../data/train --out ../anatomy_out/trainset_all --train_only 1 --train_seed all')


if __name__ == '__main__':
    main()
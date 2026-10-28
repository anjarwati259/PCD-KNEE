import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import eval_scores as ES
import extract_osteophyte_bank as EB
import make_synthetic_oa as SG
import patchcore_k1 as PC

BELTON_SEEDS = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
PRESETS = {
    'utama': dict(heads='S,P,R', bank='../k2_out/bank_bersih', sumber='bank', komponen='taji,jsn'),
    'besar': dict(heads='S,P,R', bank='../k2_out/bank_besar', sumber='bank', komponen='taji,jsn'),
    'acak': dict(heads='S,P,R', bank='../k2_out/bank_bersih', sumber='acak', komponen='taji,jsn'),
    'taji_saja': dict(heads='S,P,R', bank='../k2_out/bank_bersih', sumber='bank', komponen='taji'),
    'jsn_saja': dict(heads='S,P,R', bank='../k2_out/bank_bersih', sumber='bank', komponen='jsn'),
    'tanpa_sintetis': dict(heads='S,R', bank='', sumber='bank', komponen=''),
    'pseudo_saja': dict(heads='P', bank='../k2_out/bank_bersih', sumber='bank', komponen='taji,jsn'),
    'seen_saja': dict(heads='S', bank='', sumber='bank', komponen=''),                     # hanya 30 OA asli vs normal
    'tanpa_residual': dict(heads='S,P', bank='../k2_out/bank_bersih', sumber='bank', komponen='taji,jsn'),
}


# ----------------------------------------------------------------------------- model
class Head(nn.Module):
    """Kepala patch: fitur 28x28xD -> peta skor (+ opsional peta logit masker)."""

    def __init__(self, din=1024, hid=256, out=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(din, hid, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(hid, hid // 2, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(hid // 2, out, 1))

    def forward(self, f):                       # f: B x H x W x D
        return self.net(f.permute(0, 3, 1, 2))  # B x out x H x W


def topk_score(m, k):
    """m: B x H x W -> rata-rata k patch tertinggi."""
    return m.flatten(1).topk(k, dim=1).values.mean(1)


def to_tensor(ims, size=224):
    """Sama dengan PC.load_batch, tapi dari array (dipakai untuk gambar sintetis)."""
    out = []
    for im in ims:
        if im.shape != (size, size):
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        if PC.CLAHE is not None:
            im = PC.CLAHE.apply(im)
        out.append(im)
    x = torch.from_numpy(np.stack(out)).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - PC.MEAN) / PC.STD


def nn_residual(feat, bank, bank_pat, excl, chunk=4096):
    """feat: B x H x W x D (device); bank: M x D (device); bank_pat: M (pasien sumber, device);
    excl: daftar pasien per gambar (-1 = tanpa pengecualian). -> sisa fitur (B x H x W x D), jarak (B x H x W)."""
    B, H, W, D = feat.shape
    res, dist = [], []
    bb = (bank * bank).sum(1)
    for b in range(B):
        q = feat[b].reshape(-1, D)
        best_d = torch.full((q.shape[0],), float('inf'), device=q.device)
        best_i = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        qq = (q * q).sum(1, keepdim=True)
        for i in range(0, bank.shape[0], chunk):
            d2 = qq + bb[i:i + chunk][None] - 2 * q @ bank[i:i + chunk].T
            if excl[b] >= 0:
                d2 = d2.masked_fill((bank_pat[i:i + chunk] == excl[b])[None], float('inf'))
            v, j = d2.min(1)
            upd = v < best_d
            best_d = torch.where(upd, v, best_d)
            best_i = torch.where(upd, j + i, best_i)
        res.append((q - bank[best_i]).reshape(H, W, D))
        dist.append(best_d.clamp_min(0).sqrt().reshape(H, W))
    return torch.stack(res), torch.stack(dist)


# ----------------------------------------------------------------------------- data per seed
def seed_lists(meta_path, seed):
    n = [PC.norm_id(i) for i in pd.read_csv(os.path.join(meta_path, 'train_seed_{}.csv'.format(seed)))['id']]
    f = os.path.join(meta_path, 'oa_seed_{}.csv'.format(seed))
    if not os.path.exists(f):
        raise SystemExit('Tidak ada {}. Jalankan dulu: python make_oa_pool.py'.format(f))
    a = [PC.norm_id(i) for i in pd.read_csv(f)['id']]
    return n, a


def train_seed(seed, args, model, device, out_dir):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    heads = args.heads
    normals, oas = seed_lists(args.meta_path, seed)
    path = lambda i: os.path.join(args.img_root, 'train', *i.split('/'))
    pats = sorted({EB.patient_of(i) for i in normals + oas})
    pid = {p: k for k, p in enumerate(pats)}

    # fitur gambar asli (asli + cermin) disimpan sekali; float16 hemat memori
    def feats_of(ids):
        f = PC.extract(model, [path(i) for i in ids], device, args.batch)
        x, _ = PC.load_batch([path(i) for i in ids])
        ff = []
        for k in range(0, len(ids), args.batch):
            with torch.no_grad():
                ff.append(model(torch.flip(x[k:k + args.batch], dims=[3]).to(device)).cpu())
        return torch.stack([f, torch.cat(ff)], 1).half()           # N x 2 x H x W x D
    fN, fA = feats_of(normals), feats_of(oas)
    tN = np.zeros(len(normals), np.float32)
    tA = np.array([float(i.split('/')[0]) for i in oas], np.float32)  # KL2 -> 2, KL3 -> 3
    pN = np.array([pid[EB.patient_of(i)] for i in normals])
    pA = np.array([pid[EB.patient_of(i)] for i in oas])

    # bank normal (coreset 10%, gambar asli tanpa cermin) + pasien sumber, untuk kepala R
    bank, src = EB.build_bank(fN[:, 0].float(), args.coreset, device)
    bank = bank.to(device)
    bank_pat = torch.as_tensor(pN[src.numpy()], device=device)

    # generator OA sintetis (hanya dari 30 normal seed ini)
    gen, syn_targets = None, []
    if args.use_syn:
        taji_ids = oas if args.taji_dari == 'seed' else None
        gen = SG.OAGenerator(args.bank or '../k2_out/bank_bersih', args.img_root, args.landmarks,
                             meta_path=args.meta_path, sumber=args.sumber, blend=args.blend,
                             komponen=[k for k in args.komponen.split(',') if k], taji_ids=taji_ids)
        syn_targets = [k for k, i in enumerate(normals) if gen.valid_target(i)]
        if not syn_targets:
            raise SystemExit('Tidak ada target normal valid untuk sintesis (cek landmarks)')

    hd = {h: Head(out=2 if h == 'P' else 1).to(device) for h in heads}
    params = [p for m in hd.values() for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    k = args.topk
    logs = []
    nb_n, nb_a, nb_s = args.n_normal, args.n_oa, (args.n_syn if args.use_syn else 0)
    if not args.use_syn:
        nb_a = args.n_oa + args.n_syn
    use_real = any(h in heads for h in ('S', 'R'))
    t0 = time.time()
    for it in range(args.iters):
        # --- ambil batch
        iN = rng.integers(len(normals), size=nb_n)
        fl = rng.integers(2, size=nb_n)
        xN = fN[iN, fl].float().to(device)
        parts = dict(N=(xN, torch.zeros(nb_n, device=device), torch.as_tensor(pN[iN], device=device)))
        if use_real:
            iA = rng.integers(len(oas), size=nb_a)
            fa = rng.integers(2, size=nb_a)
            parts['A'] = (fA[iA, fa].float().to(device), torch.as_tensor(tA[iA], device=device),
                          torch.as_tensor(pA[iA], device=device))
        mS = None
        if nb_s:
            ims, tg, ms, pp = [], [], [], []
            for _ in range(nb_s):
                j = syn_targets[rng.integers(len(syn_targets))]
                g = int(rng.choice(args.syn_grades))
                r = gen.synthesize(gen.read(normals[j]), normals[j], g, rng)
                im, mk = r['img'], r['mask']
                if rng.random() < 0.5:
                    im, mk = im[:, ::-1], mk[:, ::-1]
                ims.append(np.ascontiguousarray(im))
                ms.append(cv2.resize(np.ascontiguousarray(mk), (28, 28), interpolation=cv2.INTER_AREA))
                tg.append(float(g))
                pp.append(pN[j])
            with torch.no_grad():
                xs = model(to_tensor(ims).to(device))
            parts['S'] = (xs, torch.tensor(tg, device=device), torch.as_tensor(np.array(pp), device=device))
            mS = torch.from_numpy(np.stack(ms)).float().to(device)

        # --- loss per kepala
        loss = 0.0
        rec = dict(iter=it + 1)
        for h, m in hd.items():
            use = {'S': ['N', 'A'], 'P': ['N', 'S'], 'R': ['N', 'A', 'S']}[h]
            use = [u for u in use if u in parts]
            x = torch.cat([parts[u][0] for u in use])
            t = torch.cat([parts[u][1] for u in use])
            if h == 'R':
                excl = torch.cat([parts[u][2] for u in use]).tolist()
                x, _ = nn_residual(x, bank, bank_pat, excl)
            out = m(x)
            s = topk_score(out[:, 0], k)
            lh = F.smooth_l1_loss(s, t)
            if h == 'P' and 'S' in parts:     # supervisi masker: normal = 0, sintetis = masker generator
                mask_t = torch.cat([torch.zeros(nb_n, 28, 28, device=device), mS])
                lm = F.binary_cross_entropy_with_logits(out[:, 1], mask_t)
                lh = lh + args.w_mask * lm
                rec['loss_mask'] = round(float(lm.detach()), 4)
            rec['loss_' + h] = round(float(lh.detach()), 4)
            loss = loss + lh
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        logs.append(rec)
        if (it + 1) % 50 == 0 or it == 0:
            print('    seed {} iter {}/{}  {}  ({:.1f} mnt)'.format(
                seed, it + 1, args.iters, '  '.join('{}={}'.format(a, b) for a, b in rec.items() if a != 'iter'),
                (time.time() - t0) / 60), end='\r')
    print()
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(heads={h: m.state_dict() for h, m in hd.items()}, bank=bank.cpu().half(),
                    bank_pat=bank_pat.cpu(), pats=pats, args=vars(args)), os.path.join(out_dir, 'heads.pt'))
    pd.DataFrame(logs).to_csv(os.path.join(out_dir, 'train_log.csv'), index=False)


def load_seed(out_dir, device):
    ck = torch.load(os.path.join(out_dir, 'heads.pt'), map_location='cpu')
    hd = {}
    for h, sd in ck['heads'].items():
        m = Head(out=2 if h == 'P' else 1)
        m.load_state_dict(sd)
        hd[h] = m.to(device).eval()
    return dict(heads=hd, bank=ck['bank'].float().to(device), bank_pat=ck['bank_pat'].to(device),
                pid={p: k for k, p in enumerate(ck['pats'])})


@torch.no_grad()
def score_split(split, seeds, models, args, model, device, out_root):
    files, ids, labels = PC.split_files(args.img_root, split)
    res = {s: {h: [] for h in models[s]['heads']} for s in seeds}
    for i in range(0, len(files), args.batch):
        f = model(PC.load_batch(files[i:i + args.batch])[0].to(device))
        bid = ids[i:i + args.batch]
        for s in seeds:
            M = models[s]
            for h, m in M['heads'].items():
                x = f
                if h == 'R':
                    # training split: pasien yang sama dikeluarkan dari bank (leave-one-pasien-out)
                    excl = [M['pid'].get(EB.patient_of(j), -1) if split == 'train' else -1 for j in bid]
                    x, _ = nn_residual(f, M['bank'], M['bank_pat'], excl)
                res[s][h].append(topk_score(m(x)[:, 0], args.topk).cpu().numpy())
        print('  skor {} {}/{}'.format(split, min(i + args.batch, len(files)), len(files)), end='\r')
    print()
    for s in seeds:
        df = pd.DataFrame(dict(id=ids, label=labels, **{h: np.concatenate(v) for h, v in res[s].items()}))
        df.to_csv(os.path.join(out_root, 'scores_{}_seed_{}.csv'.format(split, s)), index=False)
    return ids, np.asarray(labels)


# ----------------------------------------------------------------------------- utama
def main():
    ap = argparse.ArgumentParser(description='K2 tahap 3: DRA-OA (10 seed + ensemble + pilihan di val)')
    ap.add_argument('--preset', default='utama', choices=list(PRESETS))
    ap.add_argument('--name', default=None, help='nama folder output (default = nama preset)')
    ap.add_argument('--heads', default=None, help='override preset, mis. S,P,R')
    ap.add_argument('--bank', default=None, help='override preset: folder bank taji')
    ap.add_argument('--bank_root', default=None,
                    help='folder induk bank taji (berisi bank_bersih/, bank_besar/); menggantikan ../k2_out di preset')
    ap.add_argument('--sumber', default=None, choices=['bank', 'acak'])
    ap.add_argument('--komponen', default=None, help='override preset: taji,jsn | taji | jsn')
    ap.add_argument('--taji_dari', default='pool', choices=['pool', 'seed'],
                    help="pool = bank taji dari pool OA (default); seed = hanya taji dari 30 OA seed itu")
    ap.add_argument('--blend', default='poisson', choices=['poisson', 'mixed', 'alpha'])
    ap.add_argument('--syn_grades', default='1,2,3,4')
    ap.add_argument('--seeds', default='all')
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--n_normal', type=int, default=16, help='normal per batch')
    ap.add_argument('--n_oa', type=int, default=8, help='OA asli per batch')
    ap.add_argument('--n_syn', type=int, default=8, help='OA sintetis per batch')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--topk', type=int, default=16, help='skor gambar = rata-rata k patch tertinggi (dari 784)')
    ap.add_argument('--w_mask', type=float, default=1.0)
    ap.add_argument('--coreset', type=float, default=0.1)
    ap.add_argument('--img_root', default='../data')
    ap.add_argument('--meta_path', default='../meta/')
    ap.add_argument('--landmarks', default='../anatomy_out/train/landmarks.csv')
    ap.add_argument('--out_root', default='../k2_out/dra')
    ap.add_argument('--score_train', type=int, default=0, help='1 = skor juga split train (ekspor Belton)')
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--rerun', type=int, default=0)
    ap.add_argument('--pretrained', type=int, default=1)
    ap.add_argument('--hanya_val', type=int, default=0,
                    help='1 = laporan hanya berisi tuning di VAL (tanpa skor test/train DRA-OA sendiri; '
                         'skor tetap disimpan untuk fusi)')
    args = ap.parse_args()

    pr = PRESETS[args.preset]
    args.heads = [h.strip() for h in (args.heads or pr['heads']).split(',') if h.strip()]
    args.bank = args.bank if args.bank is not None else pr['bank']
    if args.bank and args.bank_root and args.bank == pr['bank']:
        args.bank = os.path.join(args.bank_root, os.path.basename(os.path.normpath(args.bank)))
    args.sumber = args.sumber or pr['sumber']
    args.komponen = args.komponen if args.komponen is not None else pr['komponen']
    args.use_syn = bool(args.komponen) and any(h in args.heads for h in ('P', 'R'))
    if 'P' in args.heads and not args.use_syn:
        raise SystemExit('Kepala P butuh OA sintetis (--komponen tidak boleh kosong)')
    args.syn_grades = [int(g) for g in args.syn_grades.split(',')]
    name = args.name or args.preset
    out = os.path.join(args.out_root, name)
    os.makedirs(out, exist_ok=True)
    seeds = BELTON_SEEDS if args.seeds == 'all' else [int(s) for s in args.seeds.split(',')]
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    if device != args.device:
        print('  CUDA tidak tersedia -> CPU (sangat lambat)')
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=1)
    print('DRA-OA [{}] kepala {} | sintetis: {} | bank {} | sumber {} | {} seed'.format(
        name, ','.join(args.heads), args.komponen or '-', args.bank or '-', args.sumber, len(seeds)))

    model = PC.FeatureExtractor(pretrained=bool(args.pretrained)).to(device)

    # --- 1) latih tiap seed
    for s in seeds:
        sd = os.path.join(out, 'seed_{}'.format(s))
        if os.path.exists(os.path.join(sd, 'heads.pt')) and not args.rerun:
            print('  seed {}: sudah dilatih, dilewati'.format(s))
            continue
        print('  melatih seed {} ...'.format(s))
        train_seed(s, args, model, device, sd)

    # --- 2) skor val & test (satu kali ekstraksi fitur untuk semua seed)
    models = {s: load_seed(os.path.join(out, 'seed_{}'.format(s)), device) for s in seeds}
    splits = ['val', 'test'] + (['train'] if args.score_train else [])
    data, labels, ids = {}, {}, {}
    for sp in splits:
        ids[sp], labels[sp] = score_split(sp, seeds, models, args, model, device, out)
        for s in seeds:
            data[(sp, s)] = pd.read_csv(os.path.join(out, 'scores_{}_seed_{}.csv'.format(sp, s)))

    # --- 3) kandidat skor per seed: tiap kepala + gabungan (rata-rata z kepala; z dari VAL tanpa label)
    def cand_scores(s):
        out_s = {}
        zs = {}
        for sp in splits:
            zs[sp] = []
        for h in args.heads:
            mu, sd = data[('val', s)][h].mean(), data[('val', s)][h].std() + 1e-12
            for sp in splits:
                out_s[(h, sp)] = data[(sp, s)][h].values
                zs[sp].append((data[(sp, s)][h].values - mu) / sd)
        if len(args.heads) > 1:
            for sp in splits:
                out_s[('gabungan', sp)] = np.mean(zs[sp], axis=0)
        return out_s
    cs = {s: cand_scores(s) for s in seeds}
    cands = sorted({c for s in seeds for (c, _) in cs[s]}, key=lambda c: (c != 'gabungan', c))
    pilih = lambda m: (m['AUC_KL'] + m['SRC']) / 2
    lines = []

    def log(t=''):
        print(t)
        lines.append(t)
    log('=' * 78)
    log('DRA-OA [{}] - {} seed | kepala {} | sintetis {} | bank {} | sumber {} | taji dari {}'.format(
        name, len(seeds), ','.join(args.heads), args.komponen or '-', args.bank or '-', args.sumber, args.taji_dari))
    log('=' * 78)
    log('Kandidat skor di VAL (rata-rata atas seed):')
    vm = {}
    for c in cands:
        ms = [ES.metrics(cs[s][(c, 'val')], labels['val']) for s in seeds]
        vm[c] = np.mean([pilih(m) for m in ms])
        log('  {:<10} AUC_KL {:.4f}  SRC {:.4f}  pilih {:.4f}'.format(
            c, np.mean([m['AUC_KL'] for m in ms]), np.mean([m['SRC'] for m in ms]), vm[c]))
    chosen = max(vm, key=vm.get)
    log('-> terpilih (VAL): {}'.format(chosen))
    with open(os.path.join(out, 'keputusan.json'), 'w', encoding='utf-8') as f:   # dipakai export_belton.py
        json.dump(dict(terpilih=chosen, kepala=args.heads, seeds=seeds,
                       val_pilih={c: float(v) for c, v in vm.items()}), f, indent=1)

    # --- 4) per seed & ensemble untuk kandidat terpilih (dan semua kandidat disimpan)
    ens = {}
    for c in cands:
        for sp in splits:
            z = []
            for s in seeds:
                v = cs[s][(c, 'val')]
                z.append((cs[s][(c, sp)] - v.mean()) / (v.std() + 1e-12))
            ens[(c, sp)] = np.mean(z, axis=0)
    for sp in splits:
        df = pd.DataFrame(dict(id=ids[sp], label=labels[sp], score=ens[(chosen, sp)],
                               **{'score_' + c: ens[(c, sp)] for c in cands}))
        df.to_csv(os.path.join(out, 'ensemble_{}.csv'.format(sp)), index=False)
    det = []
    rep_splits = ('val',) if args.hanya_val else ('val', 'test')
    for s in seeds:
        for sp in rep_splits:
            m = ES.metrics(cs[s][(chosen, sp)], labels[sp])
            det.append(dict(seed=s, split=sp, AUC_KL=m['AUC_KL'], SRC=m['SRC'], AUC_KL4=m['AUC_KL4']))
    det = pd.DataFrame(det)
    det.to_csv(os.path.join(out, 'detail_per_seed.csv'), index=False)
    log('\n{:<5} {:>22} {:>22} {:>22}'.format('split', 'AUC_KL mean+-SD', 'SRC mean+-SD', 'ENSEMBLE AUC/SRC/KL4'))
    for sp in rep_splits:
        d = det[det.split == sp]
        me = ES.metrics(ens[(chosen, sp)], labels[sp])
        log('{:<5} {:>14.4f} +- {:.4f} {:>14.4f} +- {:.4f}   {:.4f} / {:.4f} / {:.4f}'.format(
            sp, d.AUC_KL.mean(), d.AUC_KL.std(), d.SRC.mean(), d.SRC.std(), me['AUC_KL'], me['SRC'], me['AUC_KL4']))
    if args.hanya_val:
        log('(skor test/train DRA-OA disimpan untuk fusi, tetapi tidak dievaluasi sendiri: --hanya_val 1)')
    else:
        b = ES.bootstrap(ens[(chosen, 'test')], labels['test'], n_boot=args.n_boot)
        log('TEST ensemble CI 95%: AUC_KL [{:.3f}-{:.3f}]  SRC [{:.3f}-{:.3f}]'.format(
            b['AUC_KL_lo'], b['AUC_KL_hi'], b['SRC_lo'], b['SRC_hi']))
        log('\nKeputusan antar varian (dan vs dual_global_c) dibuat oleh report_dra_oa.py berdasarkan VAL.')
    with open(os.path.join(out, 'laporan.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('Tersimpan:', os.path.join(out, 'laporan.txt'))


if __name__ == '__main__':
    main()
import argparse
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- data
def norm_id(x):
    return '/'.join(str(x).replace('\\', '/').split('/')[-2:])


def normal_train_ids(meta_path, seed):
    f = os.path.join(meta_path, 'train_seed_{}.csv'.format(seed))
    return [norm_id(i) for i in pd.read_csv(f)['id']]


def split_files(img_root, split):
    root = os.path.join(img_root, split)
    files = sorted(f for f in glob.glob(os.path.join(root, '*', '*')) if f.lower().endswith(IMG_EXT))
    ids = [norm_id(os.path.relpath(f, root)) for f in files]
    labels = [int(i.split('/')[0]) for i in ids]
    return files, ids, labels


CLAHE = None   # diisi di main() bila --clahe 1


def load_batch(paths, size=224):
    ims = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise SystemExit('Gagal membaca ' + p)
        if im.shape != (size, size):
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        if CLAHE is not None:   # samakan kontras antar gambar sebelum ekstraksi fitur
            im = CLAHE.apply(im)
        ims.append(im)
    x = torch.from_numpy(np.stack(ims)).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - MEAN) / STD, ims


# ----------------------------------------------------------------------------- fitur
class FeatureExtractor(torch.nn.Module):
    """WideResNet-50-2: layer2 (28x28) + layer3 (14x14 -> 28x28), agregasi 3x3, proyeksi ke `dim`."""

    def __init__(self, pretrained=True, dim=1024):
        super().__init__()
        import torchvision
        w = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.wide_resnet50_2(weights=w)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3
        self.dim = dim
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        f = torch.cat([f2, f3], 1)                                  # B x 1536 x 28 x 28
        B, C, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, 1, C)
        f = F.adaptive_avg_pool1d(f, self.dim).reshape(B, H, W, self.dim)   # proyeksi PatchCore
        return f


def extract(model, paths, device, batch):
    out = []
    for i in range(0, len(paths), batch):
        x, _ = load_batch(paths[i:i + batch])
        out.append(model(x.to(device)).cpu())
    return torch.cat(out)                                           # N x H x W x D


# ----------------------------------------------------------------------------- memory bank & skor
def patient_of(i):
    """'2/9011420L.png' -> '9011420' (lutut kiri & kanan satu pasien)."""
    stem = os.path.splitext(str(i).replace('\\', '/').split('/')[-1])[0]
    return stem[:-1] if stem[-1:].upper() in ('L', 'R') else stem


def same_patient_idx(query_id, bank_ids):
    """Indeks gambar bank yang berasal dari pasien yang sama dengan query (untuk leave-one-out)."""
    p = patient_of(query_id)
    return [k for k, i in enumerate(bank_ids) if patient_of(i) == p]


def greedy_coreset(X, ratio, device, proj_dim=128, seed=0, return_idx=False):
    """Subsampling coreset greedy (PatchCore) dengan proyeksi acak untuk percepatan.
    return_idx=True -> juga mengembalikan indeks baris terpilih (dipakai untuk leave-one-out)."""
    n = X.shape[0]
    k = max(1, int(n * ratio))
    if k >= n:
        return (X, torch.arange(n)) if return_idx else X
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
    return (X[idx], idx) if return_idx else X[idx]


def score_global(feat, bank, device, chunk=8192, src=None, exclude=None):
    """feat: H x W x D; bank: M x D -> peta jarak H x W ke patch normal terdekat (di mana saja).
    src (M, opsional): gambar asal tiap patch bank; exclude: daftar gambar bank yang diabaikan
    (leave-one-out per pasien saat menskor split train)."""
    H, W, D = feat.shape
    q = feat.reshape(-1, D).to(device)
    best = torch.full((q.shape[0],), float('inf'), device=device)
    qq = (q * q).sum(1, keepdim=True)
    ex = torch.as_tensor(list(exclude)) if exclude else None
    for i in range(0, bank.shape[0], chunk):
        b = bank[i:i + chunk].to(device)
        d2 = qq + (b * b).sum(1)[None] - 2 * q @ b.T
        if ex is not None and src is not None:
            drop = torch.isin(src[i:i + chunk], ex).to(device)
            d2 = d2.masked_fill(drop[None], float('inf'))
        best = torch.minimum(best, d2.min(1).values)
    return best.clamp_min(0).sqrt().reshape(H, W).cpu()


def score_location(feat, bank, radius, device, valid=None):
    """feat: H x W x D; bank: N x H x W x D (gambar tersejajar).
    Patch (i,j) hanya dibandingkan dengan patch bank di (i+di, j+dj), |di|,|dj| <= radius.
    valid (opsional): N x H x W bool, patch bank yang boleh dipakai (mis. hanya area sendi gambar OA).
    Posisi tanpa kandidat valid -> inf."""
    N, H, W, D = bank.shape
    f = feat.to(device)
    b = bank.to(device)
    v = valid.to(device) if valid is not None else None
    best = torch.full((H, W), float('inf'), device=device)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            # geser bank: shifted[n,i,j] = bank[n, i+di, j+dj]
            ys0, ys1 = max(0, -di), min(H, H - di)
            xs0, xs1 = max(0, -dj), min(W, W - dj)
            fb = f[ys0:ys1, xs0:xs1]                                 # h x w x D
            bb = b[:, ys0 + di:ys1 + di, xs0 + dj:xs1 + dj]          # N x h x w x D
            d2 = ((bb - fb[None]) ** 2).sum(-1)                      # N x h x w
            if v is not None:
                d2 = d2.masked_fill(~v[:, ys0 + di:ys1 + di, xs0 + dj:xs1 + dj], float('inf'))
            d2 = d2.min(0).values
            best[ys0:ys1, xs0:xs1] = torch.minimum(best[ys0:ys1, xs0:xs1], d2)
    return best.sqrt().cpu()


def dual_score(d_norm, d_oa, how):
    """Gabungkan jarak ke bank normal & bank OA. Posisi tanpa referensi OA -> netral."""
    has = torch.isfinite(d_oa)
    if how == 'ratio':                      # 0..1, makin besar = makin mirip OA daripada normal
        s = torch.full_like(d_norm, 0.5)
        s[has] = d_norm[has] / (d_norm[has] + d_oa[has] + 1e-8)
    else:                                   # 'diff': seperti skor DCRL Belton (d_norm - d_anom)
        s = torch.zeros_like(d_norm)
        s[has] = d_norm[has] - d_oa[has]
    return s


def grade_score(d_norm, d_by_grade, how='expected', lam=0.5, temp=0.1):
    """K1C: skor keparahan dari jarak ke bank normal + bank OA per grade.
    d_by_grade: list (grade, jarak H x W) urut grade naik, mis. [(2, d2), (3, d3)]; inf = tanpa referensi.
      'expected' : perkiraan grade = sum_k g_k * softmax(-d_k / tau), k = {normal(0), KL2, KL3}
                   tau = temp x rata-rata jarak di patch itu (bebas skala).
      'diff2'    : (d_normal - jarak OA terdekat) + lam x (d_KL2 - d_KL3)
                   -> mirip OA menaikkan skor, lebih mirip KL3 daripada KL2 menaikkan lagi.
    Patch tanpa referensi OA -> netral (0)."""
    D = torch.stack([d for _, d in d_by_grade])                        # G x H x W
    has = torch.isfinite(D).all(0)
    s = torch.zeros_like(d_norm)
    if how == 'diff2':
        val = (d_norm - D.min(0).values) + lam * (D[0] - D[-1])
    else:
        allD = torch.cat([d_norm[None], D])                            # (1+G) x H x W
        g = torch.tensor([0.0] + [float(k) for k, _ in d_by_grade], dtype=allD.dtype)[:, None, None]
        safe = torch.where(torch.isfinite(allD), allD, torch.zeros_like(allD))
        tau = (temp * safe[1:].mean(0).clamp_min(1e-6) + temp * safe[0].clamp_min(1e-6)) / 2
        w = torch.softmax(-allD / tau, dim=0)
        val = (w * g).sum(0)
    s[has] = val[has]
    return s


def split_by_grade(ids):
    """{grade: [indeks]} dari id 'grade/file'."""
    out = {}
    for k, i in enumerate(ids):
        out.setdefault(int(i.split('/')[0]), []).append(k)
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------- evaluasi (sama dengan Belton)
def belton_metrics(score, label):
    score, label = np.asarray(score), np.asarray(label)
    auc = lambda y: roc_auc_score(y, score) if 0 < y.sum() < len(y) else float('nan')
    return dict(spearman=float(stats.spearmanr(score, label)[0]),
                auc_kl_gt0=auc((label > 0).astype(int)),
                auc_kl_ge2=auc((label > 1).astype(int)),
                auc_kl_ge3=auc((label > 2).astype(int)),
                auc_kl_eq4=auc((label == 4).astype(int)))


def save_map(path, img, smap, vmax, vmin=0.0):
    h = cv2.resize(smap.numpy().astype(np.float32), img.shape[::-1], interpolation=cv2.INTER_LINEAR)
    h = cv2.GaussianBlur(h, (0, 0), 4)
    h = np.clip((h - vmin) / (vmax - vmin + 1e-8), 0, 1)
    col = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path, np.hstack([base, cv2.addWeighted(base, 0.55, col, 0.45, 0)]))


def main():
    ap = argparse.ArgumentParser(description='PatchCore baseline & Kontribusi 1A (bank per lokasi)')
    ap.add_argument('--mode', default='global', choices=['global', 'location', 'dual'])
    ap.add_argument('--img_root', default='../data', help='../data (asli) atau ../aligned (tersejajar)')
    ap.add_argument('--meta_path', default='../meta/')
    ap.add_argument('--train_seed', type=int, default=1001, help='daftar 30 normal Belton: meta/train_seed_<seed>.csv')
    ap.add_argument('--eval_split', default='test', choices=['test', 'val', 'train'],
                    help="train = skor semua gambar train (untuk ekspor ke Stage 2 Belton); gambar yang ada di "
                         "memory bank dinilai leave-one-out per pasien")
    ap.add_argument('--out', required=True)
    ap.add_argument('--radius', type=int, default=1, help='[location/dual] radius tetangga (sel 8 px)')
    ap.add_argument('--oa_region', default='joint_band', choices=['joint_band', 'margins', 'all'],
                    help='[dual] bagian gambar OA yang masuk bank OA (default: pita garis sendi)')
    ap.add_argument('--band', type=float, default=32, help='[dual] setengah tinggi pita garis sendi (px)')
    ap.add_argument('--pad', type=float, default=8, help='[dual] perluasan ROI tepi sendi (px)')
    ap.add_argument('--oa_split', default='none', choices=['none', 'kl'],
                    help='[dual] kl = K1C: bank OA dipisah per grade (KL2 & KL3)')
    ap.add_argument('--sev_score', default='expected', choices=['expected', 'diff2'],
                    help='[K1C] expected = perkiraan grade (softmin) ; diff2 = selisih jarak berbobot')
    ap.add_argument('--sev_lambda', type=float, default=0.5, help='[K1C diff2] bobot (d_KL2 - d_KL3)')
    ap.add_argument('--sev_temp', type=float, default=0.1, help='[K1C expected] suhu softmin relatif')
    ap.add_argument('--dual_score', default='diff', choices=['diff', 'ratio'],
                    help='[dual] diff = d_normal - d_OA ; ratio = d_normal / (d_normal + d_OA)')
    ap.add_argument('--coreset', type=float, default=0.1, help='[global] rasio coreset (PatchCore asli: 0.1)')
    ap.add_argument('--dim', type=int, default=1024)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--save_maps', type=int, default=0, help='simpan peta anomali N gambar per grade')
    ap.add_argument('--pretrained', type=int, default=1, help='1 = bobot ImageNet (WAJIB untuk hasil nyata)')
    ap.add_argument('--summary', default='../k1_out/ringkasan_hasil.csv')
    ap.add_argument('--clahe', type=int, default=0,
                    help='1 = CLAHE sebelum ekstraksi fitur (menormalkan kecerahan antar gambar); 0 = asli')
    ap.add_argument('--metrik', type=int, default=1,
                    help='0 = hanya simpan skor & peta (tanpa menghitung metrik), dipakai untuk split test/train '
                         'saat memory bank tidak dievaluasi sendiri')
    args = ap.parse_args()
    global CLAHE
    if args.clahe:
        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    if device != args.device:
        print('  CUDA tidak tersedia -> pakai CPU (lebih lambat)')
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    model = FeatureExtractor(pretrained=bool(args.pretrained), dim=args.dim).to(device)

    # 1) memory bank dari 30 gambar normal
    tr_ids = normal_train_ids(args.meta_path, args.train_seed)
    tr_paths = [os.path.join(args.img_root, 'train', *i.split('/')) for i in tr_ids]
    miss = [p for p in tr_paths if not os.path.exists(p)]
    if miss:
        raise SystemExit('{} gambar training tidak ada, mis. {}. Untuk --mode location pastikan '
                         'align_joint.py sudah dijalankan untuk train.'.format(len(miss), miss[0]))
    bank = extract(model, tr_paths, device, args.batch)                        # N x H x W x D
    if args.mode == 'global':
        flat = bank.reshape(-1, bank.shape[-1])
        bank_g, idx_g = greedy_coreset(flat, args.coreset, device, return_idx=True)
        src_g = torch.arange(bank.shape[0]).repeat_interleave(bank.shape[1] * bank.shape[2])[idx_g]
        print('  Memory bank global: {} patch dari {} gambar (coreset {:.0%} -> {})'.format(
            flat.shape[0], len(tr_ids), args.coreset, bank_g.shape[0]))
    else:
        print('  Memory bank per lokasi: {} gambar x {}x{} posisi, radius {} -> {} kandidat per patch'.format(
            bank.shape[0], bank.shape[1], bank.shape[2], args.radius, bank.shape[0] * (2 * args.radius + 1) ** 2))

    bank_oa, valid_oa, sub, oa_ids = None, None, None, []
    if args.mode == 'dual':
        # bank kedua: patch dari 30 gambar OA (KL2/3) training, hanya di region sendi
        f_oa = os.path.join(args.meta_path, 'oa_seed_{}.csv'.format(args.train_seed))
        if not os.path.exists(f_oa):
            raise SystemExit('Tidak ada {}. Jalankan dulu: localize_joint.py --train_only 1'.format(f_oa))
        oa_ids = [norm_id(i) for i in pd.read_csv(f_oa)['id']]
        oa_paths = [os.path.join(args.img_root, 'train', *i.split('/')) for i in oa_ids]
        miss = [p for p in oa_paths if not os.path.exists(p)]
        if miss:
            raise SystemExit('{} gambar OA tidak ada di {}, mis. {}'.format(len(miss), args.img_root, miss[0]))
        bank_oa = extract(model, oa_paths, device, args.batch)
        N_, H_, W_ = bank_oa.shape[:3]
        if args.oa_region == 'all':
            valid_oa = torch.ones(N_, H_, W_, dtype=torch.bool)
        else:
            import eval_scores as ES
            lm_path = os.path.join(args.img_root, 'train', 'aligned_landmarks.csv')
            if not os.path.exists(lm_path):
                raise SystemExit('Tidak ada {} (mode dual butuh gambar tersejajar: --img_root ../aligned)'.format(lm_path))
            lm = pd.read_csv(lm_path, keep_default_na=False)
            lm['id'] = lm['id'].map(norm_id)
            lm = lm.drop_duplicates('id').set_index('id').reindex(oa_ids).reset_index()
            jb, mg = ES.region_masks(lm, H_, W_, args.band, args.pad)
            valid_oa = torch.from_numpy(jb if args.oa_region == 'joint_band' else mg)
        sub = None
        if args.oa_split == 'kl':
            sub = [(g, bank_oa[idx], valid_oa[idx], idx) for g, idx in split_by_grade(oa_ids).items()]
            print('  K1C: bank OA dipisah per grade -> ' + ', '.join('KL{}: {} gambar'.format(g, b.shape[0]) for g, b, _, _ in sub))
        print('  Memory bank OA: {} gambar KL{} , region {} -> {} patch OA ({:.0f}% posisi punya referensi OA)'.format(
            N_, '/'.join(sorted({i.split('/')[0] for i in oa_ids})), args.oa_region, int(valid_oa.sum()),
            100 * float(valid_oa.any(0).float().mean())))

    # 2) skor setiap gambar di split evaluasi
    files, ids, labels = split_files(args.img_root, args.eval_split)
    if not files:
        raise SystemExit('Tidak ada gambar di ' + os.path.join(args.img_root, args.eval_split))
    smax, stop, maps, allmaps = [], [], {}, []
    want_maps = {g: args.save_maps for g in range(5)}
    for i in range(0, len(files), args.batch):
        feats = model(load_batch(files[i:i + args.batch])[0].to(device)).cpu()
        for k, f in enumerate(feats):
            # leave-one-out per pasien (hanya split train: gambar bank ada di sana)
            ex_n = same_patient_idx(ids[i + k], tr_ids) if args.eval_split == 'train' else []
            ex_a = same_patient_idx(ids[i + k], oa_ids) if (args.eval_split == 'train' and bank_oa is not None) else []
            vn = None
            if ex_n:
                vn = torch.ones(bank.shape[:3], dtype=torch.bool)
                vn[ex_n] = False
            va = valid_oa
            if ex_a:
                va = valid_oa.clone()
                va[ex_a] = False
            if args.mode == 'global':
                m = score_global(f, bank_g, device, src=src_g, exclude=ex_n)
            elif args.mode == 'location':
                m = score_location(f, bank, args.radius, device, vn)
            elif sub is not None:   # K1C
                dn = score_location(f, bank, args.radius, device, vn)
                dg = []
                for g, b, v, gidx in sub:
                    loc = [j for j, o in enumerate(gidx) if o in ex_a]
                    if loc:
                        v = v.clone()
                        v[loc] = False
                    dg.append((g, score_location(f, b, args.radius, device, v)))
                m = grade_score(dn, dg, args.sev_score, args.sev_lambda, args.sev_temp)
            else:
                m = dual_score(score_location(f, bank, args.radius, device, vn),
                               score_location(f, bank_oa, args.radius, device, va), args.dual_score)
            allmaps.append(m.numpy().astype(np.float16))
            v = m.flatten()
            smax.append(float(v.max()))
            stop.append(float(v.topk(max(1, len(v) // 100)).values.mean()))
            g = labels[i + k]
            if want_maps.get(g, 0) > 0:
                maps[files[i + k]] = m
                want_maps[g] -= 1
        print('  {}/{} gambar'.format(min(i + args.batch, len(files)), len(files)), end='\r')
    print()

    df = pd.DataFrame(dict(id=ids, label=labels, score_max=smax, score_top1pct=stop))
    df.to_csv(os.path.join(args.out, 'scores_{}.csv'.format(args.eval_split)), index=False)
    # peta skor lengkap (28x28 per gambar) -> eval_scores.py bisa mencoba region/agregasi lain tanpa menjalankan ulang
    np.savez_compressed(os.path.join(args.out, 'maps_{}.npz'.format(args.eval_split)),
                        ids=np.array(ids), labels=np.array(labels), maps=np.stack(allmaps))

    if maps:
        md = os.path.join(args.out, 'maps_{}'.format(args.eval_split))
        os.makedirs(md, exist_ok=True)
        allv = np.stack(allmaps).astype(np.float32)
        vmax = float(np.percentile(allv, 99.5))
        vmin = float(np.percentile(allv, 50)) if args.mode == 'dual' else 0.0
        for p, m in maps.items():
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            save_map(os.path.join(md, 'KL{}_{}'.format(norm_id(p).split('/')[0], os.path.basename(p))), img, m, vmax, vmin)

    # 3) evaluasi
    if not args.metrik:
        print('Skor & peta disimpan di {} (metrik tidak dihitung: --metrik 0). Waktu: {:.1f} menit'.format(
            args.out, (time.time() - t0) / 60))
        return
    res = belton_metrics(df['score_max'], df['label'])
    print('\n----- HASIL {} ({}, {} gambar) -----'.format(args.mode.upper(), args.eval_split, len(df)))
    print('Spearman rank correlation (SRC)  : {:.4f}'.format(res['spearman']))
    print('OA AUC (KL>=2, = AUC_KL)         : {:.4f}'.format(res['auc_kl_ge2']))
    print('AUC KL>0 | KL>=3 | KL=4          : {:.4f} | {:.4f} | {:.4f}'.format(
        res['auc_kl_gt0'], res['auc_kl_ge3'], res['auc_kl_eq4']))
    print('Rata-rata skor per grade KL      : ' + '  '.join(
        'KL{}={:.3f}'.format(g, v) for g, v in df.groupby('label')['score_max'].mean().items()))
    print('Waktu: {:.1f} menit'.format((time.time() - t0) / 60))

    row = dict(waktu=time.strftime('%Y-%m-%d %H:%M'), mode=args.mode, img_root=args.img_root,
               split=args.eval_split, seed=args.train_seed, radius=args.radius if args.mode != 'global' else '',
               oa_region=args.oa_region if args.mode == 'dual' else '',
               dual_score=args.dual_score if args.mode == 'dual' else '',
               oa_split=args.oa_split if args.mode == 'dual' else '',
               sev_score=args.sev_score if (args.mode == 'dual' and args.oa_split == 'kl') else '',
               coreset=args.coreset if args.mode == 'global' else '', clahe=args.clahe, pretrained=args.pretrained,
               n=len(df), **{k: round(v, 4) for k, v in res.items()}, out=args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(args.summary):   # gabung dengan kolom lama (versi skrip lama mungkin kolomnya berbeda)
        try:
            new = pd.concat([pd.read_csv(args.summary), new], ignore_index=True, sort=False)
        except Exception:
            print('  [!] {} tidak terbaca (format lama rusak) -> disimpan sebagai cadangan .bak'.format(args.summary))
            os.replace(args.summary, args.summary + '.bak')
    cols = list(row.keys()) + [c for c in new.columns if c not in row]
    new[cols].to_csv(args.summary, index=False)
    print('Ringkasan ditambahkan ke', args.summary)


if __name__ == '__main__':
    main()
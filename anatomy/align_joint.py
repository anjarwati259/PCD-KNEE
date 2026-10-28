import argparse
import os

import cv2
import numpy as np
import pandas as pd


def _rot_params(row, W):
    """Parameter transformasi untuk satu gambar (setelah flip bila perlu)."""
    flip = row['medial_on'] == 'right'
    xl, xr = row['x_left'], row['x_right']
    yl, yr = row['y_joint_left'], row['y_joint_right']
    xc = row['x_center']
    if flip:   # cerminkan landmark: x -> W-1-x, kiri <-> kanan
        xl, xr = W - 1 - xr, W - 1 - xl
        yl, yr = yr, yl
        xc = W - 1 - xc
    ang = np.arctan2(yr - yl, xr - xl)            # radian; >0 = turun ke kanan
    cx, cy = (xl + xr) / 2.0, (yl + yr) / 2.0     # pusat rotasi = tengah garis sendi
    return flip, ang, cx, cy, xc


def forward_points(x, y, prm, W, joint_y, center_x):
    """Titik di gambar asli -> koordinat tersejajar (untuk landmark & ROI)."""
    flip, ang, cx, cy, xc_f = prm
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if flip:
        x = W - 1 - x
    c, s = np.cos(-ang), np.sin(-ang)                         # putar balik kemiringan
    xr = c * (x - cx) - s * (y - cy) + cx
    yr = s * (x - cx) + c * (y - cy) + cy
    xc_r = c * (xc_f - cx) + cx                               # titik tengah setelah rotasi (pada y=cy)
    yo = yr - cy + joint_y
    # piecewise linear: [0, xc_r] -> [0, center_x], [xc_r, W-1] -> [center_x, W-1]
    xo = np.where(xr <= xc_r, xr * center_x / max(xc_r, 1e-6),
                  center_x + (xr - xc_r) * (W - 1 - center_x) / max(W - 1 - xc_r, 1e-6))
    return xo, yo


def warp_image(img, prm, joint_y, center_x, size):
    """Bangun peta inverse (output -> sumber) lalu cv2.remap."""
    H, W = img.shape
    flip, ang, cx, cy, xc_f = prm
    c, s = np.cos(-ang), np.sin(-ang)
    xc_r = c * (xc_f - cx) + cx
    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    # inverse piecewise
    xr = np.where(u <= center_x, u * xc_r / center_x,
                  xc_r + (u - center_x) * (W - 1 - xc_r) / (size - 1 - center_x))
    yr = v - joint_y + cy
    # inverse rotasi (rotasi +ang)
    c2, s2 = np.cos(ang), np.sin(ang)
    x = c2 * (xr - cx) - s2 * (yr - cy) + cx
    y = s2 * (xr - cx) + c2 * (yr - cy) + cy
    if flip:
        x = W - 1 - x
    return cv2.remap(img, x.astype(np.float32), y.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser(description='Penyejajaran lutut ke kerangka anatomi kanonik')
    ap.add_argument('--input', required=True, help='folder gambar yang sama dengan saat localize_joint.py')
    ap.add_argument('--landmarks', required=True, help='folder output localize_joint.py (berisi landmarks.csv)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--joint_y', type=float, default=133, help='baris kanonik garis sendi (ujung kompartemen)')
    ap.add_argument('--center_x', type=float, default=112, help='kolom kanonik pembagi medial/lateral')
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--skip_flagged', type=int, default=0,
                    help='1 = lewati gambar ber-flag (selain edge_not_found); 0 = tetap disejajarkan')
    args = ap.parse_args()

    lm = pd.read_csv(os.path.join(args.landmarks, 'landmarks.csv'), keep_default_na=False)
    if (lm['medial_on'] == '').any():
        print('  [!] {} gambar tanpa info L/R -> tidak di-flip'.format((lm['medial_on'] == '').sum()))
    os.makedirs(args.out, exist_ok=True)

    rows, mean_before, mean_after, n = [], None, None, 0
    for _, r in lm.iterrows():
        if args.skip_flagged and r['flags'] and any(not f.startswith('edge_not_found') for f in r['flags'].split(';')):
            continue
        src = os.path.join(args.input, *r['id'].split('/'))
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('  [!] gagal dibaca:', src); continue
        H, W = img.shape
        prm = _rot_params(r, W)
        out = warp_image(img, prm, args.joint_y, args.center_x, args.size)
        dst = os.path.join(args.out, *r['id'].split('/'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, out)

        rec = dict(id=r['id'], kl=r['kl'], knee=r['knee'], flipped=int(prm[0]),
                   angle_deg=round(float(np.degrees(prm[1])), 3), rot_cx=round(prm[2], 2), rot_cy=round(prm[3], 2),
                   center_src=round(float(prm[4]), 2), joint_y=args.joint_y, center_x=args.center_x)
        if 'role' in r:
            rec['role'] = r['role']
        # ROI di koordinat tersejajar (medial selalu kiri) -> bounding box sudut-sudut yang ditransformasi
        for anat in ('medial', 'lateral'):
            for bone in ('fem', 'tib'):
                k = '{}_{}'.format(anat, bone)
                if k + '_x' not in r or r[k + '_x'] == '':
                    continue
                x0, y0, w, h = (float(r[k + s]) for s in ('_x', '_y', '_w', '_h'))
                xs, ys = forward_points([x0, x0 + w, x0, x0 + w], [y0, y0, y0 + h, y0 + h], prm, W,
                                        args.joint_y, args.center_x)
                bx0, by0 = int(np.clip(xs.min(), 0, args.size - 1)), int(np.clip(ys.min(), 0, args.size - 1))
                bx1, by1 = int(np.clip(xs.max(), 0, args.size - 1)), int(np.clip(ys.max(), 0, args.size - 1))
                rec.update({k + '_x': bx0, k + '_y': by0, k + '_w': bx1 - bx0, k + '_h': by1 - by0})
        # residu: posisi garis sendi di gambar tersejajar (harus ~joint_y di kedua ujung)
        key = r['id'].replace('/', '__').rsplit('.', 1)[0]
        cpath = os.path.join(args.landmarks, 'curves', key + '.npz')
        if os.path.exists(cpath):
            cv = np.load(cpath)
            xo, yo = forward_points(cv['xs'], cv['joint'], prm, W, args.joint_y, args.center_x)
            q = max(3, len(xo) // 8)
            rec['res_joint_y_medial'] = round(float(np.median(yo[np.argsort(xo)][:q]) - args.joint_y), 2)
            rec['res_joint_y_lateral'] = round(float(np.median(yo[np.argsort(xo)][-q:]) - args.joint_y), 2)
            os.makedirs(os.path.join(args.out, 'curves'), exist_ok=True)
            np.savez_compressed(os.path.join(args.out, 'curves', key + '.npz'),
                                xs=xo, joint=yo,
                                femur=forward_points(cv['xs'], cv['femur'], prm, W, args.joint_y, args.center_x)[1],
                                tibia=forward_points(cv['xs'], cv['tibia'], prm, W, args.joint_y, args.center_x)[1])
        rows.append(rec)

        a = cv2.resize(img, (args.size, args.size)).astype(np.float64)
        if r['medial_on'] == 'right':   # pembanding adil: "sebelum" juga di-flip, hanya tanpa rotasi/geser
            a = a[:, ::-1]
        mean_before = a if mean_before is None else mean_before + a
        mean_after = out.astype(np.float64) if mean_after is None else mean_after + out
        n += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'aligned_landmarks.csv'), index=False)

    # cek visual + angka: rata-rata gambar sebelum vs sesudah
    mb, ma = (mean_before / n).astype(np.uint8), (mean_after / n).astype(np.uint8)
    def sharp(m):
        """Kontras celah sendi pada gambar rata-rata: energi gradien vertikal di pita +-24 px sekitar
        garis sendi kanonik. Kalau semua gambar sejajar, tepi femur/tibia bertumpuk -> gradien tinggi."""
        g = np.abs(np.diff(m.astype(np.float64), axis=0))
        j = int(args.joint_y)
        return float(g[max(0, j - 24):j + 24].mean())
    panel = np.hstack([mb, np.full((args.size, 6), 255, np.uint8), ma])
    panel = cv2.cvtColor(cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    jy = int(args.joint_y * 2); cx2 = int((args.size + 6 + args.center_x) * 2)
    cv2.line(panel, ((args.size + 6) * 2, jy), (panel.shape[1], jy), (0, 200, 0), 1)
    cv2.line(panel, (cx2, 0), (cx2, panel.shape[0]), (0, 200, 0), 1)
    for x, t in ((6, 'Sebelum (flip saja)  kontras sendi {:.2f}'.format(sharp(mb))),
                 ((args.size + 6) * 2 + 6, 'Sesudah alignment  kontras sendi {:.2f}'.format(sharp(ma)))):
        cv2.putText(panel, t, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.out, 'alignment_check.png'), panel)

    print('Selesai: {} gambar -> {}'.format(len(df), args.out))
    print('  di-flip (medial dipindah ke kiri): {}'.format(int(df['flipped'].sum())))
    if 'res_joint_y_medial' in df:
        print('  residu garis sendi (px, ideal 0): medial {:.2f} ± {:.2f} | lateral {:.2f} ± {:.2f}'.format(
            df['res_joint_y_medial'].mean(), df['res_joint_y_medial'].std(),
            df['res_joint_y_lateral'].mean(), df['res_joint_y_lateral'].std()))
    print('  kontras celah sendi pada gambar rata-rata: sebelum {:.2f} -> sesudah {:.2f} (naik = lebih sejajar)'.format(
        sharp(mb), sharp(ma)))
    print('  cek visual: alignment_check.png')


if __name__ == '__main__':
    main()
"""
main_stage2baru.py - Jalankan main.py Belton APA ADANYA, tetapi pseudo-label dari Stage 1 diganti aturan Stage 2 baru.

Kode Belton (main.py, utils.py, ...) TIDAK diubah. Skrip ini hanya mengganti fungsi utils.get_pseudo_labels
sebelum main.py dijalankan:
  * panggilan dari skor Stage 1 (outputs/dfs/ss) untuk 30 pseudo-label  -> daftar dari --pseudo
  * panggilan dari skor Stage 1 (outputs/dfs/ss) untuk 3 pseudo-label parah -> daftar dari --parah
  * panggilan lain (mis. 263 pseudo-label iterasi 2 dari skor DCRL iterasi 1) -> fungsi asli Belton
Daftar --pseudo / --parah dibuat oleh simulasi_stage2_baru.py (pseudo_label_baru.csv, pseudo_parah_baru.csv).

Semua argumen setelah "--" diteruskan ke main.py tanpa diubah. Contoh (dijalankan dari folder kerja baru):
  python main_stage2baru.py --belton_main D:\\PCD\\PCD-KNEE\\ss_fewsome\\main.py ^
      --pseudo ..\\anatomy\\simulasi_stage2\\pseudo_label_baru.csv --parah ..\\anatomy\\simulasi_stage2\\pseudo_parah_baru.csv ^
      -- --train_ss 0 --model_name usulan_ --data_path D:/PCD/PCD-KNEE/ss_fewsome/data/ ^
         --train_ids_path D:/PCD/PCD-KNEE/hasil_stage1/meta/ --device cuda:0
"""

import argparse
import inspect
import os
import runpy
import sys

import pandas as pd


class BerhentiIter1(Exception):
    pass


def norm_id(x):
    return '/'.join(str(x).replace('\\', '/').split('/')[-2:])


def ambil_arg(argv, nama, default=None):
    return argv[argv.index(nama) + 1] if nama in argv else default


def main():
    if '--' not in sys.argv:
        raise SystemExit('Pisahkan argumen main.py dengan "--". Lihat contoh di bagian atas file ini.')
    k = sys.argv.index('--')
    ap = argparse.ArgumentParser(description='main.py Belton dengan pseudo-label Stage 2 baru')
    ap.add_argument('--belton_main', required=True, help='path main.py Belton')
    ap.add_argument('--pseudo', required=True,
                    help='pseudo_label_baru.csv (aturan Stage 2 baru), atau "belton" = aturan asli Belton dengan '
                         'pencarian margin cepat (pakai bersama --stage2_N, mis. 147)')
    ap.add_argument('--iter2', default='belton', choices=['belton', 'gabungan'],
                    help='263 pseudo-label iterasi 2: belton = dari skor DCRL iterasi 1 (asli), '
                         'gabungan = dari gabungan skor Stage 1 + DCRL iterasi 1 (ide A, lihat ideA.py)')
    ap.add_argument('--sampai_iter1', type=int, default=0,
                    help='1 = berhenti otomatis setelah DCRL iterasi 1 selesai dan hasilnya tercetak '
                         '(sebelum pseudo-label iterasi 2). Lanjutkan nanti dengan --stage2 0.')
    ap.add_argument('--tambahan_oa', default='',
                    help='C2: folder berisi seed_<seed>.csv (kolom path). Path itu ditambahkan sebagai contoh OA '
                         'DCRL iterasi 1 (stage2) untuk seed tersebut, di samping pseudo-label.')
    ap.add_argument('--parah', default='belton',
                    help='pseudo_parah_baru.csv (3 gambar), atau "belton" = 3 pseudo-label parah memakai aturan asli Belton')
    a = ap.parse_args(sys.argv[1:k])
    belton_argv = sys.argv[k + 1:]
    data_path = ambil_arg(belton_argv, '--data_path')
    prefix = ambil_arg(belton_argv, '--model_name', '')
    if not data_path:
        raise SystemExit('Argumen main.py harus berisi --data_path')
    data_path = data_path.replace('\\', '/')
    if not data_path.endswith('/'):
        data_path += '/'

    pseudo = None if a.pseudo.lower() == 'belton' else [norm_id(i) for i in pd.read_csv(a.pseudo)['id']]
    parah = None if a.parah.lower() == 'belton' else [norm_id(i) for i in pd.read_csv(a.parah)['id']]
    print('[stage2 baru] pseudo-label iterasi 2: {}'.format(
        'GABUNGAN Stage 1 + DCRL iterasi 1 (ide A)' if a.iter2 == 'gabungan' else 'aturan asli Belton'))
    print('[stage2 baru] pseudo-label iterasi 1: {} | parah: {}'.format(
        'aturan asli Belton (jumlah dari --stage2_N)' if pseudo is None else
        '{} gambar dari {}'.format(len(pseudo), os.path.abspath(a.pseudo)),
        'aturan asli Belton' if parah is None else '{} gambar dari {}'.format(len(parah), a.parah)))

    belton_dir = os.path.dirname(os.path.abspath(a.belton_main))
    sys.path.insert(0, belton_dir)
    try:
        import utils as belton
    except Exception as e:                       # scipy baru tidak punya scipy.ndimage.filters
        if 'filters' not in str(e):
            raise
        import scipy.ndimage as nd
        sys.modules['scipy.ndimage.filters'] = nd
        import utils as belton

    if a.tambahan_oa:
        import datasets.oa_knee as OK
        init_asli = OK.oa.__init__

        def init_baru(self, root, task, stage='ss', *args, **kwargs):
            init_asli(self, root, task, stage, *args, **kwargs)
            seed = kwargs.get('seed', args[8] if len(args) > 8 else 0)
            f = os.path.join(a.tambahan_oa, 'seed_{}.csv'.format(seed))
            if task == 'train' and stage == 'stage2':
                if not os.path.exists(f):
                    raise SystemExit('[GAGAL] tidak ada {}'.format(f))
                tamb = [p for p in pd.read_csv(f)['path'].tolist() if p not in self.paths2]
                self.paths2 += tamb
                self.targets2 += [5] * len(tamb)
                self.shots = len(self.paths2) - len(self.paths)
                print('[stage2 baru] C2: +{} contoh OA (berlabel + sintetis) untuk seed {} -> total contoh OA {}'.format(
                    len(tamb), seed, self.shots), flush=True)
        OK.oa.__init__ = init_baru

    asli = belton.get_pseudo_labels
    sig = inspect.signature(asli)

    def tulis(sub, tag, ids):
        d = os.path.join('outputs', 'label_details', sub)
        os.makedirs(d, exist_ok=True)
        pd.DataFrame({'id': ids}).to_csv(os.path.join(d, '{}{}_stage2baru.csv'.format(prefix, tag)), index=False)

    def get_pseudo_labels_baru(*args, **kw):
        b = sig.bind_partial(*args, **kw).arguments
        teks = [str(v).replace('\\', '/') for v in b.values() if isinstance(v, str)]
        dari_stage1 = any('/dfs/ss' in t for t in teks)
        n = b.get('num_pseudo_labels')
        if dari_stage1 and n is not None and n >= 10 and pseudo is not None:
            print('[stage2 baru] {} pseudo-label iterasi 1 memakai aturan baru (Belton meminta {}; bukan pencarian margin)'
                  .format(len(pseudo), n), flush=True)
            tulis('stage2', 'iter1_pseudo_anom_labels', pseudo)
            return [data_path + 'train/' + i for i in pseudo], 1.0
        if dari_stage1 and n is not None and n < 10 and parah is not None:
            print('[stage2 baru] 3 pseudo-label parah memakai aturan baru', flush=True)
            tulis('stage_severe_pred', 'sev_pseudo_anom_labels', parah)
            return [data_path + 'train/' + i for i in parah], 1.0
        dari_it1 = any('/dfs/stage2' in t for t in teks)
        if dari_it1 and a.sampai_iter1:
            raise BerhentiIter1()
        if dari_it1 and a.iter2 == 'gabungan' and n is not None:
            import ideA
            it1_dir = [t for t in teks if '/dfs/stage2' in t][0]
            meta_dir = str(b.get('train_ids_path', '')).replace('\\', '/')
            ids, m = ideA.pilih_gabungan('./outputs/dfs/ss/', it1_dir, b.get('current_epoch'), prefix, meta_dir, int(n))
            print('[stage2 baru] {} pseudo-label iterasi 2 dari GABUNGAN skor Stage 1 + DCRL iterasi 1 (ide A)'.format(len(ids)),
                  flush=True)
            tulis('stage3', 'iter2_pseudo_anom_labels', ids)
            return [data_path + 'train/' + i for i in ids], m
        print('[stage2 baru] panggilan lain (n={}) -> aturan asli Belton (pencarian margin cepat)'.format(n), flush=True)
        return cari_margin(b, n)

    def jumlah(b, m):
        """Pseudo-label Belton pada margin tetap m (num_pseudo_labels=None -> tanpa pencarian), urut av menurun."""
        arg = dict(b)
        arg['margin'], arg['num_pseudo_labels'] = float(m), None
        ids, _ = asli(**arg)
        return ids

    def cari_margin(b, n):
        """Hasil SAMA dengan pencarian Belton bila jumlah tepat n bisa dicapai (himpunan pseudo-label untuk margin
        yang lebih besar selalu bagian dari margin yang lebih kecil, jadi himpunan berukuran n itu tunggal).
        Bedanya: dicari dengan bagi-dua (puluhan langkah, bukan ribuan) dan TIDAK berhenti bila n terlewati
        (jumlah melompat, mis. 264 -> 262): diambil n dengan 'av' tertinggi dari himpunan terkecil yang >= n."""
        if n is None:
            return asli(**b)
        n = int(n)
        lo, hi = float(b.get('margin', 0.8)), float(getattr(belton, 'MAX_MARGIN', 4))
        step = float(getattr(belton, 'precision', 0.0001))
        ids_lo = jumlah(b, lo)
        while len(ids_lo) < n and lo > 0:
            lo = max(0.0, round(lo - 0.2, 6))
            print('[stage2 baru]   jumlah di margin awal < {}, margin awal diturunkan ke {:.2f}'.format(n, lo), flush=True)
            ids_lo = jumlah(b, lo)
        if len(ids_lo) < n:
            print('[stage2 baru]   [!] hanya {} kandidat bahkan di margin 0; semuanya dipakai'.format(len(ids_lo)), flush=True)
            return ids_lo, lo
        if len(ids_lo) == n:
            return ids_lo, lo
        ids_hi = jumlah(b, hi)
        if len(ids_hi) >= n:
            return ids_hi[:n], hi
        while hi - lo > step:                   # invarian: jumlah(lo) > n > jumlah(hi)
            mid = (lo + hi) / 2
            ids = jumlah(b, mid)
            print('[stage2 baru]   margin {:.5f} -> {} pseudo-label'.format(mid, len(ids)), flush=True)
            if len(ids) == n:
                print('[stage2 baru]   tepat {} di margin {:.5f} (sama dengan pencarian Belton)'.format(n, mid), flush=True)
                return ids, mid
            if len(ids) > n:
                lo, ids_lo = mid, ids
            else:
                hi = mid
        print('[stage2 baru]   [!] jumlah tepat {} tidak bisa dicapai (melompat dari {} ke lebih kecil); '
              'diambil {} dengan av tertinggi dari {} kandidat di margin {:.5f}'.format(n, len(ids_lo), n, len(ids_lo), lo),
              flush=True)
        return ids_lo[:n], lo

    belton.get_pseudo_labels = get_pseudo_labels_baru
    sys.argv = [a.belton_main] + belton_argv
    print('[stage2 baru] menjalankan {} di folder {}'.format(a.belton_main, os.getcwd()), flush=True)
    try:
        runpy.run_path(a.belton_main, run_name='__main__')
    except BerhentiIter1:
        print('\n[stage2 baru] BERHENTI setelah DCRL iterasi 1 (--sampai_iter1 1). Hasil iterasi 1 ada di atas '
              '(For stage stage2) dan tersimpan di outputs\\. Untuk melanjutkan: jalankan lagi tanpa --sampai_iter1 '
              'dan tambahkan --stage2 0 setelah "--".', flush=True)
        return
    except SystemExit as e:                     # Belton memanggil sys.exit() tanpa kode error saat gagal
        if e.code in (None, 0):
            raise SystemExit('[GAGAL] main.py berhenti sebelum selesai (sys.exit). Lihat pesan di atas.')
        raise


if __name__ == '__main__':
    main()

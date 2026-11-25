"""
run_C1C2.py - Percobaan C1 + C2: anatomi Stage 1 dibawa ke DCRL.

  C1: semua gambar disejajarkan lalu dipotong ke pita sendi (DCRL hanya melihat daerah sendi)
  C2: DCRL iterasi 1 mendapat contoh OA tambahan per seed: 30 OA berlabel + 30 OA sintetis (KL2/3/4)
  Stage 1, aturan Stage 2 (30 pseudo-label, margin Belton) dan kode Belton TIDAK diubah.

Dari folder anatomy:
  python run_C1C2.py                  # siapkan data + DCRL iterasi 1 saja, lalu BERHENTI otomatis (sekitar 1 hari)
  python run_C1C2.py --tahap lanjut   # kalau iterasi 1 bagus: iterasi 2 + prediktor parah + FINAL + evaluasi
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))


def main():
    ap = argparse.ArgumentParser(description='C1 + C2: potong sendi + OA berlabel/sintetis untuk DCRL')
    ap.add_argument('--tahap', default='iter1', choices=['iter1', 'lanjut'])
    old = os.path.join(ROOT, 'hasil_stage1', 'pipeline_old')
    ap.add_argument('--pipeline', default=old if os.path.isdir(old) else os.path.join(ROOT, 'hasil_stage1', 'pipeline'),
                    help='run usulan lama (sumber skor Stage 1: outputs/dfs/ss)')
    ap.add_argument('--stage1', default=os.path.join(ROOT, 'hasil_stage1'))
    ap.add_argument('--data', default=os.path.join(ROOT, 'ss_fewsome', 'data'))
    ap.add_argument('--belton_main', default=os.path.join(ROOT, 'ss_fewsome', 'main.py'))
    ap.add_argument('--work', default=os.path.join(ROOT, 'hasil_C1C2'))
    ap.add_argument('--usulan_lama', default=os.path.join(HERE, 'hasil_usulan'), help='hasil evaluasi usulan lama (pembanding)')
    ap.add_argument('--prefix', default='usulan_')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    W = os.path.abspath(a.work)
    PIPE = os.path.join(W, 'pipeline')
    CROP = os.path.join(W, 'data_crop')
    TAMB = os.path.join(W, 'tambahan_oa')
    meta = os.path.join(a.stage1, 'meta').replace('\\', '/').rstrip('/') + '/'
    data_crop = CROP.replace('\\', '/').rstrip('/') + '/'
    os.makedirs(os.path.join(W, 'log'), exist_ok=True)

    def run(nama, script, args, cwd=HERE):
        cmd = [sys.executable, '-u', script if os.path.isabs(script) else os.path.join(HERE, script)] + [str(x) for x in args]
        print('\n$ ' + ' '.join(cmd) + ('   (di {})'.format(cwd) if cwd != HERE else ''), flush=True)
        with open(os.path.join(W, 'log', nama + '.log'), 'a', encoding='utf-8') as log:
            p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                sys.stdout.write(line)
                log.write(line)
            p.wait()
        if p.returncode != 0:
            raise SystemExit('\n[GAGAL] langkah "{}". Lihat {}'.format(nama, os.path.join(W, 'log', nama + '.log')))

    common = ['--belton_main', os.path.abspath(a.belton_main), '--pseudo', 'belton', '--iter2', 'belton', '--parah', 'belton']
    belton = ['--train_ss', 0, '--model_name', a.prefix, '--data_path', data_crop, '--train_ids_path', meta, '--device', a.device]

    if a.tahap == 'iter1':
        # 1. data C1 + C2
        if os.path.exists(os.path.join(TAMB, 'seed_244959.csv')) and os.path.isdir(os.path.join(CROP, 'test', '4')):
            print('[siapkan] data C1 + C2 sudah ada, dilewati')
        else:
            run('siapkan', 'siapkan_C1C2.py', ['--data', a.data, '--stage1', a.stage1, '--out', W])
        # 2. skor Stage 1 (sama dengan run usulan lama -> 30 pseudo-label yang sama)
        for sub in (('dfs', 'ss'), ('results', 'ss')):
            src, dst = os.path.join(a.pipeline, 'outputs', *sub), os.path.join(PIPE, 'outputs', *sub)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
                print('[siapkan] disalin: {} -> {}'.format(src, dst))
        # 3. DCRL iterasi 1 saja
        run('iter1', 'main_stage2baru.py', common + ['--tambahan_oa', TAMB, '--sampai_iter1', 1, '--'] + belton, cwd=PIPE)
        print('\nBandingkan hasil "For stage stage2 / RESULTS ON TEST SET" di atas dengan: usulan lama 0,745 / 0,458, '
              'baseline 0,788 / 0,535 (OA AUC / Spearman).')
    else:
        run('lanjut', 'main_stage2baru.py', common + ['--'] + belton + ['--stage2', 0], cwd=PIPE)
        hasil = os.path.join(W, 'hasil')
        run('evaluasi', 'evaluasi_pipeline.py', ['--folder', PIPE, '--prefix', a.prefix, '--meta', os.path.join(a.stage1, 'meta'),
                                                 '--out', hasil])
        run('laporan', 'perbaikan_stage23.py', ['--hasil', hasil, '--out', os.path.join(W, 'perbaikan')])
        if os.path.exists(os.path.join(a.usulan_lama, 'per_seed.csv')):
            run('laporan', 'bandingkan_hasil.py', ['--baseline', a.usulan_lama, '--usulan', hasil,
                                                   '--out', os.path.join(W, 'vs_usulan_lama')])
        print('\nHasil: {}\\ringkasan_stage1_3.txt dan {}\\vs_usulan_lama\\perbandingan.txt'.format(hasil, W))


if __name__ == '__main__':
    main()

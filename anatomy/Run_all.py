import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BELTON_SEEDS = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
STEPS = ['meta', 'lokalisasi', 'penyejajaran', 'memory_bank', 'bank_taji', 'sintesis', 'dra_oa', 'stage1',
         'stage23', 'evaluasi', 'laporan']
SPLITS = ['train', 'val', 'test']

# komponen metode usulan (tetap, ditetapkan sebelum melihat hasil)
K1_VARIAN = 'k1c'           # memory bank: dual, bank OA per grade (KL2/KL3), pada gambar tersejajar
DRA_PRESET = 'utama'        # DRA-OA: kepala S, P, R; OA sintetis dari taji asli + penyempitan celah
MODEL_NAME = 'usulan_'      # --model_name untuk main.py Belton


class Runner:
    def __init__(self, work):
        self.logdir = os.path.join(work, 'log')
        os.makedirs(self.logdir, exist_ok=True)

    def __call__(self, step, script, args, cwd=HERE):
        """Jalankan satu skrip, tampilkan & simpan keluarannya; berhenti bila gagal."""
        path = script if os.path.isabs(script) else os.path.join(HERE, script)
        cmd = [sys.executable, '-u', path] + [str(a) for a in args]
        print('\n$ ' + ' '.join(cmd) + ('   (di {})'.format(cwd) if cwd != HERE else ''), flush=True)
        log = open(os.path.join(self.logdir, step + '.log'), 'a', encoding='utf-8')
        log.write('\n$ ' + ' '.join(cmd) + '\n')
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        p.wait()
        log.close()
        if p.returncode != 0:
            raise SystemExit('\n[GAGAL] langkah "{}" ({}). Lihat {}'.format(
                step, os.path.basename(script), os.path.join(self.logdir, step + '.log')))


def main():
    ap = argparse.ArgumentParser(description='Metode usulan utuh: Stage 1 -> Stage 2-3 Belton, hasil tiap stage dicatat')
    ap.add_argument('--data', required=True, help='folder dataset (berisi train/ val/ test/, subfolder 0..4)')
    ap.add_argument('--meta', required=True, help='folder meta Belton (train_seed_*.csv, train_ids.csv, sim_scores.csv)')
    ap.add_argument('--belton_root', default=None, help='folder ss_fewsome Belton (berisi main.py); perlu untuk stage23')
    ap.add_argument('--work', default=os.path.join(HERE, '..', 'hasil_stage1'), help='folder semua hasil')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--pretrained', type=int, default=1, help='1 = bobot ImageNet (WAJIB untuk hasil nyata)')
    ap.add_argument('--oa_pool_size', type=int, default=150)
    ap.add_argument('--bank_pct', type=float, default=90, help='ambang bank taji: persentil skor ROI normal')
    ap.add_argument('--iters', type=int, default=1000, help='iterasi training DRA-OA per seed')
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--belton_main', default=None, help='default <belton_root>/main.py')
    ap.add_argument('--belton_args', default='', help='argumen tambahan untuk main.py, mis. "--start_margin 0.9"')
    ap.add_argument('--cepat', type=int, default=0, help='1 = uji cepat (DRA 30 iterasi); hasil tidak untuk dilaporkan')
    ap.add_argument('--mulai_dari', default=STEPS[0], choices=STEPS)
    ap.add_argument('--sampai', default=STEPS[-1], choices=STEPS)
    ap.add_argument('--ulang', default='', help='langkah yang dipaksa diulang (laporan/skor dihitung ulang; model yang '
                                                'sudah dilatih dipakai ulang - hapus foldernya untuk melatih ulang)')
    a = ap.parse_args()
    if a.cepat:
        a.iters, a.n_boot = min(a.iters, 30), min(a.n_boot, 50)

    W, D = os.path.abspath(a.work), os.path.abspath(a.data)
    M = os.path.join(W, 'meta')
    AO, AL = os.path.join(W, 'anatomy_out'), os.path.join(W, 'aligned')
    K1, K2 = os.path.join(W, 'k1_out', 'seeds'), os.path.join(W, 'k2_out')
    DRA = os.path.join(K2, 'dra')
    PIPE = os.path.join(W, 'pipeline')                     # main.py Belton dijalankan di sini (outputs/)
    HASIL = os.path.join(W, 'hasil_akhir')
    LM_TRAIN = os.path.join(AO, 'train', 'landmarks.csv')
    BR = os.path.abspath(a.belton_root) if a.belton_root else None
    BMAIN = os.path.abspath(a.belton_main) if a.belton_main else (os.path.join(BR, 'main.py') if BR else None)
    os.makedirs(W, exist_ok=True)
    run = Runner(W)
    ulang = {s.strip() for s in a.ulang.split(',') if s.strip()}
    i0, i1 = STEPS.index(a.mulai_dari), STEPS.index(a.sampai)
    common = ['--device', a.device, '--pretrained', a.pretrained]
    t_all = time.time()

    def active(step):
        return i0 <= STEPS.index(step) <= i1

    def done(step, *paths):
        if step.split()[0] in ulang:
            return False
        ok = all(os.path.exists(p) for p in paths)
        if ok:
            print('[{}] sudah ada, dilewati'.format(step))
        return ok

    def header(step):
        print('\n' + '#' * 78 + '\n# {:>2}. {}\n'.format(STEPS.index(step) + 1, step.upper()) + '#' * 78, flush=True)

    with open(os.path.join(W, 'run_all_config.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(vars(a), k1_varian=K1_VARIAN, dra_preset=DRA_PRESET, model_name=MODEL_NAME), f, indent=1)
    for sp in SPLITS:
        if not os.path.isdir(os.path.join(D, sp)):
            raise SystemExit('Tidak ada folder {} (dataset harus berisi train/ val/ test/)'.format(os.path.join(D, sp)))

    # 1 ---------------------------------------------------------------- meta + pool OA
    if active('meta'):
        header('meta')
        os.makedirs(M, exist_ok=True)
        for f in os.listdir(a.meta):
            src = os.path.join(a.meta, f)
            if os.path.isfile(src) and not os.path.exists(os.path.join(M, f)):
                shutil.copy2(src, os.path.join(M, f))
        need = [os.path.join(M, 'oa_seed_{}.csv'.format(s)) for s in BELTON_SEEDS] + [os.path.join(M, 'oa_pool.csv')]
        if not done('meta', *need):
            run('meta', 'make_oa_pool.py', ['--train_root', os.path.join(D, 'train'), '--meta_path', M,
                                            '--pool_size', a.oa_pool_size])

    # 2 ---------------------------------------------------------------- (1) lokalisasi anatomi
    if active('lokalisasi'):
        header('lokalisasi')
        for sp in SPLITS:
            if not done('lokalisasi ' + sp, os.path.join(AO, sp, 'landmarks.csv')):
                run('lokalisasi', 'localize_joint.py', ['--input', os.path.join(D, sp), '--out', os.path.join(AO, sp),
                                                        '--overlay', 0, '--qa_per_grade', 0])
        if not done('lokalisasi cek_trainset', os.path.join(AO, 'cek_trainset', 'landmarks.csv')):
            run('lokalisasi', 'localize_joint.py', ['--input', os.path.join(D, 'train'), '--out',
                                                    os.path.join(AO, 'cek_trainset'), '--train_only', 1,
                                                    '--train_seed', 'all', '--meta_path', M, '--overlay', 1])

    # 3 ---------------------------------------------------------------- (1) penyejajaran
    if active('penyejajaran'):
        header('penyejajaran')
        for sp in SPLITS:
            if not done('penyejajaran ' + sp, os.path.join(AL, sp, 'aligned_landmarks.csv')):
                run('penyejajaran', 'align_joint.py', ['--input', os.path.join(D, sp), '--landmarks', os.path.join(AO, sp),
                                                       '--out', os.path.join(AL, sp)])

    # 4 ---------------------------------------------------------------- (2) memory bank anatomi
    if active('memory_bank'):
        header('memory_bank')
        kj = os.path.join(K1, 'keputusan_k1.json')
        have = set(json.load(open(kj, encoding='utf-8'))['varian']) if os.path.exists(kj) else set()
        if not done('memory_bank', kj) or K1_VARIAN not in have:
            run('memory_bank', 'run_seeds.py', ['--seeds', 'all', '--variants', K1_VARIAN, '--out', K1, '--meta_path', M,
                                                '--data', D, '--aligned', AL, '--anatomy_out', AO,
                                                '--n_boot', a.n_boot, '--hanya_val', 1] + common)
        v = json.load(open(kj, encoding='utf-8'))['varian'][K1_VARIAN]
        for s in BELTON_SEEDS:                               # skor seluruh train (dibutuhkan Stage 2 Belton)
            od = os.path.join(K1, v['folder'], 'seed_{}_train'.format(s))
            if not done('memory_bank_train seed {}'.format(s), os.path.join(od, 'maps_train.npz')):
                run('memory_bank', v['script'], v['args'] + ['--train_seed', s, '--eval_split', 'train', '--out', od,
                                                             '--meta_path', M, '--metrik', 0, '--summary',
                                                             os.path.join(K1, 'ringkasan_runs.csv')] + common)

    # 5 ---------------------------------------------------------------- (3) penambangan taji
    if active('bank_taji'):
        header('bank_taji')
        if not done('bank_taji', os.path.join(K2, 'bank_bersih', 'osteophyte_bank.csv')):
            run('bank_taji', 'extract_osteophyte_bank.py', ['--img_root', D, '--meta_path', M, '--landmarks', LM_TRAIN,
                                                            '--out', os.path.join(K2, 'bank_bersih'), '--pct', a.bank_pct,
                                                            '--auto_rule', 'kandidat_tonjol'] + common)

    # 6 ---------------------------------------------------------------- (4) sintesis OA (cek & validasi)
    if active('sintesis'):
        header('sintesis')
        out = os.path.join(K2, 'sintetis_bank_bersih')
        if not done('sintesis', os.path.join(out, 'ringkasan.txt')):
            extra = ['--n_valid', 10, '--n_stat', 40, '--n_save', 10] if a.cepat else []
            run('sintesis', 'make_synthetic_oa.py', ['--bank', os.path.join(K2, 'bank_bersih'), '--out', out,
                                                     '--img_root', D, '--meta_path', M, '--landmarks', LM_TRAIN] + extra)

    # 7 ---------------------------------------------------------------- (5) DRA-OA
    if active('dra_oa'):
        header('dra_oa')
        need = [os.path.join(DRA, DRA_PRESET, 'scores_{}_seed_{}.csv'.format(sp, s)) for sp in SPLITS for s in BELTON_SEEDS]
        if not done('dra_oa', os.path.join(DRA, DRA_PRESET, 'keputusan.json'), *need):
            run('dra_oa', 'train_dra_oa.py', ['--preset', DRA_PRESET, '--bank_root', K2, '--out_root', DRA, '--img_root', D,
                                              '--meta_path', M, '--landmarks', LM_TRAIN, '--seeds', 'all',
                                              '--iters', a.iters, '--n_boot', a.n_boot, '--score_train', 1,
                                              '--hanya_val', 1] + common)

    # 8 ---------------------------------------------------------------- (6) fusi -> hasil Stage 1
    if active('stage1'):
        header('stage1')
        if not done('stage1', os.path.join(HASIL, 'stage1', 'ekspor.json')):
            run('stage1', 'export_belton.py', ['--work', W, '--winner', 'fusi_{}+{}'.format(DRA_PRESET, K1_VARIAN),
                                               '--belton_root', PIPE, '--belton_data_path', D + '/',
                                               '--model_name', MODEL_NAME, '--hasil_dir', os.path.join(HASIL, 'stage1'),
                                               '--n_boot', a.n_boot])

    # 9 ---------------------------------------------------------------- Stage 2-3 (main.py Belton)
    if active('stage23'):
        header('stage23')
        if not BMAIN or not os.path.exists(BMAIN):
            raise SystemExit('Langkah stage23 butuh --belton_root (folder ss_fewsome berisi main.py)')
        sel, eks = os.path.join(PIPE, 'SELESAI'), os.path.join(HASIL, 'stage1', 'ekspor.json')
        if os.path.exists(sel) and os.path.exists(eks) and os.path.getmtime(eks) > os.path.getmtime(sel):
            print('[stage23] skor Stage 1 lebih baru dari hasil main.py -> main.py dijalankan ulang')
            os.remove(sel)
        if not done('stage23', sel):
            args = ['--train_ss', 0, '--model_name', MODEL_NAME, '--data_path', D + '/', '--train_ids_path', M + '/',
                    '--device', 'cuda:0' if a.device == 'cuda' else a.device] + (a.belton_args.split() if a.belton_args else [])
            run('stage23', BMAIN, args, cwd=PIPE)
            open(os.path.join(PIPE, 'SELESAI'), 'w').write(time.strftime('%Y-%m-%d %H:%M'))

    # 10 --------------------------------------------------------------- hasil Stage 1, 2, 3
    if active('evaluasi'):
        header('evaluasi')
        if not os.path.exists(os.path.join(PIPE, 'SELESAI')):
            print('[evaluasi] main.py belum selesai (langkah stage23), dilewati')
        else:
            run('evaluasi', 'evaluasi_pipeline.py', ['--folder', PIPE, '--prefix', MODEL_NAME, '--meta', M,
                                                     '--out', os.path.join(HASIL, 'stage1_3'), '--n_boot', a.n_boot])

    # 11 --------------------------------------------------------------- laporan gabungan
    if active('laporan'):
        header('laporan')
        parts = [('HASIL STAGE 1 - 3 (metode usulan)', os.path.join(HASIL, 'stage1_3', 'ringkasan_stage1_3.txt')),
                 ('STAGE 1: SKOR GABUNGAN + PENGATURAN METODE (tuning di VAL)', os.path.join(HASIL, 'stage1', 'ringkasan_stage1.txt')),
                 ('ANGGARAN LABEL', os.path.join(M, 'label_budget.txt')),
                 ('(2) MEMORY BANK ANATOMI - tuning region:agregasi di VAL', os.path.join(K1, 'laporan_seeds.txt')),
                 ('(3) BANK TAJI', os.path.join(K2, 'bank_bersih', 'ringkasan.txt')),
                 ('(4) SINTESIS OA', os.path.join(K2, 'sintetis_bank_bersih', 'ringkasan.txt')),
                 ('(5) DRA-OA - tuning kandidat skor di VAL', os.path.join(DRA, DRA_PRESET, 'laporan.txt'))]
        txt = ['LAPORAN METODE USULAN (Stage 1 -> Stage 3)', 'dibuat: ' + time.strftime('%Y-%m-%d %H:%M'),
               'data: {} | pretrained: {} | cepat: {}'.format(D, a.pretrained, a.cepat),
               'komponen: memory bank {} + DRA-OA {} (fusi z-score)'.format(K1_VARIAN, DRA_PRESET)]
        if a.cepat or not a.pretrained:
            txt.append('[!] mode uji (cepat / tanpa pretrained): angka TIDAK untuk dilaporkan')
        for title, f in parts:
            if os.path.exists(f):
                txt += ['', '=' * 78, title, '=' * 78, open(f, encoding='utf-8').read().rstrip()]
        txt += ['', 'Cek visual: {}/cek_trainset/qa_sheet.png, {}/<split>/alignment_check.png,'.format(AO, AL),
                '            {}/bank_bersih/qa_taji_*.png, {}/sintetis_bank_bersih/preview_grid.png'.format(K2, K2)]
        out = os.path.join(W, 'LAPORAN.txt')
        with open(out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt) + '\n')
        print('Laporan gabungan: ' + out)
    print('\nSelesai dalam {:.1f} menit.'.format((time.time() - t_all) / 60))


if __name__ == '__main__':
    main()
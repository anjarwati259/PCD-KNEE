"""
Cek bahwa output Stage 1 usulan bisa dibaca fungsi Stage 2 Belton (tanpa menjalankan DCRL).

  python -m stage1_usulan.tools.check_belton_compat --model_name usulan_A9_ --data_path ./data/ --meta_path ../meta/

Yang dicek:
  1. print_ensemble_results Belton membaca 10 file seed dan menghitung metrik.
  2. get_pseudo_labels Belton dijalankan pada beberapa margin tetap -> berapa pseudo-label yang lolos.
  3. (opsional, --search) pencarian margin seperti main.py sampai tepat --n_pseudo label.
     Pencarian asli memakai langkah 0.0001 sehingga lambat; di sini langkahnya bisa diperbesar
     dengan --precision hanya untuk pengecekan. main.py sendiri tetap memakai nilai Belton.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # folder ss_fewsome
sys.path.insert(0, HERE)


def _grades(ids):
    g = [int(p.split("/")[-2]) for p in ids]
    return ", ".join(f"KL{k}={g.count(k)}" for k in range(5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--meta_path", default="../meta/")
    ap.add_argument("--n_pseudo", type=int, default=30)
    ap.add_argument("--margins", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.5])
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--precision", type=float, default=0.01)
    a = ap.parse_args()

    import utils as belton  # utils.py milik Belton
    os.makedirs("./outputs/label_details/stage2", exist_ok=True)
    kw = dict(metric="centre_mean", current_epoch=400, model_name_prefix=a.model_name,
              model_name="stage2/" + a.model_name)

    print("== 1. print_ensemble_results (Belton) ==", flush=True)
    belton.print_ensemble_results("./outputs/dfs/ss/", 400, "ss", "centre_mean", None, 0, model_name_prefix=a.model_name)

    print("\n== 2. get_pseudo_labels (Belton) pada margin tetap ==", flush=True)
    for m in a.margins:
        ids, _ = belton.get_pseudo_labels(a.meta_path, "./outputs/dfs/ss/", a.data_path, margin=m,
                                          num_pseudo_labels=None, **kw)
        print(f"  margin {m:.2f}: {len(ids):4d} pseudo-label  ({_grades(ids)})", flush=True)

    if a.search:
        belton.precision = a.precision
        print(f"\n== 3. pencarian margin sampai {a.n_pseudo} label (langkah {a.precision}) ==", flush=True)
        ids, margin = belton.get_pseudo_labels(a.meta_path, "./outputs/dfs/ss/", a.data_path, margin=min(a.margins),
                                               num_pseudo_labels=a.n_pseudo, **kw)
        print(f"  margin terpilih {margin:.4f}; {len(ids)} pseudo-label ({_grades(ids)})")
    print("\nOK: format output kompatibel dengan Stage 2 Belton.")


if __name__ == "__main__":
    main()

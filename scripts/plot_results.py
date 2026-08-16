"""Generate the paper's result figures from the sweep CSVs.

Usage:  python plot_results.py <build_dir> <figs_dir>

Reads  <build_dir>/aof_bench_results.csv  and  <build_dir>/cert_bench_results.csv
plus   <build_dir>/runs/itree_*.txt       (baseline logs, if present)
Writes <figs_dir>/q_*.png (query/VO), c_*.png (construction/storage), v_*.png
(verifiable construction) — one panel per dimension, consistent styling.
"""
import sys, re, pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AOF_COLS = ["n","d","seed","construct_s","nodes","leaves","internals","maxdepth",
            "ads_bytes","idx_pruned","topk_server_ms","topk_verify_ms","topk_vo_b",
            "topk_gap","range_server_ms","range_verify_ms","range_vo_b",
            "topk_acc","range_acc","tamper_rej"]
CERT_COLS = ["n","d","construct_s","n_part","n_nonpart","cert_bytes","nnz_avg",
             "nnz_max","gen_s","verify_s","commit_s","vfail","tamper"]

def read_itree(build_dir):
    """Baseline numbers via parse_itree.py's CSV (BOM-aware parsing)."""
    p = pathlib.Path(build_dir) / "itree_results.csv"
    if not p.exists():
        return pd.DataFrame(columns=["n", "d", "storage_b", "fc_s", "time_s"])
    df = pd.read_csv(p)
    # Wall time preferred; Simplex feasibility time as fallback (it dominates).
    df["plot_time"] = df["time_s"].fillna(df["fc_s"])
    return df

def style(ax, xlabel, ylabel, logy=False):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if logy: ax.set_yscale("log")
    ax.legend(fontsize=9)

def save(fig, figs, name):
    fig.tight_layout()
    fig.savefig(figs / name, dpi=160)
    plt.close(fig)
    print("wrote", name)

def main(build, figs):
    build, figs = pathlib.Path(build), pathlib.Path(figs)
    figs.mkdir(exist_ok=True)
    aof = pd.read_csv(build / "aof_bench_results.csv", names=AOF_COLS)
    aof = aof.drop_duplicates(subset=["n","d"], keep="last").sort_values(["d","n"])
    try:
        cert = pd.read_csv(build / "cert_bench_results.csv", names=CERT_COLS)
        cert = cert.drop_duplicates(subset=["n","d"], keep="last").sort_values(["d","n"])
    except FileNotFoundError:
        cert = pd.DataFrame(columns=CERT_COLS)
    itree = read_itree(build)

    for d in sorted(aof["d"].unique()):
        A = aof[aof.d == d]
        C = cert[cert.d == d] if len(cert) else cert
        I = itree[itree.d == d] if len(itree) else itree

        # --- Construction time: AOF vs I-tree baseline ---
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot(A.n, A.construct_s, "o-", label="AOF-tree (vertex-based)")
        if len(C): ax.plot(C.n, C.verify_s + C.commit_s, "s-", label="Owner: verify certificates + commit")
        if len(I): ax.plot(I.n, I.plot_time, "^-", label="I-tree (Simplex)")
        style(ax, "n", "seconds", logy=True)
        save(fig, figs, f"c_construct_{d}.png")

        # --- Storage: pruned vs unpruned index + ADS vs I-tree ---
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot(A.n, A.idx_pruned / 1024, "o-", label="Pruned index")
        unpruned = A.idx_pruned + A.leaves * (A.n - 1) * 4
        ax.plot(A.n, unpruned / 1024, "s-", label="Unpruned index")
        ax.plot(A.n, A.ads_bytes / 1024, "^-", label="ADS (commitments)")
        if len(I): ax.plot(I.n, I.storage_b / 1024, "v-", label="I-tree total")
        style(ax, "n", "KiB", logy=True)
        save(fig, figs, f"c_storage_{d}.png")

        # --- Query: VO size ---
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot(A.n, A.topk_vo_b / 1024, "o-", label="top-$k$ VO")
        ax.plot(A.n, A.range_vo_b / 1024, "s-", label="range VO")
        style(ax, "n", "KiB")
        save(fig, figs, f"q_vo_{d}.png")

        # --- Query: latency (server + client verify) ---
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot(A.n, A.topk_server_ms, "o-", label="top-$k$ server")
        ax.plot(A.n, A.topk_verify_ms, "o--", label="top-$k$ client verify")
        ax.plot(A.n, A.range_server_ms, "s-", label="range server")
        ax.plot(A.n, A.range_verify_ms, "s--", label="range client verify")
        style(ax, "n", "milliseconds")
        save(fig, figs, f"q_latency_{d}.png")

        # --- Verifiable construction: cloud gen vs owner verify vs local build ---
        if len(C):
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.plot(C.n, C.gen_s, "o-", label="Cloud: generate certificates")
            ax.plot(C.n, C.verify_s, "s-", label="Owner: verify certificates")
            ax.plot(C.n, C.construct_s, "^-", label="Local construction")
            style(ax, "n", "seconds", logy=True)
            save(fig, figs, f"v_time_{d}.png")

            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.plot(C.n, C.cert_bytes / 1024, "o-", label="Certificate volume")
            ax.plot(C.n, (C.n_part + C.n_nonpart), "s--", label="Certificate count")
            style(ax, "n", "KiB / count", logy=True)
            save(fig, figs, f"v_size_{d}.png")

    # Summary table for the paper text.
    with open(figs / "summary.txt", "w") as f:
        f.write(aof.to_string(index=False) + "\n\n")
        if len(cert): f.write(cert.to_string(index=False) + "\n\n")
        if len(itree): f.write(itree.to_string(index=False) + "\n")
    print("wrote summary.txt")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".",
         sys.argv[2] if len(sys.argv) > 2 else "figs_out")

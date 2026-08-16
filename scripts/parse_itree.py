"""Parse I-tree baseline logs (UTF-16 aware) and print speedups vs AofBench."""
import re, pathlib, sys
import pandas as pd

build = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

def readtxt(p: pathlib.Path) -> str:
    raw = p.read_bytes()
    # BOM-based detection: PowerShell *> redirects write UTF-16LE with BOM;
    # the C++ programs write plain ASCII/UTF-8.  Decoding ASCII as UTF-16
    # "succeeds" with garbage, so never guess UTF-16 without the BOM.
    if raw[:2] == b"\xff\xfe":
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")

rows = []
for p in (build / "runs").glob("itree_*.txt"):
    m = re.match(r"itree_(\d+)_(\d+)", p.stem)
    t = readtxt(p)
    st = re.search(r"Storage:\s+(\d+)", t)
    fc = re.search(r"Feasibility Chekcing Time:\s+([\d.]+)", t)
    row = {"n": int(m.group(1)), "d": int(m.group(2)),
           "storage_b": int(st.group(1)) if st else None,
           "fc_s": float(fc.group(1)) if fc else None, "time_s": None}
    lp = build / "results" / f"itree_result_{row['n']}_{row['d']}_1"
    if lp.exists():
        tm = re.search(r"Time:\s+([\d.]+)", readtxt(lp))
        row["time_s"] = float(tm.group(1)) if tm else None
    rows.append(row)

df = pd.DataFrame(rows).sort_values(["d", "n"])
df.to_csv(build / "itree_results.csv", index=False)

AOF_COLS = ["n","d","seed","construct_s","nodes","leaves","internals","maxdepth",
            "ads_bytes","idx_pruned","topk_server_ms","topk_verify_ms","topk_vo_b",
            "topk_gap","range_server_ms","range_verify_ms","range_vo_b",
            "topk_acc","range_acc","tamper_rej"]
a = pd.read_csv(build / "aof_bench_results.csv", names=AOF_COLS).drop_duplicates(["n","d"], keep="last")

for _, r in df.iterrows():
    m = a[(a.n == r.n) & (a.d == r.d)]
    if len(m) and r.time_s:
        mm = m.iloc[0]
        aof_total = (mm.ads_bytes + mm.idx_pruned) / 1024
        print(f"n={r.n:.0f} d={r.d:.0f}: itree={r.time_s:.3f}s (fc={r.fc_s})"
              f" aof={mm.construct_s:.4f}s speedup={r.time_s/mm.construct_s:.0f}x"
              f" | itree_store={r.storage_b/1024:.0f}KiB aof_total={aof_total:.0f}KiB")

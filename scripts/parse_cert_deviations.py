"""Extract certificate-protocol deviation counts from CertBench run logs.

Usage:  python parse_cert_deviations.py <build_dir>

Reads  <build_dir>/runs/cert_*.txt   (BOM-aware: Windows PowerShell writes
                                      UTF-16 via *> redirection, pwsh UTF-8)
Writes <build_dir>/cert_deviations.csv with, per configuration, the number of
partition and non-partition certificates, the number of constructor
deviations the certificate replay detected, and the deviation rate as a
percentage of all certified events (partition + non-partition + deviations)
-- the quantity quoted in the paper's verifiable-construction section.
"""
import re, sys, pathlib

build = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

def readtxt(p: pathlib.Path) -> str:
    raw = p.read_bytes()
    if raw[:2] == b"\xff\xfe":
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")

rows = []
for p in sorted((build / "runs").glob("cert_*.txt")):
    m = re.match(r"cert_(\d+)_(\d+)", p.stem)
    s = re.search(r"CERT_PARTITION=(\d+) CERT_NONPARTITION=(\d+) DEVIATIONS_DETECTED=(\d+)",
                  readtxt(p))
    if not (m and s):
        continue
    part, nonpart, dev = map(int, s.groups())
    total = part + nonpart + dev
    rows.append((int(m.group(1)), int(m.group(2)), part, nonpart, dev,
                 100.0 * dev / total if total else 0.0))

rows.sort(key=lambda r: (r[1], r[0]))
out = build / "cert_deviations.csv"
with open(out, "w") as f:
    f.write("n,d,n_partition,n_nonpartition,deviations,rate_pct\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f}\n")

print(f"{'n':>5} {'d':>2} {'partition':>10} {'nonpartition':>12} {'deviations':>10} {'rate':>9}")
for r in rows:
    print(f"{r[0]:>5} {r[1]:>2} {r[2]:>10} {r[3]:>12} {r[4]:>10} {r[5]:>8.4f}%")
print("wrote", out)

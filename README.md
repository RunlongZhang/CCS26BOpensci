# AOF-tree: Authenticated Function Queries — Artifact

This artifact contains the complete implementation and experiment pipeline for
the paper *"Authentication-Oriented Function Tree: Efficient and Verifiable
Function Query Processing"*: the AOF-tree builder, its commitment scheme and
signed ADS, authenticated top-k and range query processing with an independent
client-side verifier, the verifiable-construction certificate protocol
(generation and owner-side verification), and the Simplex-based I-tree
baseline used for comparison.

Every experiment in the paper is reproducible end-to-end with two scripts:
a ~3-minute smoke test and a full reproduction run. All datasets are
synthetic and generated from fixed, recorded seeds.

## Layout

```
CMakeLists.txt        build definition (no machine-specific paths)
vcpkg.json            dependency manifest (OpenSSL, HiGHS, Eigen)
LICENSE, .gitignore
src/
  AofAds.h            AOF-tree builder, commitments, signer, server, client verifier
  AofBenchMain.cpp    construction/ADS/query benchmark driver
  CertBenchMain.cpp   verifiable-construction certificate benchmark
  ITreeSimplex*.{h,cpp}, SimplexSolver.h, MerkleTree.h   I-tree baseline
  PolytopeOps.h, PolytopeStructs.h                       geometric kernel
  FunctionPairGenerator.{h,cpp}, FunctionPairMain.cpp    seeded dataset generator
  CompactIO.h         dataset file format
scripts/
  build.ps1                 configure + build (CMake + vcpkg manifest mode)
  quick_test.ps1            ~3-minute smoke test of every program + sample figures
  reproduce_all.ps1         full experiment grids (~2-3 h) + all paper figures
  parse_itree.py            parses baseline logs into itree_results.csv
  parse_cert_deviations.py  parses CertBench logs into cert_deviations.csv
  plot_results.py           draws every result figure + summary.txt
```

The scripts create `out/` (build output) and `figs/` (figures) plus datasets,
logs, and CSVs inside the build directory; all of these are generated and
git-ignored.

## Requirements

- **Windows 11 x64.** (The baseline `ITreeSimplexMain.cpp` uses Windows
  headers; all measurements in the paper were taken on Windows.)
- **Visual Studio 2022 or later** with the *Desktop development with C++*
  workload (provides MSVC, CMake ≥ 3.28, and Ninja).
- **vcpkg** — any checkout; dependencies are declared in `vcpkg.json`
  (manifest mode) and are downloaded/built automatically at configure time.
  There are **no paths to edit** in any file.
- **Python 3.10+** with `pandas` and `matplotlib` (`pip install pandas matplotlib`)
  for parsing and plotting.

Reference machine for the paper's timings: Intel Core i9-14900K, 64 GB RAM.
Absolute timings will differ on other hardware; the trends and the
verification outcomes (acceptance, tamper rejection, certificate checks)
are hardware-independent.

## Build

From a **Developer PowerShell for VS** prompt:

```powershell
git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat            # skip both steps if you already have vcpkg
scripts\build.ps1 -VcpkgRoot <path-to-vcpkg>
```

(`-VcpkgRoot` can be omitted if the `VCPKG_ROOT` environment variable is set.)
The first configure installs OpenSSL, HiGHS, and Eigen through vcpkg, which
can take several minutes; subsequent builds are fast.

## Quick validation (~3 minutes)

```powershell
scripts\quick_test.ps1
```

Runs the generator, `AofBench`, `CertBench`, and the `ITreeSimplex` baseline
on small instances, then parses and plots. The two benchmarks **self-verify**
and gate their exit codes on it: `AofBench` exits non-zero unless all 200
authenticated queries verify and every injected tampering (dropped answer,
flipped bucket bit, bumped rank) is rejected; `CertBench` exits non-zero
unless every construction certificate verifies and every falsified
certificate is caught. (Both also exit non-zero on missing input files or bad
usage; the failing run's log in `runs\` pinpoints the cause. `ITreeSimplex`
is a performance baseline without an exit-code gate.) The script prints
`QUICK TEST PASSED` on success.

## Full reproduction (~2–3 hours)

```powershell
scripts\reproduce_all.ps1
```

Runs the paper's full grids:

| Experiment | Grid | Paper section |
|---|---|---|
| `AofBench` (construction, ADS, storage, top-k/range queries, VO sizes, client verification) | d=2: n=10..500; d=3: n=10..250; d=4: n=10..60 | §5.3, §5.4 |
| `CertBench` (certificate generation, owner verification, sparsity, deviation detection) | d=2: n=10..250; d=3: n=10..100; d=4: n=10..30 | §5.5 |
| `ITreeSimplex` baseline (construction time, feasibility-check time, storage) | the scales the baseline reaches: d=2: n≤50; d=3: n≤25; d=4: n≤16 | §5.4 |

and then produces every result figure in `figs/`:

| Files | Content |
|---|---|
| `q_vo_{2,3,4}.png`, `q_latency_{2,3,4}.png` | VO sizes; server latency and client verification time |
| `c_construct_{2,3,4}.png`, `c_storage_{2,3,4}.png` | construction time vs. the baseline; storage decomposition vs. the baseline |
| `v_time_{2,3,4}.png`, `v_size_{2,3,4}.png` | verifiable construction: generation/verification time; certificate volume |
| `summary.txt` | every raw measurement row, for cross-checking numbers quoted in the paper |

Raw measurements land in the build directory: `aof_bench_results.csv` and
`cert_bench_results.csv` (headerless; column names are listed at the top of
`scripts/plot_results.py`), `itree_results.csv` and `cert_deviations.csv`
(with header rows). `cert_deviations.csv` reports, per configuration, the
constructor deviations detected by the certificate replay and their rate
among all certified events — the quantity discussed in the paper's
verifiable-construction section.

## Determinism and numerical notes

- Dataset seeds are fixed by the scripts (`1000 + 10n + d` per configuration);
  the generator rejects duplicate functions so the paper's pairwise-distinct
  assumption holds. Query workloads use 200 queries with seed 42.
- Exact results vary with hardware and OS scheduling only in *timings*; VO
  sizes, tree shapes, certificate counts, and all verification outcomes are
  deterministic given the seeds.
- Geometric predicates are evaluated in floating point. All certificate data
  is rational, so a production owner can verify certificates in exact
  arithmetic; this implementation's owner check uses a scale-aware tolerance
  as a stand-in. The certificate protocol's *detection* behavior on
  floating-point deviations of the constructor is itself an experiment
  reported in the paper (§5.5) and is reproduced by `CertBench`.
- Windows PowerShell 5.1 writes UTF-16 log files via `*>` redirection, while
  PowerShell 7 writes UTF-8; the parsing scripts detect the encoding by BOM,
  so both parse correctly.

## Program reference

```
FunctionGenerator.exe <n> <d> [seed]        # writes <n>_functions_<d>d.bin + <n>_pairwise_<d>d.bin
                                            # seed omitted or 0 = nondeterministic;
                                            # the scripts always pass 1000 + 10n + d
AofBench.exe          <n> <d> [queries=200] [seed=42]
CertBench.exe         <n> <d>
ITreeSimplex.exe      <n> <d> <ads: 0/1>    # the paper's baseline runs use ads=1
```

All programs read datasets from and write results to the current working
directory (the scripts run them from the build directory).

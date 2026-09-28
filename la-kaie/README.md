# LA-KAIE: Landscape-Aware Knowledge-Guided Adaptive Information Exchange

Research code for an HBA–MPA hybrid metaheuristic with a landscape-conditioned
information-exchange controller, evaluated against a faithful reproduction of
MPHBS (Flores & Olivares, *Knowledge-Based Systems* 352 (2026) 117070) on the
CEC2022 suite (D = 20) and eight FIR filter-design cases (D = 31).

> **Status:** see `results/FINAL_EXPERIMENT_REPORT.md` (or, before the main
> campaign, `results/PILOT_REPORT.md`). The novelty of the method is under
> review — read `literature/NOVELTY_ASSESSMENT.md` before making any claim.
> The title *"Landscape-Aware Knowledge-Guided Information Exchange for
> Adaptive Hybrid Metaheuristic Optimization"* is provisional.

This directory is self-contained; it lives inside an unrelated repository.
All commands are run from this directory (`la-kaie/`).

## Layout

| path | content |
|---|---|
| `reference/` | MPHBS paper (PDF) and `MPHBS_NOTES.md` (full reproduction notes, missing details, paper-vs-code discrepancies) |
| `third_party/cec2022_official/` | official CEC2022 C++ code + data (P. N. Suganthan, 2022-SO-BO), minimal POSIX build patch (`build_patch.diff`) |
| `third_party/mphbs_reference/` | authors' MATLAB reference code (MIT) and their published per-run results |
| `benchmarks/fir/fir_cases.yaml` | the eight FIR cases exactly as in the paper |
| `lakaie/` | Python package: benchmarks, HBA/MPA/MPHB/MPHBS ports, LA-KAIE, statistics |
| `configs/` | every parameter (`cec2022.yaml`, `fir.yaml`, `controller.yaml`, `ablation.yaml`, `sensitivity.yaml`, `reproducibility.yaml`) |
| `experiments/` | `LAKAIE_SPEC.md` (exact equations), `baseline_validation.md` |
| `literature/` | `novelty_matrix.csv`, `NOVELTY_ASSESSMENT.md` |
| `results/raw/` | one JSON + NPZ per run (never overwritten) |
| `results/{statistics,tables,figures}/` | generated outputs |
| `logs/` | timestamped project log, development changes, campaign logs |

## 1. Environment

```bash
python3 -m pip install -r requirements.txt       # or: conda env create -f environment.yml
# g++ is required: the official CEC2022 C++ code is compiled on first use
python3 -c "from lakaie.benchmarks.cec2022 import build_library; build_library(force=True)"
python3 scripts/capture_environment.py           # -> results/ENVIRONMENT.json
```

Tested with Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1 on Linux x86_64 (4 CPUs).
BLAS is pinned to one thread per process (`lakaie/__init__.py`).

## 2. Benchmark validation

```bash
python3 scripts/validate_cec2022.py   # F(o)=F*, official C vs official Python, data hashes, guards
python3 scripts/validate_fir.py       # cases vs paper, FFT vs direct DTFT, analytic values
```

## 3. Static checks and unit tests (phases 0-1)

```bash
python3 -m compileall -q lakaie scripts tests && python3 -m pyflakes lakaie scripts tests
python3 -m pytest -q
```

## 4. Baseline validation (MPHBS reproduction)

```bash
python3 scripts/run_experiment.py --benchmark CEC2022 --campaign baseline_validation --methods HBA MPA MPHB MPHBS --runs 10
python3 scripts/run_experiment.py --benchmark FIR     --campaign baseline_validation --methods HBA MPA MPHB MPHBS --runs 10
python3 scripts/compare_baseline_validation.py        # -> experiments/baseline_validation_results.md
```

## 5. Smoke tests (phases 2-3)

```bash
ALL="HBA MPA MPHB MPHBS LA-KAIE-Full LA-KAIE-NoLandscape LA-KAIE-NoInteraction LA-KAIE-NoKnowledgeReward LA-KAIE-DimensionOnly LA-KAIE-RandomExchange LA-KAIE-FixedPolicy LA-KAIE-NoAdaptiveController"
python3 scripts/run_experiment.py --benchmark CEC2022 --campaign smoke --methods $ALL --instances 1 --runs 5 --max-fes 20000
python3 scripts/run_experiment.py --benchmark FIR     --campaign smoke --methods $ALL LA-KAIE-FIRPrior --instances 1 --runs 5 --max-fes 15000
python3 scripts/validate_experiment.py --campaign smoke --runs 5 --max-fes 20000 --instances 1   # (CEC part; FIR budget differs)
```

## 6. Pilot experiments (phases 4-5)

```bash
python3 scripts/run_experiment.py --benchmark CEC2022 --campaign pilot --methods $ALL --runs 5
python3 scripts/run_experiment.py --benchmark FIR     --campaign pilot --methods $ALL LA-KAIE-FIRPrior --runs 5
python3 scripts/validate_experiment.py --campaign pilot --runs 5
```

## 7. Sensitivity analysis (pre-registered protocol, `configs/sensitivity.yaml`)

```bash
python3 scripts/run_sensitivity.py        # CEC2022 D=10 tuning set, one-factor-at-a-time
python3 scripts/select_parameters.py      # applies the pre-registered selection rule
```

## 8. Full experiments (30 runs, common seeds, resumable)

```bash
python3 scripts/run_experiment.py --benchmark CEC2022 --campaign main --methods $ALL --runs 30
python3 scripts/run_experiment.py --benchmark FIR     --campaign main --methods $ALL LA-KAIE-FIRPrior --runs 30
python3 scripts/validate_experiment.py --campaign main --runs 30
```

Interrupted campaigns resume by re-running the same command: finished runs
(JSON present) are skipped and never overwritten.

## 9. Statistics, tables, figures

```bash
for B in CEC2022 FIR; do
  python3 scripts/analyze.py      --campaign main --benchmark $B   # -> results/statistics, results/tables
  python3 scripts/make_figures.py --campaign main --benchmark $B   # -> results/figures
done
```

## Seeds

`seed = master_seed[campaign] + benchmark_offset + 1000*instance + run`
(`configs/reproducibility.yaml`); the same seed is used by every method for a
given (instance, run), so all comparisons are paired. The main campaign uses
the MPHBS authors' master seed 20260405; MATLAB random streams are not
reproducible in NumPy, so MPHBS is reproduced distributionally.

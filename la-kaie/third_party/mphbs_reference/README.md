# MPHBS: SARSA-Guided HBA-MPA Information Exchange

This repository contains the public MATLAB implementation and reproducibility
package for MPHBS, a hybrid Honey Badger Algorithm (HBA) and Marine Predators
Algorithm (MPA) metaheuristic with SARSA-style information exchange.

The package includes the source code, paired-seed experimental runners,
saved CEC2022 and FIR benchmark outputs, statistical analyses, convergence
figures, and validation scripts used for the revised manuscript.

## Main MATLAB Entry Point

The final MPHBS implementation is:

```text
code/MPHBS_main.m
```

Function signature:

```matlab
[bestValue, bestPosition, convergenceCurve] = MPHBS_main( ...
    N, T, LB, UB, Dim, F_obj, K, Ni, rho_sub, w_best, gamma, ...
    fad_before_p2, fad_after_p2);
```

Final manuscript configurations:

```text
CEC2022 B-C1:
K = 7, Ni = 3, rho_sub = 0.25, w_best = 0.050,
gamma = 0.30, fad_before_p2 = true, fad_after_p2 = false

FIR A-C1:
K = 3, Ni = 9, rho_sub = 0.25, w_best = 0.050,
gamma = 0.30, fad_before_p2 = false, fad_after_p2 = true
```

Related implementations:

```text
code/MPHBS_random_mirror.m   matched RANDOM mirror of MPHBS
code/MPHB_baseline.m         HBA-MPA backbone without the mediator
```

## Reproduce the Manuscript Experiments

Start MATLAB in the `code/` directory:

```matlab
cd('/path/to/MPHBS-SARSA-information-exchange/code')
```

Run all six paired-seed campaigns:

```matlab
run_all_manuscript_experiments
```

Validate saved outputs:

```matlab
validate_campaign_outputs
validate_paired_seed_results
audit_final_fe_budgets
```

CSV-level validation can also be run from a terminal:

```bash
python3 code/validate_result_files.py
```

## Package Contents

```text
code/                 MATLAB algorithms, runners, validation scripts,
                      CEC2022 MEX, and official input data
results/              Saved outputs from the six paired-seed campaigns
manuscript_figures/   Final 400-dpi PNG and editable MATLAB FIG files
documentation/        Validation reports and protocol notes
```

The full reproducibility protocol is documented in
`README_REPRODUCIBILITY.md`. A short usage guide is provided in
`QUICK_START.md`.

## Reproducibility Status

- MATLAB release: R2024b
- Runs: 30 paired runs per benchmark instance
- Population size: 30
- CEC2022: 12 functions, D = 20, 200,000 FEs
- FIR: 8 filter-design cases, D = 31, 150,000 FEs
- Final CEC2022 configuration: B-C1
- Final FIR configuration: A-C1
- Statistical analyses: paired Wilcoxon tests, Holm correction,
  Friedman ranks, Kendall's W, and W/T/L summaries

The complete package was re-executed from a clean results directory on
2026-07-10. The regenerated outputs match the saved public outputs, excluding
wall-clock time by design.

## Citation

If you use this package, please cite it as:

```bibtex
@software{FloresOlivares2026MPHBS,
  author    = {Flores, Emilio and Olivares, Rodrigo},
  title     = {{efloresaraya/MPHBS-SARSA-information-exchange: MPHBS v1.0.0 - Reproducibility Package for SARSA-Guided HBA-MPA Information Exchange}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.21326361},
  url       = {https://doi.org/10.5281/zenodo.21326361},
  note      = {Computer software}
}
```

## License

This repository is released under the MIT License.

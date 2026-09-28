# LA-KAIE project log (UTC timestamps)

- 2026-09-28T09:41:10Z Project skeleton created under la-kaie/ (the enclosing repository is an unrelated ADHD-EEG project).
- 2026-09-28T09:41:10Z Read MPHBS.pdf in full (29 pages, text extracted with pypdfium2).
- 2026-09-28T09:41:10Z Zenodo (10.5281/zenodo.21326361) blocked by network policy (HTTP 403 at proxy); authors' package obtained from GitHub efloresaraya/MPHBS-SARSA-information-exchange (MIT).
- 2026-09-28T09:41:10Z Official CEC2022 code cloned from github.com/P-N-Suganthan/2022-SO-BO; C version compiled with minimal POSIX patch (build_patch.diff). C and MATLAB-MEX function bodies identical (only a comment differs).
- 2026-09-28T09:41:10Z Environment: Python 3.11.15, no MATLAB/Octave; 4 CPUs, 15 GB RAM.
- 2026-09-28T09:41:10Z scripts/validate_cec2022.py: OVERALL PASS (F(o)=F*, C vs official Python <=1.7e-16 rel.). Note: official Python port has two defects (shift-file indexing, always-true dimension message); cross-check used an in-memory one-line fix.
- 2026-09-28T09:41:10Z scripts/validate_fir.py: OVERALL PASS.
- 2026-09-28T09:41:10Z reference/MPHBS_NOTES.md written: 3 paper-vs-code discrepancies, 7 missing-from-paper items, all resolved by the authors' code; no STOP condition.

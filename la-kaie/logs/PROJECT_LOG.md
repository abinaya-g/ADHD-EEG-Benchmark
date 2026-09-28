# LA-KAIE project log (UTC timestamps)

- 2026-09-28T09:41:10Z Project skeleton created under la-kaie/ (the enclosing repository is an unrelated ADHD-EEG project).
- 2026-09-28T09:41:10Z Read MPHBS.pdf in full (29 pages, text extracted with pypdfium2).
- 2026-09-28T09:41:10Z Zenodo (10.5281/zenodo.21326361) blocked by network policy (HTTP 403 at proxy); authors' package obtained from GitHub efloresaraya/MPHBS-SARSA-information-exchange (MIT).
- 2026-09-28T09:41:10Z Official CEC2022 code cloned from github.com/P-N-Suganthan/2022-SO-BO; C version compiled with minimal POSIX patch (build_patch.diff). C and MATLAB-MEX function bodies identical (only a comment differs).
- 2026-09-28T09:41:10Z Environment: Python 3.11.15, no MATLAB/Octave; 4 CPUs, 15 GB RAM.
- 2026-09-28T09:41:10Z scripts/validate_cec2022.py: OVERALL PASS (F(o)=F*, C vs official Python <=1.7e-16 rel.). Note: official Python port has two defects (shift-file indexing, always-true dimension message); cross-check used an in-memory one-line fix.
- 2026-09-28T09:41:10Z scripts/validate_fir.py: OVERALL PASS.
- 2026-09-28T09:41:10Z reference/MPHBS_NOTES.md written: 3 paper-vs-code discrepancies, 7 missing-from-paper items, all resolved by the authors' code; no STOP condition.
- 2026-09-28T09:53:37Z Baseline-validation campaign started (10 runs/instance, seeds master 31000000). Development changes DC1-DC3 recorded in logs/DEVELOPMENT_CHANGES.md.
- 2026-09-28T10:13:41Z Baseline validation complete (CEC 480 + FIR 320 runs, validator PASS, 0/80 distributional differences after Holm). experiments/baseline_validation.md written. Starting phase 2-3 smoke tests.
- 2026-09-28T10:14:41Z Smoke tests (phases 2-3): CEC 60/60, FIR 65/65 ok; validator PASS except expected FE-granularity utilisation of MPHBS at 15k budget (14893/15000). Pilots started.
- 2026-09-28T11:20:03Z Pilots complete (CEC 720, FIR 520 runs; validator PASS). results/PILOT_REPORT.md written. Full experiment NOT started: STOP conditions (novelty overlap; main mechanism inactive, no advantage over MPHBS). Awaiting user decision.
- 2026-09-28T11:30:51Z User paused the project (no full experiment, no manuscript). RESEARCH_DECISION_REPORT.md written with diagnostics in logs/decision_diagnostics/. Awaiting instruction.
- 2026-09-28T12:52:48Z Novelty investigation: F1/F2/F3 rejected as core contributions (DIRECT overlaps: GAwLL/RVIntX/LEL; ACoS/MCDE/DE-eig; AMALGAM/MOS/CBCC). Candidate N1 (evidence-based structural-hypothesis selection) documented conditionally in NEXT_METHOD_DECISION.md; full texts blocked by network policy (only GitHub reachable; GAwLL code verified). Stopped awaiting instruction.
- 2026-09-28T13:00:54Z N1 investigation: N1_NOVELTY_ASSESSMENT.md (classification B; 7 items to verify with full texts) and N1_CONCEPT.md (concept + equations; no code). Stopped awaiting instruction.
- 2026-09-28T13:50:54Z Final N1 novelty verification: classification B (unresolved overlaps U1-U7; closest: EMT ensemble KT framework AIE+MAS, MFEA-II verified in code). No implementation. Files: N1_FINAL_NOVELTY_DECISION.md, N1_NOVELTY_EVIDENCE_LOG.md.

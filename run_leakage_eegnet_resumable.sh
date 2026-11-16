#!/bin/bash
# resumable: one process per EEGNet outer fold; skips folds whose checkpoint exists
cd /home/user/ADHD-EEG-Benchmark
export ADHD_EEG_DATA_ROOT=/tmp/claude-0/-home-user-ADHD-EEG-Benchmark/c64fa27c-2c96-5d01-9671-b1714f8540d1/scratchpad/kdata/raw TF_CPP_MIN_LOG_LEVEL=3
for f in 0 1 2 3 4; do
  [ -f results_revision/leakage_epoch_level_predictions_EEGNet_rep0_folds${f}.csv ] || python3 run_revision_deep.py --task leakage --repeats 1 --archs EEGNet --folds $f >> results_revision/leakage_EEGNet_fold${f}.log 2>&1
done
n=$(ls results_revision/leakage_epoch_level_predictions_EEGNet_rep0_folds*.csv 2>/dev/null | wc -l)
[ "$n" -eq 5 ] && echo ALLDONE >> results_revision/leakage_status.log

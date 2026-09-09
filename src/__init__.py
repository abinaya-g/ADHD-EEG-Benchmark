"""
Package init. Also centralizes log/warning suppression so real output
(metrics, sanity-check reports) isn't buried under harmless version-
compatibility chatter from TensorFlow/absl/sklearn. This changes ONLY
what gets printed to the console -- no seeds, no model logic, no metric
computation is touched here, and nothing here silences real errors
(exceptions still propagate normally; only informational/benign log
lines and deprecation-style FutureWarnings are suppressed).
"""
import os
import warnings

# Must be set before `import tensorflow` anywhere in the process for it to
# take effect -- TF's C++ logging level is read once at native library
# load time. This hides the repeated
# "NodeDef mentions attribute use_unbounded_threadpool ..." lines, which
# TensorFlow itself labels as expected/benign when the graph-generating
# binary is newer than the runtime binary -- not an error.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

# sklearn emits FutureWarnings for a couple of parameters used here
# (SVC(probability=True), LogisticRegression(penalty=...)) that are
# scheduled for removal in a future sklearn release but are still fully
# functional in the versions this code targets.
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    # Silences "WARNING:tensorflow: N out of the last M calls ... triggered
    # tf.function retracing." This is expected here by design: a fresh
    # model is built and compiled for every inner-CV fold and every final
    # outer-fold fit (see AUDIT_REPORT.md / evaluation.py), so TensorFlow
    # retraces its graph each time -- it is overhead, not a correctness
    # issue, and is already documented as a known cost in FINAL_REPORT.md.
except ImportError:
    pass  # modules that don't need TensorFlow (e.g. src.statistics alone) still import fine

# One line -- "WARNING: All log messages before absl::InitializeLog() is
# called are written to STDERR" -- is emitted by absl's C++ logging
# framework before Python ever gets control, and cannot be reliably
# suppressed from here. It is harmless and appears at most once per
# process.

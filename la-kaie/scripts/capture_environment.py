"""Record the software/hardware environment to results/ENVIRONMENT.json."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, cwd=ROOT).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


cpu = sh("lscpu | grep -E 'Model name|^CPU\\(s\\)' | tr -s ' '")
env = {
    "git_commit": sh("git rev-parse HEAD"), "git_dirty": sh("git status --porcelain | wc -l"),
    "python": sys.version, "numpy": numpy.__version__, "scipy": scipy.__version__,
    "pandas": pandas.__version__, "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__,
    "os": platform.platform(), "machine": platform.machine(), "cpu": cpu, "n_cpu": os.cpu_count(),
    "memory": sh("free -g | head -2 | tail -1"), "gpu": "none used (CPU only)",
    "gxx": sh("g++ --version | head -1"),
    "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "set to 1 by lakaie/__init__.py"),
    "reproducibility_config": yaml.safe_load((ROOT / "configs" / "reproducibility.yaml").read_text()),
}
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "ENVIRONMENT.json").write_text(json.dumps(env, indent=2))
print(json.dumps(env, indent=2))

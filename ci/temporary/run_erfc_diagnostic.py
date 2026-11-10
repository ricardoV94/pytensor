"""Compare PR 2643 base/head with fresh source and compilation directories."""

# ruff: noqa: T201

import io
import json
import os
import platform
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import llvmlite
import numba
import numpy as np
import scipy


REVISIONS = {
    "base": "c57580646e23e55973f5ec3c0a6cc61a8ca96157",
    "head": "9c050e434e11196b61f82676826726dcc96b0a8c",
}


def main():
    output = Path("erfc-diagnostic-results").resolve()
    output.mkdir(exist_ok=True)
    metadata = {
        "python": sys.version,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "numba": numba.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "llvmlite": llvmlite.__version__,
        "revisions": REVISIONS,
    }
    if sys.platform == "darwin":
        metadata["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(allowed)})
        metadata["affinity"] = sorted(os.sched_getaffinity(0))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    script = Path(__file__).with_name("erfc_diagnostic.py").resolve()
    runs = []
    with tempfile.TemporaryDirectory(prefix="erfc-diagnostic-") as directory:
        root = Path(directory)
        for name, revision in REVISIONS.items():
            source = root / name
            source.mkdir()
            archive = subprocess.check_output(["git", "archive", revision, "pytensor"])
            with tarfile.open(fileobj=io.BytesIO(archive)) as files:
                files.extractall(source, filter="data")
        for index, name in enumerate(("base", "head", "head", "base")):
            source = root / name
            run_output = output / f"run-{index}-{name}"
            env = dict(
                os.environ, PYTHONPATH=str(source), DIAGNOSTIC_OUTPUT=str(run_output)
            )
            env["DUMP_IR"] = "1" if index < 2 else "0"
            env["PYTENSOR_FLAGS"] = (
                f"base_compiledir={root / ('cache-' + name)},numba__fastmath=True"
            )
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=source,
                env=env,
                text=True,
                capture_output=True,
            )
            run_output.mkdir(exist_ok=True)
            (run_output / "stdout.txt").write_text(result.stdout)
            (run_output / "stderr.txt").write_text(result.stderr)
            if result.returncode:
                print(result.stderr)
                result.check_returncode()
            row = json.loads((run_output / "results.json").read_text())
            assert Path(row["pytensor"]).is_relative_to(source), row["pytensor"]
            runs.append({"arm": name, "round": index, **row})
            print(name, json.dumps(row["seconds"]), flush=True)
    summary = []
    for case in runs[0]["seconds"]:
        medians = {
            name: statistics.median(
                run["seconds"][case] for run in runs if run["arm"] == name
            )
            for name in REVISIONS
        }
        row = {
            "case": case,
            "base_ms": medians["base"] * 1000,
            "head_ms": medians["head"] * 1000,
            "speedup": medians["base"] / medians["head"],
        }
        summary.append(row)
        print(json.dumps(row), flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "runs.json").write_text(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()

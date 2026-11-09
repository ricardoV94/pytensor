"""Run the unchanged PR review benchmark in isolated source snapshots."""

# ruff: noqa: T201

import argparse
import contextlib
import importlib.metadata
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
import psutil
import scipy


REVISIONS = {
    "base": "c57580646e23e55973f5ec3c0a6cc61a8ca96157",
    "head": "9c050e434e11196b61f82676826726dcc96b0a8c",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    machine = platform.machine().lower()
    architecture = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    expected = os.environ.get("EXPECTED_ARCHITECTURE", architecture)
    assert architecture == expected, (architecture, expected)

    process = psutil.Process()
    affinity = None
    if hasattr(process, "cpu_affinity"):
        allowed = process.cpu_affinity()
        process.cpu_affinity([allowed[0]])
        affinity = process.cpu_affinity()

    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "libc": platform.libc_ver(),
        "affinity": affinity,
        "numba": numba.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "llvmlite": llvmlite.__version__,
        "revisions": REVISIONS,
    }
    if sys.platform.startswith("linux"):
        metadata["cpu_info"] = subprocess.check_output(["lscpu"], text=True)
        metadata["processor"] = next(
            line.split(":", 1)[1].strip()
            for line in metadata["cpu_info"].splitlines()
            if line.startswith("Model name:")
        )
    elif sys.platform == "darwin":
        metadata["cpu_info"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    if sys.platform == "darwin" or sys.platform == "win32":
        metadata["processor"] = metadata["cpu_info"]
    else:
        metadata["cpu_info"] = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ],
            text=True,
        ).strip()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(
            {key: value for key, value in metadata.items() if key != "cpu_info"}
        ),
        flush=True,
    )
    build_config = io.StringIO()
    with contextlib.redirect_stdout(build_config):
        np.show_config()
        scipy.show_config()
    (output / "build-config.txt").write_text(build_config.getvalue())
    (output / "installed-packages.txt").write_text(
        "\n".join(
            sorted(
                f"{dist.name}=={dist.version}"
                for dist in importlib.metadata.distributions()
            )
        )
        + "\n"
    )

    runs = []
    reviewer = Path(__file__).with_name("reviewer2643.py").resolve()
    with tempfile.TemporaryDirectory(prefix="pr2643-") as directory:
        root = Path(directory)
        for name, revision in REVISIONS.items():
            source = root / name
            source.mkdir()
            archive = subprocess.check_output(["git", "archive", revision, "pytensor"])
            with tarfile.open(fileobj=io.BytesIO(archive)) as files:
                files.extractall(source, filter="data")

        # Three pairs, reversing the middle pair to reduce order bias.
        for index, name in enumerate(("base", "head", "head", "base", "base", "head")):
            source = root / name
            env = dict(os.environ, PYTHONPATH=str(source))
            env["PYTENSOR_FLAGS"] = (
                f"base_compiledir={root / ('cache-' + name)},numba__fastmath=True"
            )
            result = subprocess.run(
                [sys.executable, str(reviewer), name],
                cwd=source,
                env=env,
                text=True,
                capture_output=True,
            )
            (output / f"run-{index}-{name}.stdout.txt").write_text(result.stdout)
            (output / f"run-{index}-{name}.stderr.txt").write_text(result.stderr)
            if result.returncode:
                print(result.stderr, file=sys.stderr)
                result.check_returncode()
            row = json.loads(result.stdout.strip().splitlines()[-1])
            assert Path(row["pytensor"]).is_relative_to(source), row
            row["round"] = index
            runs.append(row)
            (output / "runs.json").write_text(json.dumps(runs, indent=2))
            print(json.dumps(row), flush=True)

    summary = []
    for case in runs[0]["seconds"]:
        samples = {
            name: [row["seconds"][case] for row in runs if row["arm"] == name]
            for name in REVISIONS
        }
        medians = {name: statistics.median(times) for name, times in samples.items()}
        pairs = [
            {row["arm"]: row["seconds"][case] for row in runs[i : i + 2]}
            for i in range(0, len(runs), 2)
        ]
        summary.append(
            {
                "case": case,
                "base_ms": medians["base"] * 1000,
                "head_ms": medians["head"] * 1000,
                "speedup": medians["base"] / medians["head"],
                "pair_speedups": [pair["base"] / pair["head"] for pair in pairs],
                "round_ms": {
                    name: [time * 1000 for time in times]
                    for name, times in samples.items()
                },
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        f"# PR 2643: {platform.system()} {platform.machine()}",
        "",
        f"CPU: {metadata['processor']}",
        "",
        f"Numba {numba.__version__}, SciPy {scipy.__version__}, "
        f"NumPy {np.__version__}; Python {platform.python_version()}.",
        "Median of three process runs per commit; each uses nine batches of three calls.",
        "Speedup is base / head; below 1 means regression.",
        "",
        "| Case | Base ms | Head ms | Speedup | Pair speedups |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary:
        pairs = ", ".join(f"{ratio:.2f}x" for ratio in row["pair_speedups"])
        lines.append(
            f"| {row['case']} | {row['base_ms']:.3f} | {row['head_ms']:.3f} | "
            f"{row['speedup']:.2f}x | {pairs} |"
        )
    markdown = "\n".join(lines) + "\n"
    (output / "summary.md").write_text(markdown)
    print(markdown)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a") as stream:
            stream.write(markdown)


if __name__ == "__main__":
    main()

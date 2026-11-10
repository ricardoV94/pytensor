"""Separate argument-range effects from fused invprobit code generation."""

# ruff: noqa: T201

import json
import os
import time
from pathlib import Path

import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode


N = 200_000
BATCHES = 9
CALLS = 3


def median_seconds(fn, values):
    fn(values)
    samples = []
    for _ in range(BATCHES):
        start = time.perf_counter()
        for _ in range(CALLS):
            fn(values)
        samples.append((time.perf_counter() - start) / CALLS)
    return float(np.median(samples))


def main():
    output = Path(os.environ["DIAGNOSTIC_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rng.uniform(0.1, 20, N)  # Match the original script's RNG sequence.
    wide = rng.uniform(-6, 6, N)
    half_values = rng.uniform(0.01, 6, N)
    inv_values = rng.uniform(-6, 6, N)
    inv_args = -inv_values / np.sqrt(2)
    positive = np.abs(inv_args)
    tail = rng.uniform(1, 6 / np.sqrt(2), N)
    outer = rng.uniform(6 / np.sqrt(2), 6, N) * rng.choice([-1, 1], N)
    x = pt.dvector("x")
    plain = pytensor.function([x], pt.erfc(x), mode="NUMBA", trust_input=True)
    invprobit = 0.5 * pt.erfc(-x / np.sqrt(2))
    inv_total = invprobit.sum()
    inv_grad = pt.grad(inv_total, x)
    half_total = pt.log1p(-pt.erfc(x / np.sqrt(2))).sum()
    half_grad = pt.grad(half_total, x)
    unfused = get_mode("NUMBA").excluding("fusion", "fuse_indexed_into_elemwise")
    bare = {
        "erfc_wide": wide,
        "erfc_inv_arguments": inv_args,
        "erfc_half_arguments": half_values / np.sqrt(2),
        "erfc_matched_positive": positive,
        "erfc_matched_negative": -positive,
        "erfc_central": rng.uniform(-1, 1, N),
        "erfc_positive_tail": tail,
        "erfc_negative_tail": -tail,
        "erfc_outer": outer,
    }
    cases = {name: (plain, values) for name, values in bare.items()}
    for name, outputs, values, mode in (
        ("inv_forward_vector", invprobit, inv_values, "NUMBA"),
        ("inv_forward_sum", inv_total, inv_values, "NUMBA"),
        ("inv_gradient_only", inv_grad, inv_values, "NUMBA"),
        ("inv_both", [inv_total, inv_grad], inv_values, "NUMBA"),
        ("inv_both_unfused", [inv_total, inv_grad], inv_values, unfused),
        ("half_forward_sum", half_total, half_values, "NUMBA"),
        ("half_both", [half_total, half_grad], half_values, "NUMBA"),
    ):
        cases[name] = (
            pytensor.function([x], outputs, mode=mode, trust_input=True),
            values,
        )

    timings = {}
    graphs = {}
    checksums = {}
    for name, (fn, values) in cases.items():
        result = fn(values)
        outputs = result if isinstance(result, list) else [result]
        checksums[name] = [float(np.sum(value)) for value in outputs]
        timings[name] = median_seconds(fn, values)
        graphs[name] = [str(node.op) for node in fn.maker.fgraph.toposort()]
        if os.environ.get("DUMP_IR") == "1" and (
            name not in bare or name == "erfc_wide"
        ):
            jit = fn.vm.jit_fn
            for index, llvm in enumerate(jit.inspect_llvm().values()):
                (output / f"{name}-{index}.ll").write_text(llvm)
            for index, assembly in enumerate(jit.inspect_asm().values()):
                (output / f"{name}-{index}.s").write_text(assembly)
            (output / f"{name}.graph.txt").write_text(
                pytensor.dprint(fn, file="str", print_type=True)
            )
    results = {
        "pytensor": pytensor.__file__,
        "seconds": timings,
        "graphs": graphs,
        "checksums": checksums,
        "bare_ranges": {
            name: {
                "min": float(values.min()),
                "max": float(values.max()),
                "fraction_abs_lt_1": float(np.mean(np.abs(values) < 1)),
            }
            for name, values in bare.items()
        },
    }
    (output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results))


if __name__ == "__main__":
    main()

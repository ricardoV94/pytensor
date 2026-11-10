"""Interleave erfc implementations on identical data within one process."""

# ruff: noqa: T201

import ctypes
import json
import math
import platform
import sys
import time
from pathlib import Path

import numba
import numpy as np
import scipy
import scipy.special.cython_special as cs
from numba.extending import get_cython_function_address

import pytensor
import pytensor.scalar.math as psm
import pytensor.tensor as pt
from pytensor.link.numba.dispatch.basic import (
    numba_funcify_and_cache_key,
    numba_njit,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.cython_support import wrap_cython_function
from pytensor.link.numba.dispatch.scalar import (
    numba_funcify_ScalarOp,
    register_scalar_funcify,
    scalar_op_cache_key,
)


def main():
    output = Path("erfc-diagnostic-results").resolve()
    output.mkdir(exist_ok=True)
    metadata = {
        "python": sys.version,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "numba": numba.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pytensor": pytensor.__file__,
    }
    print(json.dumps(metadata), flush=True)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    spec = wrap_cython_function(cs.erfc, np.dtype("float64"), [np.dtype("float64")])
    address = get_cython_function_address(spec.module_name, spec.capi_name)
    if spec.has_pyx_skip_dispatch():
        pointer = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_int)(
            address
        )

        @numba_njit(cache=False, inline="always")
        def direct_erfc(x):
            return pointer(x, 1)

    else:
        pointer = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(address)

        @numba_njit(cache=False, inline="always")
        def direct_erfc(x):
            return pointer(x)

    def lower_libm(op, node, **kwargs):
        @numba_njit
        def libm_erfc(x):
            return math.erfc(x)

        return libm_erfc, scalar_op_cache_key(op, paired="libm")

    def lower_direct(op, node, **kwargs):
        return direct_erfc, None

    n = 200_000
    rng = np.random.default_rng(0)
    rng.uniform(0.1, 20, n)
    wide = rng.uniform(-6, 6, n)
    half_values = rng.uniform(0.01, 6, n)
    values = rng.uniform(-6, 6, n)
    arguments = -values / np.sqrt(2)
    x = pt.dvector("x")
    invprobit = 0.5 * pt.erfc(-x / np.sqrt(2))
    inv_total = invprobit.sum()
    inv_grad = pt.grad(inv_total, x)
    half_total = pt.log1p(-pt.erfc(x / np.sqrt(2))).sum()
    expressions = {
        "inv_forward": invprobit,
        "inv_both": [inv_total, inv_grad],
        "inv_gradient_only": inv_grad,
        "half_both": [half_total, pt.grad(half_total, x)],
    }
    functions = {}
    original = numba_funcify_and_cache_key.dispatch(psm.Erfc)
    try:
        for label, lower in (
            ("libm", lower_libm),
            ("scipy_cached", numba_funcify_ScalarOp),
            ("scipy_direct", lower_direct),
        ):
            register_scalar_funcify(psm.Erfc)(lower)
            functions[label] = {
                name: pytensor.function([x], expr, mode="NUMBA", trust_input=True)
                for name, expr in expressions.items()
            }
            functions[label]["bare"] = pytensor.function(
                [x], pt.erfc(x), mode="NUMBA", trust_input=True
            )
    finally:
        register_funcify_and_cache_key(psm.Erfc)(original)

    cases = {
        "erfc_wide": ("bare", wide),
        "erfc_inv_arguments": ("bare", arguments),
        "erfc_half_arguments": ("bare", half_values / np.sqrt(2)),
        "erfc_negative": ("bare", -np.abs(arguments)),
        "erfc_central": ("bare", rng.uniform(-1, 1, n)),
        "inv_forward": ("inv_forward", values),
        "inv_both": ("inv_both", values),
        "inv_gradient_only": ("inv_gradient_only", values),
        "half_both": ("half_both", half_values),
    }
    labels = list(functions)
    # Complete all JIT compilation and validate before starting any timers.
    for name, (kind, data) in cases.items():
        reference = functions["libm"][kind](data)
        reference = reference if isinstance(reference, list) else [reference]
        for label in labels:
            result = functions[label][kind](data)
            result = result if isinstance(result, list) else [result]
            for actual, expected in zip(result, reference, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=1e-12)
    summary = []
    for name, (kind, data) in cases.items():
        timings = {label: [] for label in labels}
        for _ in range(25):
            for label in rng.permutation(labels):
                fn = functions[label][kind]
                start = time.perf_counter()
                for _ in range(3):
                    fn(data)
                timings[label].append((time.perf_counter() - start) / 3)
        medians = {label: float(np.median(times)) for label, times in timings.items()}
        row = {
            "case": name,
            "milliseconds": {label: value * 1000 for label, value in medians.items()},
            "speedup": {
                label: medians["libm"] / value for label, value in medians.items()
            },
            "samples": timings,
        }
        summary.append(row)
        print(
            json.dumps({key: value for key, value in row.items() if key != "samples"}),
            flush=True,
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    for label in labels:
        for name in (
            "inv_forward",
            "inv_both",
            "inv_gradient_only",
            "half_both",
            "bare",
        ):
            fn = functions[label][name]
            for index, llvm in enumerate(fn.vm.jit_fn.inspect_llvm().values()):
                (output / f"{label}-{name}-{index}.ll").write_text(llvm)
            for index, assembly in enumerate(fn.vm.jit_fn.inspect_asm().values()):
                (output / f"{label}-{name}-{index}.s").write_text(assembly)


if __name__ == "__main__":
    main()

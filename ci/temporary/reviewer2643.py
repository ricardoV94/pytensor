# ruff: noqa: T201

import json
import sys
import time

import numpy as np


SIZE = 200_000
BATCHES = 9
CALLS_PER_BATCH = 3


def cases():
    import pytensor.tensor as pt

    rng = np.random.default_rng(0)
    x = pt.dvector("x")
    sqrt_two = np.sqrt(2.0)

    half_normal_logcdf = pt.log1p(-pt.erfc(x / sqrt_two)).sum()
    invprobit = (0.5 * pt.erfc(-x / sqrt_two)).sum()

    return {
        "gammaln(x), x ~ U[0.1, 20]": (x, [pt.gammaln(x)], rng.uniform(0.1, 20, SIZE)),
        "erfc(x), x ~ U[-6, 6]": (x, [pt.erfc(x)], rng.uniform(-6, 6, SIZE)),
        "HalfNormal log-CDF, sum + grad": (
            x,
            [half_normal_logcdf, pt.grad(half_normal_logcdf, x)],
            rng.uniform(0.01, 6, SIZE),
        ),
        "invprobit, sum + grad": (
            x,
            [invprobit, pt.grad(invprobit, x)],
            rng.uniform(-6, 6, SIZE),
        ),
    }


def median_batch_seconds(function, value):
    function(value)
    batch_times = []
    for _ in range(BATCHES):
        start = time.perf_counter()
        for _ in range(CALLS_PER_BATCH):
            function(value)
        batch_times.append((time.perf_counter() - start) / CALLS_PER_BATCH)
    return float(np.median(batch_times))


def main():
    import pytensor

    results = {}
    for name, (x, outputs, value) in cases().items():
        function = pytensor.function([x], outputs, mode="NUMBA", trust_input=True)
        results[name] = median_batch_seconds(function=function, value=value)

    print(
        json.dumps(
            {"arm": sys.argv[1], "pytensor": pytensor.__file__, "seconds": results}
        )
    )


if __name__ == "__main__":
    main()

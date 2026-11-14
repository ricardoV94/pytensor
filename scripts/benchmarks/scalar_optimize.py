"""PR #2592 scalar solver tests, including the implicit solution gradient."""

import numpy as np
from scipy import optimize

import pytensor
import pytensor.tensor as pt
from pytensor.tensor.optimize import minimize_scalar, root_scalar


def cases(size):
    # These are scalar workloads; the template's array-size option is unused.
    for dtype in ("float32", "float64"):
        x = pt.scalar("x", dtype=dtype)
        a = pt.scalar("a", dtype=dtype)
        points = {
            f"a={value}": (np.asarray(0, dtype), np.asarray(value, dtype))
            for value in (0.75, 1.0, 1.25)
        }
        tolerance = (
            dict(rtol=2e-3, atol=2e-5)
            if dtype == "float32"
            else dict(rtol=1e-6, atol=1e-8)
        )
        root_kwargs = {"xtol": 1e-5} if dtype == "float32" else {}
        for method in (
            "newton",
            "halley",
            "secant",
            "brentq",
            "bisect",
            "brenth",
            "ridder",
            "toms748",
        ):
            # The PR base cannot unbox float32 bracketing roots (fixed by #2592).
            if dtype == "float32" and method not in ("newton", "halley", "secant"):
                continue
            options = {}
            if method in ("newton", "halley"):
                options["jac"] = True
                if method == "halley":
                    options["hess"] = True
            elif method != "secant":
                options["bracket"] = pt.stack([-2 * a, 0.0])
            solution, success = root_scalar(
                equation=x + 2 * a * pt.cos(x),
                variable=x,
                method=method,
                optimizer_kwargs=root_kwargs,
                **options,
            )
            outputs = [solution, success, pt.grad(solution, a)]

            def build(x=x, a=a, outputs=outputs):
                return pytensor.function([x, a], outputs, mode="NUMBA")

            def check(result, inputs, tolerance=tolerance):
                value, succeeded, gradient = result
                parameter = float(inputs[1])
                reference = optimize.brentq(
                    lambda z: z + 2 * parameter * np.cos(z), -2 * parameter, 0
                )
                assert succeeded
                np.testing.assert_allclose(value, reference, **tolerance)
                np.testing.assert_allclose(
                    gradient,
                    -2 * np.cos(reference) / (1 - 2 * parameter * np.sin(reference)),
                    **tolerance,
                )

            yield f"root/{method}/{dtype}", build, points, check

        for name, method, options in (
            ("brent", "brent", {}),
            ("brent_search", "brent", {"bracket": pt.stack([a - 1.0, a])}),
            ("brent_bracket", "brent", {"bracket": pt.stack([2.0, 3 * a, 4.0])}),
            ("golden", "golden", {"bracket": pt.stack([2.0, 3 * a, 4.0])}),
            ("bounded", "bounded", {"bounds": pt.stack([0.0, 5 * a])}),
        ):
            optimizer_kwargs = {"tol": 1e-5} if dtype == "float32" else {}
            # xatol is supported by the PR base as well as the head.
            if method == "bounded" and dtype == "float32":
                optimizer_kwargs = {"options": {"xatol": 1e-5}}
            solution, success = minimize_scalar(
                objective=pt.cos(x) + 0.1 * a * x,
                x=x,
                method=method,
                optimizer_kwargs=optimizer_kwargs,
                **options,
            )
            outputs = [solution, success, pt.grad(solution, a)]

            def build(x=x, a=a, outputs=outputs):
                return pytensor.function([x, a], outputs, mode="NUMBA")

            def check(result, inputs, tolerance=tolerance):
                value, succeeded, gradient = result
                parameter = float(inputs[1])
                assert succeeded
                np.testing.assert_allclose(
                    value, np.pi - np.arcsin(0.1 * parameter), **tolerance
                )
                np.testing.assert_allclose(
                    gradient,
                    -0.1 / np.sqrt(1 - (0.1 * parameter) ** 2),
                    **tolerance,
                )

            yield f"minimize/{name}/{dtype}", build, points, check

"""XRV constructor helpers for building xtensor random variable expressions."""

from collections.abc import Sequence
from typing import Literal

from pytensor.graph.basic import Variable
from pytensor.tensor.random.op import RandomVariable
from pytensor.xtensor.type import as_xtensor
from pytensor.xtensor.vectorization import XRV


__all__ = [
    "as_xrv",
    "bernoulli",
    "beta",
    "betabinom",
    "binomial",
    "categorical",
    "cauchy",
    "chisquare",
    "dirichlet",
    "exponential",
    "gamma",
    "gengamma",
    "geometric",
    "gumbel",
    "halfcauchy",
    "halfnormal",
    "hypergeometric",
    "integers",
    "invgamma",
    "laplace",
    "logistic",
    "lognormal",
    "multinomial",
    "multivariate_normal",
    "nbinom",
    "negative_binomial",
    "normal",
    "pareto",
    "poisson",
    "rayleigh",
    "standard_normal",
    "t",
    "triangular",
    "truncexpon",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
]


def as_xrv(
    core_op: RandomVariable,
    core_inps_dims_map: Sequence[Sequence[int]] | None = None,
    core_out_dims_map: Sequence[int] | None = None,
    name: str | None = None,
):
    """Create an XRV constructor function for a given core RandomVariable op.

    Parameters
    ----------
    core_op : RandomVariable
        The core random variable operation to wrap.
    core_inps_dims_map : Sequence[Sequence[int]] | None
        Mapping of core dimensions for each input parameter.
        If None, assumes positional left-to-right.
    core_out_dims_map : Sequence[int] | None
        Mapping of core dimensions for the output.
        If None, assumes positional left-to-right.
    name : str | None
        Display name for the XRV op.
    """
    if core_inps_dims_map is None:
        core_inps_dims_map = [tuple(range(ndim)) for ndim in core_op.ndims_params]
    if core_out_dims_map is None:
        core_out_dims_map = tuple(range(core_op.ndim_supp))

    core_dims_needed = max(
        max(
            (
                max((entry + 1 for entry in dims_map), default=0)
                for dims_map in core_inps_dims_map
            ),
            default=0,
        ),
        max((entry + 1 for entry in core_out_dims_map), default=0),
    )

    def xrv_constructor(
        *params,
        core_dims: Sequence[str] | str | None = None,
        extra_dims: dict[str, Variable] | None = None,
        rng: Variable | None = None,
        return_next_rng: bool = False,
    ):
        if core_dims is None:
            core_dims_tuple: tuple[str, ...] = ()
            if core_dims_needed:
                raise ValueError(
                    f"{core_op.name} needs {core_dims_needed} core_dims to be specified"
                )
        elif isinstance(core_dims, str):
            core_dims_tuple = (core_dims,)
        else:
            core_dims_tuple = tuple(core_dims)

        if len(core_dims_tuple) != core_dims_needed:
            raise ValueError(
                f"{core_op.name} needs {core_dims_needed} core_dims, but got {len(core_dims_tuple)}"
            )

        full_input_core_dims = tuple(
            tuple(core_dims_tuple[i] for i in inp_dims_map)
            for inp_dims_map in core_inps_dims_map
        )
        full_output_core_dims = tuple(core_dims_tuple[i] for i in core_out_dims_map)
        full_core_dims = (full_input_core_dims, full_output_core_dims)

        if extra_dims is None:
            extra_dims = {}

        import warnings

        if not return_next_rng:
            warnings.warn(
                "XRV Ops will stop hiding the rng output in a future version. "
                "Set return_next_rng=True to suppress this warning.",
                DeprecationWarning,
                stacklevel=2,
            )

        out = XRV(
            core_op,
            core_dims=full_core_dims,
            extra_dims=tuple(extra_dims.keys()),
            name=name,
        )(rng, *extra_dims.values(), *params)
        if return_next_rng:
            next_rng = out.owner.outputs[0]
            return next_rng, out
        return out

    return xrv_constructor


def multivariate_normal(
    mean,
    cov,
    *,
    core_dims: Sequence[str],
    extra_dims=None,
    rng=None,
    method: Literal["cholesky", "svd", "eigh"] = "cholesky",
    return_next_rng: bool = False,
):
    """Multivariate normal random variable for xtensors."""
    import pytensor.tensor.random.basic as ptrb

    mean = as_xtensor(mean)
    if len(core_dims) != 2:
        raise ValueError(
            f"multivariate_normal requires 2 core_dims, got {len(core_dims)}"
        )

    # Align core_dims so the dim in mean comes first (output core dim)
    if core_dims[0] not in mean.type.dims:
        core_dims = core_dims[::-1]

    xop = as_xrv(ptrb.MvNormalRV(method=method))
    return xop(
        mean,
        cov,
        core_dims=core_dims,
        extra_dims=extra_dims,
        rng=rng,
        return_next_rng=return_next_rng,
    )


def _make_xrv_fn(core_op_name):
    """Create a module-level XRV function for a given core op name."""
    import pytensor.tensor.random.basic as ptrb

    core_op = getattr(ptrb, core_op_name)
    return as_xrv(core_op)


def _lazy_xrv_fn(core_op_name):
    """Lazily create an XRV function to avoid circular imports at module level."""

    def wrapper(*args, **kwargs):
        fn = _make_xrv_fn(core_op_name)
        return fn(*args, **kwargs)

    wrapper.__name__ = core_op_name
    wrapper.__qualname__ = core_op_name
    return wrapper


# Named distribution constructors (functional API)
bernoulli = _lazy_xrv_fn("bernoulli")
beta = _lazy_xrv_fn("beta")
betabinom = _lazy_xrv_fn("betabinom")
binomial = _lazy_xrv_fn("binomial")
categorical = _lazy_xrv_fn("categorical")
cauchy = _lazy_xrv_fn("cauchy")
dirichlet = _lazy_xrv_fn("dirichlet")
exponential = _lazy_xrv_fn("exponential")
gamma = _lazy_xrv_fn("_gamma")
gengamma = _lazy_xrv_fn("gengamma")
geometric = _lazy_xrv_fn("geometric")
gumbel = _lazy_xrv_fn("gumbel")
halfcauchy = _lazy_xrv_fn("halfcauchy")
halfnormal = _lazy_xrv_fn("halfnormal")
hypergeometric = _lazy_xrv_fn("hypergeometric")
integers = _lazy_xrv_fn("integers")
invgamma = _lazy_xrv_fn("invgamma")
laplace = _lazy_xrv_fn("laplace")
logistic = _lazy_xrv_fn("logistic")
lognormal = _lazy_xrv_fn("lognormal")
multinomial = _lazy_xrv_fn("multinomial")
negative_binomial = _lazy_xrv_fn("negative_binomial")
nbinom = negative_binomial
normal = _lazy_xrv_fn("normal")
pareto = _lazy_xrv_fn("pareto")
poisson = _lazy_xrv_fn("poisson")
t = _lazy_xrv_fn("t")
triangular = _lazy_xrv_fn("triangular")
truncexpon = _lazy_xrv_fn("truncexpon")
uniform = _lazy_xrv_fn("uniform")
vonmises = _lazy_xrv_fn("vonmises")
wald = _lazy_xrv_fn("wald")
weibull = _lazy_xrv_fn("weibull")


def standard_normal(
    extra_dims=None,
    rng=None,
    return_next_rng=False,
):
    return normal(0, 1, extra_dims=extra_dims, rng=rng, return_next_rng=return_next_rng)


def chisquare(
    df,
    extra_dims=None,
    rng=None,
    return_next_rng=False,
):
    return gamma(
        df / 2.0, 2.0, extra_dims=extra_dims, rng=rng, return_next_rng=return_next_rng
    )


def rayleigh(
    scale,
    extra_dims=None,
    rng=None,
    return_next_rng=False,
):
    from pytensor.xtensor.math import sqrt

    df = scale * 0 + 2
    next_rng, chisquare_draws = chisquare(
        df, extra_dims=extra_dims, rng=rng, return_next_rng=True
    )
    rayleigh_draws = sqrt(chisquare_draws) * scale
    if return_next_rng:
        return next_rng, rayleigh_draws
    return rayleigh_draws

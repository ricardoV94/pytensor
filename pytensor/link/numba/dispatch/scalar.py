import math

import numpy as np

from pytensor.compile.ops import TypeCastingOp
from pytensor.graph.basic import Variable
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    create_numba_signature,
    generate_fallback_impl,
    numba_funcify,
)
from pytensor.link.numba.dispatch.cython_support import wrap_cython_function
from pytensor.link.utils import (
    compile_function_src,
    get_name_for_object,
    unique_name_generator,
)
from pytensor.scalar.basic import (
    Add,
    Cast,
    Clip,
    Composite,
    Identity,
    Mul,
    Reciprocal,
    ScalarOp,
    Second,
    Switch,
)
from pytensor.scalar.math import Erf, Erfc, GammaLn, Log1mexp, Sigmoid, Softplus


@numba_funcify.register(ScalarOp)
def numba_funcify_ScalarOp(op, node, **kwargs):
    # TODO: Do we need to cache these functions so that we don't end up
    # compiling the same Numba function over and over again?

    if not hasattr(op, "nfunc_spec"):
        return generate_fallback_impl(op, node, **kwargs)

    scalar_func_path = op.nfunc_spec[0]
    scalar_func_numba = None

    *module_path, scalar_func_name = scalar_func_path.split(".")
    if not module_path:
        # Assume it is numpy, and numba has an implementation
        scalar_func_numba = getattr(np, scalar_func_name)

    input_dtypes = [np.dtype(input.type.dtype) for input in node.inputs]
    output_dtypes = [np.dtype(output.type.dtype) for output in node.outputs]

    if len(output_dtypes) != 1:
        raise ValueError("ScalarOps with more than one output are not supported")

    output_dtype = output_dtypes[0]

    input_inner_dtypes = None
    output_inner_dtype = None

    # Cython functions might have an additional argument
    has_pyx_skip_dispatch = False

    if scalar_func_path.startswith("scipy.special"):
        import scipy.special.cython_special

        cython_func = getattr(scipy.special.cython_special, scalar_func_name, None)
        if cython_func is not None:
            scalar_func_numba = wrap_cython_function(
                cython_func, output_dtype, input_dtypes
            )
            has_pyx_skip_dispatch = scalar_func_numba.has_pyx_skip_dispatch
            input_inner_dtypes = scalar_func_numba.numpy_arg_dtypes()
            output_inner_dtype = scalar_func_numba.numpy_output_dtype()

    if scalar_func_numba is None:
        scalar_func_numba = generate_fallback_impl(op, node, **kwargs)

    scalar_op_fn_name = get_name_for_object(scalar_func_numba)

    global_env = {"scalar_func_numba": scalar_func_numba}

    if input_inner_dtypes is None and output_inner_dtype is None:
        unique_names = unique_name_generator(
            [scalar_op_fn_name, "scalar_func_numba"], suffix_sep="_"
        )
        input_names = ", ".join(unique_names(v, force_unique=True) for v in node.inputs)
        if not has_pyx_skip_dispatch:
            scalar_op_src = f"""
def {scalar_op_fn_name}({input_names}):
    return scalar_func_numba({input_names})
            """
        else:
            scalar_op_src = f"""
def {scalar_op_fn_name}({input_names}):
    return scalar_func_numba({input_names}, np.intc(1))
            """

    else:
        global_env["direct_cast"] = numba_basic.direct_cast
        global_env["output_dtype"] = np.dtype(output_inner_dtype)
        input_tmp_dtype_names = {
            f"inp_tmp_dtype_{i}": i_dtype
            for i, i_dtype in enumerate(input_inner_dtypes)
        }
        global_env.update(input_tmp_dtype_names)

        unique_names = unique_name_generator(
            [scalar_op_fn_name, "scalar_func_numba", *global_env.keys()],
            suffix_sep="_",
        )

        input_names = [unique_names(v, force_unique=True) for v in node.inputs]
        converted_call_args = ", ".join(
            f"direct_cast({i_name}, {i_tmp_dtype_name})"
            for i_name, i_tmp_dtype_name in zip(
                input_names, input_tmp_dtype_names, strict=False
            )
        )
        if not has_pyx_skip_dispatch:
            scalar_op_src = f"""
def {scalar_op_fn_name}({', '.join(input_names)}):
    return direct_cast(scalar_func_numba({converted_call_args}), output_dtype)
            """
        else:
            scalar_op_src = f"""
def {scalar_op_fn_name}({', '.join(input_names)}):
    return direct_cast(scalar_func_numba({converted_call_args}, np.intc(1)), output_dtype)
            """

    scalar_op_fn = compile_function_src(
        scalar_op_src, scalar_op_fn_name, {**globals(), **global_env}
    )

    signature = create_numba_signature(node, force_scalar=True)

    return numba_basic.numba_njit(
        signature,
        # Functions that call a function pointer can't be cached
        cache=False,
    )(scalar_op_fn)


@numba_funcify.register(Switch)
def numba_funcify_Switch(op, node, **kwargs):
    @numba_basic.numba_njit
    def switch(condition, x, y):
        if condition:
            return x
        else:
            return y

    return switch


def binary_to_nary_func(inputs: list[Variable], binary_op_name: str, binary_op: str):
    """Create a Numba-compatible N-ary function from a binary function."""
    unique_names = unique_name_generator(["binary_op_name"], suffix_sep="_")
    input_names = [unique_names(v, force_unique=True) for v in inputs]
    input_signature = ", ".join(input_names)
    output_expr = binary_op.join(input_names)

    nary_src = f"""
def {binary_op_name}({input_signature}):
    return {output_expr}
    """
    nary_fn = compile_function_src(nary_src, binary_op_name, globals())

    return nary_fn


@numba_funcify.register(Add)
def numba_funcify_Add(op, node, **kwargs):
    signature = create_numba_signature(node, force_scalar=True)
    nary_add_fn = binary_to_nary_func(node.inputs, "add", "+")

    return numba_basic.numba_njit(signature)(nary_add_fn)


@numba_funcify.register(Mul)
def numba_funcify_Mul(op, node, **kwargs):
    signature = create_numba_signature(node, force_scalar=True)
    nary_add_fn = binary_to_nary_func(node.inputs, "mul", "*")

    return numba_basic.numba_njit(signature)(nary_add_fn)


@numba_funcify.register(Cast)
def numba_funcify_Cast(op, node, **kwargs):
    dtype = np.dtype(op.o_type.dtype)

    @numba_basic.numba_njit
    def cast(x):
        return numba_basic.direct_cast(x, dtype)

    return cast


@numba_funcify.register(Identity)
@numba_funcify.register(TypeCastingOp)
def numba_funcify_type_casting(op, **kwargs):
    @numba_basic.numba_njit
    def identity(x):
        return x

    return identity


@numba_funcify.register(Clip)
def numba_funcify_Clip(op, **kwargs):
    @numba_basic.numba_njit
    def clip(x, min_val, max_val):
        x = numba_basic.to_scalar(x)
        min_scalar = numba_basic.to_scalar(min_val)
        max_scalar = numba_basic.to_scalar(max_val)

        if x < min_scalar:
            return min_scalar
        elif x > max_scalar:
            return max_scalar
        else:
            return x

    return clip


@numba_funcify.register(Composite)
def numba_funcify_Composite(op, node, **kwargs):
    signature = create_numba_signature(op.fgraph, force_scalar=True)

    _ = kwargs.pop("storage_map", None)

    composite_fn = numba_basic.numba_njit(signature)(
        numba_funcify(op.fgraph, squeeze_output=True, **kwargs)
    )
    return composite_fn


@numba_funcify.register(Second)
def numba_funcify_Second(op, node, **kwargs):
    @numba_basic.numba_njit
    def second(x, y):
        return y

    return second


@numba_funcify.register(Reciprocal)
def numba_funcify_Reciprocal(op, node, **kwargs):
    @numba_basic.numba_njit
    def reciprocal(x):
        # TODO FIXME: This isn't really the behavior or `numpy.reciprocal` when
        # `x` is an `int`
        return 1 / x

    return reciprocal


@numba_funcify.register(Sigmoid)
def numba_funcify_Sigmoid(op, node, **kwargs):
    @numba_basic.numba_njit
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    return sigmoid


@numba_funcify.register(GammaLn)
def numba_funcify_GammaLn(op, node, **kwargs):
    @numba_basic.numba_njit
    def gammaln(x):
        return math.lgamma(x)

    return gammaln


@numba_funcify.register(Log1mexp)
def numba_funcify_Log1mexp(op, node, **kwargs):
    @numba_basic.numba_njit
    def logp1mexp(x):
        if x < np.log(0.5):
            return np.log1p(-np.exp(x))
        else:
            return np.log(-np.expm1(x))

    return logp1mexp


@numba_funcify.register(Erf)
def numba_funcify_Erf(op, **kwargs):
    @numba_basic.numba_njit
    def erf(x):
        return math.erf(x)

    return erf


@numba_funcify.register(Erfc)
def numba_funcify_Erfc(op, **kwargs):
    @numba_basic.numba_njit
    def erfc(x):
        return math.erfc(x)

    return erfc


@numba_funcify.register(Softplus)
def numba_funcify_Softplus(op, node, **kwargs):
    out_dtype = np.dtype(node.outputs[0].type.dtype)

    @numba_basic.numba_njit
    def softplus(x):
        if x < -37.0:
            value = np.exp(x)
        elif x < 18.0:
            value = np.log1p(np.exp(x))
        elif x < 33.3:
            value = x + np.exp(-x)
        else:
            value = x
        return numba_basic.direct_cast(value, out_dtype)

    return softplus

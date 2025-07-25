import warnings

import numba
import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    get_numba_type,
    int_to_float_fn,
    numba_funcify,
)
from pytensor.tensor.nlinalg import (
    SVD,
    Det,
    Eig,
    Eigh,
    MatrixInverse,
    MatrixPinv,
    SLogDet,
)


@numba_funcify.register(SVD)
def numba_funcify_SVD(op, node, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv
    out_dtype = np.dtype(node.outputs[0].dtype)

    inputs_cast = int_to_float_fn(node.inputs, out_dtype)

    if not compute_uv:

        @numba_basic.numba_njit()
        def svd(x):
            _, ret, _ = np.linalg.svd(inputs_cast(x), full_matrices)
            return ret

    else:

        @numba_basic.numba_njit()
        def svd(x):
            return np.linalg.svd(inputs_cast(x), full_matrices)

    return svd


@numba_funcify.register(Det)
def numba_funcify_Det(op, node, **kwargs):
    out_dtype = node.outputs[0].type.numpy_dtype
    inputs_cast = int_to_float_fn(node.inputs, out_dtype)

    @numba_basic.numba_njit(inline="always")
    def det(x):
        return np.array(np.linalg.det(inputs_cast(x))).astype(out_dtype)

    return det


@numba_funcify.register(SLogDet)
def numba_funcify_SLogDet(op, node, **kwargs):
    out_dtype_1 = node.outputs[0].type.numpy_dtype
    out_dtype_2 = node.outputs[1].type.numpy_dtype

    inputs_cast = int_to_float_fn(node.inputs, out_dtype_1)

    @numba_basic.numba_njit
    def slogdet(x):
        sign, det = np.linalg.slogdet(inputs_cast(x))
        return (
            np.array(sign).astype(out_dtype_1),
            np.array(det).astype(out_dtype_2),
        )

    return slogdet


@numba_funcify.register(Eig)
def numba_funcify_Eig(op, node, **kwargs):
    out_dtype_1 = node.outputs[0].type.numpy_dtype
    out_dtype_2 = node.outputs[1].type.numpy_dtype

    inputs_cast = int_to_float_fn(node.inputs, out_dtype_1)

    @numba_basic.numba_njit
    def eig(x):
        out = np.linalg.eig(inputs_cast(x))
        return (out[0].astype(out_dtype_1), out[1].astype(out_dtype_2))

    return eig


@numba_funcify.register(Eigh)
def numba_funcify_Eigh(op, node, **kwargs):
    uplo = op.UPLO

    if uplo != "L":
        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`UPLO` argument to `numpy.linalg.eigh`."
            ),
            UserWarning,
        )

        out_dtypes = tuple(o.type.numpy_dtype for o in node.outputs)
        ret_sig = numba.types.Tuple(
            [get_numba_type(node.outputs[0].type), get_numba_type(node.outputs[1].type)]
        )

        @numba_basic.numba_njit
        def eigh(x):
            with numba.objmode(ret=ret_sig):
                out = np.linalg.eigh(x, UPLO=uplo)
                ret = (out[0].astype(out_dtypes[0]), out[1].astype(out_dtypes[1]))
            return ret

    else:

        @numba_basic.numba_njit(inline="always")
        def eigh(x):
            return np.linalg.eigh(x)

    return eigh


@numba_funcify.register(MatrixInverse)
def numba_funcify_MatrixInverse(op, node, **kwargs):
    out_dtype = node.outputs[0].type.numpy_dtype
    inputs_cast = int_to_float_fn(node.inputs, out_dtype)

    @numba_basic.numba_njit(inline="always")
    def matrix_inverse(x):
        return np.linalg.inv(inputs_cast(x)).astype(out_dtype)

    return matrix_inverse


@numba_funcify.register(MatrixPinv)
def numba_funcify_MatrixPinv(op, node, **kwargs):
    out_dtype = node.outputs[0].type.numpy_dtype
    inputs_cast = int_to_float_fn(node.inputs, out_dtype)

    @numba_basic.numba_njit(inline="always")
    def matrixpinv(x):
        return np.linalg.pinv(inputs_cast(x)).astype(out_dtype)

    return matrixpinv

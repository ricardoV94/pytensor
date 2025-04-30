from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal, cast

from numpy import convolve as numpy_convolve

from pytensor.graph import Apply
from pytensor.link.c.op import COp
from pytensor.scalar.basic import upcast
from pytensor.tensor.basic import as_tensor_variable, join, zeros
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import maximum, minimum
from pytensor.tensor.type import complex_dtypes, vector
from pytensor.tensor.variable import TensorVariable


if TYPE_CHECKING:
    from pytensor.tensor import TensorLike


class Convolve1d(COp):
    __props__ = ("mode",)
    gufunc_signature = "(n),(k)->(o)"

    def __init__(self, mode: Literal["full", "valid"] = "full"):
        if mode not in ("full", "valid"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def make_node(self, in1, in2):
        in1 = as_tensor_variable(in1)
        in2 = as_tensor_variable(in2)

        assert in1.ndim == 1
        assert in2.ndim == 1

        dtype = upcast(in1.dtype, in2.dtype)

        n = in1.type.shape[0]
        k = in2.type.shape[0]

        if n is None or k is None:
            out_shape = (None,)
        elif self.mode == "full":
            out_shape = (n + k - 1,)
        else:  # mode == "valid":
            out_shape = (max(n, k) - min(n, k) + 1,)

        out = vector(dtype=dtype, shape=out_shape)
        return Apply(self, [in1, in2], [out])

    def perform(self, node, inputs, outputs):
        # We use numpy_convolve as that's what scipy would use if method="direct" was passed.
        # And mode != "same", which this Op doesn't cover anyway.
        outputs[0][0] = numpy_convolve(*inputs, mode=self.mode)

    def infer_shape(self, fgraph, node, shapes):
        in1_shape, in2_shape = shapes
        n = in1_shape[0]
        k = in2_shape[0]
        if self.mode == "full":
            shape = n + k - 1
        else:  # mode == "valid":
            shape = maximum(n, k) - minimum(n, k) + 1
        return [[shape]]

    def L_op(self, inputs, outputs, output_grads):
        in1, in2 = inputs
        [grad] = output_grads

        if self.mode == "full":
            valid_conv = type(self)(mode="valid")
            in1_bar = valid_conv(grad, in2[::-1])
            in2_bar = valid_conv(grad, in1[::-1])

        else:  # mode == "valid":
            full_conv = type(self)(mode="full")
            n = in1.shape[0]
            k = in2.shape[0]
            kmn = maximum(0, k - n)
            nmk = maximum(0, n - k)
            # We need mode="full" if k >= n else "valid" for `in1_bar` (opposite for `in2_bar`), but mode is not symbolic.
            # Instead, we always use mode="full" and slice the result so it behaves like "valid" for the input that's shorter.
            # There is a rewrite that optimizes this case when n, k are static
            in1_bar = full_conv(grad, in2[::-1])
            in1_bar = in1_bar[kmn : in1_bar.shape[0] - kmn]
            in2_bar = full_conv(grad, in1[::-1])
            in2_bar = in2_bar[nmk : in2_bar.shape[0] - nmk]

        return [in1_bar, in2_bar]

    def c_code(self, node, name, inputs, outputs, sub):
        raise NotImplementedError
        in1, in2 = inputs
        if any(inp.type.dtype in complex_dtypes for inp in node.inputs):
            raise NotImplementedError
        [out] = outputs
        fail = sub["fail"]

        mode = 2 if self.mode == "full" else 0
        return f"""
        Py_XDECREF({out});
        
        PyArrayObject *in1 = (PyArrayObject*){in1};
        PyArrayObject *in2 = (PyArrayObject*){in2};
        // Numpy correlate swaps inputs
        if (PyArray_SHAPE(in1)[0] < PyArray_SHAPE(in2)[0]){{
            in1 = (PyArrayObject*){in2};
            in2 = (PyArrayObject*){in1};
        }}
        
        // Create a reverse view of in2 by negating the 
        npy_intp stride = PyArray_STRIDES(in2)[0];
        npy_intp reverse_strides[1] = {{-stride}};
        npy_intp reverse_offset = (PyArray_SHAPE(in2)[0] - 1) * stride;
        
        Py_INCREF(PyArray_DESCR(in2));
        PyArrayObject* rev_in2 = (PyArrayObject*)PyArray_NewFromDescr(
            &PyArray_Type,
            PyArray_DESCR(in2),
            1,
            PyArray_SHAPE(in2),
            reverse_strides,
            PyArray_BYTES(in2) + reverse_offset,
            PyArray_FLAGS(in2),
            NULL);
                        
        if (!rev_in2){{
            {fail}
        }}

        Py_INCREF(in2);
        PyArray_SetBaseObject(rev_in2, (PyObject*)in2);        
        {out} = (PyArrayObject *) PyArray_Correlate((PyObject*)in1, (PyObject*)rev_in2, {mode});
        Py_DECREF(rev_in2);
        
        if (!{out}){{
            {fail}
        }}
        """

    def c_code_cache_version(self) -> tuple[Hashable, ...]:
        return None


def convolve1d(
    in1: "TensorLike",
    in2: "TensorLike",
    mode: Literal["full", "valid", "same"] = "full",
) -> TensorVariable:
    """Convolve two one-dimensional arrays.

    Convolve in1 and in2, with the output size determined by the mode argument.

    Parameters
    ----------
    in1 : (..., N,) tensor_like
        First input.
    in2 : (..., M,) tensor_like
        Second input.
    mode : {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        - 'full': The output is the full discrete linear convolution of the inputs, with shape (..., N+M-1,).
        - 'valid': The output consists only of elements that do not rely on zero-padding, with shape (..., max(N, M) - min(N, M) + 1,).
        - 'same': The output is the same size as in1, centered with respect to the 'full' output.

    Returns
    -------
    out: tensor_variable
        The discrete linear convolution of in1 with in2.

    """
    in1 = as_tensor_variable(in1)
    in2 = as_tensor_variable(in2)

    if mode == "same":
        # We implement "same" as "valid" with padded `in1`.
        in1_batch_shape = tuple(in1.shape)[:-1]
        zeros_left = in2.shape[-1] // 2
        zeros_right = (in2.shape[-1] - 1) // 2
        in1 = join(
            -1,
            zeros((*in1_batch_shape, zeros_left), dtype=in2.dtype),
            in1,
            zeros((*in1_batch_shape, zeros_right), dtype=in2.dtype),
        )
        mode = "valid"

    return cast(TensorVariable, Blockwise(Convolve1d(mode=mode))(in1, in2))

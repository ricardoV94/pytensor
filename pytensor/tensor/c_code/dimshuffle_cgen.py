from collections.abc import Sequence
from textwrap import dedent


def expand_dims_cgen(
    *,
    inp: str,
    out: str | None,
    nd_in: int,
    axis: Sequence[int],
    fail: str,
    check_ndim: bool = True,
):
    if out is None:
        # Works inplace
        out = inp
    else:
        assert inp != out
        raise NotImplementedError

    nd_out = nd_in + len(axis)
    code = ""
    code += f"PyArrayObject *original_inp = {inp};\n"
    code += f"npy_intp dimensions[{nd_out}];\n"
    code += f"npy_intp strides[{nd_out}];\n"

    if check_ndim:
        code += dedent(
            f"""
            if (PyArray_NDIM({inp}) != {nd_in}) {{
                PyErr_SetString(PyExc_ValueError, "ExpandDims: Input dimensions do not match expected.");
                {fail}
            }}
            """
        )

    j = 0
    for i in range(nd_out):
        if i in axis:
            code += f"dimensions[{i}] = 1;\n"
            code += f"strides[{i}] = PyArray_ITEMSIZE({inp});\n"
        else:
            code += f"dimensions[{i}] = PyArray_DIMS({inp})[{j}];\n"
            code += f"strides[{i}] = PyArray_STRIDES({inp})[{j}];\n"
            j += 1

    code += dedent(
        f"""
        // if not inplace: Py_XDECREF({out});
        Py_INCREF(PyArray_DESCR({inp}));
        {out} = (PyArrayObject*)PyArray_NewFromDescr(&PyArray_Type,
                                            PyArray_DESCR({inp}),
                                            {nd_out}, dimensions,
                                            strides,
                                            PyArray_DATA({inp}),
                                            (PyArray_FLAGS({inp}) & ~NPY_ARRAY_OWNDATA),
                                            NULL);

        if ({out} == NULL) {{
            {fail}
        }}
        // if not inplace: Py_INCREF((PyObject*){inp});
        PyArray_SetBaseObject({out}, (PyObject*)original_inp);
        """
    )
    return code

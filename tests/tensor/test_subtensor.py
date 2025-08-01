import logging
import sys
from io import StringIO

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from packaging import version

import pytensor
import pytensor.scalar as scal
import pytensor.tensor.basic as ptb
from pytensor import function
from pytensor.compile import DeepCopyOp, shared
from pytensor.compile.io import In
from pytensor.compile.mode import Mode
from pytensor.configdefaults import config
from pytensor.gradient import grad
from pytensor.graph import Constant
from pytensor.graph.basic import equal_computations
from pytensor.graph.op import get_test_value
from pytensor.graph.rewriting.utils import is_same_graph
from pytensor.printing import pprint
from pytensor.scalar.basic import as_scalar, int16
from pytensor.tensor import as_tensor, constant, get_vector_length, vectorize
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import exp, isinf, lt, switch
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.shape import specify_broadcastable, specify_shape
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedIndexingError,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    advanced_inc_subtensor,
    advanced_inc_subtensor1,
    advanced_set_subtensor,
    advanced_set_subtensor1,
    advanced_subtensor,
    advanced_subtensor1,
    as_index_literal,
    basic_shape,
    flip,
    get_canonical_form_slice,
    inc_subtensor,
    index_vars_to_types,
    indexed_result_shape,
    set_subtensor,
    slice_at_axis,
    take,
)
from pytensor.tensor.type import (
    TensorType,
    col,
    ctensor3,
    dmatrix,
    dscalar,
    dtensor3,
    dtensor4,
    dvector,
    fmatrix,
    fscalar,
    ftensor4,
    fvector,
    iscalar,
    lmatrix,
    lrow,
    lscalar,
    lvector,
    matrix,
    tensor,
    tensor3,
    tensor4,
    tensor5,
    vector,
)
from pytensor.tensor.type_other import (
    NoneConst,
    SliceConstant,
    as_symbolic_slice,
    make_slice,
    slicetype,
)
from tests import unittest_tools as utt
from tests.tensor.utils import inplace_func, integers_ranged, random


subtensor_ops = (
    Subtensor,
    IncSubtensor,
    AdvancedSubtensor1,
    AdvancedIncSubtensor1,
)


def test_as_index_literal():
    res = as_index_literal(slice(None, ptb.as_tensor(1)))
    assert res == slice(None, 1)
    res = as_index_literal(slice(ptb.as_tensor(1), None))
    assert res == slice(1, None)
    res = as_index_literal(slice(None, None, ptb.as_tensor(2)))
    assert res == slice(None, None, 2)
    res = as_index_literal(SliceConstant(slicetype, slice(None)))
    assert res == slice(None)
    res = as_index_literal(make_slice(None, ptb.as_tensor(1)))
    assert res == slice(None, 1)

    res = as_index_literal(ptb.as_tensor(2))
    assert res == 2

    res = as_index_literal(np.newaxis)
    assert res is np.newaxis
    res = as_index_literal(NoneConst)
    assert res is np.newaxis
    res = as_index_literal(NoneConst.clone())
    assert res is np.newaxis


class TestGetCanonicalFormSlice:
    @pytest.mark.parametrize(
        "idx",
        [
            NoneConst,
            None,
            as_symbolic_slice(slice(3, 7, 2)),
            as_symbolic_slice(slice(3, int16(), 2)),
            vector(),
        ],
    )
    def test_unsupported_inputs(self, idx):
        with pytest.raises(ValueError, match="not a supported slice"):
            get_canonical_form_slice(idx, 5)

    def test_scalar_constant(self):
        a = as_scalar(0)
        length = lscalar()
        res, direction = get_canonical_form_slice(a, length)
        assert res == 0
        assert direction == 1

        b = as_scalar(-1)
        res, direction = get_canonical_form_slice(b, length)
        assert equal_computations([res], [as_tensor(-1) + length])
        assert direction == 1

    def test_tensor_constant(self):
        a = as_tensor(0)
        length = lscalar()
        res, direction = get_canonical_form_slice(a, length)
        assert equal_computations([res], [a])
        assert direction == 1

        b = as_tensor(-1)
        res, direction = get_canonical_form_slice(b, length)
        assert equal_computations([res], [b + length])
        assert direction == 1

    def test_symbolic_scalar(self):
        a = int16()
        length = lscalar()
        res, direction = get_canonical_form_slice(a, length)
        a_t = as_tensor(a)
        assert equal_computations([res], [switch(lt(a_t, 0), a_t + length, a_t)])
        assert direction == 1

    def test_symbolic_tensor(self):
        a = lscalar()
        length = lscalar()
        res, direction = get_canonical_form_slice(a, length)
        assert equal_computations([res], [switch(lt(a, 0), a + length, a)])
        assert direction == 1

    @pytest.mark.parametrize("int_fn", [int, np.int64, as_tensor, as_scalar])
    def test_all_integer(self, int_fn):
        res = get_canonical_form_slice(
            slice(int_fn(1), int_fn(5), int_fn(2)), int_fn(7)
        )
        assert isinstance(res[0], slice)
        assert res[1] == 1

    def test_all_symbolic(self):
        start = iscalar("b")
        stop = iscalar("e")
        step = iscalar("s")
        length = iscalar("l")
        cnf = get_canonical_form_slice(slice(start, stop, step), length)
        f = pytensor.function(
            [start, stop, step, length],
            [
                ptb.as_tensor_variable(cnf[0].start),
                ptb.as_tensor_variable(cnf[0].stop),
                ptb.as_tensor_variable(cnf[0].step),
                ptb.as_tensor_variable(cnf[1]),
            ],
        )

        length = 5
        a = np.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
                for step in [-6, -3, -1, 2, 5]:
                    out = f(start, stop, step, length)
                    t_out = a[out[0] : out[1] : out[2]][:: out[3]]
                    v_out = a[start:stop:step]
                    assert np.all(t_out == v_out)
                    assert np.all(t_out.shape == v_out.shape)

    def test_start_None(self):
        stop = iscalar("e")
        step = iscalar("s")
        length = iscalar("l")
        cnf = get_canonical_form_slice(slice(None, stop, step), length)
        f = pytensor.function(
            [stop, step, length],
            [
                ptb.as_tensor_variable(cnf[0].start),
                ptb.as_tensor_variable(cnf[0].stop),
                ptb.as_tensor_variable(cnf[0].step),
                ptb.as_tensor_variable(cnf[1]),
            ],
        )

        length = 5
        a = np.arange(length)
        for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for step in [-6, -3, -1, 2, 5]:
                out = f(stop, step, length)
                t_out = a[out[0] : out[1] : out[2]][:: out[3]]
                v_out = a[:stop:step]
                assert np.all(t_out == v_out)
                assert np.all(t_out.shape == v_out.shape)

    def test_stop_None(self):
        start = iscalar("b")
        step = iscalar("s")
        length = iscalar("l")
        cnf = get_canonical_form_slice(slice(start, None, step), length)
        f = pytensor.function(
            [start, step, length],
            [
                ptb.as_tensor_variable(cnf[0].start),
                ptb.as_tensor_variable(cnf[0].stop),
                ptb.as_tensor_variable(cnf[0].step),
                ptb.as_tensor_variable(cnf[1]),
            ],
        )

        length = 5
        a = np.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for step in [-6, -3, -1, 2, 5]:
                out = f(start, step, length)
                t_out = a[out[0] : out[1] : out[2]][:: out[3]]
                v_out = a[start:None:step]
                assert np.all(t_out == v_out)
                assert np.all(t_out.shape == v_out.shape)

    def test_step_None(self):
        start = iscalar("b")
        stop = iscalar("e")
        length = iscalar("l")
        cnf = get_canonical_form_slice(slice(start, stop, None), length)
        f = pytensor.function(
            [start, stop, length],
            [
                ptb.as_tensor_variable(cnf[0].start),
                ptb.as_tensor_variable(cnf[0].stop),
                ptb.as_tensor_variable(cnf[0].step),
                ptb.as_tensor_variable(cnf[1]),
            ],
        )

        length = 5
        a = np.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
                out = f(start, stop, length)
                t_out = a[out[0] : out[1] : out[2]][:: out[3]]
                v_out = a[start:stop:None]
                assert np.all(t_out == v_out)
                assert np.all(t_out.shape == v_out.shape)

    def test_start_stop_None(self):
        step = iscalar("s")
        length = iscalar("l")
        cnf = get_canonical_form_slice(slice(None, None, step), length)
        f = pytensor.function(
            [step, length],
            [
                ptb.as_tensor_variable(cnf[0].start),
                ptb.as_tensor_variable(cnf[0].stop),
                ptb.as_tensor_variable(cnf[0].step),
                ptb.as_tensor_variable(cnf[1]),
            ],
        )

        length = 5
        a = np.arange(length)
        for step in [-6, -3, -1, 2, 5]:
            out = f(step, length)
            t_out = a[out[0] : out[1] : out[2]][:: out[3]]
            v_out = a[None:None:step]
            assert np.all(t_out == v_out)
            assert np.all(t_out.shape == v_out.shape)

    def test_stop_step_None(self):
        start = iscalar("b")
        length = iscalar("l")
        cnf = get_canonical_form_slice(slice(start, None, None), length)
        f = pytensor.function(
            [start, length],
            [
                ptb.as_tensor_variable(cnf[0].start),
                ptb.as_tensor_variable(cnf[0].stop),
                ptb.as_tensor_variable(cnf[0].step),
                ptb.as_tensor_variable(cnf[1]),
            ],
        )

        length = 5
        a = np.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            out = f(start, length)
            t_out = a[out[0] : out[1] : out[2]][:: out[3]]
            v_out = a[start:None:None]
            assert np.all(t_out == v_out)
            assert np.all(t_out.shape == v_out.shape)

    def test_start_step_None(self):
        stop = iscalar("e")
        length = iscalar("l")
        cnf = get_canonical_form_slice(slice(None, stop, None), length)
        f = pytensor.function(
            [stop, length],
            [
                ptb.as_tensor_variable(cnf[0].start),
                ptb.as_tensor_variable(cnf[0].stop),
                ptb.as_tensor_variable(cnf[0].step),
                ptb.as_tensor_variable(cnf[1]),
            ],
        )

        length = 5
        a = np.arange(length)
        for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            out = f(stop, length)
            t_out = a[out[0] : out[1] : out[2]][:: out[3]]
            v_out = a[None:stop:None]
            assert np.all(t_out == v_out)
            assert np.all(t_out.shape == v_out.shape)


class TestSubtensor(utt.OptimizationTestMixin):
    """
    This is designed to be sub-classed (e.g. by the GPU tests).
    """

    def setup_method(self):
        self.shared = shared
        self.dtype = config.floatX
        mode = pytensor.compile.mode.get_default_mode()
        self.mode = mode.including(
            "local_replace_AdvancedSubtensor",
            "local_AdvancedIncSubtensor_to_AdvancedIncSubtensor1",
            "local_useless_subtensor",
        )
        self.fast_compile = config.mode == "FAST_COMPILE"

    def function(
        self,
        inputs,
        outputs,
        accept_inplace=False,
        op=None,
        mode=None,
        N=1,
        N_fast=None,
    ):
        """
        wrapper around pytensor.function that also check the output

        :param N: the number of op expected in the toposort
                  if tuple of length 2, (expected if fast_compile,
                                         if not fast_compile)
        """
        if self.fast_compile and N_fast is not None:
            N = N_fast
        if mode is None:
            mode = self.mode
        if op is None:
            op = Subtensor

        f = pytensor.function(inputs, outputs, mode=mode, accept_inplace=accept_inplace)
        self.assertFunctionContainsClassN(f, op, N)
        return f

    def eval_output_and_check(self, t, op_type=None, mode=None, length=1):
        if op_type is None:
            op_type = Subtensor
        if mode is None:
            mode = self.mode
        f = inplace_func([], t, mode=mode)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, DeepCopyOp)]
        assert len(topo_) == length
        if length == 1:
            assert isinstance(topo_[0].op, op_type)
        tval = f()
        return tval

    def test_err_invalid(self):
        # it is impossible to retrieve a view of a 0-d tensor
        n = self.shared(np.ones((), dtype=self.dtype))
        with pytest.raises(IndexError):
            n.__getitem__(0)

    @config.change_flags(compute_test_value="off")
    def test_err_bounds(self):
        n = self.shared(np.ones(3, dtype=self.dtype))
        t = n[7]
        assert isinstance(t.owner.op, Subtensor)
        # Silence expected error messages
        _logger = logging.getLogger("pytensor.graph.rewriting.basic")
        oldlevel = _logger.level
        _logger.setLevel(logging.CRITICAL)
        try:
            with pytest.raises(IndexError):
                self.eval_output_and_check(t)
        finally:
            _logger.setLevel(oldlevel)

    def test_err_subslice(self):
        n = self.shared(np.ones(3, dtype=self.dtype))
        with pytest.raises(Exception):
            n[slice(0, slice(1, 2, None), None)]

    def test_ok_range_finite(self):
        n = self.shared(np.arange(3, dtype=self.dtype))
        t = n[0:2]
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == (2,)
        assert (tval == [0, 1]).all()

    def test_ok_range_finite_2(self):
        n = self.shared(np.arange(12, dtype=self.dtype).reshape((3, 4)))
        # Also check negative index
        for idx in [(slice(0, 2), 3), ((slice(0, 2), -1)), (slice(0, 2), -4)]:
            t = n[idx]  # l]#0:2,3]
            assert isinstance(t.owner.op, Subtensor)
            tval = self.eval_output_and_check(t)
            assert tval.shape == (2,)
            assert np.allclose(tval, n.get_value()[idx])

    def test_0_dims(self):
        n = self.shared(np.ones((), dtype=self.dtype))
        t = Subtensor([])(n)
        assert isinstance(t.owner.op, Subtensor)
        self.eval_output_and_check(
            t,
            mode=self.mode.excluding("local_useless_subtensor", "local_useless_slice"),
        )

    def test_err_invalid_2(self):
        n = self.shared(np.ones(1, dtype=self.dtype))
        with pytest.raises(IndexError):
            n.__getitem__((0, 0))

    def test_ok_elem(self):
        n = self.shared(np.ones(1, dtype=self.dtype) * 5)
        t = n[0]
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == ()
        assert tval == 5.0

    def test_ok_range_infinite(self):
        n = self.shared(np.arange(3, dtype=self.dtype))
        t = n[1:]
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == (2,)
        assert (tval == [1.0, 2.0]).all()

    def test_ok_strided(self):
        n = self.shared(np.arange(5, dtype=self.dtype))
        t = n[1::2]
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == (2,)
        assert (tval == [1.0, 3.0]).all()

        t = n[0:-1:2]  # 0 to 1 from the end stepping by 2
        tval = self.eval_output_and_check(t)
        assert tval.shape == (2,)
        assert (tval == [0.0, 2.0]).all()

    @config.change_flags(compute_test_value="off")
    def test_err_bounds0(self):
        n = self.shared(np.ones((2, 3), dtype=self.dtype) * 5)
        for idx in [(0, 4), (0, -4)]:
            t = n[idx]
            assert isinstance(t.owner.op, Subtensor)
            # Silence expected warnings
            _logger = logging.getLogger("pytensor.graph.rewriting.basic")
            oldlevel = _logger.level
            _logger.setLevel(logging.CRITICAL)
            try:
                with pytest.raises(IndexError):
                    self.eval_output_and_check([t])
            finally:
                _logger.setLevel(oldlevel)

    @config.change_flags(compute_test_value="off")
    def test_err_bounds1(self):
        n = self.shared(np.ones((2, 3), dtype=self.dtype) * 5)
        t = n[4:5, 3]
        assert isinstance(t.owner.op, Subtensor)
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            with pytest.raises(IndexError):
                self.eval_output_and_check([t])
        finally:
            sys.stderr = old_stderr

    def test_ok_elem_2(self):
        n = self.shared(np.arange(6, dtype=self.dtype).reshape((2, 3)))
        t = n[0, 2]
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == ()
        assert np.all(tval == 2)

    def test_ok_row(self):
        n = self.shared(np.arange(6, dtype=self.dtype).reshape((2, 3)))
        t = n[1]
        assert not any(s == 1 for s in n.type.shape)
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == (3,)
        assert np.all(tval == [3, 4, 5])

    def test_ok_col(self):
        n = self.shared(np.arange(6, dtype=self.dtype).reshape((2, 3)))
        t = n[:, 0]
        assert isinstance(t.owner.op, Subtensor)
        assert not any(s == 1 for s in n.type.shape)
        tval = self.eval_output_and_check(t)
        assert tval.shape == (2,)
        assert np.all(tval == [0, 3])

    def test_ok_rows_finite(self):
        n = self.shared(np.arange(12, dtype=self.dtype).reshape((4, 3)))
        t = n[1:3, 0]
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == (2,)
        assert np.all(tval == [3, 6])

    def test_ok_cols_infinite(self):
        n = self.shared(np.arange(12, dtype=self.dtype).reshape((4, 3)))
        t = n[1, 2:]
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == (1,)
        assert np.all(tval == 5)

    def test_ok_strided_2(self):
        n = self.shared(np.arange(20, dtype=self.dtype).reshape((4, 5)))
        t = n[1:4:2, 1:5:2]
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == (2, 2)
        assert np.all(tval == [[6, 8], [16, 18]])

    def test_ok_mat(self):
        n = self.shared(np.arange(24, dtype=self.dtype).reshape((2, 3, 4)))
        t = n[0, 0, 0]
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == ()
        assert np.all(tval == 0)

    def test_long(self):
        n = self.shared(np.arange(12, dtype=self.dtype).reshape((4, 3)))
        t = n[1:4:2, 1]
        assert isinstance(t.owner.op, Subtensor)
        tval = self.eval_output_and_check(t)
        assert tval.shape == (2,)
        assert np.all(tval == [4, 10])

    def test_long_too_big(self):
        # Currently, we cast Python longs to int64 when used for indexing.
        # This test checks that using a long that does not fit raises an error.
        n = self.shared(np.arange(12, dtype=self.dtype).reshape((4, 3)))
        with pytest.raises(Exception):
            n[: (2**63)]

    def test_slice_symbol(self):
        x = self.shared(np.random.random((5, 4)).astype(self.dtype))
        y = self.shared(np.random.random((1, 2, 3)).astype(self.dtype))
        o = x[: y.shape[0], None, :]
        f = pytensor.function([], o, mode=self.mode)
        ret = f()
        assert ret.shape == (1, 1, 4)

    @pytest.mark.parametrize(
        "length, op_type_opt, slice_",
        [
            (0, Subtensor, np.index_exp[...]),
            (1, Subtensor, np.index_exp[..., 1]),
            (1, Subtensor, np.index_exp[1, ...]),
            (1, Subtensor, np.index_exp[..., 1, 2, 3]),
            (1, Subtensor, np.index_exp[1, ..., 2, 3]),
            (1, Subtensor, np.index_exp[1, 2, 3, ...]),
            (3, DimShuffle, np.index_exp[..., [0, 2, 3]]),
            (1, DimShuffle, np.index_exp[np.newaxis, ...]),
            (
                1,
                AdvancedSubtensor,
                np.index_exp[..., np.newaxis, [1, 2]],
            ),
        ],
    )
    def test_ellipsis(self, length, op_type_opt, slice_):
        numpy_n = np.arange(24, dtype=self.dtype).reshape((2, 3, 4))
        n = self.shared(numpy_n)
        numpy_tval = numpy_n[slice_]
        t = n[slice_]
        tval = self.eval_output_and_check(t, op_type=op_type_opt, length=length)
        assert tval.shape == numpy_tval.shape
        assert_array_equal(tval, numpy_tval)

    def test_boolean(self):
        def numpy_inc_subtensor(x, idx, a):
            x = x.copy()
            x[idx] += a
            return x

        test_array_np = np.arange(6, dtype=self.dtype).reshape((2, 3))
        test_array = self.shared(test_array_np)

        # indexing with a mask for some dimensions
        mask = np.array([True, False])
        val = self.eval_output_and_check(test_array[mask], op_type=AdvancedSubtensor)
        assert_array_equal(test_array_np[mask], val)
        val = self.eval_output_and_check(
            inc_subtensor(test_array[mask], 1), op_type=AdvancedIncSubtensor
        )
        assert_array_equal(numpy_inc_subtensor(test_array_np, mask, 1), val)
        assert_array_equal(
            numpy_inc_subtensor(test_array_np, mask, test_array_np[mask]),
            inc_subtensor(test_array[mask], test_array[mask]).eval(),
        )

        # test gradient
        utt.verify_grad(lambda m: m[mask], [test_array_np])
        utt.verify_grad(lambda m: inc_subtensor(m[mask], 1), [test_array_np])

        # indexing with a comparison (should translate to a boolean mask)
        assert_array_equal(
            test_array_np[test_array_np > 2], test_array[test_array > 2].eval()
        )
        assert_array_equal(
            test_array_np[[0], test_array_np[0] > 2],
            test_array[[0], test_array[0] > 2].eval(),
        )
        assert_array_equal(
            test_array_np[[1], test_array_np[0] > 2],
            test_array[[1], test_array[0] > 2].eval(),
        )

        # indexing with a mask for the second dimension
        mask = np.array([True, False, True])
        assert_array_equal(test_array_np[0, mask], test_array[0, mask].eval())
        assert_array_equal(test_array_np[:, mask], test_array[:, mask].eval())
        assert_array_equal(
            test_array_np[:, mask], test_array[:, self.shared(mask)].eval()
        )
        assert_array_equal(test_array_np[1:, mask], test_array[1:, mask].eval())
        assert_array_equal(test_array_np[:1, mask], test_array[:1, mask].eval())
        assert_array_equal(
            test_array_np[1:, mask, np.newaxis], test_array[1:, mask, np.newaxis].eval()
        )
        assert_array_equal(
            test_array_np[np.newaxis, 1:, mask], test_array[np.newaxis, 1:, mask].eval()
        )
        assert_array_equal(
            numpy_inc_subtensor(test_array_np, (0, mask), 1),
            inc_subtensor(test_array[(0, *mask.nonzero())], 1).eval(),
        )
        assert_array_equal(
            numpy_inc_subtensor(test_array_np, (0, mask), 1),
            inc_subtensor(test_array[0, mask], 1).eval(),
        )
        assert_array_equal(
            numpy_inc_subtensor(test_array_np, (slice(None), mask), 1),
            inc_subtensor(test_array[:, mask], 1).eval(),
        )

        # indexing with a boolean ndarray
        mask = np.array([[True, False, True], [False, False, True]])
        assert_array_equal(test_array_np[mask], test_array[mask].eval())
        assert_array_equal(test_array_np[mask], test_array[self.shared(mask)].eval())
        assert_array_equal(
            numpy_inc_subtensor(test_array_np, mask, 1),
            inc_subtensor(test_array[mask], 1).eval(),
        )

        # indexing with ellipsis
        numpy_n4 = np.arange(48, dtype=self.dtype).reshape((2, 3, 4, 2))
        n4 = self.shared(numpy_n4)
        assert_array_equal(
            numpy_n4[test_array_np > 2, ...], n4[test_array > 2, ...].eval()
        )
        assert_array_equal(
            numpy_n4[test_array_np > 2, ..., 1], n4[test_array > 2, ..., 1].eval()
        )
        assert_array_equal(
            numpy_n4[test_array_np > 2, ..., 0, 1], n4[test_array > 2, ..., 0, 1].eval()
        )
        assert_array_equal(
            numpy_inc_subtensor(numpy_n4, (test_array_np > 2, Ellipsis), 1),
            inc_subtensor(n4[test_array > 2, ...], 1).eval(),
        )
        assert_array_equal(
            numpy_inc_subtensor(numpy_n4, (test_array_np > 2, Ellipsis, 1), 1),
            inc_subtensor(n4[test_array > 2, ..., 1], 1).eval(),
        )
        assert_array_equal(
            numpy_inc_subtensor(numpy_n4, (test_array_np > 2, Ellipsis, 0, 1), 1),
            inc_subtensor(n4[test_array > 2, ..., 0, 1], 1).eval(),
        )

        with config.change_flags(compute_test_value="off"):
            # the boolean mask should have the correct shape
            # - too large, padded with True
            mask = np.array([True, False, True])
            with pytest.raises(IndexError):
                test_array[mask].eval()
            with pytest.raises(IndexError):
                test_array[mask, ...].eval()
            with pytest.raises(IndexError):
                inc_subtensor(test_array[mask], 1).eval()
            with pytest.raises(IndexError):
                inc_subtensor(test_array[mask, ...], 1).eval()
            mask = np.array([[True, False, False, True], [False, True, False, True]])
            with pytest.raises(IndexError):
                test_array[mask].eval()
            with pytest.raises(IndexError):
                inc_subtensor(test_array[mask], 1).eval()
            # - too large, padded with False (this works in NumPy < 0.13.0)
            mask = np.array([True, False, False])
            with pytest.raises(IndexError):
                test_array[mask].eval()
            with pytest.raises(IndexError):
                test_array[mask, ...].eval()
            with pytest.raises(IndexError):
                inc_subtensor(test_array[mask], 1).eval()
            with pytest.raises(IndexError):
                inc_subtensor(test_array[mask, ...], 1).eval()
            mask = np.array([[True, False, False, False], [False, True, False, False]])
            with pytest.raises(IndexError):
                test_array[mask].eval()
            with pytest.raises(IndexError):
                inc_subtensor(test_array[mask], 1).eval()
            # - mask too small (this works in NumPy < 0.13.0)
            mask = np.array([True])
            with pytest.raises(IndexError):
                test_array[mask].eval()
            with pytest.raises(IndexError):
                test_array[mask, ...].eval()
            with pytest.raises(IndexError):
                inc_subtensor(test_array[mask], 1).eval()
            with pytest.raises(IndexError):
                inc_subtensor(test_array[mask, ...], 1).eval()
            mask = np.array([[True], [True]])
            with pytest.raises(IndexError):
                test_array[mask].eval()
            with pytest.raises(IndexError):
                inc_subtensor(test_array[mask], 1).eval()
            # - too many dimensions
            mask = np.array([[[True, False, False], [False, True, False]]])
            with pytest.raises(IndexError):
                test_array.__getitem__(mask)
            with pytest.raises(IndexError):
                test_array.__getitem__(mask)

            # special cases: Python bools and bools nested in Python arrays are not supported
            with pytest.raises(TypeError):
                test_array.__getitem__((True,))
            with pytest.raises(TypeError):
                test_array.__getitem__((False,))
            with pytest.raises(TypeError):
                test_array.__getitem__((True, False))
            with pytest.raises(TypeError):
                test_array.__getitem__(([0, 1], [0, False]))
            with pytest.raises(TypeError):
                test_array.__getitem__(([0, 1], [0, pytensor.shared(True)]))

    def test_grad_1d(self):
        subi = 0
        data = np.asarray(random(2, 3), dtype=self.dtype)
        n = self.shared(data)
        z = scal.constant(subi).astype("int32")
        t = n[z:, z]
        gn = pytensor.grad(pt_sum(exp(t)), n)

        f = inplace_func([], gn, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, DeepCopyOp)]
        if not self.fast_compile:
            assert len(topo_) == 6
        assert any(isinstance(node.op, IncSubtensor) for node in topo_)
        assert any(isinstance(node.op, Subtensor) for node in topo_)
        gval = f()

        good = np.zeros_like(data)
        good[subi:, subi] = np.exp(data[subi:, subi])
        assert np.allclose(gval, good), (gval, good)

    def test_grad_2d_inc_set_subtensor(self):
        for n_shape, m_shape in [
            [(2, 3), (2, 2)],
            [(3, 2), (2, 2)],
            [(3, 2), (1, 2)],
            [(3, 2), (2,)],
        ]:
            for op in [inc_subtensor, set_subtensor]:
                subi = 2
                data = np.asarray(random(*n_shape), dtype=self.dtype)
                n = self.shared(data)
                z = scal.constant(subi)
                m = matrix("m", dtype=self.dtype)
                mv = np.asarray(random(*m_shape), dtype=self.dtype)

                t = op(n[:z, :z], m)
                gn, gm = pytensor.grad(pt_sum(t), [n, m])
                utt.verify_grad(lambda m: op(n[:z, :z], m), [mv], mode=self.mode)
                utt.verify_grad(lambda nn: op(nn[:z, :z], mv), [data], mode=self.mode)

    def test_grad_0d(self):
        data = np.asarray(random(2, 3), dtype=self.dtype)
        n = self.shared(data)
        t = n[1, 0]
        gn = pytensor.grad(pt_sum(exp(t)), n)
        f = self.function([], gn)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, DeepCopyOp)]
        if not self.fast_compile:
            assert len(topo_) == 6
        assert sum(isinstance(node.op, IncSubtensor) for node in topo_) == 1
        assert sum(isinstance(node.op, Subtensor) for node in topo_) == 1

        gval = f()
        good = np.zeros_like(data)
        good[1, 0] = np.exp(data[1, 0])
        assert np.allclose(gval, good), (gval, good)

    def test_ok_list(self):
        for data, idx in [
            (random(4), [1, 0]),
            (random(4, 5), [2, 3, -1]),
            (random(4, 2, 3), [0, 3]),
            (random(4, 2, 3), [3, 3, 1, 1, 2, 2, 0, 0]),
            (random(4, 2, 3), [3, 3, 1, 1, 2, 2, 0, 0, -1, -2, -3, -4]),
            # Test 4 dims as gpu code use another algo
            # in that case This new algo is not as much
            # optimized for that case.
            (random(4, 4, 2, 3), [3, 3, 1, 1, 2, 2, 0, 0, -1, -2, -3, -4]),
            # Test with TensorConstant index.
            (random(4, 2, 3), ptb.constant([3, 3, 1, 1, 2, 2, 0, 0])),
        ]:
            data = np.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[idx]

            val = self.eval_output_and_check(t, op_type=AdvancedSubtensor1)
            if isinstance(idx, list):
                good = data[idx]
            else:
                good = data[idx.data]
            assert val.ndim == data.ndim
            assert np.allclose(val, good), (val, good)

            # Test reuse of output memory
            if type(AdvancedSubtensor1) is AdvancedSubtensor1:
                op = AdvancedSubtensor1()
                # When idx is a TensorConstant.
                if hasattr(idx, "data"):
                    idx = idx.data
                test_out = [[None]]
                op.perform(None, [data, idx], test_out)
                out1 = test_out[0][0]
                op.perform(None, [data, idx], test_out)
                out2 = test_out[0][0]
                assert out1 is out2

            # test the grad
            gn = pytensor.grad(t.sum(), n)
            g = self.function([], gn, op=AdvancedIncSubtensor1)
            utt.verify_grad(
                lambda m: m[[1, 3]],
                [np.random.random((5, 5)).astype(self.dtype)],
                mode=self.mode,
            )
            g()
            utt.verify_grad(lambda m: m[idx], [data], mode=self.mode)

    def test_noncontiguous_idx(self):
        data = random(4, 2, 3)
        idx = [2, 2, 0, 0, 1, 1]
        n = self.shared(data)
        t = n[self.shared(np.asarray(idx).astype("int64"))[::2]]
        val = self.eval_output_and_check(t, op_type=AdvancedSubtensor1, length=2)
        utt.assert_allclose(data[idx[::2]], val)

    def test_err_invalid_list(self):
        n = self.shared(np.asarray(5, dtype=self.dtype))
        with pytest.raises(IndexError):
            n.__getitem__([0, 0])

    def test_err_invalid_2list_dtype(self):
        n = self.shared(np.ones((3, 3), dtype=self.dtype) * 5)
        with pytest.raises(TypeError):
            n.__getitem__(([0.0, 0], [1, 1]))

    def test_err_bound_list(self):
        n = self.shared(np.ones((2, 3), dtype=self.dtype) * 5)
        l = lvector()
        t = n[l]

        f = self.function([l], t, op=AdvancedSubtensor1)

        g = self.function(
            [l],
            inc_subtensor(t, np.asarray([[1.0]], self.dtype)),
            op=AdvancedIncSubtensor1,
        )

        for shp in [[0, 4], [0, -3], [-10]]:
            with pytest.raises(IndexError):
                f(shp)
            with pytest.raises(IndexError):
                g(shp)

    def test_adv_sub1_broadcast(self):
        v = np.arange(3, dtype=self.dtype).reshape((1, 3))
        n = self.shared(v * 5, shape=(1, None))
        idx = lvector()
        t = n[idx]

        f = self.function([idx], t, op=AdvancedSubtensor1)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, DeepCopyOp)]
        assert len(topo_) == 1
        assert isinstance(topo_[0].op, AdvancedSubtensor1)
        f_0 = f([0])
        assert f_0.shape == (1, 3)
        assert np.allclose(f_0, v * 5)
        f_00 = f([0, 0])
        assert f_00.shape == (2, 3)
        assert np.allclose(f_00, v * 5)
        with pytest.raises(IndexError):
            f([0, 1])

        # Test the gradient
        c = t.sum()
        gn = pytensor.grad(c, n)
        g = self.function([idx], gn, op=AdvancedIncSubtensor1)
        g_0 = g([0])
        assert g_0.shape == (1, 3)
        assert np.allclose(g_0, 1)
        g_00 = g([0, 0])
        assert g_00.shape == (1, 3)
        assert np.allclose(g_00, 2)

        utt.verify_grad(
            lambda m: m[[1, 3]],
            [np.random.random((5, 5)).astype(self.dtype)],
            mode=self.mode,
        )

        def fun(x, y):
            return advanced_inc_subtensor1(x, y, [1, 3])

        utt.verify_grad(
            fun,
            [
                np.random.random((5, 5)).astype(self.dtype),
                np.random.random((2, 5)).astype(self.dtype),
            ],
            mode=self.mode,
        )

        def fun(x, y):
            return advanced_set_subtensor1(x, y, [1, 3])

        utt.verify_grad(
            fun,
            [
                np.random.random((5, 5)).astype(self.dtype),
                np.random.random((2, 5)).astype(self.dtype),
            ],
            mode=self.mode,
        )

        # test set_subtensor broadcast
        self.dtype = "float32"

        x = tensor4("x", dtype=self.dtype)
        indexes = pytensor.shared(np.int32([1, 2, 3, 4]))
        W = self.shared(np.random.random((10, 10, 3, 3)).astype(self.dtype))

        h = x + W
        h = set_subtensor(h[indexes], h[indexes])
        g = pytensor.grad(h.sum(), W)
        N = 2
        if (
            config.mode == "FAST_COMPILE"
            and AdvancedIncSubtensor1 is AdvancedIncSubtensor1
        ):
            N = 3
        f = self.function([x], g, op=AdvancedIncSubtensor1, N=N)

        f(np.random.random((10, 10, 3, 3)).astype(self.dtype))

    def test_adv_sub1_idx_broadcast(self):
        # The idx can be a broadcastable vector.
        ones = np.ones((4, 3), dtype=self.dtype)
        n = self.shared(ones * 5)
        idx = TensorType(dtype="int64", shape=(1,))()
        assert idx.type.shape == (1,)
        t = n[idx]

        f = self.function([idx], t, op=AdvancedSubtensor1)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, DeepCopyOp)]
        assert len(topo_) == 1
        assert isinstance(topo_[0].op, AdvancedSubtensor1)
        f_0 = f([0])
        assert f_0.shape == (1, 3)
        assert np.allclose(f_0, 5)

        # Test the gradient
        c = t.sum()
        gn = pytensor.grad(c, n)
        g = self.function([idx], gn, op=AdvancedIncSubtensor1)
        g_0 = g([0])
        assert g_0.shape == (4, 3)
        assert np.allclose(g_0[0], 1)
        assert np.allclose(g_0[1:], 0)

    @pytest.mark.slow
    def test_shape_i_const(self):
        # Each axis is treated independently by shape_i/shape operators

        mode_opt = self.mode
        data = self.shared(np.array(np.arange(5), dtype=self.dtype))
        for start in [None, -8, -5, -1, 0, 1, 5, 8]:
            outs = []
            shapes = []
            for stop in [None, -8, -5, -1, 0, 1, 5, 8]:
                for step in [None, -3, -1, 2]:
                    outs += [data[start:stop:step].shape]
                    shapes += [data.get_value(borrow=True)[start:stop:step].shape]
            f = self.function([], outs, mode=mode_opt, op=subtensor_ops, N=0)
            t_shapes = f()
            for t_shape, shape in zip(t_shapes, shapes, strict=True):
                assert np.all(t_shape == shape)
            assert Subtensor not in [x.op for x in f.maker.fgraph.toposort()]

    def test_shape_i_scalar(self):
        # Each axis is treated independently by shape_i/shape operators

        mode_opt = self.mode

        v_data = np.array(np.arange(5), dtype=self.dtype)
        t_data = self.shared(v_data)
        start = iscalar("b")
        stop = iscalar("e")
        step = iscalar("s")
        f = self.function(
            [start, stop, step],
            t_data[start:stop:step].shape,
            mode=mode_opt,
            op=subtensor_ops,
            N=0,
        )
        assert Subtensor not in [x.op for x in f.maker.fgraph.toposort()]
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
                for step in [-3, -1, 2, 5]:
                    assert np.all(f(start, stop, step) == v_data[start:stop:step].shape)

    def grad_list_(self, idxs, data):
        n = self.shared(data)

        for idx in idxs:
            idx_np = np.asarray(idx)
            idx_pt = shared(idx_np, shape=(1 if idx_np.shape[0] == 1 else None,))
            t = n[idx_pt]
            gn = pytensor.grad(pt_sum(exp(t)), n)
            f = self.function([], [gn, gn.shape], op=AdvancedIncSubtensor1)
            topo = f.maker.fgraph.toposort()
            if not self.fast_compile:
                assert any(
                    isinstance(node.op, AdvancedIncSubtensor1) and node.op.inplace
                    for node in topo
                )
            else:
                assert any(isinstance(node.op, AdvancedIncSubtensor1) for node in topo)
            assert any(isinstance(node.op, AdvancedSubtensor1) for node in topo)
            gval, gshape = f()
            good = np.zeros_like(data)
            # don't work when the same index is used many time
            # good[idx] += np.exp(data[idx])
            for i in idx:
                good[i] += np.exp(data[i])
            assert gval.ndim == data.ndim
            assert np.allclose(gval, good), (gval, good)
            assert np.allclose(gshape, data.shape)

            def fct(t):
                return pt_sum(t[idx_pt])

            utt.verify_grad(fct, [data], mode=self.mode)

            # Test the grad of the grad (e.i. AdvancedIncSubtensor1.grad)
            def fct2(t):
                return pytensor.grad(pt_sum(t[idx_pt]), t)

            utt.verify_grad(fct2, [data], mode=self.mode)

            # Test shape of AdvancedIncSubtensor1 and AdvancedSubtensor1
            if not self.fast_compile:
                ops = (AdvancedIncSubtensor1, AdvancedSubtensor1)
            else:
                ops = subtensor_ops
            if idx is idxs[0]:
                # TODO FIXME: This is a very poorly specified test.
                f = self.function(
                    [], [gn.shape, n[idx_pt].shape], op=ops, N=0, N_fast=0
                )
                f()

    def test_wrong_exception_regression(self):
        a = fscalar()
        b = fscalar()
        c = vector()
        with pytest.raises(TypeError):
            c[a:b]
        with pytest.raises(TypeError):
            c[a:]
        with pytest.raises(TypeError):
            c[:b]

    @pytest.mark.slow
    def test_grad_list(self):
        data = random(4)
        data = np.asarray(data, dtype=self.dtype)
        idxs = [[i] for i in range(data.shape[0])]
        idxs.extend(
            [i, j, (i + 1) % data.shape[0]]
            for i in range(data.shape[0])
            for j in range(0, data.shape[0], 2)
        )
        self.grad_list_(idxs, data)

        data = random(4, 3)
        data = np.asarray(data, dtype=self.dtype)
        self.grad_list_(idxs, data)

        data = random(4, 3, 2)
        data = np.asarray(data, dtype=self.dtype)
        self.grad_list_(idxs, data)

    def test_shape_list(self):
        # TODO for all type of subtensor shape
        for data, idx in [
            (random(4), [1, 0]),
            (random(4, 2), [2, 3]),
            (random(4, 2, 3), [0, 3]),
            (
                random(4, 2, 3),
                [
                    3,
                    3,
                    1,
                    2,
                    2,
                ],
            ),
        ]:
            data = np.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[idx]
            f = self.function([], t.shape, op=subtensor_ops, N=0, N_fast=0)
            val = f()
            assert np.allclose(val, data[idx].shape)

    def test_inc_and_set_subtensor(self):
        # Test increment and set with broadcast

        X = self.shared(np.ones((9, 9)).astype(self.dtype))
        y = set_subtensor(X[1::, 1::], 0)
        f = self.function([], [y], op=IncSubtensor, N=1)
        out = f()

        res = np.ones((9, 9))
        res[1::, 1::] = 0
        assert np.allclose(out, res)

    def test_advanced1_inc_and_set(self):
        # Test advanced increment and set.

        rng = np.random.default_rng(seed=utt.fetch_seed())
        all_inputs_var = []
        all_inputs_num = []
        all_outputs_var = []
        all_outputs_num = []
        all_params = []
        for set_instead_of_inc in (False, True):
            for inplace in (False, True):
                for data_shape in ((10,), (4, 5), (1, 2, 3), (4, 5, 6, 7)):
                    data_n_dims = len(data_shape)
                    data_size = np.prod(data_shape)
                    # Corresponding numeric variable.
                    data_num_init = np.arange(data_size, dtype=self.dtype)
                    data_num_init = data_num_init.reshape(data_shape)
                    inc_shapes = [data_shape[i:] for i in range(0, len(data_shape) + 1)]
                    for inc_shape in inc_shapes:
                        # We copy the numeric value to be 100% sure there is no
                        # risk of accidentally sharing it.
                        data_num = data_num_init.copy()
                        # Symbolic variable to be incremented.
                        # We create a new one every time in order not to
                        # have duplicated variables in the function's inputs
                        data_var = TensorType(
                            shape=(None,) * data_n_dims, dtype=self.dtype
                        )()
                        # Symbolic variable with rows to be incremented.
                        idx_var = vector(dtype="int64")
                        n_to_inc = rng.integers(data_shape[0])
                        if (
                            n_to_inc == 1
                            and len(inc_shape) > 0
                            and inc_shape[0] == 1
                            and data_shape[0] > 1
                        ):
                            n_to_inc = 2
                        # Corresponding numeric variable.
                        # If set_instead_of_inc, we want to avoid repeating
                        # indices, as the order is not guaranteed.
                        idx_num = rng.choice(
                            np.arange(data_shape[0]),
                            n_to_inc,
                            replace=(not set_instead_of_inc),
                        )
                        idx_num = idx_num.astype("int64")

                        # Trick for the case where `inc_shape` is the same as
                        # `data_shape`: what we actually want is the first
                        # shape element to be equal to the number of rows to
                        # increment.
                        if len(inc_shape) == len(data_shape) and (
                            len(inc_shapes) == 0 or inc_shape[0] != 1
                        ):
                            inc_shape = (n_to_inc,) + inc_shape[1:]

                        # Symbolic variable with increment value.
                        inc_var_static_shape = tuple(
                            1 if dim_length == 1 else None for dim_length in inc_shape
                        )
                        inc_var = TensorType(
                            shape=inc_var_static_shape, dtype=self.dtype
                        )()

                        # The param dtype is needed when inc_shape is empty.
                        # By default, it would return a float and rng.uniform
                        # with NumPy 1.10 will raise a Deprecation warning.
                        inc_size = np.prod(inc_shape, dtype="int")
                        # Corresponding numeric variable.
                        inc_num = rng.uniform(size=inc_size).astype(self.dtype)
                        inc_num = inc_num.reshape(inc_shape)
                        # Result of the incrementation.
                        # (i) PyTensor
                        if set_instead_of_inc:
                            op = set_subtensor
                        else:
                            op = inc_subtensor
                        output = op(data_var[idx_var], inc_var, inplace=inplace)
                        # (ii) Numpy (note that Numpy increments only once
                        # duplicated indices, so we cannot directly use +=).
                        data_copy = data_num.copy()
                        for j, idx in enumerate(idx_num):
                            if len(inc_shape) == len(data_shape):
                                if inc_shape[0] == 1:
                                    # Allow broadcasting of y[0]
                                    inc_num0 = inc_num[0]
                                    if set_instead_of_inc:
                                        data_copy[idx] = inc_num0
                                    else:
                                        data_copy[idx] += inc_num0
                                else:
                                    # Special case where there is no broadcasting.
                                    if set_instead_of_inc:
                                        data_copy[idx] = inc_num[j]
                                    else:
                                        data_copy[idx] += inc_num[j]
                            else:
                                if set_instead_of_inc:
                                    data_copy[idx] = inc_num
                                else:
                                    data_copy[idx] += inc_num
                        data_var = In(data_var, mutable=True)

                        # Remember data for the PyTensor function (see below).
                        all_inputs_var += [data_var, idx_var, inc_var]
                        all_inputs_num += [data_num, idx_num, inc_num]
                        all_outputs_var.append(output)
                        all_outputs_num.append(data_copy)
                        all_params.append(
                            (set_instead_of_inc, inplace, data_shape, inc_shape)
                        )

        # Actual test (we compile a single PyTensor function to make it faster).
        f = self.function(
            all_inputs_var,
            all_outputs_var,
            accept_inplace=True,
            op=AdvancedIncSubtensor1,
            N=len(all_outputs_var),
        )

        f_outs = f(*all_inputs_num)
        assert len(f_outs) == len(all_outputs_num)
        for params, f_out, output_num in zip(
            all_params, f_outs, all_outputs_num, strict=True
        ):
            # NB: if this assert fails, it will probably be easier to debug if
            # you enable the debug code above.
            assert np.allclose(f_out, output_num), (params, f_out, output_num)

    @pytest.mark.skipif(
        version.parse(np.__version__) < version.parse("2.0"),
        reason="Legacy C-implementation did not check for runtime broadcast",
    )
    @pytest.mark.parametrize("func", (advanced_inc_subtensor1, advanced_set_subtensor1))
    def test_advanced1_inc_runtime_broadcast(self, func):
        y = matrix("y", dtype="float64", shape=(None, None))

        x = ptb.zeros((10, 5))
        idxs = np.repeat(np.arange(10), 2)
        out = func(x, y, idxs)

        f = function([y], out)
        f(np.ones((20, 5)))  # Fine
        with pytest.raises(
            ValueError,
            match="Runtime broadcasting not allowed. AdvancedIncSubtensor1 was asked",
        ):
            f(np.ones((1, 5)))
        with pytest.raises(
            ValueError,
            match="Runtime broadcasting not allowed. AdvancedIncSubtensor1 was asked",
        ):
            f(np.ones((20, 1)))

    def test_adv_constant_arg(self):
        # Test case provided (and bug detected, gh-607) by John Salvatier
        m = matrix("m")
        gv = np.array([0, 1, 3])
        g = ptb.constant(gv)
        i = lvector("i")

        # s1 used to fail
        s1 = m[gv, i]
        s2 = m[g, i]

        assert is_same_graph(s1, s2)

    def test_adv1_inc_sub_notlastdim(self):
        # Test that taking 1-dimensional advanced indexing
        # over a dimension that's not the first (outer-most) works.
        m = matrix("m")
        i = lvector("i")

        m1 = set_subtensor(m[:, i], 0)
        m2 = inc_subtensor(m[:, i], 1)
        f = pytensor.function([m, i], [m1, m2], mode=self.mode)

        m_val = random(3, 5)
        i_val = integers_ranged(min=0, max=4, shape=(4,))
        m1_ref = m_val.copy()
        m2_ref = m_val.copy()

        m1_val, m2_val = f(m_val, i_val)
        for idx in i_val:
            m1_ref[:, idx] = 0
            m2_ref[:, idx] += 1

        assert np.allclose(m1_val, m1_ref), (m1_val, m1_ref)
        assert np.allclose(m2_val, m2_ref), (m2_val, m2_ref)

    def test_adv1_inc_sub_notlastdim_2didx(self):
        # Test that taking 1-dimensional advanced indexing
        # over a dimension that's not the first (outer-most) works,
        # if the index is a matrix.
        m = matrix("m")
        i = lmatrix("i")

        m1 = set_subtensor(m[:, i], 0)
        m2 = inc_subtensor(m[:, i], 1)

        f = pytensor.function([m, i], [m1, m2], mode=self.mode)

        m_val = random(5, 7)
        i_val = integers_ranged(min=0, max=6, shape=(4, 2))
        m1_ref = m_val.copy()
        m2_ref = m_val.copy()

        m1_val, m2_val = f(m_val, i_val)
        for idx in i_val.ravel():
            m1_ref[:, idx] = 0
            m2_ref[:, idx] += 1

        assert np.allclose(m1_val, m1_ref), (m1_val, m1_ref)
        assert np.allclose(m2_val, m2_ref), (m2_val, m2_ref)

    def test_adv1_inc_sub_notlastdim_1_2dval_broadcast(self):
        # Test that taking 1-dimensional advanced indexing
        # over a dimension that's not the first (outer-most),
        # and incrementing/setting with broadcast
        m = matrix("m")

        # Test for both vector and matrix as index
        sym_i = (lvector("i"), lmatrix("i"))
        shape_i = ((4,), (4, 2))
        shape_val = ((3, 1), (3, 1, 1))

        for i, shp_i, shp_v in zip(sym_i, shape_i, shape_val, strict=True):
            sub_m = m[:, i]
            m1 = set_subtensor(sub_m, np.zeros(shp_v))
            m2 = inc_subtensor(sub_m, np.ones(shp_v))
            f = pytensor.function([m, i], [m1, m2], mode=self.mode)

            m_val = random(3, 5)
            i_val = integers_ranged(min=0, max=4, shape=shp_i)
            m1_ref = m_val.copy()
            m2_ref = m_val.copy()

            m1_val, m2_val = f(m_val, i_val)
            for idx in i_val.ravel():
                m1_ref[:, idx] = 0
                m2_ref[:, idx] += 1

            assert np.allclose(m1_val, m1_ref), (m1_val, m1_ref)
            assert np.allclose(m2_val, m2_ref), (m2_val, m2_ref)

    def test_adv1_inc_sub_notlastdim_1_2dval_no_broadcast(self):
        # Test that taking 1-dimensional advanced indexing
        # over a dimension that's not the first (outer-most),
        # and incrementing/setting without broadcast
        m = matrix("m")

        # Test for both vector and matrix as index
        sym_i = (lvector("i"), lmatrix("i"))
        shape_i = ((4,), (4, 2))
        shape_val = ((3, 4), (3, 4, 2))

        for i, shp_i, shp_v in zip(sym_i, shape_i, shape_val, strict=True):
            sub_m = m[:, i]
            m1 = set_subtensor(sub_m, np.zeros(shp_v))
            m2 = inc_subtensor(sub_m, np.ones(shp_v))
            f = pytensor.function([m, i], [m1, m2], mode=self.mode)

            m_val = random(3, 5)
            i_val = integers_ranged(min=0, max=4, shape=shp_i)
            m1_ref = m_val.copy()
            m2_ref = m_val.copy()

            m1_val, m2_val = f(m_val, i_val)
            # We have to explicitly loop over all individual indices,
            # not as a list or array, numpy only increments the indexed
            # elements once even if the indices are repeated.
            for idx in i_val.ravel():
                m1_ref[:, idx] = 0
                m2_ref[:, idx] += 1

            assert np.allclose(m1_val, m1_ref), (m1_val, m1_ref)
            assert np.allclose(m2_val, m2_ref), (m2_val, m2_ref)


def test_take_basic():
    with pytest.raises(TypeError):
        take(matrix(), lvector(), axis=lscalar())


@pytest.mark.parametrize(
    "a_shape, index, axis, mode",
    [
        ((4, 5, 6), np.array([[1, 2, 3], [1, 2, 3]]), -1, None),
        ((4, 5, 6), np.array([[1, 2, 3], [5, 6, 7]]), None, None),
        ((4, 5, 6), np.array([[1, 2, 3], [1, 2, 3]]), 1, None),
        ((4, 5, 6), np.array([[1, 2, 3], [5, 6, 7]]), 1, "clip"),
        ((4, 5, 6), np.array([[1, 2, 3], [5, 6, 7]]), 1, "wrap"),
    ],
)
def test_take_cases(a_shape, index, axis, mode):
    a_val = np.random.random(size=a_shape).astype(config.floatX)
    py_res = a_val.take(index, axis=axis, mode=mode)

    a = ptb.as_tensor_variable(a_val)
    index = ptb.as_tensor_variable(index)

    f = pytensor.function([], a.take(index, axis=axis, mode=mode))
    f_res = f()
    assert np.array_equal(py_res, f_res)


class TestIncSubtensor:
    """
    Partial testing.

    What could be tested:
    - increment vs set
    - thing incremented: scalar, vector, matrix,
    - increment/set: constant, scalar, vector, matrix
    - indices: scalar vs slice, constant vs variable, out of bound, ...
    - inplace

    NOTE: these are the same tests as test_incsubtensor.py, but using
    the new (read: not deprecated) inc_subtensor, set_subtensor
    functions.
    """

    def test_simple_2d(self):
        # Increments or sets part of a tensor by a scalar using full slice and
        # a partial slice depending on a scalar.

        a = dmatrix()
        increment = dscalar()
        sl1 = slice(None)
        sl2_end = lscalar()
        sl2 = slice(sl2_end)

        for do_set in [False, True]:
            if do_set:
                result = set_subtensor(a[sl1, sl2], increment)
            else:
                result = inc_subtensor(a[sl1, sl2], increment)

            f = pytensor.function([a, increment, sl2_end], result)

            val_a = np.ones((5, 5))
            val_inc = 2.3
            val_sl2_end = 2

            result = f(val_a, val_inc, val_sl2_end)

            expected_result = np.copy(val_a)
            if do_set:
                expected_result[:, :val_sl2_end] = val_inc
            else:
                expected_result[:, :val_sl2_end] += val_inc

            utt.assert_allclose(result, expected_result)

    def test_wrong_dims(self):
        a = matrix()
        increment = matrix()
        index = 0

        with pytest.raises(TypeError):
            set_subtensor(a[index], increment)
        with pytest.raises(TypeError):
            inc_subtensor(a[index], increment)

    def test_wrong_broadcast(self):
        a = col()
        increment = vector()

        # These symbolic graphs legitimate, as long as increment has exactly
        # one element. So it should fail at runtime, not at compile time.
        rng = np.random.default_rng(utt.fetch_seed())

        def rng_randX(*shape):
            return rng.random(shape).astype(pytensor.config.floatX)

        for op in (set_subtensor, inc_subtensor):
            for base in (a[:], a[0]):
                out = op(base, increment)
                f = pytensor.function([a, increment], out)
                # This one should work
                f(rng_randX(3, 1), rng_randX(1))
                # These ones should not
                with pytest.raises(AssertionError):
                    f(rng_randX(3, 1), rng_randX(2))
                with pytest.raises(AssertionError):
                    f(rng_randX(3, 1), rng_randX(3))
                with pytest.raises(AssertionError):
                    f(rng_randX(3, 1), rng_randX(0))

    def test_simple_3d(self):
        # Increments or sets part of a tensor by a scalar using full slice and
        # a partial slice depending on a scalar.

        a = dtensor3()
        increment = dscalar()
        sl1 = slice(None)
        sl2_end = lscalar()
        sl2 = slice(sl2_end)
        sl3 = 2

        val_a = np.ones((5, 3, 4))
        val_inc = 2.3
        val_sl2_end = 2

        for method in [set_subtensor, inc_subtensor]:
            result = method(a[sl1, sl3, sl2], increment)
            f = pytensor.function([a, increment, sl2_end], result)

            expected_result = np.copy(val_a)
            result = f(val_a, val_inc, val_sl2_end)

            if method is set_subtensor:
                expected_result[:, sl3, :val_sl2_end] = val_inc
            else:
                expected_result[:, sl3, :val_sl2_end] += val_inc

            utt.assert_allclose(result, expected_result)

            # Test when we broadcast the result
            result = method(a[sl1, sl2], increment)

            f = pytensor.function([a, increment, sl2_end], result)

            expected_result = np.copy(val_a)
            result = f(val_a, val_inc, val_sl2_end)

            if method is set_subtensor:
                expected_result[:, :val_sl2_end] = val_inc
            else:
                expected_result[:, :val_sl2_end] += val_inc

            utt.assert_allclose(result, expected_result)

    def test_grad_inc_set(self):
        def inc_slice(*s):
            def just_numeric_args(a, b):
                return inc_subtensor(a[s], b)

            return just_numeric_args

        def set_slice(*s):
            def just_numeric_args(a, b):
                return set_subtensor(a[s], b)

            return just_numeric_args

        for f_slice in [inc_slice, set_slice]:
            # vector
            utt.verify_grad(
                f_slice(slice(2, 4, None)),
                (
                    np.asarray([0, 1, 2, 3, 4, 5.0]),
                    np.asarray([9, 9.0]),
                ),
            )

            # matrix
            utt.verify_grad(
                f_slice(slice(1, 2, None), slice(None, None, None)),
                (
                    np.asarray([[0, 1], [2, 3], [4, 5.0]]),
                    np.asarray([[9, 9.0]]),
                ),
            )

            # single element
            utt.verify_grad(
                f_slice(2, 1),
                (
                    np.asarray([[0, 1], [2, 3], [4, 5.0]]),
                    np.asarray(9.0),
                ),
            )

            # broadcast
            utt.verify_grad(
                f_slice(2),
                (
                    np.asarray([[0, 1], [2, 3], [4, 5.0]]),
                    np.asarray(9.0),
                ),
            )

            # Broadcastable leading dim
            utt.verify_grad(
                f_slice(slice(None, None), slice(1, 3)),
                (
                    np.asarray([0, 1, 2, 3, 4, 5.0])[None, ...],
                    np.asarray([9, 9.0]),
                ),
            )

    def test_grad_broadcastable_specialization(self):
        # Make sure gradient does not fail when gx has a more precise static_shape after indexing.
        # This is a regression test for a bug reported in
        # https://discourse.pymc.io/t/marginalized-mixture-wont-begin-sampling-throws-assertion-error/15969

        x = vector("x")  # Unknown write time shape = (2,)
        out = x.zeros_like()

        # Update a subtensor of unknown write time shape = (1,)
        out = out[1:].set(exp(x[1:]))
        out = specify_shape(out, 2)
        gx = grad(out.sum(), x)

        mode = Mode(linker="py", optimizer=None)
        np.testing.assert_allclose(
            gx.eval({x: [1, 1]}, mode=mode),
            [0, np.e],
        )


class TestAdvancedIncSubtensor1:
    def setup_method(self):
        self.rng = np.random.default_rng(seed=utt.fetch_seed())

        self.s = iscalar()
        self.v = fvector()
        self.m = dmatrix()
        self.t = ctensor3()

        self.adv1q = lvector()  # advanced 1d query

    def test_cant_adv_idx_into_scalar(self):
        with pytest.raises(IndexError):
            self.s[self.adv1q]

    def test_index_into_vec_w_vec(self):
        a = self.v[self.adv1q]
        assert a.type == self.v.type

    def test_1d_set_adv_selection(self):
        a = set_subtensor(self.v[self.adv1q], self.v[self.adv1q])

        assert a.type == self.v.type

        # TODO: compile a function and verify that the subtensor is removed
        #      completely, because the whole expression is redundant.

        f = pytensor.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([0.4, 0.9, 0.1], [1, 2])
        assert np.allclose(aval, [0.4, 0.9, 0.1])

    @pytest.mark.parametrize("ignore_duplicates", [True, False])
    def test_inc_subtensor_AdvancedSubtensor1(self, ignore_duplicates):
        x = AdvancedSubtensor1()(self.v, self.adv1q)
        a = inc_subtensor(x, self.v[self.adv1q], ignore_duplicates=ignore_duplicates)
        assert isinstance(a.owner.op, AdvancedIncSubtensor1 | AdvancedIncSubtensor)
        assert getattr(a.owner.op, "ignore_duplicates", False) == ignore_duplicates

    def test_1d_inc_adv_selection(self):
        a = inc_subtensor(self.v[self.adv1q], self.v[self.adv1q])

        assert a.type == self.v.type
        f = pytensor.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([0.4, 0.9, 0.1], [1, 2])
        assert np.allclose(aval, [0.4, 1.8, 0.2])

    def test_1d_inc_adv_selection_w_broadcasting(self):
        a = inc_subtensor(self.v[self.adv1q], 3.0)

        assert a.type == self.v.type
        f = pytensor.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([0.4, 0.9, 0.1], [1, 2])
        assert np.allclose(aval, [0.4, 3.9, 3.1])

    def test_assigning_matrix_to_vector_selection(self):
        with pytest.raises(TypeError):
            inc_subtensor(self.v[self.adv1q](fmatrix()))

    def test_matrix_idx(self):
        idx = lmatrix()
        a = self.m[idx]
        a2 = inc_subtensor(a, a)
        f = pytensor.function([self.m, idx], a2)

        mval = self.rng.random((4, 10))
        idxval = np.array([[1, 2], [3, 2]])
        a2val = f(mval, idxval)

        utt.assert_allclose(a2val[0], mval[0])
        utt.assert_allclose(a2val[1], mval[1] * 2)
        utt.assert_allclose(a2val[2], mval[2] * 3)
        utt.assert_allclose(a2val[3], mval[3] * 2)

    def test_inc_bcastableidx(self):
        idx = ptb.constant([0])
        c_inc = col()
        m_inc = matrix()
        out1 = inc_subtensor(self.m[:, idx], c_inc)
        out2 = inc_subtensor(self.m[:, idx], m_inc)

        f = pytensor.function([self.m, c_inc, m_inc], [out1, out2])
        mval = self.rng.random((10, 5))
        incval = self.rng.random((10, 1)).astype(config.floatX)

        out1val, out2val = f(mval, incval, incval)
        utt.assert_allclose(out1val, out2val)

    def test_empty_index(self):
        x = fvector()
        idx = constant([], dtype="int64")
        y = idx.astype("float32")
        out = advanced_inc_subtensor1(x, y, idx)

        test_x = np.array([1, 2, 3], dtype="float32")
        res = out.eval({x: test_x}, mode=Mode(optimizer=None))
        np.testing.assert_array_equal(res, test_x)


class TestAdvancedSubtensor:
    """Test inc_subtensor and set_subtensor."""

    def setup_method(self):
        self.shared = shared
        self.dtype = config.floatX
        self.mode = pytensor.compile.mode.get_default_mode()

        self.s = iscalar()
        self.v = fvector()
        self.m = dmatrix()
        self.t = ctensor3()
        self.ft4 = ftensor4()

        self.ix1 = lvector()  # advanced 1d query
        self.ix12 = lvector()
        self.ix2 = lmatrix()
        self.ixr = lrow()

    @pytest.mark.parametrize(
        "inplace",
        [
            True,
            False,
        ],
    )
    def test_advinc_subtensor(self, inplace):
        x_shp = (20, 15, 10, 5)

        def check(idx, y_val, x_val, true):
            x = self.shared(x_val, name="x")
            y = tensor(dtype="float32", shape=(None,) * len(y_val.shape), name="y")
            sym_idx = [ptb.as_tensor_variable(ix) for ix in idx]
            expr = AdvancedIncSubtensor(inplace=inplace)(x, y, *sym_idx)
            f = pytensor.function(
                [y], expr, mode=self.mode.excluding("inplace"), accept_inplace=inplace
            )
            fgraph = f.maker.fgraph
            assert fgraph.outputs[0].owner.op.inplace == inplace
            rval = f(y_val)
            assert np.allclose(rval, true)

        idxs_y_shp_pairs = [
            ((0, [1, 3, 5], 1), (3, 5)),
            (([1, 2, 4, 8],), (4, 15, 10, 5)),
            (([0, 1, 2], 0, [0, 1, 2]), (3, 3, 5)),
            (([[0, 1], [2, 3]], [[0, 1], [2, 3]]), (2, 2, 10, 5)),
        ]

        for idx, y_shps in idxs_y_shp_pairs:
            for i in range(len(y_shps) - 1):
                y_shp = y_shps[i:]
                x_val = np.arange(np.prod(x_shp), dtype="float32").reshape(x_shp) + 1
                y_val = np.arange(np.prod(y_shp), dtype="float32").reshape(y_shp) + 1
                rep = x_val.copy()

                try:
                    rep[idx] += y_val
                except ValueError:
                    continue

                check(idx, y_val, x_val, rep)

            x_val = np.arange(np.prod(x_shp), dtype="float32").reshape(x_shp) + 1
            y_val = np.array(1).astype(np.float32)
            rep = x_val.copy()
            rep[idx] += y_val
            check(idx, y_val, x_val, rep)

    def eval_output_and_check(self, t, op):
        f = inplace_func([], t, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, DeepCopyOp)]
        assert len(topo_) == 1
        assert isinstance(topo_[0].op, op)
        tval = f()
        return tval

    def test_cant_adv_idx_into_scalar(self):
        with pytest.raises(IndexError):
            self.s[self.ix1]

    def test_index_into_vec_w_vec(self):
        a = self.v[self.ix1]
        assert a.type == self.v.type, (a.type, self.v.type)

    def test_index_into_vec_w_matrix(self):
        a = self.v[self.ix2]
        assert a.dtype == self.v.dtype, (a.dtype, self.v.dtype)
        assert a.type.ndim == self.ix2.type.ndim
        assert all(
            s1 == s2
            for s1, s2 in zip(a.type.shape, self.ix2.type.shape, strict=True)
            if s1 == 1 or s2 == 1
        )

    def test_index_into_mat_w_row(self):
        a = self.m[self.ixr]
        assert a.dtype == self.m.dtype, (a.dtype, self.m.dtype)
        assert a.type.shape == (1, None, None)

    def test_index_w_int_and_vec(self):
        # like test_ok_list, but with a single index on the first one
        # data has to have at least 2 dimensions
        for data, idx in [
            (random(4, 5), [2, 3]),
            (random(2, 4, 3), [0, 3]),
            (random(2, 4, 3), [3, 3, 1, 1, 2, 2, 0, 0]),
            (random(2, 4, 3), [3, 3, 1, 1, 2, 2, 0, 0, -1, -2, -3, -4]),
            # Test 4 dims as gpu code use another algo
            # in that case This new algo is not as much
            # optimized for that case.
            (random(4, 4, 2, 3), [3, 3, 1, 1, 2, 2, 0, 0, -1, -2, -3, -4]),
            # Test with TensorConstant index.
            (random(2, 4, 3), ptb.constant([3, 3, 1, 1, 2, 2, 0, 0])),
        ]:
            data = np.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[0, idx]

            assert isinstance(t.owner.op, AdvancedSubtensor)

            val = self.eval_output_and_check(t, AdvancedSubtensor)
            if isinstance(idx, list):
                good = data[0, idx]
            else:
                good = data[0, idx.data]
            assert val.ndim == data.ndim - 1
            assert np.allclose(val, good), (val, good)

    def test_inc_adv_subtensor_w_matrix(self):
        subt = self.v[self.ix2]
        a = inc_subtensor(subt, subt)

        assert a.type == self.v.type, (a.type, self.v.type)
        f = pytensor.function(
            [self.v, self.ix2], a, allow_input_downcast=True, mode=self.mode
        )
        aval = f([0.4, 0.9, 0.1], [[1, 2], [1, 2]])
        assert np.allclose(aval, [0.4, 0.9 * 3, 0.1 * 3])

    def test_adv_subtensor_w_int_and_matrix(self):
        subt = self.ft4[0, :, self.ix2, :]
        f = pytensor.function([self.ft4, self.ix2], subt, mode=self.mode)
        ft4v = np.random.random((2, 3, 4, 5)).astype("float32")
        ix2v = np.asarray([[0, 1], [1, 0]])
        aval = f(ft4v, ix2v)
        rval = ft4v[0, :, ix2v, :]
        utt.assert_allclose(rval, aval)

    def test_adv_subtensor_w_none_and_matrix(self):
        subt = self.ft4[:, None, :, self.ix2, :]
        f = pytensor.function([self.ft4, self.ix2], subt, mode=self.mode)
        ft4v = np.random.random((2, 3, 4, 5)).astype("float32")
        ix2v = np.asarray([[0, 1], [1, 0]])
        aval = f(ft4v, ix2v)
        rval = ft4v[:, None, :, ix2v, :]
        utt.assert_allclose(rval, aval)

    def test_adv_subtensor_w_slice_and_matrix(self):
        subt = self.ft4[:, 0:1, self.ix2, :]
        f = pytensor.function([self.ft4, self.ix2], subt, mode=self.mode)
        ft4v = np.random.random((2, 3, 4, 5)).astype("float32")
        ix2v = np.asarray([[0, 1], [1, 0]])
        aval = f(ft4v, ix2v)
        rval = ft4v[:, 0:1, ix2v, :]
        utt.assert_allclose(rval, aval)

    def test_adv_subtensor_w_matrix_and_int(self):
        subt = self.ft4[:, :, self.ix2, 0]
        f = pytensor.function([self.ft4, self.ix2], subt, mode=self.mode)
        ft4v = np.random.random((2, 3, 4, 5)).astype("float32")
        ix2v = np.asarray([[0, 1], [1, 0]])
        aval = f(ft4v, ix2v)
        rval = ft4v[:, :, ix2v, 0]
        utt.assert_allclose(rval, aval)

    def test_adv_subtensor_w_matrix_and_none(self):
        subt = self.ft4[:, :, self.ix2, None, :]
        f = pytensor.function([self.ft4, self.ix2], subt, mode=self.mode)
        ft4v = np.random.random((2, 3, 4, 5)).astype("float32")
        ix2v = np.asarray([[0, 1], [1, 0]])
        aval = f(ft4v, ix2v)
        rval = ft4v[:, :, ix2v, None, :]
        utt.assert_allclose(rval, aval)

    @pytest.mark.parametrize(
        "ignore_duplicates",
        [
            True,
            False,
        ],
    )
    def test_inc_adv_subtensor_w_2vec(self, ignore_duplicates):
        subt = self.m[self.ix1, self.ix12]
        a = inc_subtensor(subt, subt, ignore_duplicates=ignore_duplicates)

        typ = TensorType(
            self.m.type.dtype,
            shape=tuple(1 if s == 1 else None for s in self.ix2.type.shape),
        )
        assert a.type == typ

        f = pytensor.function(
            [self.m, self.ix1, self.ix12], a, allow_input_downcast=True, mode=self.mode
        )

        m_val = [[0.4, 0.9, 0.1], [5, 6, 7], [0.5, 0.3, 0.15]]
        ix1_val = [1, 2, 1]
        ix2_val = [0, 1, 0]

        exp_aval = np.array(m_val)
        if ignore_duplicates:
            exp_aval[ix1_val, ix2_val] += exp_aval[ix1_val, ix2_val]
        else:
            np.add.at(exp_aval, (ix1_val, ix2_val), exp_aval[ix1_val, ix2_val])

        aval = f(m_val, ix1_val, ix2_val)
        assert np.allclose(aval, exp_aval)

    @pytest.mark.parametrize(
        "ignore_duplicates",
        [
            True,
            False,
        ],
    )
    def test_inc_adv_subtensor_with_broadcasting(self, ignore_duplicates):
        inc = dscalar()
        a = inc_subtensor(
            self.m[self.ix1, self.ix12], inc, ignore_duplicates=ignore_duplicates
        )
        g_inc = pytensor.grad(a.sum(), inc)

        assert a.type == self.m.type

        f = pytensor.function(
            [self.m, self.ix1, self.ix12, inc],
            [a, g_inc],
            allow_input_downcast=True,
            mode=self.mode,
        )

        m_val = [[0.4, 0.9, 0.1], [5, 6, 7], [0.5, 0.3, 0.15]]
        ix1_val = [1, 2, 1]
        ix2_val = [0, 1, 0]
        inc_val = 2.1

        exp_aval = np.array(m_val)
        if ignore_duplicates:
            exp_aval[ix1_val, ix2_val] += inc_val
        else:
            np.add.at(exp_aval, (ix1_val, ix2_val), inc_val)

        aval, gval = f(m_val, ix1_val, ix2_val, inc_val)
        assert np.allclose(aval, exp_aval)
        assert np.allclose(gval, 3.0)

    @pytest.mark.parametrize(
        "ignore_duplicates",
        [
            True,
            False,
        ],
    )
    def test_inc_adv_subtensor1_with_broadcasting(self, ignore_duplicates):
        inc = dscalar()
        a = inc_subtensor(self.m[self.ix1], inc, ignore_duplicates=ignore_duplicates)
        g_inc = pytensor.grad(a.sum(), inc)

        assert a.type == self.m.type

        f = pytensor.function(
            [self.m, self.ix1, inc],
            [a, g_inc],
            allow_input_downcast=True,
            mode=self.mode,
        )

        m_val = [[0.4, 0.9, 0.1], [5, 6, 7], [0.5, 0.3, 0.15]]
        ix1_val = [0, 1, 0]
        inc_val = 2.1

        exp_aval = np.array(m_val).copy()
        if ignore_duplicates:
            exp_aval[ix1_val] += inc_val
        else:
            np.add.at(exp_aval, ix1_val, inc_val)

        aval, gval = f(m_val, ix1_val, inc_val)

        assert np.allclose(aval, exp_aval)
        assert np.allclose(gval, 9.0)

    @pytest.mark.parametrize(
        "ignore_duplicates",
        [
            True,
            False,
        ],
    )
    def test_inc_adv_subtensor_with_index_broadcasting(self, ignore_duplicates):
        a = inc_subtensor(
            self.m[self.ix1, self.ix2], 2.1, ignore_duplicates=ignore_duplicates
        )

        assert a.type == self.m.type

        f = pytensor.function(
            [self.m, self.ix1, self.ix2], a, allow_input_downcast=True, mode=self.mode
        )

        m_val = [[0.4, 0.9, 0.1], [5, 6, 7], [0.5, 0.3, 0.15]]
        ix1_val = [0, 2, 0]
        ix2_val = [[0, 1, 0], [2, 2, 2]]

        inc_val = 2.1
        exp_aval = np.array(m_val)
        if ignore_duplicates:
            exp_aval[ix1_val, ix2_val] += inc_val
        else:
            np.add.at(exp_aval, (ix1_val, ix2_val), inc_val)

        aval = f(m_val, ix1_val, ix2_val)
        assert np.allclose(aval, exp_aval)

    def test_2d_3d_tensors(self):
        rng = np.random.default_rng(utt.fetch_seed())
        a = rng.uniform(size=(3, 3))
        b = pytensor.shared(a)
        i = iscalar()
        j = iscalar()
        z = b[[i, j], :]
        f1 = pytensor.function([i, j], z, mode=self.mode)
        cmd = f1(0, 1) == a[[0, 1], :]
        assert cmd.all()

        aa = rng.uniform(size=(4, 2, 3))
        bb = pytensor.shared(aa)
        k = iscalar()
        z = bb[[i, j, k], :, i:k]
        f2 = pytensor.function([i, j, k], z, mode=self.mode)
        cmd = f2(0, 1, 2) == aa[[0, 1, 2], :, 0:2]
        assert cmd.all()

    def test_adv_sub_3d(self):
        # Reported in https://github.com/Theano/Theano/issues/5674
        X = tensor3("X")

        xx = np.zeros((3, 2, 2), config.floatX)
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    xx[i, j, k] = 100 * i + 10 * j + k

        b_idx = np.zeros((2, 2), "int32")
        b_idx[0, 1] = 1
        b_idx[1, 1] = 2

        r_idx = np.arange(xx.shape[1])[:, np.newaxis]
        c_idx = np.arange(xx.shape[2])[np.newaxis, :]

        f = pytensor.function([X], X[b_idx, r_idx, c_idx], mode=self.mode)
        out = f(xx)
        utt.assert_allclose(out, xx[b_idx, r_idx, c_idx])

    def test_adv_sub_slice(self):
        # Reported in https://github.com/Theano/Theano/issues/5898
        var = self.shared(np.zeros([3, 3], dtype=config.floatX))
        slc = slicetype()
        f = pytensor.function([slc], var[slc], mode=self.mode)
        s = slice(1, 3)
        assert f(s).shape == (2, 3)

        f_shape0 = pytensor.function([slc], var[slc].shape[0], mode=self.mode)
        assert f_shape0(s) == 2

        f_shape1 = pytensor.function([slc], var[slc].shape[1], mode=self.mode)
        assert not any(
            isinstance(node.op, AdvancedSubtensor)
            for node in f_shape1.maker.fgraph.toposort()
        )
        assert f_shape1(s) == 3

    def test_adv_grouped(self):
        # Reported in https://github.com/Theano/Theano/issues/6152
        rng = np.random.default_rng(utt.fetch_seed())
        var_v = rng.random((3, 63, 4)).astype(config.floatX)
        var = self.shared(var_v)
        idx1_v = rng.integers(0, 61, size=(5, 4)).astype("int32")
        idx1 = self.shared(idx1_v)
        idx2 = ptb.arange(4)
        out = var[:, idx1, idx2]
        f = pytensor.function([], out, mode=self.mode)
        out_v = f()
        assert out_v.shape == (3, 5, 4)

        out_np = var_v[:, idx1_v, np.arange(4)]
        utt.assert_allclose(out_v, out_np)

    def test_grad(self):
        ones = np.ones((1, 3), dtype=self.dtype)
        n = self.shared(ones * 5, shape=(1, None))
        idx = lvector()
        idx2 = lvector()
        t = n[idx, idx2]
        assert isinstance(t.owner.op, AdvancedSubtensor)

        utt.verify_grad(
            lambda m: m[[1, 3], [2, 4]],
            [np.random.random((5, 5)).astype(self.dtype)],
            mode=self.mode,
        )

        def fun(x, y):
            return advanced_inc_subtensor(x, y, [1, 3], [2, 4])

        utt.verify_grad(
            fun,
            [
                np.random.random((5, 5)).astype(self.dtype),
                np.random.random(2).astype(self.dtype),
            ],
            mode=self.mode,
        )

        def fun(x, y):
            return advanced_set_subtensor(x, y, [1, 3], [2, 4])

        utt.verify_grad(
            fun,
            [
                np.random.random((5, 5)).astype(self.dtype),
                np.random.random(2).astype(self.dtype),
            ],
            mode=self.mode,
        )

        # Test boolean gradients
        def fun(x, y):
            return advanced_inc_subtensor(
                x, y, ptb.as_tensor(np.array([[True, False], [False, True]]))
            )

        utt.verify_grad(
            fun,
            [
                np.random.random((2, 2)).astype(self.dtype),
                np.random.random(2).astype(self.dtype),
            ],
            mode=self.mode,
        )

        def fun(x, y):
            return advanced_set_subtensor(
                x, y, ptb.as_tensor(np.array([[True, False], [False, True]]))
            )

        utt.verify_grad(
            fun,
            [
                np.random.random((2, 2)).astype(self.dtype),
                np.random.random(2).astype(self.dtype),
            ],
            mode=self.mode,
        )

    def test_boolean_scalar_raises(self):
        x = vector("x")
        with pytest.raises(NotImplementedError):
            x[np.array(True)]


class TestInferShape(utt.InferShapeTester):
    @staticmethod
    def random_bool_mask(shape, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.binomial(n=1, p=0.5, size=shape).astype(bool)

    def test_IncSubtensor(self):
        admat = dmatrix()
        bdmat = dmatrix()
        advec = dvector()
        adscal = dscalar()
        admat_val = random(5, 4)
        self._compile_and_check(
            [admat, bdmat],
            [inc_subtensor(admat[2:4], bdmat)],
            [admat_val, [[1, 2, 3, 4]]],
            IncSubtensor,
        )

        self._compile_and_check(
            [admat, advec],
            [inc_subtensor(admat[2], advec)],
            [admat_val, [1, 2, 3, 4]],
            IncSubtensor,
        )

        self._compile_and_check(
            [admat, adscal],
            [inc_subtensor(admat[2, 3], adscal)],
            [admat_val, 1],
            IncSubtensor,
        )

        self._compile_and_check(
            [admat, adscal],
            [inc_subtensor(admat[1:3, 2], adscal)],
            [admat_val, 1],
            IncSubtensor,
        )

        self._compile_and_check(
            [admat, bdmat],
            [set_subtensor(admat[2:4], bdmat)],
            [admat_val, [[1, 2, 3, 4]]],
            IncSubtensor,
        )

        self._compile_and_check(
            [admat, advec],
            [set_subtensor(admat[2], advec)],
            [admat_val, [1, 2, 3, 4]],
            IncSubtensor,
        )

        self._compile_and_check(
            [admat, adscal],
            [set_subtensor(admat[2, 3], adscal)],
            [admat_val, 1],
            IncSubtensor,
        )

        self._compile_and_check(
            [admat, adscal],
            [set_subtensor(admat[1:3, 2], adscal)],
            [admat_val, 1],
            IncSubtensor,
        )

        adtens4 = dtensor4()
        bdtens4 = dtensor4()
        adtens4_val = random(3, 4, 2, 5)
        self._compile_and_check(
            [adtens4, bdtens4],
            [inc_subtensor(adtens4[::, 2:4, ::, ::], bdtens4)],
            [adtens4_val, [[[[1, 2, 3, 4, 5]]]]],
            IncSubtensor,
            warn=False,
        )
        self._compile_and_check(
            [adtens4, bdmat],
            [inc_subtensor(adtens4[2, 2:4, 1, ::], bdmat)],
            [adtens4_val, [[1, 2, 3, 4, 5]]],
            IncSubtensor,
        )

        self._compile_and_check(
            [adtens4, advec],
            [inc_subtensor(adtens4[0, 1, ::, 4], advec)],
            [adtens4_val, [1, 2]],
            IncSubtensor,
        )

        self._compile_and_check(
            [adtens4, adscal],
            [inc_subtensor(adtens4[1:3, 1, ::, 2:4], adscal)],
            [adtens4_val, 1],
            IncSubtensor,
        )

        self._compile_and_check(
            [adtens4, bdtens4],
            [set_subtensor(adtens4[::, 2:4, ::, ::], bdtens4)],
            [adtens4_val, [[[[1, 2, 3, 4, 5]]]]],
            IncSubtensor,
            warn=False,
        )

        self._compile_and_check(
            [adtens4, bdmat],
            [set_subtensor(adtens4[2, 2:4, 1, ::], bdmat)],
            [adtens4_val, [[1, 2, 3, 4, 5]]],
            IncSubtensor,
        )

        self._compile_and_check(
            [adtens4, advec],
            [set_subtensor(adtens4[0, 1, ::, 4], advec)],
            [adtens4_val, [1, 2]],
            IncSubtensor,
        )

        self._compile_and_check(
            [adtens4, adscal],
            [set_subtensor(adtens4[1:3, 1, ::, 2:4], adscal)],
            [adtens4_val, 1],
            IncSubtensor,
        )

    def test_AdvancedIncSubtensor1(self):
        admat = dmatrix()
        bdmat = dmatrix()
        advec = dvector()
        adscal = dscalar()
        admat_val = random(5, 4)
        aivec_val = [2, 3]
        self._compile_and_check(
            [admat, bdmat],
            [
                advanced_set_subtensor1(
                    admat, specify_broadcastable(bdmat, 0), aivec_val
                )
            ],
            [admat_val, [[1, 2, 3, 4]]],
            AdvancedIncSubtensor1,
        )

        aivec_val = [1, 3, 2]
        self._compile_and_check(
            [admat, advec],
            [advanced_set_subtensor1(admat, advec, aivec_val)],
            [admat_val, [1, 2, 3, 4]],
            AdvancedIncSubtensor1,
        )

        aivec_val = [0, 3, 0]
        self._compile_and_check(
            [admat, adscal],
            [advanced_set_subtensor1(admat, adscal, aivec_val)],
            [admat_val, 1],
            AdvancedIncSubtensor1,
        )

        adtens4 = dtensor4()
        bdtens4 = dtensor4()
        adtens4_val = random(4, 3, 2, 5)
        aivec_val = [2, 3]
        self._compile_and_check(
            [adtens4, bdtens4],
            [
                advanced_set_subtensor1(
                    adtens4, specify_broadcastable(bdtens4, 0, 1, 2), aivec_val
                )
            ],
            [adtens4_val, [[[[1, 2, 3, 4, 5]]]]],
            AdvancedIncSubtensor1,
            warn=False,
        )

        aivec_val = [1, 3, 2]
        self._compile_and_check(
            [adtens4, advec],
            [advanced_set_subtensor1(adtens4, advec, aivec_val)],
            [adtens4_val, [1, 2, 3, 4, 5]],
            AdvancedIncSubtensor1,
        )

        aivec_val = [0, 3, 0]
        self._compile_and_check(
            [adtens4, adscal],
            [advanced_set_subtensor1(adtens4, adscal, aivec_val)],
            [adtens4_val, 1],
            AdvancedIncSubtensor1,
        )

        aivec_val = [2, 3]
        self._compile_and_check(
            [admat, bdmat],
            [advanced_set_subtensor1(admat, bdmat, aivec_val)],
            [admat_val, [[1, 2, 3, 4], [5, 6, 7, 8]]],
            AdvancedIncSubtensor1,
        )

        aivec_val = [1, 3, 2]
        self._compile_and_check(
            [admat, advec],
            [advanced_set_subtensor1(admat, advec, aivec_val)],
            [admat_val, [1, 2, 3, 4]],
            AdvancedIncSubtensor1,
        )

        aivec_val = [0, 3, 0]
        self._compile_and_check(
            [admat, adscal],
            [advanced_set_subtensor1(admat, adscal, aivec_val)],
            [admat_val, 1],
            AdvancedIncSubtensor1,
        )

        bdtens4 = dtensor4()
        adtens4_val = random(4, 3, 2, 5)
        aivec_val = [2, 3]
        self._compile_and_check(
            [adtens4, bdtens4],
            [
                advanced_set_subtensor1(
                    adtens4, specify_broadcastable(bdtens4, 1, 2), aivec_val
                )
            ],
            [adtens4_val, [[[[1, 2, 3, 4, 5]]], [[[6, 7, 8, 9, 10]]]]],
            AdvancedIncSubtensor1,
            warn=False,
        )

        aivec_val = [1, 2, 1]
        self._compile_and_check(
            [adtens4, advec],
            [advanced_set_subtensor1(adtens4, advec, aivec_val)],
            [adtens4_val, [1, 2, 3, 4, 5]],
            AdvancedIncSubtensor1,
        )

        aivec_val = [0, 3, 0]
        self._compile_and_check(
            [adtens4, adscal],
            [advanced_set_subtensor1(adtens4, adscal, aivec_val)],
            [adtens4_val, 2],
            AdvancedIncSubtensor1,
        )

    def test_AdvancedIncSubtensor(self):
        admat = dmatrix()
        advec = dvector()
        admat_val = random(5, 4)
        aivec_val = [1, 3, 2]
        bivec_val = [0, 3, 3]
        advec_val = [23, 24, 25]
        self._compile_and_check(
            [admat, advec],
            [set_subtensor(admat[aivec_val, bivec_val], advec)],
            [admat_val, advec_val],
            AdvancedIncSubtensor,
        )

    def test_AdvancedSubtensor(self):
        admat = dmatrix()
        aivec = lvector()
        bivec = lvector()

        admat_val = random(5, 4)
        aivec_val = [1, 3, 2]
        bivec_val = [0, 3, 3]
        self._compile_and_check(
            [admat, aivec, bivec],
            [admat[aivec, bivec]],
            [admat_val, aivec_val, bivec_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [admat, aivec],
            [admat[aivec, 1:3]],
            [admat_val, aivec_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [admat, aivec],
            [admat[1:3, aivec]],
            [admat_val, aivec_val],
            AdvancedSubtensor,
        )

        admat.tag.test_value = admat_val
        aivec.tag.test_value = aivec_val
        bivec.tag.test_value = bivec_val

        # Make sure it doesn't complain about test values
        with config.change_flags(compute_test_value="raise"):
            self._compile_and_check(
                [admat, aivec],
                [admat[1:3, aivec]],
                [admat_val, aivec_val],
                AdvancedSubtensor,
            )

    def test_AdvancedSubtensor_bool(self):
        n = dmatrix()
        n_val = np.arange(6).reshape((2, 3))

        # Shape inference requires runtime broadcasting between the nonzero() shapes
        self._compile_and_check(
            [n],
            [n[n[:, 0] > 2, n[0, :] > 2]],
            [n_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [n],
            [n[n[:, 0] > 2]],
            [n_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [n],
            [n[:, np.array([True, False, True])]],
            [n_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [n],
            [n[np.array([False, False]), 1:]],
            [n_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [n],
            [n[np.array([True, True]), 0]],
            [n_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [n],
            [n[self.random_bool_mask(n_val.shape)]],
            [n_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [n],
            [n[None, self.random_bool_mask(n_val.shape), None]],
            [n_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [n],
            [n[slice(5, None), self.random_bool_mask(n_val.shape[1])]],
            [n_val],
            AdvancedSubtensor,
        )

        abs_res = n[~isinf(n)]
        assert abs_res.type.shape == (None,)

    def test_AdvancedSubtensor_bool_mixed(self):
        n = tensor5("x", dtype="float64")
        shape = (18, 3, 4, 5, 6)
        n_val = np.arange(np.prod(shape)).reshape(shape)
        self._compile_and_check(
            [n],
            # Consecutive advanced index
            [n[1:, self.random_bool_mask((3, 4)), 0, 1:]],
            [n_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [n],
            # Non-consecutive advanced index
            [n[1:, self.random_bool_mask((3, 4)), 1:, 0]],
            [n_val],
            AdvancedSubtensor,
        )
        self._compile_and_check(
            [n],
            # Non-consecutive advanced index
            [n[1:, self.random_bool_mask((3,)), 1:, None, np.zeros((6, 1), dtype=int)]],
            [n_val],
            AdvancedSubtensor,
        )

    def test_advanced_subtensor_constant_slice(self):
        x = dmatrix("x")
        constant_slice = pytensor.as_symbolic(slice(1, None, None))
        assert isinstance(constant_slice, Constant)
        adv_indices = ptb.constant(np.zeros((2, 3)), dtype="int")
        y = advanced_subtensor(x, constant_slice, adv_indices)
        assert tuple(y.shape.eval({x: np.zeros((10, 10))})) == (9, 2, 3)


@config.change_flags(compute_test_value="raise")
def test_basic_shape():
    test_shape = (5, 4)
    test_indices = (make_slice(1, 3, None),)
    res = basic_shape(test_shape, test_indices)
    assert get_test_value(res) == (2,)


def idx_as_tensor(x):
    if isinstance(x, slice | type(None)):
        return x
    else:
        return ptb.as_tensor(x)


def bcast_shape_tuple(x):
    if not hasattr(x, "shape"):
        return x
    return tuple(
        s if ss != 1 else 1 for s, ss in zip(tuple(x.shape), x.type.shape, strict=True)
    )


test_idx = np.ix_(np.array([True, True]), np.array([True]), np.array([True, True]))


@pytest.mark.parametrize(
    "test_array, test_idx",
    [
        (np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)), (slice(None, None),)),
        (np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)), (2,)),
        (np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)), test_idx[:1]),
        (np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)), test_idx[:2]),
        (
            np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)),
            test_idx[:2] + (slice(None, None),),
        ),
        (
            np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)),
            (slice(None, None),) + test_idx[:1],
        ),
        (
            np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)),
            (slice(None, None), None) + test_idx[1:2],
        ),
        (
            np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)),
            (np.array(1), slice(None, None), None),
        ),
        (
            np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)),
            (slice(None, None), None, np.array(1)),
        ),
        (
            np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)),
            test_idx[:1] + (slice(None, None),) + test_idx[1:2],
        ),
        (
            np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)),
            test_idx[:1] + (slice(None, None),) + test_idx[1:2] + (slice(None, None),),
        ),
        (
            np.arange(np.prod((5, 6, 7, 8))).reshape((5, 6, 7, 8)),
            test_idx[:1] + (None,) + test_idx[1:2],
        ),
        (np.arange(np.prod((5, 4))).reshape((5, 4)), ([1, 3, 2], slice(1, 3))),
        (np.arange(np.prod((5, 4))).reshape((5, 4)), (slice(1, 3), [1, 3, 2])),
        (
            np.arange(np.prod((5, 6, 7))).reshape((5, 6, 7)),
            (slice(None, None), [1, 2, 3], slice(None, None)),
        ),
    ],
)
@config.change_flags(compute_test_value="raise")
def test_indexed_result_shape(test_array, test_idx):
    res = indexed_result_shape(
        ptb.as_tensor(test_array).shape, [idx_as_tensor(i) for i in test_idx]
    )
    exp_res = test_array[test_idx].shape
    assert np.array_equal(tuple(get_test_value(r) for r in res), exp_res)

    # Test shape-only version
    res = indexed_result_shape(
        ptb.as_tensor(test_array).shape,
        [bcast_shape_tuple(idx_as_tensor(i)) for i in test_idx],
        indices_are_shapes=True,
    )
    exp_res = test_array[test_idx].shape
    assert np.array_equal(tuple(get_test_value(r) for r in res), exp_res)


def test_symbolic_slice():
    x = tensor4("x")
    a, b = x.shape[:2]
    output = a.eval({x: np.zeros((5, 4, 3, 2), dtype=config.floatX)})
    assert output == np.array(5)


def test_get_vector_length():
    x = ptb.as_tensor_variable(np.arange(4))
    assert get_vector_length(x[2:4]) == 2
    assert get_vector_length(x[2:]) == 2
    assert get_vector_length(x[1:4]) == 3
    assert get_vector_length(x[2:2]) == 0
    assert get_vector_length(x[1:10]) == 3
    # Test step
    assert get_vector_length(x[1:10:2]) == 2
    # Test neg start
    assert get_vector_length(x[-1:4]) == 1
    assert get_vector_length(x[-6:4]) == 4
    # test neg stop
    assert get_vector_length(x[1:-2]) == 1
    assert get_vector_length(x[1:-1]) == 2
    assert get_vector_length(lvector()[1:1]) == 0
    assert get_vector_length(lvector()[-1:-1:3]) == 0

    with pytest.raises(ValueError, match="^Length of .*"):
        get_vector_length(x[lscalar() :])


@pytest.mark.parametrize(
    "indices, exp_res",
    [
        ((0,), "x[0]"),
        # TODO: The numbers should be printed
        ((slice(None, 2),), "x[:int64]"),
        ((slice(0, None),), "x[int64:]"),
        ((slice(0, 2),), "x[int64:int64]"),
        ((slice(0, 2, 2),), "x[int64:int64:int64]"),
        ((slice(0, 2), 0, slice(0, 2)), "x[int64:int64, 2, int64:int64]"),
    ],
)
def test_pprint_Subtensor(indices, exp_res):
    x = tensor4("x")
    y = x[indices]
    assert pprint(y) == exp_res


@pytest.mark.parametrize(
    "indices, set_instead_of_inc, exp_res",
    [
        ((0,), False, "inc_subtensor(x[0], z)"),
        ((0,), True, "set_subtensor(x[0], z)"),
        ((slice(0, 2),), True, "set_subtensor(x[int64:int64], z)"),
    ],
)
def test_pprint_IncSubtensor(indices, set_instead_of_inc, exp_res):
    x = tensor4("x")
    z = tensor3("z")
    y = inc_subtensor(x[indices], z, set_instead_of_inc=set_instead_of_inc)
    assert pprint(y) == exp_res


def test_index_vars_to_types():
    x = ptb.as_tensor_variable(np.array([True, False]))

    with pytest.raises(AdvancedIndexingError):
        index_vars_to_types(x)

    with pytest.raises(TypeError):
        index_vars_to_types(1)

    res = index_vars_to_types(iscalar)
    assert isinstance(res, scal.ScalarType)

    x = scal.constant(1, dtype=np.uint8)
    assert isinstance(x.type, scal.ScalarType)
    res = index_vars_to_types(x)
    assert res == x.type


@pytest.mark.parametrize(
    "x_shape, indices, expected",
    [
        [(None,), (slice(None, None),), (None,)],
        [(13,), (slice(None, 100),), (13,)],
        [(13,), (slice(-1, None),), (1,)],
        [(7, 13), (slice(None, None, 2), slice(-1, 1, -1)), (4, 11)],
    ],
)
def test_subtensor_static_shapes(x_shape, indices, expected):
    x = ptb.tensor(dtype="float64", shape=x_shape)
    y = x[indices]
    assert y.type.shape == expected


@pytest.mark.parametrize(
    "x_shape, indices, expected",
    [
        [(None, 5, None, 3), vector(shape=(1,)), (1, 5, None, 3)],
        [(None, 5, None, 3), vector(shape=(2,)), (2, 5, None, 3)],
        [(None, 5, None, 3), vector(shape=(None,)), (None, 5, None, 3)],
    ],
)
def test_advanced_subtensor1_static_shapes(x_shape, indices, expected):
    x = ptb.tensor(dtype="float64", shape=x_shape)
    y = advanced_subtensor1(x, indices.astype(int))
    assert y.type.shape == expected


def test_vectorize_subtensor_without_batch_indices():
    signature = "(t1,t2,t3),()->(t1,t3)"

    def core_fn(x, start):
        return x[:, start, :]

    x = tensor(shape=(11, 7, 5, 3))
    start = tensor(shape=(), dtype="int")
    vectorize_pt = function(
        [x, start], vectorize(core_fn, signature=signature)(x, start)
    )
    assert not any(
        isinstance(node.op, Blockwise) for node in vectorize_pt.maker.fgraph.apply_nodes
    )
    x_test = np.random.normal(size=x.type.shape).astype(x.type.dtype)
    start_test = np.random.randint(0, x.type.shape[-2])
    vectorize_np = np.vectorize(core_fn, signature=signature)
    np.testing.assert_allclose(
        vectorize_pt(x_test, start_test),
        vectorize_np(x_test, start_test),
    )

    # If we vectorize start, we should get a Blockwise that still works
    x = tensor(shape=(11, 7, 5, 3))
    start = tensor(shape=(11,), dtype="int")
    vectorize_pt = function(
        [x, start], vectorize(core_fn, signature=signature)(x, start)
    )
    assert any(
        isinstance(node.op, Blockwise) for node in vectorize_pt.maker.fgraph.apply_nodes
    )
    x_test = np.random.normal(size=x.type.shape).astype(x.type.dtype)
    start_test = np.random.randint(0, x.type.shape[-2], size=start.type.shape[0])
    vectorize_np = np.vectorize(core_fn, signature=signature)
    np.testing.assert_allclose(
        vectorize_pt(x_test, start_test),
        vectorize_np(x_test, start_test),
    )


@pytest.mark.parametrize(
    "core_idx_fn, signature, x_shape, idx_shape, uses_blockwise",
    [
        # Core case
        ((lambda x, idx: x[:, idx, :]), "(7,5,3),(2)->(7,2,3)", (7, 5, 3), (2,), False),
        # Batched x, core idx
        (
            (lambda x, idx: x[:, idx, :]),
            "(7,5,3),(2)->(7,2,3)",
            (11, 7, 5, 3),
            (2,),
            False,
        ),
        (
            (lambda x, idx: x[idx, None]),
            "(5,7,3),(2)->(2,1,7,3)",
            (11, 5, 7, 3),
            (2,),
            False,
        ),
        # (this is currently failing because PyTensor tries to vectorize the slice(None) operation,
        # due to the exact same None constant being used there and in the np.newaxis)
        pytest.param(
            (lambda x, idx: x[:, idx, None]),
            "(7,5,3),(2)->(7,2,1,3)",
            (11, 7, 5, 3),
            (2,),
            False,
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
        (
            (lambda x, idx: x[:, idx, idx, :]),
            "(7,5,5,3),(2)->(7,2,3)",
            (11, 7, 5, 5, 3),
            (2,),
            False,
        ),
        # (not supported, because fallback Blocwise can't handle slices)
        pytest.param(
            (lambda x, idx: x[:, idx, :, idx]),
            "(7,5,3,5),(2)->(2,7,3)",
            (11, 7, 5, 3, 5),
            (2,),
            True,
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
        # Core x, batched idx
        ((lambda x, idx: x[idx]), "(t1),(idx)->(tx)", (7,), (11, 2), True),
        # Batched x, batched idx
        ((lambda x, idx: x[idx]), "(t1),(idx)->(tx)", (11, 7), (11, 2), True),
        # (not supported, because fallback Blocwise can't handle slices)
        pytest.param(
            (lambda x, idx: x[:, idx, :]),
            "(t1,t2,t3),(idx)->(t1,tx,t3)",
            (11, 7, 5, 3),
            (11, 2),
            True,
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
    ],
)
def test_vectorize_adv_subtensor(
    core_idx_fn, signature, x_shape, idx_shape, uses_blockwise
):
    x = tensor(shape=x_shape, dtype="float64")
    idx = tensor(shape=idx_shape, dtype="int64")
    vectorize_pt = function(
        [x, idx], vectorize(core_idx_fn, signature=signature)(x, idx)
    )

    has_blockwise = any(
        isinstance(node.op, Blockwise) for node in vectorize_pt.maker.fgraph.apply_nodes
    )
    assert has_blockwise == uses_blockwise

    x_test = np.random.normal(size=x.type.shape).astype(x.type.dtype)
    # Idx dimension should be length 5
    idx_test = np.random.randint(0, 5, size=idx.type.shape)
    vectorize_np = np.vectorize(core_idx_fn, signature=signature)
    np.testing.assert_allclose(
        vectorize_pt(x_test, idx_test),
        vectorize_np(x_test, idx_test),
    )


def test_slice_at_axis():
    x = ptb.tensor("x", shape=(3, 4, 5))
    x_sliced = x[slice_at_axis(slice(None, 1), axis=0)]
    assert x_sliced.type.shape == (1, 4, 5)

    # Negative axis
    x_sliced = x[slice_at_axis(slice(None, 1), axis=-2)]
    assert x_sliced.type.shape == (3, 1, 5)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
def test_flip(size: tuple[int]):
    from itertools import combinations

    ATOL = RTOL = 1e-8 if config.floatX == "float64" else 1e-4

    x = np.random.normal(size=size).astype(config.floatX)
    x_pt = pytensor.tensor.tensor(shape=size, name="x")
    expected = np.flip(x, axis=None)
    z = flip(x_pt, axis=None)
    f = pytensor.function([x_pt], z, mode="FAST_COMPILE")
    np.testing.assert_allclose(expected, f(x), atol=ATOL, rtol=RTOL)

    # Test all combinations of axes
    flip_options = [
        axes for i in range(1, x.ndim + 1) for axes in combinations(range(x.ndim), r=i)
    ]
    for axes in flip_options:
        expected = np.flip(x, axis=list(axes))
        z = flip(x_pt, axis=list(axes))
        f = pytensor.function([x_pt], z, mode="FAST_COMPILE")
        np.testing.assert_allclose(expected, f(x), atol=ATOL, rtol=RTOL)


class TestBenchmarks:
    @pytest.mark.parametrize(
        "static_shape", (False, True), ids=lambda x: f"static_shape={x}"
    )
    @pytest.mark.parametrize("gc", (False, True), ids=lambda x: f"gc={x}")
    def test_advanced_subtensor1(self, static_shape, gc, benchmark):
        x = vector("x", shape=(85 if static_shape else None,))

        x_values = np.random.normal(size=(85,))
        idxs_values = np.arange(85).repeat(11)

        # With static shape and constant indices we know all idxs are valid
        # And can use faster mode in numpy.take
        out = x[idxs_values]

        fn = pytensor.function(
            [x],
            pytensor.Out(out, borrow=True),
            on_unused_input="ignore",
            trust_input=True,
        )
        fn.vm.allow_gc = gc
        benchmark(fn, x_values, idxs_values)

    @pytest.mark.parametrize(
        "static_shape", (False, True), ids=lambda x: f"static_shape={x}"
    )
    @pytest.mark.parametrize("gc", (False, True), ids=lambda x: f"gc={x}")
    @pytest.mark.parametrize("func", (inc_subtensor, set_subtensor))
    def test_advanced_incsubtensor1(self, func, static_shape, gc, benchmark):
        x = vector("x", shape=(85 if static_shape else None,))
        x_values = np.zeros((85,))
        buffer = ptb.zeros_like(x)
        y_values = np.random.normal(size=(85 * 11,))
        idxs_values = np.arange(85).repeat(11)

        # With static shape and constant indices we know all idxs are valid
        # Reuse same buffer of zeros, to check we rather allocate twice than copy inside IncSubtensor
        out1 = func(buffer[idxs_values], y_values)
        out2 = func(buffer[idxs_values[::-1]], y_values)

        fn = pytensor.function(
            [x],
            [pytensor.Out(out1, borrow=True), pytensor.Out(out2, borrow=True)],
            on_unused_input="ignore",
            trust_input=True,
        )
        fn.vm.allow_gc = gc
        benchmark(fn, x_values)

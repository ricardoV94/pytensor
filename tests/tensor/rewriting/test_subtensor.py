import random

import numpy as np
import pytest

import pytensor
import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor import shared
from pytensor.compile.function import function
from pytensor.compile.mode import Mode, get_default_mode, get_mode
from pytensor.compile.ops import DeepCopyOp
from pytensor.configdefaults import config
from pytensor.graph import rewrite_graph, vectorize_graph
from pytensor.graph.basic import Constant, Variable, ancestors, equal_computations
from pytensor.graph.rewriting.basic import check_stack_trace
from pytensor.raise_op import Assert
from pytensor.tensor.basic import Alloc, _convert_to_int8
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import Dot, dot, exp, sqr
from pytensor.tensor.rewriting.subtensor import (
    local_replace_AdvancedSubtensor,
)
from pytensor.tensor.shape import (
    SpecifyShape,
    specify_shape,
)
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    advanced_inc_subtensor1,
    inc_subtensor,
    set_subtensor,
)
from pytensor.tensor.type import (
    bmatrix,
    col,
    dmatrix,
    fmatrix,
    iscalar,
    ivector,
    matrix,
    scalar,
    tensor,
    tensor3,
    tensor4,
    vector,
)
from pytensor.tensor.type_other import make_slice
from tests import unittest_tools as utt
from tests.unittest_tools import create_pytensor_param


mode_opt = config.mode
if mode_opt == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
mode_opt = get_mode(mode_opt)


y = create_pytensor_param(np.random.default_rng().integers(0, 4, size=(2,)))
z = create_pytensor_param(np.random.default_rng().integers(0, 4, size=(2, 2)))


@pytest.mark.parametrize(
    ("indices", "is_none"),
    [
        ((slice(None), y, y), True),
        ((y, y, slice(None)), True),
        ((y,), False),
        ((slice(None), y), False),
        ((y, slice(None)), False),
        ((slice(None), y, slice(None)), False),
        ((slice(None), z, slice(None)), False),
        ((slice(None), z), False),
        ((z, slice(None)), False),
        ((slice(None), z, slice(None)), False),
    ],
)
def test_local_replace_AdvancedSubtensor(indices, is_none):
    X_val = np.random.normal(size=(4, 4, 4))
    X = tensor(dtype=np.float64, shape=(None, None, None), name="X")
    X.tag.test_value = X_val

    Y = X[indices]

    res_pt = local_replace_AdvancedSubtensor.transform(None, Y.owner)

    if is_none:
        assert res_pt is None
    else:
        (res_pt,) = res_pt

        assert not any(
            isinstance(v.owner.op, AdvancedSubtensor)
            for v in ancestors([res_pt])
            if v.owner
        )

        inputs = [X] + [i for i in indices if isinstance(i, Variable)]

        res_fn = function(inputs, res_pt, mode=Mode("py", None, None))
        exp_res_fn = function(inputs, Y, mode=Mode("py", None, None))

        # Make sure that the expected result graph has an `AdvancedSubtensor`
        assert any(
            isinstance(v.owner.op, AdvancedSubtensor)
            for v in exp_res_fn.maker.fgraph.variables
            if v.owner
        )

        res_val = res_fn(*[i.tag.test_value for i in inputs])
        exp_res_val = exp_res_fn(*[i.tag.test_value for i in inputs])

        assert np.array_equal(res_val, exp_res_val)


@pytest.mark.parametrize("s", [slice(None), slice(None, None, -1)])
def test_local_useless_inc_subtensor(s):
    x = matrix("x")
    y = matrix("y")

    o = set_subtensor(x[:, s], y)

    mode = get_default_mode().including("local_useless_inc_subtensor")

    # Test without shape info (i.e. don't apply the opt)
    f = function([x, y], o, mode=mode)

    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, IncSubtensor)

    # Test with shape info
    o_shape = set_subtensor(x[:, s], specify_shape(y, x.shape))
    f_shape = function([x, y], o_shape, mode=mode)

    topo = f_shape.maker.fgraph.toposort()
    assert not any(isinstance(n.op, IncSubtensor) for n in topo)

    out = f_shape([[2, 3]], [[3, 4]])
    assert np.array_equal(out, np.asarray([[3, 4]])[::, s])


def test_local_useless_inc_subtensor_increment_zeros():
    r"""Make sure we remove `IncSubtensor`\s that are increments on entire zero arrays."""
    y = matrix("y")

    s = pt.zeros((2, 2))[:, :]
    o_shape = inc_subtensor(s, specify_shape(y, s.shape))

    mode = get_default_mode().including("local_useless_inc_subtensor")
    f_shape = function([y], o_shape, mode=mode)

    topo = f_shape.maker.fgraph.toposort()
    assert not any(isinstance(n.op, IncSubtensor) for n in topo)


def test_local_useless_inc_subtensor_no_opt():
    r"""Make sure we don't remove `IncSubtensor`\s that involve slices with steps that skip elements and non-zero increments."""
    x = matrix("x")
    y = matrix("y")

    s = x[:, ::2]
    o_shape = set_subtensor(s, specify_shape(y, s.shape))

    mode = get_default_mode().including("local_useless_inc_subtensor")
    f_shape = function([x, y], o_shape, mode=mode)

    topo = f_shape.maker.fgraph.toposort()
    assert any(isinstance(n.op, IncSubtensor) for n in topo)

    out = f_shape([[2, 3, 6, 7]], [[8, 9]])
    assert np.array_equal(out, np.asarray([[8, 3, 9, 7]]))

    # This is an increment with a non-constant target array
    s = x[:, :]
    o_shape = inc_subtensor(s, specify_shape(y, s.shape))

    f_shape = function([x, y], o_shape, mode=mode)

    topo = f_shape.maker.fgraph.toposort()
    assert any(isinstance(n.op, IncSubtensor) for n in topo)

    # This is an increment with a non-zero target array
    s = pt.ones((2, 2))[:, :]
    o_shape = inc_subtensor(s, specify_shape(y, s.shape))

    f_shape = function([y], o_shape, mode=mode)

    topo = f_shape.maker.fgraph.toposort()
    assert any(isinstance(n.op, IncSubtensor) for n in topo)


class TestLocalUselessSubtensor:
    x = matrix("x")
    s = ps.int32("s")
    mode = mode_opt.including(
        "local_useless_subtensor", "local_useless_AdvancedSubtensor1"
    )

    @pytest.mark.parametrize(
        "idx",
        [
            (slice(0, None),),
            (slice(0, None), slice(0, None)),
        ],
    )
    def test_local_useless_subtensor_1(self, idx):
        f = function([self.x], exp(self.x).__getitem__(idx), mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert prog[0].op == exp
        assert len(prog) == 1

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=pytensor.config.floatX)
        idx_val = idx
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val)
        assert np.allclose(res, exp_res)

    @pytest.mark.parametrize(
        "idx, res",
        [
            ((slice(0, 2),), True),
            ((slice(0, 2), slice(0, None)), True),
            ((slice(0, 2), slice(0, 3)), True),
            ((slice(0, None), slice(0, 3)), True),
            ((slice(0, 3), slice(0, 13)), True),
            ((slice(0, 3), slice(0, 2)), False),
            ((slice(0, 1), slice(0, None)), False),
            ((slice(0, 1), 1), False),
        ],
    )
    def test_local_useless_subtensor_2(self, idx, res):
        x_c = specify_shape(self.x, (2, 3))
        f = function([self.x], exp(x_c).__getitem__(idx), mode=self.mode)
        prog = f.maker.fgraph.toposort()
        if res:
            assert isinstance(prog[0].op, SpecifyShape)
            assert prog[1].op == exp
            assert len(prog) == 2
        else:
            assert any(isinstance(node.op, Subtensor) for node in prog)

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=pytensor.config.floatX)
        idx_val = idx
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val)
        assert np.allclose(res, exp_res)

    @pytest.mark.parametrize(
        "idx_fn, res",
        [
            (lambda x: (slice(0, x.shape[0]),), True),
            (lambda x: (slice(0, x.shape[1]),), False),
            (
                lambda x: (
                    slice(0, x.shape[0]),
                    slice(0, x.shape[1]),
                ),
                True,
            ),
            (
                lambda x: (
                    slice(0, x.shape[0]),
                    slice(0, x.shape[0]),
                ),
                False,
            ),
            (
                lambda x: (
                    slice(0, x.shape[1]),
                    slice(0, x.shape[0]),
                ),
                False,
            ),
            (
                lambda x: (
                    slice(0, x.shape[1]),
                    slice(0, x.shape[1]),
                ),
                False,
            ),
            (lambda x: (slice(0, x.shape[1]), 2), False),
            (
                lambda x: (
                    slice(0, x.shape[1]),
                    slice(x.shape[0] - x.shape[0], x.shape[1]),
                ),
                False,
            ),
            (
                lambda x: (
                    slice(
                        0,
                        pt.scalar_from_tensor(x.shape[0])
                        if isinstance(x, Variable)
                        else x.shape[0],
                    ),
                ),
                True,
            ),
        ],
    )
    def test_local_useless_subtensor_3(self, idx_fn, res):
        idx = idx_fn(self.x)
        f = function([self.x], exp(self.x).__getitem__(idx), mode=self.mode)
        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == exp
            assert len(prog) == 1
        else:
            assert any(isinstance(node.op, Subtensor) for node in prog)

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=pytensor.config.floatX)
        idx_val = idx_fn(x_val)
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val)
        assert np.allclose(res, exp_res)

    @pytest.mark.parametrize(
        "idx_fn, res",
        [
            (lambda x: (slice(0, x.shape[0]), slice(0, 3)), False),
            (lambda x: (slice(0, 3), slice(0, x.shape[1])), False),
        ],
    )
    def test_local_useless_subtensor_4(self, idx_fn, res):
        # Test mix Variable and Constant
        # Currently not supported
        x_c = specify_shape(self.x, (2, 3))
        idx = idx_fn(self.x)
        f = function([self.x], exp(x_c).__getitem__(idx), mode=self.mode)
        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == exp
            assert len(prog) == 1
        else:
            assert any(isinstance(node.op, Subtensor) for node in prog)

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=pytensor.config.floatX)
        idx_val = idx_fn(x_val)
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val)
        assert np.allclose(res, exp_res)

    @pytest.mark.parametrize(
        "idx_fn, res",
        [
            (lambda s: (slice(0, s),), False),
        ],
    )
    def test_local_useless_subtensor_5(self, idx_fn, res):
        # Test scalar variable
        idx = idx_fn(self.s)
        f = function([self.x, self.s], exp(self.x).__getitem__(idx), mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == exp
            assert len(prog) == 1
        else:
            assert any(isinstance(node.op, Subtensor) for node in prog)

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=pytensor.config.floatX)
        idx_val = idx_fn(1)
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val, 1)
        assert np.allclose(res, exp_res)

        idx_val = idx_fn(3)
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val, 3)
        assert np.allclose(res, exp_res)

    @pytest.mark.parametrize(
        "idx, res",
        [
            ([0, 1], True),
            ([1, 0], False),
            ([0, 0], False),
            ([0, 0, 1], False),
            (pt.arange(2), True),
            (pt.arange(0, 2), True),
            (pt.arange(0, 2, 2), False),
            (pt.arange(0, 2, -1), False),
            (pt.arange(1, 2), False),
        ],
    )
    def test_local_useless_subtensor_6(self, idx, res):
        # Test AdvancedSubtensor1 case when all rows are selected by a list/vector
        # or ARange op
        x_c = specify_shape(self.x, (2, 3))
        f = function([self.x], exp(x_c).__getitem__(idx), mode=mode_opt)
        prog = f.maker.fgraph.toposort()
        if res:
            assert isinstance(prog[0].op, SpecifyShape)
            assert prog[1].op == exp
            assert len(prog) == 2
        else:
            assert any(isinstance(node.op, AdvancedSubtensor1) for node in prog)

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=pytensor.config.floatX)
        idx_val = idx.eval() if isinstance(idx, Variable) else idx
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val)
        assert np.allclose(res, exp_res)


def test_local_subtensor_remove_broadcastable_index():
    # testing local_subtensor_remove_broadcastable_index optimization
    #
    # tests removing broadcastable dimensions with index 0 or -1,
    # otherwise the optimization should not be applied

    mode = get_default_mode()
    mode = mode.including("local_subtensor_remove_broadcastable_index")
    x = dmatrix("x")
    y1 = x.dimshuffle(0, "x", 1)
    y2 = x.dimshuffle("x", 1, 0, "x")
    y3 = x.dimshuffle("x", 1, "x", 0, "x")

    # testing for cases that the optimization should be applied
    z1 = y1[:, 0, :]
    z2 = y1[:, -1, :]
    z3 = y2[0, :, :, -1]
    z4 = y2[0, :, :, 0]
    z5 = y2[-1, :, :, -1]
    z6 = y3[-1, :, 0, :, -1]
    z7 = y3[-1, :, -1, :, -1]
    z8 = y3[0, :, 0, :, 0]
    f = function([x], [z1, z2, z3, z4, z5, z6, z7, z8], mode=mode)
    for elem in f.maker.fgraph.toposort():
        assert not isinstance(
            elem.op,
            Subtensor
            | AdvancedSubtensor
            | AdvancedSubtensor1
            | IncSubtensor
            | AdvancedIncSubtensor
            | AdvancedIncSubtensor1,
        )
    rng = np.random.default_rng(seed=utt.fetch_seed())
    xn = rng.random((5, 5))
    f(xn)

    # testing for cases that the optimization should not be applied
    # to verify that other subtensor usage are passed without errors
    w1 = y1[3, 0, :]
    w2 = y1[2:4, -1, :]
    w3 = y2[0, :, 4:, -1]
    w4 = y2[:, :, 0, -1]
    w5 = y2[0, 2:4, :, 0]
    w6 = y2[0, -1, :, -1]
    w7 = y2[-1, 4:, :, -1]
    w8 = y2[-1, :, :3, -1]
    w9 = y2[-1, :, -1, -1]
    w10 = y3[-1, 2, 0, :, -1]
    w11 = y3[-1, 0, -1, :, -1]
    w12 = y3[-1, :, -1, -1, -1]
    w13 = y3[0, 0, 0, :, 0]
    w14 = y3[-1, 2:4, 0, 1:5, -1]
    w15 = y3[-1, 0, -1, 0, -1]
    w16 = y3[0, 2, 0, 4, 0]
    w17 = y3[:, 0, :, 1]
    w18 = y3[0, :, :, 2]
    w19 = y3[:, 2, 0]
    w20 = y3[:, 3]
    f2 = function(
        [x],
        [
            w1,
            w2,
            w3,
            w4,
            w5,
            w6,
            w7,
            w8,
            w9,
            w10,
            w11,
            w12,
            w13,
            w14,
            w15,
            w16,
            w17,
            w18,
            w19,
            w20,
        ],
        mode=mode,
    )
    f2(xn)


class TestSubtensorIncSubtensor:
    @classmethod
    def setup_class(cls):
        cls.rng = np.random.default_rng(utt.fetch_seed())
        cls.mode = get_default_mode().including(
            "local_subtensor_inc_subtensor",
            "local_AdvancedIncSubtensor_to_AdvancedIncSubtensor1",
            "local_replace_AdvancedSubtensor",
        )

    @pytest.mark.parametrize(
        "val, indices, optype",
        [
            (vector(), (iscalar(),), IncSubtensor),
            (vector(), (ivector(),), AdvancedIncSubtensor1),
            (vector(), (ivector(), ivector()), AdvancedIncSubtensor),
        ],
    )
    def test_inplace(self, val, indices, optype):
        x = matrix("x")
        y = set_subtensor((2 * x)[indices], val, inplace=False)
        assert y.owner.op.inplace is False
        f = function(
            [x, val, *indices],
            y,
            mode=self.mode.including("inplace"),
        )
        assert isinstance(f.maker.fgraph.outputs[0].owner.op, optype)
        assert f.maker.fgraph.outputs[0].owner.op.inplace is True

    def test_basic(self):
        # basic test
        x = matrix("x")
        i = iscalar("i")
        v = vector("v")
        y = set_subtensor(x[i], v)
        z = y[i]
        f = function([x, i, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        # basic test, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(
            size=[
                4,
            ]
        ).astype(config.floatX)
        i_ = 1
        assert np.array_equal(f(x_, i_, v_), v_)

    def test_multiple_idx(self):
        # complicated test
        x = tensor4("x")
        i1 = iscalar("i1")
        i2 = iscalar("i2")
        i3 = iscalar("i3")
        i4 = iscalar("i4")
        v = tensor3("v")
        y = set_subtensor(x[i1, :i2, i3:, ::i4], v)
        z = y[i1, :i2, i3:, ::i4]
        f = function([x, i1, i2, i3, i4, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        # complicated test, numerical check
        x_ = np.random.uniform(size=[3, 4, 5, 6]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 2, 2]).astype(config.floatX)
        i1_, i2_, i3_, i4_ = 1, 2, 3, 4
        assert np.array_equal(f(x_, i1_, i2_, i3_, i4_, v_), v_)

    def test_not_applied(self):
        # case not use this optimization
        x = tensor4("x")
        i1 = iscalar("i1")
        i2 = iscalar("i2")
        i3 = iscalar("i3")
        i4 = iscalar("i4")
        v = tensor3("v")
        y = set_subtensor(x[i1, :i2, i3:, ::i4], v)
        z = y[i1, :i3, i2:, ::i4]
        f = function([x, i1, i2, i3, i4, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) != 1
        assert any(isinstance(x.op, IncSubtensor) for x in prog)
        assert any(isinstance(x.op, Subtensor) for x in prog)
        # case not use this optimization, numerical check
        x_ = np.random.uniform(size=[3, 4, 5, 6]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 2, 2]).astype(config.floatX)
        i1_, i2_, i3_, i4_ = 1, 2, 3, 4
        x_[i1_, :i2_, i3_:, ::i4_] = v_
        assert np.array_equal(f(x_, i1_, i2_, i3_, i4_, v_), x_[i1_, :i3_, i2_:, ::i4_])

    def test_fewer_dims(self):
        # case when v has fewer dimensions
        x = matrix("x")
        i1 = iscalar("i")
        i2 = iscalar("i")
        v = vector("v")
        y = set_subtensor(x[:i1, :i2], v)
        z = y[:i1, :i2]
        f = function([x, i1, i2, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert any(isinstance(x.op, Alloc) for x in prog)
        # case when v is broadcastable, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(
            size=[
                2,
            ]
        ).astype(config.floatX)
        i1_, i2_ = 2, 2
        x_[:i1_, :i2_] = v_
        assert np.array_equal(f(x_, i1_, i2_, v_), x_[:i1_, :i2_])

    def test_broadcasted(self):
        # case when v has the same number of dimensions, some broadcastable
        x = matrix("x")
        i1 = iscalar("i")
        i2 = iscalar("i")
        v = col("v")
        y = set_subtensor(x[:i1, :i2], v)
        z = y[:i1, :i2]
        f = function([x, i1, i2, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert any(isinstance(x.op, Alloc) for x in prog)
        # case when v is broadcastable, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 1]).astype(config.floatX)
        i1_, i2_ = 2, 2
        x_[:i1_, :i2_] = v_
        assert np.array_equal(f(x_, i1_, i2_, v_), x_[:i1_, :i2_])

    def test_different_dtypes(self):
        # Case when the dtype differs
        x = bmatrix("x")
        i = iscalar("i")
        v = vector("v")
        y = set_subtensor(x[i], v)
        z = y[i]
        f = function([x, i, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert prog[0].op == _convert_to_int8
        # basic test, numerical check
        x_ = self.rng.integers(12, size=[3, 4]).astype("int8")
        v_ = np.random.uniform(
            12,
            size=[
                4,
            ],
        ).astype(config.floatX)
        i_ = 1
        assert np.array_equal(f(x_, i_, v_), v_.astype("int8"))


class TestLocalSubtensorMerge:
    def setup_method(self):
        self.x_shapes = [(2, 2), (5, 3), (4, 1), (1, 2), (0, 2), (2, 0), (1, 0), (0, 0)]
        self.rng = np.random.default_rng(seed=utt.fetch_seed())

    def test_const(self):
        # var[const::][-1] -> var[-1]
        x = matrix("x")
        for idx in range(-7, 6):
            f = function([x], x[idx::][-1], mode=mode_opt)
            g = function(
                [x], x[idx::][-1], mode=mode_opt.excluding("local_subtensor_merge")
            )

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)

                if idx < x_s[0] and x_s[0] > 0:
                    # The first subtensor is non-empty, so it makes sense
                    f(x_val)  # let debugmode test something
                else:
                    # A non-empty subtensor of an empty one should be
                    # an IndexError
                    with pytest.raises(IndexError):
                        f(x_val)
                    with pytest.raises(IndexError):
                        g(x_val)

    def test_scalar(self):
        # var[int::][-1] -> var[-1]
        x = matrix("x")
        y = iscalar("y")
        f = function([x, y], x[y::][-1], mode=mode_opt)
        g = function(
            [x, y], x[y::][-1], mode=mode_opt.excluding("local_subtensor_merge")
        )

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)

            for idx in range(-9, 8):
                if (idx < x_s[0]) and (x_s[0] > 0):
                    # The first subtensor is non-empty
                    f(x_val, idx)  # let debugmode test something
                else:
                    with pytest.raises(IndexError):
                        f(x_val, idx)
                    with pytest.raises(IndexError):
                        g(x_val, idx)

    @pytest.mark.slow
    def test_const2(self):
        # var[::-1][const] -> var[-1]
        x = matrix("x")
        for idx in range(-8, 7):
            f = function([x], x[::-1][idx], mode=mode_opt)
            g = function(
                [x], x[::-1][idx], mode=mode_opt.excluding("local_subtensor_merge")
            )

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                if (idx < x_s[0]) and (idx >= -x_s[0]):
                    # The first subtensor is non-empty, so it makes sense
                    f(x_val)  # let debugmode test something
                else:
                    # A non-empty subtensor of an empty one should be
                    # an IndexError
                    with pytest.raises(IndexError):
                        f(x_val)
                    with pytest.raises(IndexError):
                        g(x_val)

    def test_scalar2(self):
        # var[::-1][int] -> var[-1]
        x = matrix("x")
        y = iscalar("y")
        f = function([x, y], x[::-1][y], mode=mode_opt)
        g = function(
            [x, y], x[::-1][y], mode=mode_opt.excluding("local_subtensor_merge")
        )

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)

            for idx in range(-x_s[0], x_s[0]):
                f(x_val, idx)  # let debugmode test something
            for idx in list(range(x_s[0], 9)) + list(range(-9, -x_s[0])):
                with pytest.raises(IndexError):
                    f(x_val, idx)
                with pytest.raises(IndexError):
                    g(x_val, idx)

    def test_const3(self):
        # var[::-1][:const] -> var[-1]
        x = matrix("x")
        for idx in range(-9, 8):
            f = function([x], x[::-1][:idx], mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                f(x_val)  # let debugmode test something

    def test_scalar3(self):
        # var[::-1][:int] -> var[-1]
        x = matrix("x")
        y = iscalar("y")
        f = function([x, y], x[::-1][:y], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for idx in range(-7, 7):
                f(x_val, idx)  # let debugmode test something

    def test_const4(self):
        # var[const1::][:const2]
        x = matrix("x")
        for idx1 in range(-7, 7):
            for idx2 in range(-7, 7):
                f = function([x], x[idx1:][:idx2], mode=mode_opt)

                # Check stacktrace was copied over correctly after opt was applied
                assert check_stack_trace(f, ops_to_check=Subtensor)

                topo = f.maker.fgraph.toposort()
                assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
                assert isinstance(topo[-1].op, DeepCopyOp)

                for x_s in self.x_shapes:
                    x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                    f(x_val)  # let debugmode test something

    def test_scalar4(self):
        # var[int1:][:int2]
        x = matrix("x")
        y = iscalar("y")
        z = iscalar("y")
        f = function([x, y, z], x[y:][:z], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for idx1 in range(-11, 11):
                for idx2 in range(-11, 11):
                    f(x_val, idx1, idx2)  # let debugmode test something

    def test_const_general(self):
        # Some cases of merge: shape, (start, stop, step) of first,
        # (start, stop, step) of second subtensor
        cases = [
            ((2, 3), (None, None, None), (None, None, -1)),
            ((12, 1), (None, None, -4), (None, None, 1)),
            ((5, 3), (1, 4, 2), (None, None, -1)),
        ]
        x = matrix("x")

        for s, sl1, sl2 in cases:
            z = x[slice(*sl1)][slice(*sl2)]
            f = function([x], z, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            x_val = self.rng.uniform(size=s).astype(config.floatX)
            f(x_val)

    def test_scalar5(self):
        # General case with two real slices
        # var[b1:e1:s1][b2:e2:s2]
        x = matrix("x")
        b1 = iscalar("b1")
        e1 = iscalar("e1")
        s1 = iscalar("s1")
        b2 = iscalar("b2")
        e2 = iscalar("e2")
        s2 = iscalar("s2")
        f = function([x, b1, e1, s1, b2, e2, s2], x[b1:e1:s1][b2:e2:s2], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        b1r = self.rng.permutation(list(range(-8, 8)))[:2]
        e1r = self.rng.permutation(list(range(-8, 8)))[:2]
        b2r = self.rng.permutation(list(range(-8, 8)))[:2]
        e2r = self.rng.permutation(list(range(-8, 8)))[:2]

        s1r = self.rng.permutation([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])[
            :2
        ]
        s2r = self.rng.permutation([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])[
            :2
        ]

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for b1 in b1r:
                for e1 in e1r:
                    for s1 in s1r:
                        for b2 in b2r:
                            for e2 in e2r:
                                for s2 in s2r:
                                    f(x_val, b1, e1, s1, b2, e2, s2)

    def test_const5(self):
        # Bug reported by Razvan
        data = np.asarray(np.arange(8), dtype=config.floatX)
        x = vector("x")
        y = x[7:1:-1]
        t = shared(np.int64(0))

        fun = function([x], y[t])

        val = fun(data)
        assert val == data[7:1:-1][0]

    def test_const6(self):
        # Bug reported by Graham
        data = self.rng.uniform(size=(8, 8, 8)).astype(config.floatX)
        x = tensor3("x")

        # test 1)
        y = x[3:6, 2:6, 1:7][1]
        fun = function([x], y)
        val = fun(data)
        assert np.all(val == data[3:6, 2:6, 1:7][1])
        assert (
            len([n for n in fun.maker.fgraph.toposort() if isinstance(n.op, Subtensor)])
            == 1
        )

        # test 2)
        y = x[2, 3][1]
        fun = function([x], y)
        val = fun(data)
        assert np.all(val == data[2, 3][1])
        assert (
            len([n for n in fun.maker.fgraph.toposort() if isinstance(n.op, Subtensor)])
            == 1
        )

        # test 3)
        y = x[3:6, 2, 1:7][1]
        fun = function([x], y)
        val = fun(data)
        assert np.all(val == data[3:6, 2, 1:7][1])
        assert (
            len([n for n in fun.maker.fgraph.toposort() if isinstance(n.op, Subtensor)])
            == 1
        )

    def test_scalar6(self):
        # General case with one slice and one index
        # var[b:e:s][i]
        x = matrix("x")
        b = iscalar("b")
        e = iscalar("e")
        s = iscalar("s")
        i = iscalar("i")
        f = function([x, b, e, s, i], x[b:e:s][i], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        b_r = self.rng.permutation(list(range(-4, 4)))[:3]
        e_r = self.rng.permutation(list(range(-4, 4)))[:3]
        i_r = self.rng.permutation(list(range(-4, 4)))[:3]

        s_r = self.rng.permutation([-3, -2, -1, 1, 2, 3])[:3]

        for x_s in self.x_shapes:
            n_index_err = 0
            n_ok = 0
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for b_v in b_r:
                for e_v in e_r:
                    for s_v in s_r:
                        for i_v in i_r:
                            # The index could be out of bounds
                            # In that case, an Exception should be raised,
                            # otherwise, we let DebugMode check f
                            try:
                                x_val[b_v:e_v:s_v][i_v]
                            except IndexError:
                                n_index_err += 1
                                with pytest.raises(IndexError):
                                    f(x_val, b_v, e_v, s_v, i_v)
                            else:
                                # Executed if the "try" clause did not raise
                                # any exception
                                n_ok += 1
                                f(x_val, b_v, e_v, s_v, i_v)

    @pytest.mark.slow
    def test_none_slice(self):
        # Test case of two slices, var[b1:e1:s1][b2:e2:s2]
        # where any of the b, e, and s can be None
        x = matrix("x")
        b1 = iscalar("b1")
        e1 = iscalar("e1")
        s1 = iscalar("s1")
        b2 = iscalar("b2")
        e2 = iscalar("e2")
        s2 = iscalar("s2")

        # Generate all possible lists of positions for None in those 6 slots
        # A 1 indicates None is present, 0 that there is an PyTensor scalar.
        none_positions = np.ndindex(2, 2, 2, 2, 2, 2)

        # Ranges to be used when not None
        b1r = self.rng.permutation(list(range(-4, 4)))[:]
        e1r = self.rng.permutation(list(range(-4, 4)))[:]
        b2r = self.rng.permutation(list(range(-4, 4)))[:]
        e2r = self.rng.permutation(list(range(-4, 4)))[:]
        s1r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]
        s2r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]

        scalar_vars = [b1, e1, s1, b2, e2, s2]
        scalar_ranges = [b1r, e1r, s1r, b2r, e2r, s2r]

        # For each case, we will build a graph, function, and list of values
        # Then, we test it on each input shape.
        for none_pos in none_positions:
            slice_inputs = []
            input_vars = []
            values = []
            if sum(none_pos) == 0:
                # Those case are already tested in test_scalar4
                continue

            for i, none_i in enumerate(none_pos):
                if none_i:
                    slice_inputs.append(None)
                else:
                    slice_inputs.append(scalar_vars[i])
                    input_vars.append(scalar_vars[i])
                    values.append(scalar_ranges[i])

            slice1 = slice(*slice_inputs[:3])
            slice2 = slice(*slice_inputs[3:])
            sub_x = x[slice1][slice2]
            f = function([x, *input_vars], sub_x, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            # for some cases, the optimization may remove all Subtensors,
            # which is why we pass "bug_print='ignore'".
            assert check_stack_trace(f, ops_to_check=Subtensor, bug_print="ignore")

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) <= 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                for i_val in zip(*values, strict=True):
                    f(x_val, *i_val)

    def test_none_index(self):
        # Test the general case of indexing into a subvector,
        # like x[b:e:s][i], where any of b, e, and s can be None
        x = matrix("x")
        b = iscalar("b")
        e = iscalar("e")
        s = iscalar("s")
        i = iscalar("i")

        # Generate all possible lists of positions for None in those 6 slots
        # A 1 indicates None is present, 0 that there is an PyTensor scalar.
        # The last index (i) is never None
        none_positions = np.ndindex(2, 2, 2, 1)

        # Ranges to be used when not None
        b_r = self.rng.permutation(list(range(-4, 4)))[:]
        e_r = self.rng.permutation(list(range(-4, 4)))[:]
        i_r = self.rng.permutation(list(range(-4, 4)))[:]
        s_r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]

        scalar_vars = [b, e, s, i]
        scalar_ranges = [b_r, e_r, s_r, i_r]

        # For each case, we will build a graph, function, and list of values
        # Then, we test it on each input shape.
        for none_pos in none_positions:
            slice_inputs = []
            input_vars = []
            values = []
            if sum(none_pos) == 0:
                # Those case are already tested in test_scalar6
                continue

            for j, none_j in enumerate(none_pos):
                if none_j:
                    slice_inputs.append(None)

                else:
                    slice_inputs.append(scalar_vars[j])
                    input_vars.append(scalar_vars[j])
                    values.append(scalar_ranges[j])

            symbol_slice = slice(*slice_inputs[:3])
            sub_x = x[symbol_slice][i]
            f = function([x, *input_vars], sub_x, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) <= 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                for i_val in zip(*values, strict=True):
                    # The index could be out of bounds
                    # In that case, an Exception should be raised,
                    # otherwise, we let DebugMode check f
                    # For that, we need to create a numerical slice.
                    i_val_idx = 0
                    num_slice_inputs = []
                    for none_j in none_pos:
                        if none_j:
                            num_slice_inputs.append(None)
                        else:
                            num_slice_inputs.append(i_val[i_val_idx])
                            i_val_idx += 1
                    num_slice = slice(*num_slice_inputs[:3])
                    num_i = num_slice_inputs[3]

                    try:
                        x_val[num_slice][num_i]
                    except IndexError:
                        with pytest.raises(IndexError):
                            f(x_val, *i_val)
                    else:
                        # Executed if the "try" clause did not raise
                        # any exception
                        f(x_val, *i_val)


class TestLocalAdvSub1AdvIncSub1:
    def setup_method(self):
        mode = get_default_mode()
        self.mode = mode.including(
            "local_replace_AdvancedSubtensor",
            "local_AdvancedIncSubtensor_to_AdvancedIncSubtensor1",
            "local_adv_sub1_adv_inc_sub1",
        ).excluding("fusion")
        self.mode_no_assert = self.mode.including("local_remove_all_assert")

    def test_basic(self):
        for dtype1, dtype2 in [
            ("float32", "float32"),
            ("float32", "float64"),
            ("float64", "float32"),
            ("float64", "float64"),
        ]:
            x = matrix(dtype=dtype1)
            y = matrix(dtype=dtype2)
            idx = ivector()

            dx = np.random.random((4, 5)).astype(dtype1)
            dy = np.random.random((2, 5)).astype(dtype2)
            # Duplicate the last row of dy
            dy = np.vstack([dy, dy[-1:]])
            # Use the same index twice, with the same corresponding value.
            # That makes set_subtensor well-defined, and tests
            # duplication for inc_subtensor.
            didx = np.asarray([1, 3, 3], "int32")

            # set_subtensor
            inc = set_subtensor(x[idx], y)
            o = inc[idx]
            f = function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            utt.assert_allclose(dy, res)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, DeepCopyOp | Elemwise)

            # inc_subtensor(data[idx], y)
            inc = inc_subtensor(x[idx], y)
            o = inc[idx]
            f = function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            _dx = dx.copy()
            np.add.at(_dx, didx, dy)
            utt.assert_allclose(_dx[didx], res)
            topo = f.maker.fgraph.toposort()
            len(topo) == 2

            # inc_subtensor(0[idx], y)
            inc = inc_subtensor(x.zeros_like()[idx], y)
            o = inc[idx]
            f = function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            utt.assert_allclose(np.vstack([dy[0], 2 * dy[1], 2 * dy[2]]), res)

    def test_assert(self):
        x = matrix("x")
        y = matrix("y")
        idx = ivector()

        dx = np.random.random((4, 5)).astype(config.floatX)
        dy = np.random.random((2, 5)).astype(config.floatX)

        # set_subtensor
        inc = set_subtensor(x[idx], y)
        o = inc[idx]
        f = function([x, y, idx], o, self.mode)
        # test wrong index
        for i in [dx.shape[0], -dx.shape[0] - 1]:
            with pytest.raises((AssertionError, IndexError)):
                f(dx, dy, [i, i])
        # test wrong shape
        with pytest.raises((AssertionError, IndexError)):
            f(dx, dy, [1])

    def test_stack_trace(self):
        x = matrix("x")
        # test cases with y.dtype
        # - equal to x.dtype
        # - different from x.dtype (to trigger the cast in
        #   local_adv_sub1_adv_inc_sub1)
        ys = [matrix("y"), dmatrix("y")]
        idx = ivector()

        # set_subtensor and then subtensor with both ys
        incs = [set_subtensor(x[idx], y) for y in ys]
        outs = [inc[idx] for inc in incs]

        for y, out in zip(ys, outs, strict=True):
            f = function([x, y, idx], out, self.mode)
            assert check_stack_trace(f, ops_to_check=(Assert, ps.Cast))


class TestSubtensorAllocRewrites:
    def setup_method(self):
        mode = get_default_mode()
        self.mode = mode.including(
            "local_incsubtensor_of_zeros",
            "local_setsubtensor_of_constants",
            "local_0_dot_x",
        )

    def test_setsubtensor_allocs0(self):
        x = matrix()
        y = matrix()
        x0 = pt.zeros_like(x)
        y0 = pt.zeros_like(y)
        z = set_subtensor(x0[:4], y0)
        f = function([x, y], z, mode=self.mode)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_setsubtensor_allocs1(self):
        y = matrix()
        x0 = pt.constant(np.asarray(np.zeros((4, 4)), dtype=config.floatX))
        y0 = pt.zeros_like(y)
        z = set_subtensor(x0[:4], y0)
        f = function([y], z, mode=self.mode)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_setsubtensor_allocs1t(self):
        y = matrix()
        x0 = pt.constant(np.asarray(np.zeros((4, 4)), dtype=config.floatX))
        y0 = pt.zeros_like(y)
        z = set_subtensor(x0[:4], y0.T)
        f = function([y], z, mode=mode_opt)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_setsubtensor_allocs2(self):
        x = matrix()
        y0 = pt.constant(np.asarray(np.zeros_like((4, 4)), dtype=config.floatX))
        x0 = pt.zeros_like(x)
        z = set_subtensor(x0[:4], y0)
        f = function([x], z, mode=self.mode)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_incsubtensor_allocs0(self):
        x = matrix()
        y = matrix()
        y0 = pt.zeros_like(y)
        z = inc_subtensor(x[:4], y0)
        f = function([x, y], z, mode=self.mode)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_incsubtensor_allocs0t(self):
        x = matrix()
        y = matrix()
        y0 = pt.zeros_like(y)
        z = inc_subtensor(x[:4], y0.T)
        f = function([x, y], z, mode=mode_opt)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_incsubtensor_allocs1(self):
        x = matrix()
        y0 = pt.constant(np.asarray(np.zeros_like((4, 4)), dtype=config.floatX))
        z = inc_subtensor(x[:4], y0)
        f = function([x], z, mode=self.mode)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_incsubtensor_x_zeros(self):
        x = pt.constant(np.asarray(np.zeros((4, 4)), dtype=config.floatX))
        y = matrix()
        z = inc_subtensor(x[:4], y)
        f = function([y], z)
        inc_nodes = [
            n for n in f.maker.fgraph.toposort() if isinstance(n.op, IncSubtensor)
        ]

        assert len(inc_nodes) == 1
        node_is_set_instead_of_inc = inc_nodes[0].op.set_instead_of_inc
        assert node_is_set_instead_of_inc
        test_X = np.random.random((4, 4)).astype(config.floatX)
        utt.assert_allclose(f(test_X), test_X)

        # also check the flag doesn't get set if first input is not zeros:
        not_all_zeros = np.zeros((4, 4))
        not_all_zeros[1, 0] = 0.001
        x = pt.constant(np.asarray(not_all_zeros, dtype=config.floatX))
        y = matrix()
        z = inc_subtensor(x[:4], y)
        f = function([y], z)
        inc_nodes = [
            n for n in f.maker.fgraph.toposort() if isinstance(n.op, IncSubtensor)
        ]
        assert len(inc_nodes) == 1
        assert inc_nodes[0].op.set_instead_of_inc is False
        test_X = np.random.random((4, 4)).astype(config.floatX)
        utt.assert_allclose(f(test_X), test_X + not_all_zeros)

    def test_advancedincsubtensor1_allocs0(self):
        x = matrix()
        y = matrix()
        y0 = pt.zeros_like(y)
        z = inc_subtensor(x[[0, 1, 2, 3]], y0)
        f = function([x, y], z, mode=self.mode)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor1)
            for n in f.maker.fgraph.toposort()
        )

    def test_advancedincsubtensor1_allocs0t(self):
        x = matrix()
        y = matrix()
        y0 = pt.zeros_like(y)
        z = inc_subtensor(x[[0, 1, 2, 3]], y0.T)
        f = function([x, y], z, mode=mode_opt)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor1)
            for n in f.maker.fgraph.toposort()
        )

    def test_advancedincsubtensor1_allocs1(self):
        x = matrix()
        y0 = pt.constant(np.asarray(np.zeros_like((4, 4)), dtype=config.floatX))
        z = inc_subtensor(x[[0, 1, 2, 3]], y0)
        f = function([x], z, mode=self.mode)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor1)
            for n in f.maker.fgraph.toposort()
        )

    def test_advancedincsubtensor_allocs0(self):
        x = matrix()
        y = matrix()
        y0 = pt.zeros_like(y)
        z = inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0)
        f = function([x, y], z, mode=self.mode)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor)
            for n in f.maker.fgraph.toposort()
        )

    def test_advancedincsubtensor_allocs0t(self):
        x = matrix()
        y = matrix()
        y0 = pt.zeros_like(y)
        z = inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0.T)
        f = function([x, y], z, mode=mode_opt)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor)
            for n in f.maker.fgraph.toposort()
        )

    def test_advancedincsubtensor_allocs1(self):
        x = matrix()
        y0 = pt.constant(np.asarray(np.zeros_like((2, 2)), dtype=config.floatX))
        z = inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0)
        f = function([x], z, mode=self.mode)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor)
            for n in f.maker.fgraph.toposort()
        )

    def test_dot_allocs_0(self):
        v1 = vector("v1")
        v2 = vector("v2")
        m1 = matrix("m1")
        m2 = matrix("m2")
        vv2 = np.asarray([0, 1], dtype=config.floatX)
        vm2 = np.asarray([[1, 2], [4, 5]], dtype=config.floatX)
        vv3 = np.asarray([0, 1, 2], dtype=config.floatX)
        vm3 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=config.floatX)
        for _e1 in [(v1, vv2, vv3), (m1, vm2, vm3)]:
            for _e2 in [(v2, vv2, vv3), (m2, vm2, vm3)]:
                for p in [0, 1]:
                    if p == 0:
                        e1 = pt.zeros_like(_e1[0])
                        e2 = _e2[0]
                    else:
                        e1 = _e1[0]
                        e2 = pt.zeros_like(_e2[0])
                    o = dot(e1, e2)
                    f = function([_e1[0], _e2[0]], o, mode=self.mode)
                    f(_e1[1], _e2[1])
                    f(_e1[2], _e2[2])
                    assert all(
                        not isinstance(n.op, Dot) for n in f.maker.fgraph.toposort()
                    )

                    # test that we don't remove shape errors
                    with pytest.raises((ValueError, AssertionError)):
                        f(_e1[1], _e2[2])
                    with pytest.raises((ValueError, AssertionError)):
                        f(_e1[2], _e2[1])


def test_local_IncSubtensor_serialize():
    d = np.random.normal(0, 0.01, size=(100, 100))
    d = d.astype(config.floatX)

    W = shared(d, name="W")
    i = vector("i", dtype="int64")
    j = vector("j", dtype="int64")
    t = scalar("t")
    y = (W[i] + W[j] + W[1] + W[i, j]).sum()
    cost = sqr(t - y)
    dW = pytensor.grad(cost, W)
    mode = get_default_mode().excluding("fusion")
    mode = mode.including("local_IncSubtensor_serialize")
    f = function([i, j, t], updates=[(W, W - 0.01 * dW)], mode=mode)
    topo = f.maker.fgraph.toposort()
    adds = [
        n
        for n in topo
        if isinstance(n.op, Elemwise) and isinstance(n.op.scalar_op, ps.Add)
    ]
    for a in adds:
        assert not any(
            inp.owner
            and isinstance(
                inp.owner.op,
                IncSubtensor | AdvancedIncSubtensor | AdvancedIncSubtensor1,
            )
            for inp in a.inputs
        )

    # Now test that the stack trace is copied over properly,
    # if we return the gradients. We need to use same mode as before.
    f = function([i, j, t], dW, mode=mode)
    assert check_stack_trace(
        f,
        ops_to_check=[
            IncSubtensor,
            AdvancedIncSubtensor,
            AdvancedIncSubtensor1,
        ],
    )


def test_local_set_to_inc_subtensor():
    v = fmatrix()
    s = v[[2, 1]]
    g = s + 3
    r = set_subtensor(s, g)

    mode = get_default_mode().including(
        "local_replace_AdvancedSubtensor",
        "local_AdvancedIncSubtensor_to_AdvancedIncSubtensor1",
    )
    moder = mode.excluding("local_set_to_inc_subtensor")
    modet = mode.including("local_set_to_inc_subtensor")
    f1 = function([v], r, mode=moder)
    f2 = function([v], r, mode=modet)

    advi1 = [
        n for n in f1.maker.fgraph.toposort() if isinstance(n.op, AdvancedIncSubtensor1)
    ]

    advi2 = [
        n for n in f2.maker.fgraph.toposort() if isinstance(n.op, AdvancedIncSubtensor1)
    ]

    # We only have SetSubtensor in f1
    assert all(n.op.set_instead_of_inc for n in advi1)
    # We don't have any SetSubtensor in f2
    assert not any(n.op.set_instead_of_inc for n in advi2)

    val = np.random.standard_normal((3, 2)).astype("float32")

    r1 = f1(val)
    r2 = f2(val)

    utt.assert_allclose(r1, r2)

    # Finally, test that the stack trace is copied over properly,
    # before and after optimization.
    assert check_stack_trace(f1, ops_to_check=AdvancedIncSubtensor1)
    assert check_stack_trace(f2, ops_to_check="all")


@pytest.mark.parametrize(
    "axis, slices_fn, expected_nodes",
    [
        # Below should be merged
        (0, lambda _: ((slice(None, 5, None),), (slice(5, None, None),)), 1),
        (0, lambda _: ((slice(0, 5, 1),), (slice(5, None, 1),)), 1),
        (
            0,
            lambda _: (
                (slice(0, 2, 1),),
                (slice(2, 4, None),),
                (slice(4, None, 1)),
            ),
            1,
        ),
        (
            0,
            lambda _: (
                (slice(None, 5, None), slice(None, -1, None)),
                (slice(5, None, None), slice(None, -1, None)),
            ),
            2,
        ),
        (
            1,
            lambda step: (
                (slice(2, None, step), slice(None, 2, None)),
                (slice(2, None, step), slice(2, 4, None)),
                (slice(2, None, step), slice(4, 6, None)),
            ),
            3,
        ),
        (
            0,
            lambda stop: (
                (slice(1, stop, None),),
                (slice(stop, 5, None),),
                (slice(5, 7, None)),
            ),
            2,
        ),
        (
            0,
            lambda stop: (
                (slice(1, stop + 1, None),),
                (slice(stop + 1, 5, None),),
                (slice(5, 7, None)),
            ),
            2,
        ),
        # Below NotImplemented: These could be merged, but we would need to evaluate the
        # start and stop values
        (0, lambda _: ((slice(None, 6, 3),), (slice(6, None, 3),)), 3),
        (0, lambda step: ((slice(None, 6, step),), (slice(6, None, step),)), 4),
        # Below should not be merged
        (0, lambda _: ((slice(5, None, None),), (slice(None, 5, None),)), 3),
        (0, lambda _: ((slice(None, 5, None),), (slice(4, None, None),)), 3),
        (1, lambda _: ((slice(None, 5, None),), (slice(5, None, None),)), 3),
        (
            0,
            lambda _: (
                (slice(2, None, None), slice(None, 2, None)),
                (slice(2, None, None), slice(2, 4, None)),
                (slice(2, None, None), slice(4, 6, None)),
            ),
            4,
        ),
        (
            0,
            lambda _: (
                (slice(None, 5, 2), slice(None, -1, None)),
                (slice(5, None, 3), slice(None, -1, None)),
            ),
            3,
        ),
        (
            0,
            lambda _: (
                (slice(None, 5, None), slice(None, -1, None)),
                (slice(5, None, None), slice(1, None, None)),
            ),
            3,
        ),
        (0, lambda stop: ((slice(None, stop, None),), (slice(3, None, None),)), 4),
        (0, lambda _: ((slice(None, 5, 2),), (slice(5, None, 2),)), 3),
    ],
)
def test_local_join_subtensors(axis, slices_fn, expected_nodes):
    x = pt.dmatrix("x")
    slice_scalar = pt.iscalar("slice_scalar")
    slices = slices_fn(slice_scalar)
    y = pt.concatenate([x[slice] for slice in slices], axis=axis)
    f = pytensor.function(
        [x, slice_scalar],
        y,
        mode=Mode("py").excluding("fusion"),
        on_unused_input="ignore",
    )
    nodes = f.maker.fgraph.toposort()
    assert len(nodes) == expected_nodes, nodes

    x_val = np.arange(100).reshape(10, 10)
    stop_val = 3
    slices_val = slices_fn(stop_val)
    f_val = np.concatenate([x_val[slice_val] for slice_val in slices_val], axis=axis)

    np.testing.assert_array_equal(f(x_val, stop_val), f_val)


def test_local_uint_constant_indices():
    mode = get_default_mode().including("specialize", "local_uint_constant_indices")
    rng = np.random.default_rng(20900)

    # Subtensor, don't convert
    x = pt.vector("x")
    idx = pt.as_tensor_variable(np.array(-1, np.int64))
    z = x[idx]

    z_fn = pytensor.function([x], z, mode=mode)

    deepcopy_node = z_fn.maker.fgraph.outputs[0].owner
    subtensor_node = deepcopy_node.inputs[0].owner
    assert isinstance(subtensor_node.op, Subtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "int64"

    # `Subtensor`, one index, convert
    x = pt.vector("x")
    idx = pt.as_tensor_variable(np.array(1, np.int64))
    z = x[idx]

    z_fn = pytensor.function([x], z, mode=mode)

    deepcopy_node = z_fn.maker.fgraph.outputs[0].owner
    subtensor_node = deepcopy_node.inputs[0].owner
    assert isinstance(subtensor_node.op, Subtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # `Subtensor`, two indices, one slice, convert
    x = pt.matrix("x")
    indices = (pt.as_tensor_variable(np.array(1, np.int64)), slice(None, 10))
    z = x[indices]

    z_fn = pytensor.function([x], z, mode=mode)

    deepcopy_node = z_fn.maker.fgraph.outputs[0].owner
    subtensor_node = deepcopy_node.inputs[0].owner
    assert isinstance(subtensor_node.op, Subtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # `AdvancedSubtensor`, two indices, one symbolic slice, convert
    x = pt.matrix("x")
    indices = (
        pt.as_tensor_variable(np.array(1, np.int64)),
        make_slice(slice(None, 10)),
    )
    z = x[indices]

    z_fn = pytensor.function([x], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedSubtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # `AdvancedSubtensor1`, convert
    x = pt.vector("x")
    idx = pt.as_tensor_variable(rng.integers(0, 10, size=10).astype(np.int64))
    z = x[idx]

    z_fn = pytensor.function([x], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedSubtensor1)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # AdvancedSubtensor, empty, convert
    x = pt.matrix("x")
    idx = pt.as_tensor_variable(1, dtype=np.int64)
    z = x[idx, []]

    z_fn = pytensor.function([x], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedSubtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # AdvancedSubtensor, bool, don't convert
    x = pt.matrix("x")
    idx = pt.as_tensor_variable(np.array([True]), dtype=bool)
    z = x[idx, []]

    z_fn = pytensor.function([x], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedSubtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "bool"

    # `IncSubtensor`, convert
    x = pt.vector("x")
    y = pt.scalar("y")
    idx = pt.as_tensor_variable(1, dtype=np.int64)
    z = inc_subtensor(x[idx], y)

    z_fn = pytensor.function([x, y], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, IncSubtensor)
    new_index = subtensor_node.inputs[2]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # `AdvancedIncSubtensor1`, convert
    x = pt.vector("x")
    y = pt.vector("y")
    idx = pt.as_tensor_variable(rng.integers(0, 10, size=10).astype(np.int64))
    z = advanced_inc_subtensor1(x, y, idx)

    z_fn = pytensor.function([x, y], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedIncSubtensor1)
    new_index = subtensor_node.inputs[2]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # `AdvancedIncSubtensor1`, convert
    x = pt.vector("x")
    idx = pt.as_tensor_variable(rng.integers(0, 10, size=10).astype(np.int64))
    z = x[idx, None]

    z_fn = pytensor.function([x], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedSubtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"


@pytest.mark.parametrize("core_y_implicitly_batched", (False, True))
@pytest.mark.parametrize("set_instead_of_inc", (True, False))
def test_local_blockwise_advanced_inc_subtensor(
    set_instead_of_inc, core_y_implicitly_batched
):
    rng = np.random.default_rng([1764, set_instead_of_inc, core_y_implicitly_batched])

    def np_inplace_f(x, idx, y):
        if core_y_implicitly_batched:
            y = y[..., None]
        if set_instead_of_inc:
            x[idx] = y
        else:
            x[idx] += y

    core_y_shape = () if core_y_implicitly_batched else (3,)
    core_x = tensor("x", shape=(6,))
    core_y = tensor("y", shape=core_y_shape, dtype=int)
    core_idxs = [0, 2, 4]
    if set_instead_of_inc:
        core_graph = set_subtensor(core_x[core_idxs], core_y)
    else:
        core_graph = inc_subtensor(core_x[core_idxs], core_y)

    # Only x is batched
    x = tensor("x", shape=(5, 2, 6))
    y = tensor("y", shape=core_y_shape, dtype=int)
    out = vectorize_graph(core_graph, replace={core_x: x, core_y: y})
    assert isinstance(out.owner.op, Blockwise)

    fn = pytensor.function([x, y], out, mode="FAST_RUN")
    assert not any(
        isinstance(node.op, Blockwise) for node in fn.maker.fgraph.apply_nodes
    )

    test_x = np.ones(x.type.shape, dtype=x.type.dtype)
    test_y = rng.integers(1, 10, size=y.type.shape, dtype=y.type.dtype)
    expected_out = test_x.copy()
    np_inplace_f(expected_out, np.s_[:, :, core_idxs], test_y)
    np.testing.assert_allclose(fn(test_x, test_y), expected_out)

    # Only y is batched
    x = tensor("y", shape=(6,))
    y = tensor("y", shape=(2, *core_y_shape), dtype=int)
    out = vectorize_graph(core_graph, replace={core_x: x, core_y: y})
    assert isinstance(out.owner.op, Blockwise)

    fn = pytensor.function([x, y], out, mode="FAST_RUN")
    assert not any(
        isinstance(node.op, Blockwise) for node in fn.maker.fgraph.apply_nodes
    )

    test_x = np.ones(x.type.shape, dtype=x.type.dtype)
    test_y = rng.integers(1, 10, size=y.type.shape, dtype=y.type.dtype)
    expected_out = np.ones((2, *x.type.shape))
    np_inplace_f(expected_out, np.s_[:, core_idxs], test_y)
    np.testing.assert_allclose(fn(test_x, test_y), expected_out)

    # Both x and y are batched, and do not need to be broadcasted
    x = tensor("y", shape=(2, 6))
    y = tensor("y", shape=(2, *core_y_shape), dtype=int)
    out = vectorize_graph(core_graph, replace={core_x: x, core_y: y})
    assert isinstance(out.owner.op, Blockwise)

    fn = pytensor.function([x, y], out, mode="FAST_RUN")
    assert not any(
        isinstance(node.op, Blockwise) for node in fn.maker.fgraph.apply_nodes
    )

    test_x = np.ones(x.type.shape, dtype=x.type.dtype)
    test_y = rng.integers(1, 10, size=y.type.shape, dtype=y.type.dtype)
    expected_out = test_x.copy()
    np_inplace_f(expected_out, np.s_[:, core_idxs], test_y)
    np.testing.assert_allclose(fn(test_x, test_y), expected_out)

    # Both x and y are batched, but must be broadcasted
    x = tensor("y", shape=(5, 1, 6))
    y = tensor("y", shape=(1, 2, *core_y_shape), dtype=int)
    out = vectorize_graph(core_graph, replace={core_x: x, core_y: y})
    assert isinstance(out.owner.op, Blockwise)

    fn = pytensor.function([x, y], out, mode="FAST_RUN")
    assert not any(
        isinstance(node.op, Blockwise) for node in fn.maker.fgraph.apply_nodes
    )

    test_x = np.ones(x.type.shape, dtype=x.type.dtype)
    test_y = rng.integers(1, 10, size=y.type.shape, dtype=y.type.dtype)
    final_shape = (
        *np.broadcast_shapes(x.type.shape[:2], y.type.shape[:2]),
        x.type.shape[-1],
    )
    expected_out = np.broadcast_to(test_x, final_shape).copy()
    np_inplace_f(expected_out, np.s_[:, :, core_idxs], test_y)
    np.testing.assert_allclose(fn(test_x, test_y), expected_out)


class TestUselessSlice:
    def test_positive_step(self):
        # When steps are positive, default start and end are `0` and `len(dim)`
        x = tensor(shape=(3, 5, None, 9), dtype="float64")
        test_x = np.random.normal(size=(3, 5, 8, 9))

        y = x[0:3:1, 1:5:2, 0:7:1, 0:9:1]
        f = pytensor.function([x], y)

        # Get the DeepCopy input and assert that the Op is a DeepCopy
        deep_copy_node = f.maker.fgraph.outputs[0].owner
        assert isinstance(deep_copy_node.op, DeepCopyOp)

        rewritten_y = deep_copy_node.inputs[0]
        expected_y = x[None:None:None, 1:None:2, None:7:None]
        assert equal_computations([rewritten_y], [expected_y])

        np.testing.assert_allclose(
            f(test_x),
            # Use the unoptimized slice to make sure our rewrite logic is correct
            test_x[0:3:1, 1:5:2, 0:7:1, 0:9:1],
        )

    def test_negative_step(self):
        # When steps are negative, default start and end are `-1` and `-len(dim) - 1`
        x = tensor(shape=(3, 5, None, 9), dtype="float64")
        test_x = np.random.normal(size=(3, 5, 8, 9))

        y = x[-1:-4:-1, 0:5:-2, -1:-9:-1, 0:9:None]
        f = pytensor.function([x], y)

        # Get the DeepCopy input and assert that the Op is a DeepCopy
        deep_copy_node = f.maker.fgraph.outputs[0].owner
        assert isinstance(deep_copy_node.op, DeepCopyOp)

        rewritten_y = deep_copy_node.inputs[0]
        expected_y = x[None:None:-1, 0:5:-2, None:-9:-1]
        assert equal_computations([rewritten_y], [expected_y])

        np.testing.assert_allclose(
            f(test_x),
            test_x[-1:-4:-1, 0:5:-2, -1:-9:-1, 0:9:None],
        )

    def test_unknown_step(self):
        # If step isn't known, we can't canonicalize start and stop points
        step = pt.scalar("step", dtype=int)
        x = tensor(shape=(3, 5, None), dtype="float64")
        test_x = np.random.normal(size=(3, 5, 7))

        y = x[0:3:step, -1:-6:-step, ::]
        # Need this rewrite when `FAST_COMPILE` otherwise step = -1 * step instead of neg(step)
        mode = get_default_mode().including("local_mul_specialize")
        f = pytensor.function([x, step], y, mode=mode)

        # Get the DeepCopy input and assert that the Op is a DeepCopy
        deep_copy_node = f.maker.fgraph.outputs[0].owner
        assert isinstance(deep_copy_node.op, DeepCopyOp)

        rewritten_y = deep_copy_node.inputs[0]
        expected_y = x[0:3:step, -1:-6:-step]
        assert equal_computations([rewritten_y], [expected_y])

        np.testing.assert_allclose(
            f(test_x, 1),
            test_x[0:3:1, -1:-6:-1, ::],
        )
        np.testing.assert_allclose(
            f(test_x, -2),
            test_x[0:3:-2, -1:-6:2, ::],
        )


def test_extract_diag_of_diagonal_set_subtensor():
    A = pt.full((2, 6, 6), np.nan)
    rows = pt.arange(A.shape[-2])
    cols = pt.arange(A.shape[-1])
    write_offsets = [-2, -1, 0, 1, 2]
    # Randomize order of write operations, to make sure rewrite is not sensitive to it
    random.shuffle(write_offsets)
    for offset in write_offsets:
        value = offset + 0.1 * offset
        if offset == 0:
            A = A[..., rows, cols].set(value)
        elif offset > 0:
            A = A[..., rows[:-offset], cols[offset:]].set(value)
        else:
            offset = -offset
            A = A[..., rows[offset:], cols[:-offset]].set(value)
    # Add a partial diagonal along offset 3
    A = A[..., rows[1:-3], cols[4:]].set(np.pi)

    read_offsets = [-2, -1, 0, 1, 2, 3]
    outs = [A.diagonal(offset=offset, axis1=-2, axis2=-1) for offset in read_offsets]
    rewritten_outs = rewrite_graph(outs, include=("ShapeOpt", "canonicalize"))

    # Every output should just be an Alloc with value
    expected_outs = []
    for offset in read_offsets[:-1]:
        value = np.asarray(offset + 0.1 * offset, dtype=A.type.dtype)
        expected_outs.append(pt.full((np.int64(2), np.int8(6 - abs(offset))), value))
    # The partial diagonal shouldn't be rewritten
    expected_outs.append(outs[-1])

    assert equal_computations(rewritten_outs, expected_outs)


def test_local_convert_negative_indices():
    x = pt.tensor("x", shape=(None, 3, 1))

    # Dim length is unknown rewrite can't be applied
    rewritten_out = rewrite_graph(x[-2])
    assert equal_computations([rewritten_out], [x[-2]])

    # Rewrite applies
    rewritten_out = rewrite_graph(x[:, -2])
    assert equal_computations([rewritten_out], [x[:, 1]])

    # Rewrite doesn't apply because index is invalid
    # TODO: If Subtensor decides to raise on make_node, this test can be removed
    rewritten_out = rewrite_graph(x[:, :, -2])
    assert equal_computations([rewritten_out], [x[:, :, -2]])

"""
Provides `DebugMode`, an evaluation mode for debugging pytensor internals.

TODO: add support for IfElse Op, LazyLinker, etc.

"""

import copy
import gc
import logging
import sys
from io import StringIO
from itertools import chain
from itertools import product as itertools_product
from logging import Logger
from warnings import warn

import numpy as np

import pytensor
from pytensor.compile.function.types import (
    Function,
    FunctionMaker,
    infer_reuse_pattern,
    std_fgraph,
)
from pytensor.compile.mode import Mode, register_mode
from pytensor.compile.ops import OutputGuard, _output_guard
from pytensor.configdefaults import config
from pytensor.graph.basic import Variable, io_toposort
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.graph.features import AlreadyThere, BadOptimization
from pytensor.graph.fg import Output
from pytensor.graph.op import HasInnerGraph, Op
from pytensor.graph.utils import InconsistencyError, MethodNotDefined
from pytensor.link.basic import Container, LocalLinker
from pytensor.link.c.op import COp
from pytensor.link.utils import map_storage, raise_with_op
from pytensor.printing import _debugprint
from pytensor.tensor import TensorType
from pytensor.utils import NoDuplicateOptWarningFilter, difference, get_unbound_function


__docformat__ = "restructuredtext en"
_logger: Logger = logging.getLogger("pytensor.compile.debugmode")
_logger.addFilter(NoDuplicateOptWarningFilter())


class DebugModeError(Exception):
    """
    Generic Exception raised to indicate an internal pytensor problem.

    """


class BadThunkOutput(DebugModeError):
    """
    Exception: Calling the same Op twice gives inconsistent outputs.

    It can be raised, for instance, if an Op's c_code and perform method
    do not agree, or if one of these methods do not give the same result
    when called twice with the same inputs (but different memory layouts
    for the output).

    """

    r = None
    """
    The `Variable` instance for which conflicting values were computed.

    """

    thunk1 = ""
    val1 = None
    """
    The value computed by `thunk1`.

    """

    thunk2 = ""
    val2 = None
    """
    The value computed by `thunk2`.

    """

    def __init__(self, r, thunk1, val1, thunk2, val2, inputs_val=()):
        super().__init__()
        self.r = r
        self.thunk1 = thunk1
        self.val1 = val1
        self.thunk2 = thunk2
        self.val2 = val2
        self.inputs_val = inputs_val

    def offending_op(self):
        """
        Return the Op class whose c_code and perform implementations
        didn't match.

        """
        return type(self.r.owner.op)

    def __str__(self):
        return self.str_diagnostic()

    def str_diagnostic(self):
        """
        Return a pretty multiline string representing the cause of the exception.

        """
        sio = StringIO()
        print("BadThunkOutput", file=sio)
        print("  Apply   :", self.r.owner, file=sio)
        print("  op      :", self.offending_op(), file=sio)
        print("  Outputs Type:", self.r.type, file=sio)
        print("  Outputs Shape:", getattr(self.val1, "shape", None), file=sio)
        print("  Outputs Strides:", getattr(self.val1, "strides", None), file=sio)
        print("  Inputs Type :", [i.type for i in self.r.owner.inputs], file=sio)
        print(
            "  Inputs Shape:",
            [getattr(val, "shape", None) for val in self.inputs_val],
            file=sio,
        )
        print(
            "  Inputs Strides:",
            [getattr(val, "strides", None) for val in self.inputs_val],
            file=sio,
        )
        scalar_values = []
        for ipt in self.inputs_val:
            if getattr(ipt, "size", -1) <= 10:
                scalar_values.append(ipt)
            else:
                scalar_values.append("not shown")
        print(f"  Inputs values: {scalar_values}", file=sio)
        print("  Bad Variable:", self.r, file=sio)
        print("  thunk1  :", self.thunk1, file=sio)
        print("  thunk2  :", self.thunk2, file=sio)

        print(str_diagnostic(self.val1, self.val2, None, None), file=sio)
        ret = sio.getvalue()
        return ret


class BadOptimization(DebugModeError, BadOptimization):
    pass


class BadDestroyMap(DebugModeError):
    """
    Exception: Some perform() or c_code() modified an input that
    wasn't in the destroy_map.

    """

    def __init__(self, node, idx, old_val, new_val, perform):
        super().__init__()
        self.node = node
        self.idx = idx
        self.old_val = old_val
        self.new_val = new_val
        self.perform = perform

    def __str__(self):
        sio = StringIO()
        print("  node:", self.node, file=sio)
        print("  perform:", self.perform, file=sio)
        print("  node.inputs:", [(str(i), id(i)) for i in self.node.inputs], file=sio)
        print("  destroy_map:", self.node.op.destroy_map, file=sio)
        print("  changed input idx:", self.idx, file=sio)
        print("  changed input type:", self.node.inputs[self.idx].type, file=sio)
        print("  repr (old val):", repr(self.old_val), file=sio)
        print("  repr (new val):", repr(self.new_val), file=sio)
        try:
            npy_old_val = np.asarray(self.old_val)
            npy_new_val = np.asarray(self.new_val)
            print(
                "  value dtype (new <space> old):",
                npy_new_val.dtype,
                npy_old_val.dtype,
                file=sio,
            )
            print(
                "  value shape (new <space> old):",
                npy_new_val.shape,
                npy_old_val.shape,
                file=sio,
            )
            print(
                "  value min (new <space> old):",
                npy_new_val.min(),
                npy_old_val.min(),
                file=sio,
            )
            print(
                "  value max (new <space> old):",
                npy_new_val.max(),
                npy_old_val.max(),
                file=sio,
            )
            delta = npy_new_val - npy_old_val
            print("  value min (new-old):", delta.min(), file=sio)
            print("  value max (new-old):", delta.max(), file=sio)
            print(
                "  value argmin (new-old):",
                np.unravel_index(delta.argmin(), npy_new_val.shape),
                file=sio,
            )
            print(
                "  value argmax (new-old):",
                np.unravel_index(delta.argmax(), npy_new_val.shape),
                file=sio,
            )
            print(
                "  location of first 10 mismatches:",
                np.transpose(np.nonzero(delta))[:10],
                file=sio,
            )
            print("", file=sio)
        except Exception as e:
            print(f"(Numpy-hints failed with: {e})", file=sio)
        print(
            "  Hint: this can also be caused by a deficient "
            "values_eq_approx() or __eq__() implementation "
            "[which compared input values]",
            file=sio,
        )
        return sio.getvalue()


class BadViewMap(DebugModeError):
    """
    Exception: Some perform() or c_code() created a memory alias
    that wasn't in the view_map.

    """

    def __init__(
        self, node, output_idx, out_storage, in_alias_idx=None, out_alias_idx=None
    ):
        super().__init__()
        self.node = node
        self.output_idx = output_idx
        self.out_storage = out_storage
        self.in_alias_idx = in_alias_idx
        self.out_alias_idx = out_alias_idx

    def __str__(self):
        sio = StringIO()
        print("  node:", self.node, file=sio)
        print("  node.inputs:", [(str(i), id(i)) for i in self.node.inputs], file=sio)
        print("  node.outputs:", [(str(i), id(i)) for i in self.node.outputs], file=sio)
        print("  view_map:", self.node.op.view_map, file=sio)
        print("  destroy_map:", self.node.op.destroy_map, file=sio)
        print("  aliased output:", self.output_idx, file=sio)
        print("  aliased output storage:", self.out_storage, file=sio)
        if self.in_alias_idx:
            print("  aliased to inputs:", self.in_alias_idx, file=sio)
        if self.out_alias_idx:
            print("  aliased to outputs:", self.out_alias_idx, file=sio)
        return sio.getvalue()


class StochasticOrder(DebugModeError):
    """
    Exception: Repeated Optimizations of the same graph do not give
    identical results.

    The most common cause is that an Optimization iterates over some
    objects in a memory-address-dependent order (such as id() or
    object.hash()).

    """


class InvalidValueError(DebugModeError):
    """
    Exception: some Op an output value that is inconsistent with
    the Type of that output.

    Note: If there is only one parameter and it is a string, then we
    will use it as the error message. This is needed when we catch,
    extend, and reraise an error.
    """

    def __init__(self, r, v=None, client_node=None, hint="none", specific_hint="none"):
        super().__init__()
        self.r = r
        self.v = v
        self.client_node = client_node
        self.hint = hint
        self.specific_hint = specific_hint

        # To allow extending th error message of an existing error.
        self.full_err = None
        if isinstance(r, str):
            assert (
                v is None
                and client_node is None
                and hint == "none"
                and specific_hint == "none"
            )
            self.full_err = r

    def __str__(self):
        # We have a pre-made message
        if getattr(self, "full_err", None) is not None:
            return self.full_err

        r, v = self.r, self.v
        type_r = r.type
        type_v = type(v)
        v_val = str(v)[0:100]
        v_dtype = "N/A"
        v_shape = "N/A"
        v_min = "N/A"
        v_max = "N/A"
        v_isfinite = "N/A"
        try:
            v_shape = v.shape
            v_dtype = v.dtype
            v_min = v.min()
            v_max = v.max()
            v_isfinite = np.all(np.isfinite(v))
        except Exception:
            pass
        client_node = self.client_node
        hint = self.hint
        specific_hint = self.specific_hint
        context = _debugprint(r, prefix="  ", depth=12, file=StringIO()).getvalue()
        return f"""InvalidValueError
        type(variable) = {type_r}
        variable       = {r}
        type(value)    = {type_v}
        dtype(value)   = {v_dtype}
        shape(value)   = {v_shape}
        value          = {v_val}
        min(value)     = {v_min}
        max(value)     = {v_max}
        isfinite       = {v_isfinite}
        client_node    = {client_node}
        hint           = {hint}
        specific_hint  = {specific_hint}
        context        = ...
{context}
        """


def str_diagnostic(expected, value, rtol, atol):
    """Return a pretty multiline string representing the cause of the exception."""
    sio = StringIO()

    try:
        ssio = StringIO()
        print("           : shape, dtype, strides, min, max, n_inf, n_nan:", file=ssio)
        print("  Expected :", end=" ", file=ssio)
        print(expected.shape, end=" ", file=ssio)
        print(expected.dtype, end=" ", file=ssio)
        print(expected.strides, end=" ", file=ssio)
        print(expected.min(), end=" ", file=ssio)
        print(expected.max(), end=" ", file=ssio)
        print(np.isinf(expected).sum(), end=" ", file=ssio)
        print(np.isnan(expected).sum(), end=" ", file=ssio)
        # only if all succeeds to we add anything to sio
        print(ssio.getvalue(), file=sio)
    except Exception:
        pass
    try:
        ssio = StringIO()
        print("  Value    :", end=" ", file=ssio)
        print(value.shape, end=" ", file=ssio)
        print(value.dtype, end=" ", file=ssio)
        print(value.strides, end=" ", file=ssio)
        print(value.min(), end=" ", file=ssio)
        print(value.max(), end=" ", file=ssio)
        print(np.isinf(value).sum(), end=" ", file=ssio)
        print(np.isnan(value).sum(), end=" ", file=ssio)
        # only if all succeeds to we add anything to sio
        print(ssio.getvalue(), file=sio)
    except Exception:
        pass

    print("  expected    :", expected, file=sio)
    print("  value    :", value, file=sio)

    try:
        ov = np.asarray(expected)
        nv = np.asarray(value)
        ssio = StringIO()
        absdiff = np.absolute(nv - ov)
        print("  Max Abs Diff: ", np.max(absdiff), file=ssio)
        print("  Mean Abs Diff: ", np.mean(absdiff), file=ssio)
        print("  Median Abs Diff: ", np.median(absdiff), file=ssio)
        print("  Std Abs Diff: ", np.std(absdiff), file=ssio)
        reldiff = np.absolute(nv - ov) / np.absolute(ov)
        print("  Max Rel Diff: ", np.max(reldiff), file=ssio)
        print("  Mean Rel Diff: ", np.mean(reldiff), file=ssio)
        print("  Median Rel Diff: ", np.median(reldiff), file=ssio)
        print("  Std Rel Diff: ", np.std(reldiff), file=ssio)
        # only if all succeeds to we add anything to sio
        print(ssio.getvalue(), file=sio)
    except Exception:
        pass
    atol_, rtol_ = pytensor.tensor.math._get_atol_rtol(expected, value)
    if rtol is not None:
        rtol_ = rtol
    if atol is not None:
        atol_ = atol
    print("  rtol, atol:", rtol_, atol_, file=sio)
    return sio.getvalue()


def _optcheck_fgraph(input_specs, output_specs, accept_inplace=False):
    """
    Create a FunctionGraph for debugging.

    Parameters
    ----------
    input_specs: WRITEME
        fgraph inputs.
    output_specs: WRITEME
        fgraph outputs.
    accept_inplace : bool
        Are inplace ops permitted in the original graph?

    Returns
    -------
    FunctionGraph
        A new FunctionGraph with a cloned graph, with debugging `Feature`
        instances already installed.

    """
    equivalence_tracker = _VariableEquivalenceTracker()
    fgraph, updates = std_fgraph(
        input_specs, output_specs, accept_inplace, force_clone=True
    )
    fgraph.attach_feature(equivalence_tracker)
    return fgraph, updates, equivalence_tracker


class DataDestroyed:
    # this is a singleton class We put it in the storage_map when the
    # variable value was destroyed to prevent reusing bad value for
    # it.
    pass


data_destroyed = DataDestroyed()


def check_eq(var, val1, val2):
    if hasattr(var.tag, "values_eq_approx"):
        return var.tag.values_eq_approx(val1, val2)
    else:
        return var.type.values_eq_approx(val1, val2)


def _check_inputs(
    node,
    storage_map,
    r_vals,
    dr_vals,
    active_nodes,
    clobber_dr_vals=True,
    perform=None,
    warn_input_not_reused=True,
):
    """
    Raise BadDestroyMap if necessary, update dr_vals.

    Returns a list of output variables that actually worked inplace
    (their value is aliased to the value of at least one input).

    It modify the storage_map to remove node.inputs variable that have
    been destroyed.

    """
    destroyed_idx_list = []
    destroy_map = node.op.destroy_map
    for i_pos_list in destroy_map.values():
        destroyed_idx_list.extend(i_pos_list)
    destroyed_res_list = [node.inputs[i] for i in destroyed_idx_list]

    actually_inplace_outputs = []
    dmap = node.op.destroy_map
    for oo, ii in dmap.items():
        var = node.outputs[oo]
        out_var = storage_map[var][0]
        in_var = storage_map[node.inputs[ii[0]]][0]
        if hasattr(var.type, "may_share_memory") and var.type.may_share_memory(
            out_var, in_var
        ):
            actually_inplace_outputs.append(node.outputs[oo])

        if warn_input_not_reused and destroyed_res_list:
            if isinstance(node.op, OutputGuard):
                # The point of OutputGuard is to be declared as destructive
                # while not destroying anything
                continue
            if out_var is not in_var:
                _logger.warning(
                    f"Optimization Warning: input idx {int(ii[0])} marked "
                    f"as destroyed was not changed for node '{node}'"
                )

    vmap = node.op.view_map
    for oo, ii in vmap.items():
        var = node.outputs[oo]
        out_var = storage_map[var][0]
        in_var = storage_map[node.inputs[ii[0]]][0]
        may_share = hasattr(var.type, "may_share_memory") and var.type.may_share_memory(
            out_var, in_var
        )
        if may_share:
            actually_inplace_outputs.append(node.outputs[oo])

        if warn_input_not_reused:
            # We don't try to optimize simple scalar and empty ndarray,
            # as this is not worth our time. This happen at least in
            # Subtensor when the output is a scalar But this depend on
            # the version of numpy!
            if getattr(out_var, "size", 2) <= 1:
                continue
            if isinstance(node.op, OutputGuard):
                # This class is not in the final graph.
                continue
            if not may_share:
                _logger.warning(
                    f"Optimization Warning: input idx {int(ii[0])} marked "
                    "as viewed but new memory allocated by node "
                    f"'{node}'"
                )

    for r_idx, r in enumerate(node.inputs):
        if not r.type.values_eq(r_vals[r], storage_map[r][0]):
            # some input node 'r' got changed by running the node
            # this may or may not be ok...
            if r in destroyed_res_list:
                # ok, we expected r to be destroyed
                if node in active_nodes:
                    if dr_vals.get(r, (0, node))[1] is not node:
                        # bad: there should only be one active node
                        # that destroys any variable
                        raise Exception("failure in topological ordering")
                    if clobber_dr_vals:
                        # no copy, this is the last use of this variable
                        dr_vals[r] = (storage_map[r][0], node)
                    # make sure that dr_vals[r] doesn't get used again
                    storage_map[r][0] = data_destroyed
            else:
                raise BadDestroyMap(node, r_idx, r_vals[r], storage_map[r][0], perform)

    return actually_inplace_outputs


def _check_viewmap(fgraph, node, storage_map):
    """
    This functions raises a BadViewMap exception when it detects the
    following:
    - Output node storages aliased to input storage, with no declaration
      in view_map.
    - If not aliased to an input, check if two outputs are aliased together
      and used subsequently in the graph.

    """

    for oi, onode in enumerate(node.outputs):
        good_alias, bad_alias = {}, {}
        outstorage = storage_map[onode][0]

        # first find out which input it aliases
        view_map = node.op.view_map
        destroy_map = node.op.destroy_map

        # In theory, pytensor's view_map only allows for 1 output to
        # alias 1 input. Checking for multiple aliases just in
        # case...

        for ii, inode in enumerate(node.inputs):
            in_storage = storage_map[inode][0]
            if in_storage is data_destroyed:
                # If the input have been destroyed, it can't be a
                # view. So no need to check. Also, we don't have the
                # original value, we we wouldn't be able to do this
                # useless check.
                continue
            if hasattr(inode.type, "may_share_memory") and inode.type.may_share_memory(
                outstorage, in_storage
            ):
                nodeid = id(inode)
                bad_alias[nodeid] = ii

                # check that the aliasing was declared in [view|destroy]_map
                if [ii] == view_map.get(oi, None) or [ii] == destroy_map.get(oi, None):
                    good_alias[nodeid] = bad_alias.pop(nodeid)

        # TODO: make sure this is correct
        # According to OB, duplicate inputs are rejected on build graph time
        # if they cause problems. So if they are here it should be ok.
        for key in good_alias:
            bad_alias.pop(key, None)
        if bad_alias:
            raise BadViewMap(node, oi, outstorage, list(bad_alias.values()))

        # if its not aliased to input, check output->output aliasing
        if not good_alias and _is_used_in_graph(fgraph, onode):
            for other_oi, other_onode in enumerate(node.outputs):
                if other_oi == oi:
                    continue

                other_storage = storage_map[other_onode][0]
                # check to see if we share memory with this other output
                # this is not a problem if the node is not actually used
                if (
                    _is_used_in_graph(fgraph, other_onode)
                    and hasattr(other_onode.type, "may_share_memory")
                    and other_onode.type.may_share_memory(outstorage, other_storage)
                ):
                    raise BadViewMap(node, oi, outstorage, out_alias_idx=other_oi)


def _is_used_in_graph(fgraph, var):
    """

    Returns
    -------
    bool
        True if `var` is used by another node in the graph.

    """
    return any(
        client for client, _ in fgraph.clients[var] if not isinstance(client.op, Output)
    )


def _check_strides_match(a, b, warn_err, op):
    """

    Parameters
    ----------
    warn_err
        If 0, no warning, if 1 warning, if 2 error.

    """
    if warn_err == 0:
        return

    try:
        strides_eq = a.strides == b.strides
    except Exception:
        return  # no strides

    if not strides_eq:
        e = TypeError(
            "Stride mismatch", (a.shape, b.shape, a.strides, b.strides, str(op))
        )
        if warn_err == 2:
            raise e
        else:
            warn(str(e))


def _lessbroken_deepcopy(a):
    """

    Parameters
    ----------
    a
        Any object

    Returns
    -------
    object
        A copy of `a` that shares no internal storage with the original
        (a deep copy). This function handles numpy arrays specially, because
        copy.deepcopy() called on a 0-d array will return a numpy scalar,
        not an array.

    """
    # this exists because copy.deepcopy on numpy arrays is broken
    # This logic is also in link.py
    from pytensor.link.c.type import _cdata_type

    if isinstance(a, np.ndarray | np.memmap):
        rval = a.copy(order="K")
    elif isinstance(a, _cdata_type):
        # This is not copyable (and should be used for constant data).
        rval = a
    else:
        rval = copy.deepcopy(a)

    assert type(rval) is type(a), (type(rval), type(a))

    if isinstance(rval, np.ndarray):
        assert rval.dtype == a.dtype
    return rval


def _find_bad_optimizations(order, reasons, r_vals):
    """Iterate over variables looking for values that don't match the values of the variables they replaced.

    This is a sign of a broken optimization.

    This algorithm is simple to understand, but sometimes when there's
    a problem it identifies the wrong optimization as the culprit.
    The problem stems from the fact that results are not evaluated in
    chronological order (looking at when they were introduced to the
    graph).

    """
    for i, node in enumerate(order):
        for new_r in node.outputs:
            for reason, r, old_graph_str, new_graph_str in reasons[new_r]:
                # check if the value for new_r doesn't match the value for r
                new_r_val = r_vals[new_r]
                r_val = r_vals[r]
                assert r.type.is_super(new_r.type)

                if hasattr(new_r.tag, "values_eq_approx"):
                    check = new_r.tag.values_eq_approx(r_val, new_r_val)
                elif hasattr(new_r, "values_eq_approx"):
                    # This way will be deprecated later, but not right now
                    check = new_r.values_eq_approx(r_val, new_r_val)
                else:
                    check = r.type.values_eq_approx(r_val, new_r_val)
                if not check:
                    raise BadOptimization(
                        old_r=r,
                        new_r=new_r,
                        old_r_val=r_val,
                        new_r_val=new_r_val,
                        reason=reason,
                        old_graph=old_graph_str,
                        new_graph=new_graph_str,
                    )


def _get_preallocated_maps(
    node,
    thunk,
    prealloc_modes,
    def_val,
    storage_map,
    r_vals,
    dr_vals,
    perform,
    active_order_set,
    inplace_outs,
    init_outputs,
):
    """
    Preallocate outputs in different memory layouts.

    """
    # TODO: Sparse? Scalar does not really make sense.

    # Do not preallocate memory for outputs that actually work inplace
    considered_outputs = [r for r in node.outputs if r not in inplace_outs]

    # Output storage that was initially present in the storage_map
    if "initial" in prealloc_modes or "ALL" in prealloc_modes:
        initial_outputs = {}
        for r in considered_outputs:
            if r in init_outputs:
                initial_outputs[r] = init_outputs[r]

        if initial_outputs:
            yield ("initial", initial_outputs)

    # reuse_output: use a copy of the same storage returned the first time
    # TODO: optimization warning if the storage in reuse_outputs
    # is not reused
    if "previous" in prealloc_modes or "ALL" in prealloc_modes:
        reuse_outputs = {}
        for r in considered_outputs:
            # We want to reuse the exact same memory buffer,
            # so we keep the copy in r_vals
            new_r = _lessbroken_deepcopy(r_vals[r])
            reuse_outputs[r] = r_vals[r]
            r_vals[r] = new_r
            # Sometimes, outputs can be aliased together.
            # I'm not sure why it is legitimate, but there are tests about it.
            # So, we cannot fill r_vals[r] with def_val yet, we have to wait
            # until all output values are deepcopied.

        for r in considered_outputs:
            # There is no risk to overwrite inputs, since r does not work
            # inplace.
            if isinstance(r.type, TensorType):
                reuse_outputs[r][...] = np.asarray(def_val).astype(r.type.dtype)

        if reuse_outputs:
            yield ("previous", reuse_outputs)
        # clear memory that is not needed any more
        del reuse_outputs

    # c_cont_output: use a c-continuous array
    # (for TensorType, else None)
    if "c_contiguous" in prealloc_modes or "ALL" in prealloc_modes:
        c_cont_outputs = {}
        for r in considered_outputs:
            if isinstance(r.type, TensorType):
                # Build a C-contiguous buffer
                new_buf = np.empty(r_vals[r].shape, dtype=r.type.dtype)
                assert new_buf.flags["C_CONTIGUOUS"]
                new_buf[...] = np.asarray(def_val).astype(r.type.dtype)

                c_cont_outputs[r] = new_buf

        if len(c_cont_outputs):
            yield ("c_contiguous", c_cont_outputs)
            del c_cont_outputs

    # f_cont_output: use a fortran-continuous ndarray
    # (for TensorType, only)
    if "f_contiguous" in prealloc_modes or "ALL" in prealloc_modes:
        f_cont_outputs = {}
        for r in considered_outputs:
            if isinstance(r.type, TensorType):
                new_buf = np.zeros(
                    shape=r_vals[r].shape, dtype=r_vals[r].dtype, order="F"
                )
                new_buf[...] = def_val

                f_cont_outputs[r] = new_buf

        if len(f_cont_outputs):
            yield ("f_contiguous", f_cont_outputs)
            del f_cont_outputs

    # We assume that the different outputs of a same Op will behave
    # independently, and there is no need to test over all combinations
    # of outputs (the time taken is prohibitive).
    # When all outputs on a certain dimension are broadcastable, the Op
    # can assume that the shape is 1 on that dimension, and stride testing
    # is less relevant.
    # Dimensions should be align by the innermost index, so we iterate
    # from the end of shapes.
    if (
        "strided" in prealloc_modes
        or "wrong_size" in prealloc_modes
        or "ALL" in prealloc_modes
    ):
        max_ndim = 0
        rev_out_shape = []
        for r in considered_outputs:
            if isinstance(r.type, TensorType):
                if max_ndim < r.ndim:
                    rev_out_shape += [1] * (r.ndim - max_ndim)
                    max_ndim = r.ndim
                assert len(rev_out_shape) == max_ndim

                for i, s in enumerate(r.type.shape[::-1]):
                    rev_out_shape[i] = 1 if rev_out_shape[i] == 1 and s == 1 else None
        out_shape = rev_out_shape[::-1]

    if "strided" in prealloc_modes or "ALL" in prealloc_modes:
        check_ndim = config.DebugMode__check_preallocated_output_ndim
        # Initial allocation
        init_strided = {}
        for r in considered_outputs:
            if isinstance(r.type, TensorType):
                # Create a buffer twice as large in every dimension,
                # except if broadcastable, or for dimensions above
                # config.DebugMode__check_preallocated_output_ndim
                buf_shape = []
                for s, b in zip(r_vals[r].shape, r.broadcastable, strict=True):
                    if b or ((r.ndim - len(buf_shape)) > check_ndim):
                        buf_shape.append(s)
                    else:
                        buf_shape.append(s * 2)

                new_buf = np.empty(buf_shape, dtype=r.type.dtype)
                new_buf[...] = np.asarray(def_val).astype(r.type.dtype)
                init_strided[r] = new_buf

        # The number of combinations is exponential in the number of
        # dimensions, and some ops can have tens of outputs. To prevent
        # tests from lasting days, we use the same strides for all
        # dimensions but the last check_ndim ones.
        # Moreover, to avoid memory problems, we do not test with strides
        # 2 and -2 on those dimensions.
        step_signs_list = []
        for s in out_shape[-check_ndim:]:
            if s == 1:
                step_signs_list.append((1,))
            else:
                step_signs_list.append((-1, 1))

        # Use the same step on all dimensions before the last check_ndim.
        if all(s == 1 for s in out_shape[:-check_ndim]):
            step_signs_list = [(1,), *step_signs_list]
        else:
            step_signs_list = [(-1, 1), *step_signs_list]

        for step_signs in itertools_product(*step_signs_list):
            for step_size in (1, 2):
                strided = {}

                # First, the dimensions above check_ndim, then the other ones
                # Do not test with 2 or -2 for dimensions above check_ndim
                steps = [step_signs[0]] * len(out_shape[:-check_ndim])
                steps += [s * step_size for s in step_signs[1:]]

                name = f"strided{tuple(steps)}"
                for r in considered_outputs:
                    if r in init_strided:
                        shapes = [slice(None, size, None) for size in r_vals[r].shape]
                        strides = [
                            slice(None, None, steps[i]) for i in range(r_vals[r].ndim)
                        ]

                        r_buf = init_strided[r]

                        if r_buf.ndim > 0:
                            r_buf = r_buf[tuple(strides)][tuple(shapes)]
                        assert r_buf.shape == r_vals[r].shape

                        r_buf[...] = np.asarray(def_val).astype(r_buf.dtype)
                        strided[r] = r_buf

                if strided:
                    yield (name, strided)
                del strided

    if "wrong_size" in prealloc_modes or "ALL" in prealloc_modes:
        # For each dimension, try size-1, size, size+1
        for dim, s in enumerate(out_shape):
            if s == 1:
                # The shape has to be 1
                continue

            shape_diff = [0] * max_ndim
            for diff in (-1, 1):
                shape_diff[dim] = diff

                wrong_size = {}
                name = f"wrong_size{tuple(shape_diff)}"

                for r in considered_outputs:
                    if isinstance(r.type, TensorType):
                        r_shape_diff = shape_diff[: r.ndim]
                        new_buf_shape = [
                            max((s + sd), 0)
                            for s, sd in zip(r_vals[r].shape, r_shape_diff, strict=True)
                        ]
                        new_buf = np.empty(new_buf_shape, dtype=r.type.dtype)
                        new_buf[...] = np.asarray(def_val).astype(r.type.dtype)
                        wrong_size[r] = new_buf

                if wrong_size:
                    yield (name, wrong_size)
                del wrong_size


def _check_preallocated_output(
    fgraph,
    node,
    thunk,
    prealloc_modes,
    def_val,
    storage_map,
    r_vals,
    dr_vals,
    perform,
    active_order_set,
    inplace_outs,
    init_outputs,
):
    """
    Try to apply thunk() on different output storages.

    """

    # If node has an inner compiled PyTensor function with mode DebugMode,
    # disable memory checks in that mode, since they were already run.
    try:
        changed_inner_mode = False
        if isinstance(node.op, HasInnerGraph):
            fn = node.op.fn
            if not (fn and hasattr(fn, "maker") and hasattr(fn.maker, "mode")):
                _logger.warning(f"Expected pytensor function not found in {node.op}.fn")
            else:
                if isinstance(fn.maker.mode, DebugMode):
                    backup_mode = fn.maker.mode
                    new_mode = copy.copy(backup_mode)
                    # Disactivate as many checks as possible
                    new_mode.check_py_code = False
                    new_mode.check_isfinite = False
                    new_mode.require_matching_strides = 0
                    new_mode.check_preallocated_output = []
                    new_mode.stability_patience = 1
                    fn.maker.mode = new_mode
                    changed_inner_mode = True
                    _logger.info("changing inner mode")

        # Set of inputs that are marked as destroyed or viewed
        aliased_inputs = set()
        dmap = node.op.destroy_map
        vmap = node.op.view_map
        for i, r in enumerate(node.inputs):
            if any(i in v for v in chain(dmap.values(), vmap.values())):
                aliased_inputs.add(r)

        _logger.debug("starting preallocated output checking")
        for name, out_map in _get_preallocated_maps(
            node,
            thunk,
            prealloc_modes,
            def_val,
            storage_map,
            r_vals,
            dr_vals,
            perform,
            active_order_set,
            inplace_outs,
            init_outputs,
        ):
            _logger.debug(f"  name = {name}")

            thunk_name = f"{perform} with {name} output"

            if not out_map:
                # Map is empty, there is no need to execute thunk() again
                _logger.warning(f"{name}: out_map is empty")
                continue

            # Copy the inputs over, if they were marked as destroyed or viewed
            # (we will destroy the output at some point so it can destroy
            # the input)
            for r in aliased_inputs:
                storage_map[r][0] = _lessbroken_deepcopy(r_vals[r])

            # Get the appropriate output storages
            # (no copy)
            for r in node.outputs:
                storage_map[r][0] = out_map.get(r, None)

            thunk()

            # Check outputs
            for r in node.outputs:
                if not r.type.is_valid_value(storage_map[r][0]):
                    raise InvalidValueError(
                        r,
                        storage_map[r][0],
                        hint=thunk_name,
                        specific_hint=validity_hint(r.type, storage_map[r][0]),
                    )

            _check_inputs(
                node,
                storage_map,
                r_vals,
                dr_vals,
                active_order_set,
                clobber_dr_vals=False,
                perform=thunk_name,
                warn_input_not_reused=False,
            )

            _check_viewmap(fgraph, node, storage_map)

            for r in node.outputs:
                if not check_eq(r, r_vals[r], storage_map[r][0]):
                    # TODO: indicate it is not a C/Py problem
                    inputs_val = [storage_map[inp][0] for inp in r.owner.inputs]
                    raise BadThunkOutput(
                        r,
                        thunk1="Reference value",
                        val1=r_vals[r],
                        thunk2=thunk_name,
                        val2=storage_map[r][0],
                        inputs_val=inputs_val,
                    )

            # Clear storage_map
            for r in node.outputs:
                storage_map[r][0] = None

        _logger.debug("finished preallocated output checking")
    finally:
        if changed_inner_mode:
            _logger.info("changing mode back")
            fn.maker.mode = backup_mode


def validity_hint(type, x):
    try:
        type.filter(x, strict=True)
    except Exception as e:
        return str(e)
    return "value is valid"


class _FunctionGraphEvent:
    """
    A record of an event in the life of an FunctionGraph.

    The __eq__ function is important here, as it is the basis for
    comparing optimization runs.

    """

    kind = ""
    """
    One of 'import', 'change', 'prune'.

    """

    node = None
    """
    Either 'output' or an Apply instance.

    """

    op = None
    """Either 'output' or an Op instance"""

    idx = None
    """
    Change events involve an position index of the input variable.

    """

    reason = None
    """
    Change events sometimes have a reason.

    """

    def __init__(self, kind, node, idx=None, reason=None):
        self.kind = kind
        self.node = node
        self.op = node.op
        self.idx = idx
        self.reason = str(reason)

    def __str__(self):
        if self.kind == "change":
            if not isinstance(self.op, Output):
                msg = str(len(self.node.inputs))
            else:
                msg = ""

            return " ".join(["change", self.reason, str(self.op), str(self.idx), msg])
        else:
            return str(self.__dict__)

    def __eq__(self, other):
        rval = type(self) is type(other)
        if rval:
            # nodes are not compared because this comparison is
            # supposed to be true for corresponding events that happen
            # in different FunctionGraph instances (different graphs)
            for attr in ("kind", "op", "idx", "reason"):
                rval = rval and getattr(self, attr) == getattr(other, attr)
        return rval

    def __ne__(self, other):
        return not (self == other)


class _VariableEquivalenceTracker:
    """
    A FunctionGraph Feature that keeps tabs on an FunctionGraph and
    tries to detect problems.

    """

    fgraph = None
    """WRITEME"""

    equiv = None
    """WRITEME"""

    active_nodes = None
    """WRITEME"""

    inactive_nodes = None
    """WRITEME"""

    all_variables_ever = None
    """WRITEME"""

    reasons = None
    """WRITEME"""

    replaced_by = None
    """WRITEME"""

    event_list = None
    """WRITEME"""

    def __init__(self):
        self.fgraph = None

    def on_attach(self, fgraph):
        if self.fgraph is not None:
            raise AlreadyThere()

        self.equiv = {}
        self.active_nodes = set()
        self.inactive_nodes = set()
        self.fgraph = fgraph
        self.all_variables_ever = []
        self.reasons = {}
        self.replaced_by = {}
        self.event_list = []
        for node in fgraph.toposort():
            self.on_import(fgraph, node, "on_attach")

    def on_detach(self, fgraph):
        assert fgraph is self.fgraph
        self.fgraph = None

    def on_prune(self, fgraph, node, reason):
        self.event_list.append(_FunctionGraphEvent("prune", node, reason=str(reason)))
        assert node in self.active_nodes
        assert node not in self.inactive_nodes
        self.active_nodes.remove(node)
        self.inactive_nodes.add(node)

    def on_import(self, fgraph, node, reason):
        self.event_list.append(_FunctionGraphEvent("import", node, reason=str(reason)))

        assert node not in self.active_nodes
        self.active_nodes.add(node)

        if node in self.inactive_nodes:
            self.inactive_nodes.remove(node)
            for r in node.outputs:
                assert r in self.equiv
        else:
            for r in node.outputs:
                assert r not in self.equiv
                self.equiv[r] = {r}
                self.all_variables_ever.append(r)
                self.reasons.setdefault(r, [])
                self.replaced_by.setdefault(r, [])
            for r in node.inputs:
                self.reasons.setdefault(r, [])
                self.replaced_by.setdefault(r, [])

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        reason = str(reason)
        self.event_list.append(
            _FunctionGraphEvent("change", node, reason=reason, idx=i)
        )

        self.reasons.setdefault(new_r, [])
        self.replaced_by.setdefault(new_r, [])

        append_reason = True
        for tup in self.reasons[new_r]:
            if tup[0] == reason and tup[1] is r:
                append_reason = False

        if append_reason:
            # N.B. compute the debugprint now, because future
            # optimizations will change the graph
            done = dict()
            used_ids = dict()
            self.reasons[new_r].append(
                (
                    reason,
                    r,
                    _debugprint(
                        r,
                        prefix="  ",
                        depth=6,
                        file=StringIO(),
                        done=done,
                        print_type=True,
                        used_ids=used_ids,
                    ).getvalue(),
                    _debugprint(
                        new_r,
                        prefix="  ",
                        depth=6,
                        file=StringIO(),
                        done=done,
                        print_type=True,
                        used_ids=used_ids,
                    ).getvalue(),
                )
            )
            self.replaced_by[r].append((reason, new_r))

        if r in self.equiv:
            r_set = self.equiv[r]
        else:
            r_set = self.equiv.setdefault(r, {r})
            self.all_variables_ever.append(r)

        if new_r in self.equiv:
            new_r_set = self.equiv[new_r]
        else:
            new_r_set = self.equiv.setdefault(new_r, {new_r})
            self.all_variables_ever.append(new_r)

        assert new_r in new_r_set
        assert r in r_set

        # update one equivalence set to contain the other
        # transfer all the elements of the old one to the new one
        r_set.update(new_r_set)
        for like_new_r in new_r_set:
            self.equiv[like_new_r] = r_set
            assert like_new_r in r_set

        assert self.equiv[r] is r_set
        assert self.equiv[new_r] is r_set

    def printstuff(self):
        for key in self.equiv:
            print(key)  # noqa: T201
            for e in self.equiv[key]:
                print("  ", e)  # noqa: T201


# List of default version of make thunk.
# This is needed to know if the user overrode it.
default_make_thunk = [get_unbound_function(COp.make_thunk)]


# Debug mode cheats and initializes the linker in a different way in
# its maker so we can cheat some more by having a linker to satisfy
# the external requirements of the .linker attribute of a mode
# 1) it's a class instance
# 2) it a has a .clone() method
class _DummyLinker:
    # This is not a real linker anyway
    def clone(self, allow_gc=None):
        return self


class _Linker(LocalLinker):
    """
    Special debugging linker.

    """

    def __init__(self, maker, schedule=None):
        super().__init__()
        self.fgraph = None
        self.maker = maker
        super().__init__(scheduler=schedule)

    def accept(self, fgraph, no_recycling: list | None = None, profile=None):
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            assert type(self) is _Linker
            return type(self)(maker=self.maker).accept(fgraph, no_recycling, profile)
        self.fgraph = fgraph
        self.no_recycling: list = no_recycling
        return self

    def make_all(
        self, profiler=None, input_storage=None, output_storage=None, storage_map=None
    ):
        # can't import at toplevel because of circular import TODO:
        # don't do this ugly hacky way of setting the
        # filter_checks_isfinite

        fgraph = self.fgraph
        input_storage_ = input_storage
        output_storage_ = output_storage

        # Compute a topological ordering that IGNORES the destroy_map
        # of destructive Ops.  This will be OK, because every thunk is
        # evaluated on a copy of its input.
        fgraph_equiv = fgraph.equivalence_tracker
        order_outputs = copy.copy(fgraph_equiv.all_variables_ever)
        del fgraph_equiv
        order_outputs.reverse()
        order = io_toposort(fgraph.inputs, order_outputs)

        # an ordering of just the active nodes
        active_order = self.schedule(fgraph)
        active_order_set = set(active_order)

        # Disable no_recycling, in order to be able to use
        # check_preallocated_output even on the output of the function.
        # no_recycling in individual thunks does not really matter, since
        # the function's outputs will always be freshly allocated.
        no_recycling: list = []

        input_storage, output_storage, storage_map = map_storage(
            fgraph, order, input_storage_, output_storage_, storage_map
        )

        thunks_py = []  # python thunks
        thunks_c = []  # c thunks

        for node in order:
            compute_map = {}
            for k in node.inputs:
                compute_map[k] = [True]
            for k in node.outputs:
                compute_map[k] = [False]

            # Some Ops define a make_thunk with the expectation that
            # it will be called before the C code is compiled, because
            # the compilation of some dependency is triggered there.
            thunk_other = None

            if get_unbound_function(node.op.make_thunk) not in default_make_thunk:
                thunk = node.op.make_thunk(node, storage_map, compute_map, no_recycling)
                thunk.inputs = [storage_map[v] for v in node.inputs]
                thunk.outputs = [storage_map[v] for v in node.outputs]
                thunk_other = thunk

            debug = hasattr(node.op, "debug_perform")

            try:
                if (
                    not self.maker.mode.check_c_code
                    or debug
                    or not isinstance(node.op, COp)
                ):
                    raise MethodNotDefined()

                node.op.prepare_node(node, storage_map, compute_map, "c")
                thunk = node.op.make_c_thunk(
                    node, storage_map, compute_map, no_recycling
                )
                thunks_c.append(thunk)
            except (NotImplementedError, MethodNotDefined):
                thunks_c.append(None)

            # Pure ops don't really have a perform ( or their perform just
            # raises an not implemented exception), so in those cases we
            # consider that we don't have a python implementation
            if (
                (self.maker.mode.check_py_code or thunks_c[-1] is None)
                and node.op.perform.__code__ != Op.perform.__code__
            ) or debug:
                node.op.prepare_node(node, storage_map, compute_map, "py")
                thunk = node.op.make_py_thunk(
                    node, storage_map, compute_map, no_recycling, debug=debug
                )
                thunks_py.append(thunk)
            else:
                thunks_py.append(None)

            if (
                not self.maker.mode.check_c_code
                and thunks_py[-1] is None
                and isinstance(node.op, COp)
            ):
                _logger.warning(
                    f"Op {node.op} doesn't have a perform, forcing check of the C code"
                )
                node.op.prepare_node(node, storage_map, compute_map, "c")
                thunk = node.op.make_c_thunk(
                    node, storage_map, compute_map, no_recycling
                )
                thunks_c[-1] = thunk

            # If the op defined its own make_thunk, use the generated thunk
            if thunk_other is not None:
                if thunks_py[-1] is None:
                    thunks_py[-1] = thunk_other
                elif thunks_c[-1] is None:
                    thunks_c[-1] = thunk_other
                else:
                    _logger.warning(
                        "We won't check the perform function "
                        f"of node '{node}' but we will check its "
                        "make_thunk function"
                    )
                    thunks_py[-1] = thunk_other

        # Use self.no_recycling (that was passed in accept()) to always
        # use new memory storage when it is needed, in particular for the
        # function's outputs. no_recycling_map will be used in f() below.
        no_recycling_map: list = []
        if self.no_recycling is True:
            no_recycling_map = list(storage_map.values())
            no_recycling_map = difference(no_recycling_map, input_storage)
        else:
            no_recycling_map = [
                storage_map[r] for r in self.no_recycling if r not in fgraph.inputs
            ]

        # Precompute some things for storage pre-allocation
        def_val = int(config.unittests__rseed)

        #####
        # This is the function that runs when you evaluate the graph
        #####
        def f():
            ####
            # Note: `f` ignores the compute_map and evaluates the nodes in
            # topological order. In some sense, this is ok, and can be used
            # for now.
            #####
            _logger.debug("starting a DebugMode call")
            _logger.debug(
                f"self.maker.mode.check_preallocated_output: {self.maker.mode.check_preallocated_output}"
            )
            for x in no_recycling_map:
                x[0] = None

            # nest all this in try-finally to put storage *back* into
            # storage_map when an exception is raised
            original_storage_map_keys = [r for r in storage_map if r.owner is None]

            try:
                # r_vals are the true values associated with each
                # variable in the graph they should not change during
                # the evaluation of this function, even when the graph
                # has destructive ops in it
                #
                # This dictionary is used to populate the storage_map
                # as necessary
                r_vals = {}

                # dr_vals are the values taken by variables after
                # being destroyed
                dr_vals = {}
                assert len(thunks_py) == len(order)

                # transfer the initial values from the storage_map to
                # the r_vals
                _logger.debug("DEBUGMODE: transfer initial values")
                # r_vals_initialized keeps track of the values that have
                # actually been transferred from storage_map to r_vals
                r_vals_initialized = []
                for r in storage_map:
                    if r.owner is None:
                        if not r.type.is_valid_value(storage_map[r][0]):
                            # None may be a valid input value (for instance,
                            # for a Generic object). We only want to raise
                            # an error if it is not valid.
                            if storage_map[r][0] is None:
                                raise InvalidValueError(
                                    r,
                                    storage_map[r][0],
                                    hint=f"Graph Input '{r}' is missing",
                                )
                            raise InvalidValueError(
                                r,
                                storage_map[r][0],
                                hint=(
                                    f"Graph Input '{r}' has invalid value "
                                    f"{storage_map[r][0]}"
                                ),
                            )
                        r_vals[r] = storage_map[r][0]
                        storage_map[r][0] = None
                        r_vals_initialized.append(r)

                # store preallocated outputs in another map, and test
                # the thunks on them as output storages.
                init_outputs = {}
                for r in storage_map:
                    if r in fgraph.outputs:
                        if storage_map[r][0] is not None:
                            init_outputs[r] = storage_map[r][0]
                            storage_map[r][0] = None

                #####
                #  Precondition: the storage map is empty, transferred
                #  completely to r_vals
                #####
                for r, s in storage_map.items():
                    if s[0] is not None:
                        print(r, s)  # noqa: T201
                    assert s[0] is None

                # try:
                # compute the value of all variables
                for i, (thunk_py, thunk_c, node) in enumerate(
                    zip(thunks_py, thunks_c, order, strict=True)
                ):
                    _logger.debug(f"{i} - starting node {i} {node}")

                    # put a copy of each input into the storage_map
                    # also, check that inputs have valid values
                    for r in node.inputs:
                        assert isinstance(r, Variable)
                        assert r in r_vals
                        storage_map[r][0] = _lessbroken_deepcopy(r_vals[r])
                        if not r.type.is_valid_value(storage_map[r][0]):
                            raise InvalidValueError(
                                r, storage_map[r][0], client_node=node
                            )

                    # On the first call to thunk_py(), its output
                    # storage will be None
                    if thunk_py:
                        _logger.debug(
                            f"{i} - running thunk_py with None as output storage"
                        )
                        try:
                            thunk_py()
                        except (MethodNotDefined, NotImplementedError):
                            # shouldn't have put it into the list in
                            # the first place
                            thunk_py = None
                            thunks_py[i] = None
                        except Exception as e:
                            # I think that only 1 optimization can
                            # insert a given apply node. If that is not True,
                            # we would need to loop over all node outputs,
                            # But this make the output uglier.
                            reason = fgraph.equivalence_tracker.reasons[node.outputs[0]]
                            if not reason:
                                raise
                            opt = str(reason[0][0])
                            msg = (
                                f"An optimization (probably {opt}) inserted an "
                                "apply node that raise an error.\n"
                                "The information we have about this optimization is:"
                                f"{reason[0][1]}\n{reason[0][2]}\n"
                                f"\nThe original exception: \n{e}"
                            )
                            new_e = e.__class__(msg)
                            exc_type, exc_value, exc_trace = sys.exc_info()
                            exc_value = new_e
                            raise_with_op(
                                fgraph, node, thunk_c, (exc_type, exc_value, exc_trace)
                            )

                    if thunk_py:
                        # check output values for type-correctness
                        for r in node.outputs:
                            if not r.type.is_valid_value(storage_map[r][0]):
                                hint2 = validity_hint(r.type, storage_map[r][0])
                                raise InvalidValueError(
                                    r,
                                    storage_map[r][0],
                                    hint="perform output",
                                    specific_hint=hint2,
                                )
                        warn_inp = config.DebugMode__warn_input_not_reused
                        py_inplace_outs = _check_inputs(
                            node,
                            storage_map,
                            r_vals,
                            dr_vals,
                            active_order_set,
                            clobber_dr_vals=True,
                            perform="py",
                            warn_input_not_reused=warn_inp,
                        )
                        _check_viewmap(fgraph, node, storage_map)

                        # Retrieve each output from the storage_map.
                        # The return values of this first run will be
                        # the reference ones
                        for r in node.outputs:
                            assert r not in r_vals
                            r_vals[r] = storage_map[r][0]
                            # clear the storage_map of outputs for the thunk_c
                            storage_map[r][0] = None

                        if self.maker.mode.check_preallocated_output:
                            prealloc_modes = self.maker.mode.check_preallocated_output
                            _logger.debug(
                                f"{i} - calling _check_preallocated_output "
                                "with thunk_py"
                            )
                            _check_preallocated_output(
                                fgraph=fgraph,
                                node=node,
                                thunk=thunk_py,
                                prealloc_modes=prealloc_modes,
                                def_val=def_val,
                                storage_map=storage_map,
                                r_vals=r_vals,
                                dr_vals=dr_vals,
                                perform="py",
                                active_order_set=active_order_set,
                                inplace_outs=py_inplace_outs,
                                init_outputs=init_outputs,
                            )

                        sys.stdout.flush()

                    if thunk_c:
                        clobber = True
                        if thunk_py:
                            dmap = node.op.destroy_map
                            vmap = node.op.view_map
                            for i, r in enumerate(node.inputs):
                                # if thunk_py ran, and we still got
                                # this far, it means that the
                                # destroy_map of the Op (and view_map)
                                # are accurate so we can assume that
                                # inputs not marked as destroyed have
                                # in fact not been destroyed.
                                # Therefore... we only need to
                                # overwrite inputs that *have* been
                                # marked as destroyed.  Inputs marked
                                # as viewd are unsafe too, because the
                                # corresponding output can be
                                # destroyed.
                                if any(
                                    i in v for v in chain(dmap.values(), vmap.values())
                                ):
                                    storage_map[r][0] = _lessbroken_deepcopy(r_vals[r])

                            clobber = False

                        _logger.debug(f"{i} - running thunk_c")
                        # First time, with None in output_storage
                        try:
                            thunk_c()
                        except Exception as e:
                            # I think that only 1 optimization can
                            # insert a given apply node. If that is not True,
                            # we would need to loop over all node outputs,
                            # But this make the output uglier.
                            reason = fgraph.equivalence_tracker.reasons[node.outputs[0]]
                            if not reason:
                                raise
                            opt = str(reason[0][0])
                            msg = (
                                f"An optimization (probably {opt}) inserted an "
                                "apply node that raise an error.\n"
                                "The information we have about this optimization is:"
                                f"{reason[0][1]}\n{reason[0][2]}\n"
                                f"\nThe original exception: \n{e}"
                            )
                            new_e = e.__class__(msg)
                            exc_type, exc_value, exc_trace = sys.exc_info()
                            exc_value = new_e
                            raise_with_op(
                                fgraph, node, thunk_c, (exc_type, exc_value, exc_trace)
                            )

                        for r in node.outputs:
                            # check output values for type-correctness
                            if not r.type.is_valid_value(storage_map[r][0]):
                                raise InvalidValueError(
                                    r, storage_map[r][0], hint="c output"
                                )

                            if thunk_py:
                                # because we put it in during the
                                # thunk_py branch
                                assert r in r_vals
                                # check for stride correctness (may
                                # raise exception)
                                _check_strides_match(
                                    r_vals[r],
                                    storage_map[r][0],
                                    self.maker.mode.require_matching_strides,
                                    node.op,
                                )

                        warn_inp = config.DebugMode__warn_input_not_reused
                        c_inplace_outs = _check_inputs(
                            node,
                            storage_map,
                            r_vals,
                            dr_vals,
                            active_order_set,
                            clobber_dr_vals=clobber,
                            perform="c",
                            warn_input_not_reused=warn_inp,
                        )

                        _check_viewmap(fgraph, node, storage_map)

                        # Check with Python result
                        for r in node.outputs:
                            if r in r_vals:
                                # compares the version from thunk_py
                                # (in r_vals) to the version produced
                                # by thunk_c (in storage_map)
                                if not check_eq(r, r_vals[r], storage_map[r][0]):
                                    inputs_val = [
                                        storage_map[inp][0] for inp in r.owner.inputs
                                    ]
                                    raise BadThunkOutput(
                                        r,
                                        thunk1="perform",
                                        val1=r_vals[r],
                                        thunk2="c_code",
                                        val2=storage_map[r][0],
                                        inputs_val=inputs_val,
                                    )
                            else:
                                # retrieve each output from the storage_map
                                r_vals[r] = storage_map[r][0]
                            # clear the storage_map for the thunk_c
                            storage_map[r][0] = None

                        if self.maker.mode.check_preallocated_output:
                            prealloc_modes = self.maker.mode.check_preallocated_output

                            def thunk():
                                try:
                                    thunk_c()
                                except Exception:
                                    raise_with_op(fgraph, node, thunk_c)

                            _logger.debug(
                                f"{i} - calling _check_preallocated_output "
                                "with thunk_c",
                            )
                            _check_preallocated_output(
                                fgraph=fgraph,
                                node=node,
                                thunk=thunk,
                                prealloc_modes=prealloc_modes,
                                def_val=def_val,
                                storage_map=storage_map,
                                r_vals=r_vals,
                                dr_vals=dr_vals,
                                perform="c code",
                                active_order_set=active_order_set,
                                inplace_outs=c_inplace_outs,
                                init_outputs=init_outputs,
                            )

                        sys.stdout.flush()

                    # we're done with this thunk
                    # clear everything out of the storage_map
                    for r in node.inputs:
                        storage_map[r][0] = None
                    _logger.debug(f"{i} - done with node")
                    for r in node.outputs:
                        if r not in r_vals:
                            idx = order.index(node)
                            assert thunks_py[idx] is None, node
                            assert thunks_c[idx] is None, node
                            raise Exception(f"No code run for {node}")

                if False:
                    # This could be useful to help finding refcount problem.
                    # But it is very slow and it is not sure it will help.
                    gc.collect()

                _find_bad_optimizations(
                    order, fgraph.equivalence_tracker.reasons, r_vals
                )

                #####
                #  Postcondition: the input and output variables are
                #  in the storage map, nothing more
                #####

                # Nothing should be in storage map after evaluating
                # each the thunk (specifically the last one)
                for r, s in storage_map.items():
                    assert isinstance(s, list)
                    assert s[0] is None

                # store our output variables to their respective storage lists
                for output, storage in zip(fgraph.outputs, output_storage, strict=True):
                    storage[0] = r_vals[output]

                # transfer all inputs back to their respective storage lists
                for r in r_vals:
                    if r.owner is None:
                        if r in fgraph.inputs:
                            assert (
                                storage_map[r] is input_storage[fgraph.inputs.index(r)]
                            )
                        storage_map[r][0] = r_vals[r]

                # if an input was destroyed, the destroyed value
                # should be returned
                for r in dr_vals:
                    assert dr_vals[r][0] is not None
                    if r.owner is None:
                        assert r in fgraph.inputs
                        # HACK TO LOOK LIKE A REAL DESTRUCTIVE ACTION
                        # TOOK PLACE
                        if (
                            isinstance(dr_vals[r][0], np.ndarray | np.memmap)
                            and (dr_vals[r][0].dtype == storage_map[r][0].dtype)
                            and (dr_vals[r][0].shape == storage_map[r][0].shape)
                        ):
                            if len(dr_vals[r][0].shape):
                                storage_map[r][0][:] = dr_vals[r][0]
                            else:
                                storage_map[r][0].itemset(dr_vals[r][0])
                        else:
                            storage_map[r][0] = dr_vals[r][0]
            except Exception:
                # Restore the initial state of storage_map
                for r in storage_map:
                    if r in original_storage_map_keys:
                        # If r was transferred to r_vals, put it back
                        if r in r_vals_initialized:
                            storage_map[r][0] = r_vals[r]
                    else:
                        # clear out any partially-computed stuff
                        storage_map[r][0] = None
                raise

            for r in storage_map:
                if r.owner is None:
                    if not r.type.is_valid_value(None):
                        assert storage_map[r][0] is not None

            ###############
            # Done debugmode function call 'f'
            ##############

        def run_with_tensortype_filter_check(f):
            def deco():
                # WARNING: this is a global mechanism...
                # so it will screw up if we are trying to use
                # multiple modes at once.
                old_filter_checks_isfinite = TensorType.filter_checks_isfinite
                TensorType.filter_checks_isfinite = self.maker.mode.check_isfinite
                try:
                    return f()
                finally:
                    # put back the filter_checks_isfinite
                    TensorType.filter_checks_isfinite = old_filter_checks_isfinite

            return deco

        f = run_with_tensortype_filter_check(f)
        f.storage_map = storage_map
        f.allow_gc = True
        assert len(fgraph.inputs) == len(input_storage)
        assert len(fgraph.outputs) == len(output_storage)
        return (
            f,
            [
                Container(input, storage, readonly=False)
                for input, storage in zip(fgraph.inputs, input_storage, strict=True)
            ],
            [
                Container(output, storage, readonly=True)
                for output, storage in zip(fgraph.outputs, output_storage, strict=True)
            ],
            thunks_py,
            order,
        )


_NODEFAULT = ["NODEFAULT"]


class _Maker(FunctionMaker):  # inheritance buys a few helper functions
    """
    Special debugging FunctionMaker.

    Parameters
    ----------
    inputs : list of SymbolicInput instances
    outputs : list of SymbolicOutput instances
        Outputs may also be a single Variable (not a list), in which case
        the functions produced by FunctionMaker will return their output
        value directly.
    accept_inplace
        True iff it is acceptable to have inplace operations in the graph from
        the inputs to the outputs.
    on_unused_input
        What to do if a variable in the 'inputs' list is not used in the
        graph. Possible values are 'raise', 'warn' and 'ignore'.
    output_keys
        If the outputs argument for pytensor.function was a list, then
        output_keys is None. If the outputs argument was a dict, then
        output_keys is a sorted list of the keys from that dict.
    trust_input : bool, default False
        If True, no input validation checks are performed when the function is
        called. This includes checking the number of inputs, their types and
        that multiple inputs are not aliased to each other. Failure to meet any
        of these conditions can lead to computational errors or to the
        interpreter crashing.

    Notes
    -----
    The constructor sets TensorType.filter_checks_isfinite when
    `mode.check_isfinite` is True.

    """

    verbose = 0
    """
    Verbosity level of compile-time and run-time checks. (Default 0: silent).

    """

    def __init__(
        self,
        inputs,
        outputs,
        mode,
        accept_inplace=False,
        function_builder=Function,
        profile=None,
        on_unused_input=None,
        fgraph=None,  # If present the optimized graph. we ignore it.
        output_keys=None,
        name=None,
        no_fgraph_prep=False,
        trust_input=False,
    ):
        self.mode = mode
        self.profile = profile
        if profile:
            raise Exception("DebugMode do not support profiling.")
        optimizer = mode.optimizer
        # Handle the case where inputs and/or outputs is a single
        # Variable (not in a list)
        unpack_single = False
        return_none = False
        if outputs is None:
            return_none = True
            outputs = []
        if not isinstance(outputs, list | tuple):
            unpack_single = True
            outputs = [outputs]
        if not isinstance(inputs, list | tuple):
            inputs = [inputs]

        # Wrap them in In or Out instances if needed.
        inputs = [self.wrap_in(i) for i in inputs]
        outputs = [self.wrap_out(o) for o in outputs]

        # Check if some input variables are unused
        self.check_unused_inputs(inputs, outputs, on_unused_input)

        indices = [[input, None, [input]] for input in inputs]

        # make the fgraph
        for i in range(mode.stability_patience):
            fgraph, additional_outputs, equivalence_tracker = _optcheck_fgraph(
                inputs, outputs, accept_inplace
            )
            fgraph.equivalence_tracker = equivalence_tracker

            with config.change_flags(compute_test_value=config.compute_test_value_opt):
                optimizer(fgraph)

                pytensor.compile.function.types.insert_deepcopy(
                    fgraph, inputs, list(chain(outputs, additional_outputs))
                )

            if i == 0:
                fgraph0 = fgraph
            else:
                li = fgraph.equivalence_tracker.event_list
                l0 = fgraph0.equivalence_tracker.event_list
                if li != l0:
                    infolog = StringIO()
                    print("Optimization process is unstable...", file=infolog)
                    print(
                        "  (HINT: Ops that the nodes point to must compare equal)",
                        file=infolog,
                    )
                    print(
                        "(event index)  (one event trace)  (other event trace)",
                        file=infolog,
                    )
                    print(
                        "-----------------------------------------------------",
                        file=infolog,
                    )
                    for j in range(max(len(li), len(l0))):
                        if j >= len(li):
                            print("trailing event in optimization 0 :", j, file=infolog)
                            print("   ", str(l0[j]), file=infolog)
                        elif j >= len(l0):
                            print(
                                "trailing event in optimization",
                                i,
                                ":",
                                j,
                                file=infolog,
                            )
                            print("   ", str(li[j]), file=infolog)
                        elif li[j] != l0[j]:
                            print(
                                "non-equal optimization events", i, ":", j, file=infolog
                            )
                            print("   ", str(l0[j]), file=infolog)
                            print("   ", str(li[j]), file=infolog)
                        else:
                            pass
                    raise StochasticOrder(infolog.getvalue())
                else:
                    if self.verbose:
                        print(  # noqa: T201
                            "OPTCHECK: optimization",
                            i,
                            "of",
                            len(li),
                            "events was stable.",
                            file=sys.stderr,
                        )
        self.fgraph = fgraph
        if config.cycle_detection == "regular":
            destroy_handler_added = False
            for feature in fgraph._features:
                if isinstance(feature, DestroyHandler):
                    destroy_handler_added = True
                    break
            if not destroy_handler_added:
                fgraph.attach_feature(DestroyHandler())
            for o in fgraph.outputs:
                try:
                    with config.change_flags(
                        compute_test_value=config.compute_test_value_opt
                    ):
                        fgraph.replace_validate(
                            o, _output_guard(o), reason="output_guard"
                        )
                    raise Exception(
                        f"Output variable {o} required output_guard, "
                        "how was this output left unprotected against "
                        "destructive operations?"
                    )

                except InconsistencyError:
                    # This output is already impossible to destroy.
                    # No guard necessary
                    pass

        linker = _Linker(self)

        # the 'no_borrow' outputs are the ones for which that we can't return
        # the internal storage pointer.

        no_borrow = [
            output
            for output, spec in zip(
                fgraph.outputs, outputs + additional_outputs, strict=True
            )
            if not spec.borrow
        ]
        if no_borrow:
            self.linker = linker.accept(
                fgraph, no_recycling=infer_reuse_pattern(fgraph, no_borrow)
            )
        else:
            self.linker = linker.accept(fgraph)
        fgraph.name = name
        self.indices = indices
        self.inputs = inputs
        # TODO: Get rid of all this `expanded_inputs` nonsense
        self.expanded_inputs = inputs
        self.outputs = outputs
        self.unpack_single = unpack_single
        self.return_none = return_none
        self.accept_inplace = accept_inplace
        self.function_builder = function_builder
        self.on_unused_input = on_unused_input  # Used for the pickling/copy
        self.output_keys = output_keys
        self.name = name
        self.trust_input = trust_input

        self.required = [(i.value is None) for i in self.inputs]
        self.refeed = [
            (
                i.value is not None
                and not isinstance(i.value, Container)
                and i.update is None
            )
            for i in self.inputs
        ]


class DebugMode(Mode):
    """
    Evaluation Mode that detects internal pytensor errors.

    This mode catches several kinds of internal error:

    - Inconsistent outputs when calling the same Op twice with the same
      inputs, for instance if c_code and perform implementations, are
      inconsistent, or in case of incorrect handling of output memory
      (see `BadThunkOutput`).

    - A variable replacing another when their runtime values don't
      match.  This is a symptom of an incorrect optimization step, or
      faulty Op implementation (raises `BadOptimization`).

    - Stochastic optimization ordering (raises `StochasticOrder`).

    - Incomplete `destroy_map` specification (raises `BadDestroyMap`).

    - An op that returns an illegal value not matching the output
      Variable Type (raises InvalidValueError).

    Each of these exceptions inherits from the more generic `DebugModeError`.

    If there are no internal errors, this mode behaves like FAST_RUN
    or FAST_COMPILE, but takes a little longer and uses more memory.

    Raises
    ------
    DebugModeError
        If there are internal errors.

    Notes
    -----
    The work of debugging is implemented by the `_Maker`, `_Linker`,
    and `_VariableEquivalenceTracker` classes.

    """

    stability_patience = config.DebugMode__patience
    """
    When checking for the stability of optimization, recompile the
    graph this many times.

    """

    check_c_code = config.DebugMode__check_c
    """
    Should we evaluate (and check) the `c_code` implementations?

    """

    check_py_code = config.DebugMode__check_py
    """
    Should we evaluate (and check) the `perform` implementations?
    Always checked if no `c_code`.

    """

    check_isfinite = config.DebugMode__check_finite
    """
    Should we check for (and complain about) NaN/Inf ndarray elements?

    """

    require_matching_strides = config.DebugMode__check_strides
    """
    Should we check for (and complain about) Ops whose python and C
    outputs are ndarrays with different strides? (This can catch bugs,
    but is generally overly strict.) 0 no check, 1 warn, 2 err.

    """

    check_preallocated_output = config.DebugMode__check_preallocated_output
    check_preallocated_output = check_preallocated_output.split(":")
    """
    List of strings representing ways to pre-allocate output memory in
    tests.  Valid values are: "previous" (previously-returned memory),
    "c_contiguous", "f_contiguous", "strided" (positive and negative
    strides), "wrong_size" (larger and smaller dimensions), and "ALL"
    (all of the above).

    """

    # This function will be used to create a FunctionMaker in
    # function.types.function
    def function_maker(self, i, o, m, *args, **kwargs):
        """
        Return an instance of `_Maker` which handles much of the debugging work.

        """
        assert m is self
        return _Maker(i, o, self, *args, **kwargs)

    def __init__(
        self,
        optimizer="fast_run",
        stability_patience=None,
        check_c_code=None,
        check_py_code=None,
        check_isfinite=None,
        check_preallocated_output=None,
        require_matching_strides=None,
        linker=None,
        db=None,
    ):
        """
        If any of these arguments (except optimizer) is not None, it overrides
        the class default. The linker argument is not used. It is set there to
        allow Mode.requiring() and some other fct to work with DebugMode too.

        """
        if linker is None:
            linker = _DummyLinker()

        if not isinstance(linker, _DummyLinker):
            raise Exception(
                "DebugMode can only use its own linker! You should not provide one.",
                linker,
            )

        super().__init__(optimizer=optimizer, linker=linker, db=db)

        if stability_patience is not None:
            self.stability_patience = stability_patience

        if check_c_code is not None:
            self.check_c_code = check_c_code

        if check_py_code is not None:
            self.check_py_code = check_py_code

        if check_isfinite is not None:
            self.check_isfinite = check_isfinite

        if check_preallocated_output is not None:
            # Copy to avoid sharing the same list across different instances
            self.check_preallocated_output = check_preallocated_output[:]

        if require_matching_strides is not None:
            self.require_matching_strides = require_matching_strides

        if not (self.check_c_code or self.check_py_code):
            raise ValueError("DebugMode has to check at least one of c and py code")

    def __str__(self):
        return f"DebugMode(linker={self.provided_linker}, optimizer={self.provided_optimizer})"


register_mode("DEBUG_MODE", DebugMode(optimizer="fast_run"))

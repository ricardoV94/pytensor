"""Buffer-reduction rewrites: shrink Scan storage and iteration counts.

These rewrites drop work and storage that nothing downstream actually
needs:

* :func:`scan_merge_subtensor_chain` — fold ``scan_out[init_l:]…[const_idx]``
  chains into a direct ``raw_out[final_idx]`` so ``scan_save_mem`` can see
  through nested constant-bound subtensors.
* :func:`scan_save_mem_rewrite` (registered as ``scan_save_mem_prealloc``
  / ``scan_save_mem_no_prealloc``) — shorten outer buffers and the
  ``n_steps`` to the smallest range any client actually reads.
* :func:`scan_sit_sot_to_untraced` — convert sit_sot states whose
  history is unused into the cheaper ``untraced_sit_sot`` form.
"""

import dataclasses
from itertools import chain
from typing import cast

import numpy as np

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.fg import Output
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.graph.traversal import apply_depends_on
from pytensor.raise_op import Assert
from pytensor.scalar import ScalarConstant
from pytensor.scan.op import Scan
from pytensor.scan.utils import expand_empty
from pytensor.tensor.basic import (
    AllocEmpty,
    atleast_Nd,
    get_scalar_constant_value,
)
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import maximum, minimum
from pytensor.tensor.rewriting.basic import broadcasted_by
from pytensor.tensor.subtensor import (
    IncSubtensor,
    Subtensor,
    as_index_constant,
    basic_subtensor,
    get_canonical_form_slice,
    get_idx_list,
)
from pytensor.tensor.variable import TensorVariable


def select_min(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return minimum(x, y)


def select_max(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return maximum(x, y)


def sanitize(x):
    if x is None:
        return None
    else:
        return pt.as_tensor_variable(x)


def _const_int_or_none(v):
    """Return v as a Python int if it's a constant scalar (None passes through)."""
    if v is None:
        return None
    try:
        return int(get_scalar_constant_value(v, max_recur=4))
    except NotScalarConstantError:
        return None


def _enumerate_subtensor_chains(fgraph, root):
    """Yield ``(terminal_node, chain_indices, trailings)`` for every chain of
    ``Subtensor`` clients rooted at ``root`` whose first-dim entries are all
    constant and whose terminal first-dim entry is a scalar.

    ``chain_indices`` is a list of pure-Python slice / int objects (one per
    chain step) describing the leading-axis index. ``trailings`` is a parallel
    list of tuples — each tuple holds the idx_list entries past the first dim
    for that step, to be replayed verbatim once the leading axis has been
    folded.
    """

    def _python_slice_from_idx(entry):
        """Convert an idx_list first-entry into a pure-Python slice / int.

        Returns ``None`` if the entry has any non-constant component.
        """
        if isinstance(entry, slice):
            a = _const_int_or_none(entry.start)
            b = _const_int_or_none(entry.stop)
            s = _const_int_or_none(entry.step)
            if entry.start is not None and a is None:
                return None
            if entry.stop is not None and b is None:
                return None
            if entry.step is not None and s is None:
                return None
            return slice(a, b, s)
        return _const_int_or_none(entry)

    stack = [(root, [], [])]
    while stack:
        var, prefix, trailings = stack.pop()
        for client_node, _ in fgraph.clients.get(var, []):
            if not isinstance(client_node.op, Subtensor):
                continue
            this_slice = get_idx_list(client_node.inputs, client_node.op.idx_list)
            entry = _python_slice_from_idx(this_slice[0])
            if entry is None:
                continue
            new_prefix = [*prefix, entry]
            new_trailings = [*trailings, tuple(this_slice[1:])]
            if isinstance(entry, slice):
                stack.append((client_node.outputs[0], new_prefix, new_trailings))
            else:
                yield client_node, new_prefix, new_trailings


@dataclasses.dataclass(slots=True)
class _Affine:
    """``const + coeff * raw_length`` with ``coeff >= 0``.

    ``raw_length = n_steps + init_l`` is unknown at compile time;
    we keep ``const`` and ``coeff`` as Python ints and reason symbolically.
    """

    const: int
    coeff: int = 0

    def __add__(self, other: "_Affine") -> "_Affine":
        return _Affine(self.const + other.const, self.coeff + other.coeff)

    def __sub__(self, other: "_Affine") -> "_Affine":
        return _Affine(self.const - other.const, self.coeff - other.coeff)


class _ChainFolder:
    """Walk a chain of constant-bound subtensor slices symbolically.

    Tracks the current view as ``(base, length, direction)`` against the
    raw length, plus the smallest ``raw_length`` for which every bound seen
    so far stays in range (``n_steps_min``). Each ``apply_*`` call mutates
    the state; constraints accumulate via ``_require``.

    When ``raw_length`` is known (constant ``n_steps``), pass it in so the
    affine math collapses to plain integer arithmetic and ``n_steps_min``
    stays at ``0``. When unknown, leave it ``None`` and the folder reasons
    symbolically with ``length = 0 + 1 * raw_length``.
    """

    def __init__(self, raw_length: int | None = None):
        self.base = _Affine(0)
        self.length = _Affine(raw_length) if raw_length is not None else _Affine(0, 1)
        self.direction = +1
        self.n_steps_min = 0

    @staticmethod
    def _ceildiv(a: int, b: int) -> int:
        """``ceil(a / b)`` for ``b > 0`` (any sign of ``a``)."""
        return -(-a // b)

    def _require(self, need: int) -> None:
        if need > self.n_steps_min:
            self.n_steps_min = need

    def _normalize_forward(self, bound: int | None) -> _Affine:
        """Clamp a slice bound for ``step >= 1`` to a non-negative position."""
        if bound is None:
            return self.length
        const, coeff = self.length.const, self.length.coeff
        if bound >= 0:
            if coeff == 0:
                return _Affine(min(bound, const))
            self._require(self._ceildiv(bound - const, coeff))
            return _Affine(bound)
        # bound < 0: max(0, length + bound).
        if coeff == 0:
            return _Affine(max(0, const + bound))
        self._require(self._ceildiv(-const - bound, coeff))
        return _Affine(const + bound, coeff)

    def _normalize_reverse(self, bound: int | None, *, is_start: bool) -> _Affine:
        """Clamp a slice bound for ``step == -1``.

        For ``[start:stop:-1]`` Python clamps ``start`` into ``[-1, length-1]``
        and ``stop`` into ``[-1, length]``; defaults are ``start = length - 1``
        and ``stop = -1``.
        """
        if bound is None:
            return self.length - _Affine(1) if is_start else _Affine(-1)
        const, coeff = self.length.const, self.length.coeff
        if bound >= 0:
            if coeff == 0:
                return _Affine(min(bound, const - 1))
            self._require(self._ceildiv(bound + 1 - const, coeff))
            return _Affine(bound)
        # bound < 0: max(-1, length + bound).
        if coeff == 0:
            return _Affine(max(-1, const + bound))
        self._require(self._ceildiv(-1 - const - bound, coeff))
        return _Affine(const + bound, coeff)

    def apply_slice(self, sl: slice) -> bool:
        """Fold one slice into the current view. Returns ``False`` for
        ``sl.step`` not in ``{None, 1, -1}``.
        """
        step = 1 if sl.step is None else sl.step
        if step == 1:
            start = self._normalize_forward(sl.start)
            stop = self._normalize_forward(sl.stop)
            new_length = stop - start
        elif step == -1:
            start = self._normalize_reverse(sl.start, is_start=True)
            stop = self._normalize_reverse(sl.stop, is_start=False)
            new_length = start - stop
        else:
            return False

        if new_length.coeff == 0 and new_length.const < 0:
            new_length = _Affine(0)
        self.base = self.base + start if self.direction == +1 else self.base - start
        self.length = new_length
        if step == -1:
            self.direction = -self.direction
        return True

    def apply_index(self, k: int) -> _Affine | None:
        """Fold the terminal scalar index into a raw-buffer position.

        Returns ``None`` when ``k`` is provably out of range.
        """
        const, coeff = self.length.const, self.length.coeff
        if k >= 0:
            if coeff == 0 and k >= const:
                return None
            if coeff > 0:
                self._require(self._ceildiv(k + 1 - const, coeff))
            view_pos = _Affine(k)
        else:
            if coeff == 0 and -k > const:
                return None
            if coeff > 0:
                self._require(self._ceildiv(-k - const, coeff))
            view_pos = self.length + _Affine(k)
        return self.base + view_pos if self.direction == +1 else self.base - view_pos


@node_rewriter([Scan])
def scan_merge_subtensor_chain(fgraph, scan_node):
    """Fold ``scan_out[init_l:]…[const_idx]`` chains into ``scan_out[final_idx]``.

    ``pytensor.scan`` strips the initial state (``raw_out[init_l:]``) before
    returning. When client code writes ``result[…][k]`` the graph carries a
    chain of ``Subtensor`` nodes that ``scan_save_mem`` cannot read past the
    first strip. This rewrite walks each chain that bottoms out in a constant
    scalar index, folds the constant-step slices in between, and replaces the
    terminal client with a direct ``raw_out[final_idx]`` (plus an
    ``Assert(n_steps >= …)`` when negative-index safety requires it).

    ``local_subtensor_merge`` can't do this job: it's gated to fully-constant
    bounds and shapes (see ``_can_merge_simply``) because the unconstrained
    merge produces big switch/min/max trees on symbolic ``n_steps``, and Scan
    outputs whose strip ``raw_out[init_l:]`` has symbolic shape fall outside
    that gate. Folding here is safe because we know the structure (a leading
    strip + scalar index) and can emit an ``Assert`` to discharge the
    negative-index range check that ``local_subtensor_merge`` has no way to express.

    Only constant slice bounds and step ∈ {None, 1, -1} are handled.
    Non-foldable chains are left untouched.
    """
    op = scan_node.op

    recurrent_outputs = op.outer_mitsot_outs(scan_node.outputs) + op.outer_sitsot_outs(
        scan_node.outputs
    )
    recurrent_taps = op.info.mit_sot_in_slices + op.info.sit_sot_in_slices
    nitsot_outputs = op.outer_nitsot_outs(scan_node.outputs)

    n_steps = scan_node.inputs[0]
    n_steps_const = _const_int_or_none(n_steps)
    replacements: dict = {}

    all_outputs = [
        (x, abs(min(taps))) for x, taps in zip(recurrent_outputs, recurrent_taps)
    ] + [(x, 0) for x in nitsot_outputs]

    for scan_out, init_l in all_outputs:
        raw_length_const = n_steps_const + init_l if n_steps_const is not None else None
        for terminal_node, (
            *chain_indices,
            k,
        ), trailings in _enumerate_subtensor_chains(fgraph, scan_out):
            # Single-entry chain (just ``raw[k]``) means there's nothing to
            # flatten -- ``scan_save_mem`` / ``scan_reduce_nsteps`` already
            # handle the direct case and re-firing here would just re-wrap
            # any required ``Assert`` ad infinitum.
            if not chain_indices:
                continue

            # Walk the chain through a fresh folder. With ``raw_length_const``
            # the affine math collapses to plain int arithmetic; without it,
            # the folder reasons symbolically against an unknown raw length.
            # ``_enumerate_subtensor_chains`` guarantees every entry but the
            # last is a slice and the last is a scalar int.
            folder = _ChainFolder(raw_length_const)
            for entry in chain_indices:
                if not folder.apply_slice(entry):
                    return None
            raw_pos = folder.apply_index(k)
            if raw_pos is None:
                return None
            final_idx, coeff = raw_pos.const, raw_pos.coeff
            # ``raw_pos = const + coeff * raw_length``; we can only emit a Python int.
            # ``coeff == 0`` is a plain constant (positive or negative).
            # ``coeff == 1, const < 0`` is the negative-form.
            # Anything else is always out of range and unfoldable.
            if not (coeff == 0 or (coeff == 1 and final_idx < 0)):
                return None
            # Convert the absolute raw position to negative-from-end form
            # when raw_length is known so the while-scan check and downstream
            # save_mem reasoning recognize "last position" uniformly.
            if coeff == 0 and raw_length_const is not None:
                final_idx -= raw_length_const

            # While-scans: only views into last state (-1) or into initial buffer are safe
            # under early exit. No other case can be ensured from n_steps alone
            if op.info.as_while and not (final_idx == -1 or 0 <= final_idx <= init_l):
                continue

            # Move the final idx into the scan buffer output
            new_out = scan_out[final_idx]
            for trailing in trailings:
                # then replay each step's trailing-axis indices.
                if trailing:
                    new_out = new_out[trailing]

            # The folder tracks a static minimum on raw_length;
            # convert to a minimum on ``n_steps`` (raw_length = n_steps + init_l).
            if (n_steps_min := max(0, folder.n_steps_min - init_l)) > 0:
                # For the index to be valid scan needs at least this many steps
                assert_op = Assert(f"n_steps >= {n_steps_min}")
                new_out = assert_op(new_out, n_steps >= n_steps_min)

            copy_stack_trace(
                [terminal_node.outputs[0], terminal_node.inputs[0]], new_out
            )
            replacements[terminal_node.outputs[0]] = new_out

    return replacements or None


def _is_default_scan_buffer(final_buffer: TensorVariable, taps: int) -> bool:
    node = final_buffer.owner

    if node is None:
        return False

    op = node.op
    if not (
        isinstance(op, IncSubtensor)
        and op.set_instead_of_inc
        and op.idx_list == (slice(None, 0),)
    ):
        return False

    init_buffer, init_value, *_ = node.inputs
    if not (
        init_buffer.owner is not None and isinstance(init_buffer.owner.op, AllocEmpty)
    ):
        return False

    # The value may have been broadcast to fill in the initial taps.
    # If the user specified outputs as:
    #   x = scalar(); init = alloc(x, 2);
    #   outputs_info=[init, taps=(-2, -1)]
    # Scan will generate an initial buffer that looks like
    #   alloc_empty(2 + nsteps)[:2].set(alloc(x, 2))
    # PyTensor will then rewrite it as:
    #   alloc_empty(2 + nsteps)[:2].set(x)
    # When the initial value (x) is being broadcast by the set_subtensor
    # we can't recreate a newly sized buffer working with x alone
    # We want to check that:
    #   1. alloc_empty(2 + nsteps)[:2].broadcastable == x.broadcastable
    # But due to laziness we use the slightly more conservative check:
    #   2. alloc_empty(2 + nsteps).broadcastable == x.broadcastable
    if taps > 1:
        return not broadcasted_by(init_value, init_buffer)
    else:
        # In this case we know we have alloc_empty(1 + nsteps, ...)[:1].set(init_value)
        # The first dimension cannot possibly broadcast in the subtensor assignment,
        # so we exclude it from `broadcasted_by`. To exclude it we squeeze it out,
        # after adding any other implicit expand_dims. We select into the first entry of
        # the buffer, to check for potential broadcasting in other dimensions.
        init_value_ = atleast_Nd(init_value, n=init_buffer.ndim)
        return not broadcasted_by(init_value_.squeeze(0), init_buffer[0])


def scan_save_mem_rewrite(fgraph, node, backend_supports_output_pre_allocation: bool):
    r"""Graph optimizer that reduces scan memory consumption.

    This optimizations attempts to determine if a `Scan` node, during its execution,
    for any of its outputs, can get away with allocating a memory buffer that is
    large enough to contain some of the computed timesteps of that output but not
    all of them.

    By default, during the execution of a `Scan` node, memory buffers will be
    allocated to store the values computed for every output at every iteration.
    However, in some cases, there are outputs for which there is only really a
    need to store the most recent ``N`` values, not all of them.

    For instance, if a `Scan` node has a SITSOT output (last computed value is
    fed back as an input at the next iteration) and only the last timestep of
    that output is ever used in the outer function, the `ScanSaveMem` optimization
    could determine that there is no need to store all computed timesteps for
    that SITSOT output. Only the most recently computed timestep ever needs to
    be kept in memory.

    There are two ways in which the Scan buffer size is controlled:
    1. Each recurring output is saved in an input empty tensor x with the initial
    state written at x[:abs(min(taps))]. The remaining x[abs(min(taps)):]
    positions determine how many intermediate results should be stored.
    This rewrite shortens x[abs(min(taps)):] to the smallest possible size.
    2. Each non-recurrent output (nit-sot) is associated with a scalar integer
    input that determines how many steps should be saved in the perform method.
    This rewrite reduces this number to the smallest possible.

    The scan perform implementation takes the output sizes into consideration,
    saving the newest results over the oldest ones whenever the buffer is filled.

    This rewrite must only run at compilation time, after grad() has already
    built the backward scan. The backward scan needs all intermediate forward
    states as sequence inputs (to evaluate f'(x[t])). If this rewrite truncates
    buffers before grad() is called, the gradient will be silently wrong.
    TODO: Use a subclass that raises explicitly on `L_op`

    Paramaters
    ----------
    backend_supports_output_pre_allocation: bool
        When the backend supports output pre-allocation Scan must keep buffers
        with a length of required_states + 1, because the inner function will
        attempt to write the inner function outputs directly into the provided
        position in the outer circular buffer. This would invalidate results,
        if the input is still needed for some other output computation.
    """
    if hasattr(fgraph, "shape_feature"):
        shape_of = fgraph.shape_feature.shape_of
    else:
        # Each access to shape_of is in a try..except block in order to
        # use a default version when the variable is not in the shape_of
        # dictionary.
        shape_of = {}
    # 1. Initialization of variables
    # Note 1) We do not actually care about outputs representing shared
    # variables (those have no intermediate values) so it is safer to
    # ignore them and not change them in any way. To simplify the
    # optimizations I construct the variable ``c_outs`` ( that counts
    # outputs up to those we care) and the list ``init_l`` which for any
    # output we care says the length of its initial state. Note that
    # defining ``init_l`` for mit_mot sequences is a bit trickier but
    # it is safe to set it to 0
    op = node.op
    op_info = op.info
    c_outs = (
        op_info.n_mit_mot + op_info.n_mit_sot + op_info.n_sit_sot + op_info.n_nit_sot
    )

    init_l = [0 for x in range(op_info.n_mit_mot)]
    init_l += [
        abs(min(v)) for v in chain(op_info.mit_sot_in_slices, op_info.sit_sot_in_slices)
    ]
    init_l += [0 for x in range(op_info.n_nit_sot)]
    # 2. Check the clients of each output and see for how many steps
    # does scan need to run

    # This comparison checks if there is any uncounted output, which
    # can only be an output corresponding to a shared variable

    # 2.1 Initialize
    # global_nsteps is a dictionary having two fields ( 'real' deals
    # with int values, 'sym' with symbolic ones) or None
    # given that a scan op has k outputs o_1, .. o_k and each
    # output has n_j clients c_1^1, c_1^2, .. c_1^{n_1}, c_2^1, ..,
    # global_nsteps is None if any of the clients is different
    # from a subtensor or its real and sym field equal to
    # max(c_i_j.idx_list[0].stop), meaning store up to which maximal
    # index(step) for any output scan actually needs to compute
    # In other words n_steps should be equal to this maximal !
    # Note: if we have a shared variable that gets updated at every step
    # of the loop, reducing the number of steps will affect the
    # value of the shared variable after the loop so we cannot
    # change the number of steps in that case. To do this we set
    # global_nsteps to None which is seen as a flag that nothing needs
    # to be done.
    # Note: For simplicity while Scans also have global_nsteps set to None.
    #  All step optimizations require knowing the shape of the output, which
    #  cannot be determined from the inputs alone.
    global_nsteps: None | dict
    assert len(node.outputs) >= c_outs
    if len(node.outputs) == c_outs and not op.info.as_while:
        global_nsteps = {"real": -1, "sym": []}
    else:
        global_nsteps = None

    # Keeps track of the original slices that each client represent
    slices: list[None | list] = [None for o in node.outputs]

    # For each output: how many intermediate values to store.
    # 0 means keep all (required for mit_mot and shared outputs);
    # -1 is a "no decision" sentinel flipped to 0 after the trimming loop.
    store_steps = [0 for o in range(op_info.n_mit_mot)]
    store_steps += [-1 for o in node.outputs[op_info.n_mit_mot : c_outs]]
    # Flag that says if an input has changed and we need to do something
    # or not
    flag_store = False

    # 2.2 Loop over the clients to figure out how many steps we actually need to do in the Scan
    for i, out in enumerate(node.outputs[:c_outs]):
        # look at all its clients
        slices[i] = []
        for cl, _ in fgraph.clients[out]:
            # 2.1 outputs of the function
            # => output needs all its intermediate values
            if isinstance(cl.op, Output):
                # if the node is actually an output, then
                # we need to store the entire thing
                global_nsteps = None
                slices[i] = None
                break
            # 2.2 non-subtensor nodes
            # => output needs all its intermediate values
            elif not isinstance(cl.op, Subtensor):
                global_nsteps = None
                slices[i] = None
                break
            # 2.3 subtensor nodes
            # => output might need to store just a subset of its values
            else:
                # 2.3.1 extract idx list of subtensor
                this_slice = get_idx_list(cl.inputs, cl.op.idx_list)

                # 2.3.2 extract the begin/end of the first dimension
                if i >= op_info.n_mit_mot:
                    try:
                        length = shape_of[out][0]
                    except KeyError:
                        length = node.inputs[0] + init_l[i]
                else:
                    try:
                        length = shape_of[out][0]
                    except KeyError:
                        length = out.shape[0]
                cf_slice = get_canonical_form_slice(this_slice[0], length)
                slices[i] += [(cf_slice, this_slice)]  # type: ignore

                if isinstance(this_slice[0], slice) and this_slice[0].stop is None:
                    global_nsteps = None
                if isinstance(cf_slice[0], slice):
                    stop = get_scalar_constant_value(
                        cf_slice[0].stop, raise_not_constant=False
                    )
                else:
                    stop = (
                        get_scalar_constant_value(cf_slice[0], raise_not_constant=False)
                        + 1
                    )
                if stop == get_scalar_constant_value(length, raise_not_constant=False):
                    stop = None
                    global_nsteps = None
                else:
                    # there is a **gotcha** here ! Namely, scan returns an
                    # array that contains the initial state of the output
                    # as well. Which means that if y has an initial state of
                    # length 3, and you look for 5 steps you get an output
                    # y of length 8. If you only use y[:5], this does not
                    # mean that you only need to loop for 5 steps but
                    # actually only for 2 steps ( the first 3 are the
                    # initial state)
                    stop = stop - init_l[i]

                # 2.3.3 we might get away with fewer steps
                if stop is not None and global_nsteps is not None:
                    # yes if it is a tensor
                    if isinstance(stop, Variable):
                        global_nsteps["sym"] += [stop]
                    elif isinstance(stop, int | np.integer):
                        global_nsteps["real"] = max(global_nsteps["real"], stop)
                    else:
                        global_nsteps = None

    # 2.3. Analyze global_nsteps to figure out for how many steps scan
    # needs to iterate
    if global_nsteps is None:
        nw_steps = node.inputs[0]
    else:
        # there are some symbolic tensors that limit the number of
        # steps
        if len(global_nsteps["sym"]) == 0:
            sym_steps = None
        else:
            sym_steps = global_nsteps["sym"][0]
            for c in global_nsteps["sym"][1:]:
                sym_steps = maximum(sym_steps, c)

        if global_nsteps["real"] >= 0:
            real_steps = global_nsteps["real"]
        else:
            real_steps = None
        nw_steps = select_min(select_max(sym_steps, real_steps), node.inputs[0])

    # 2.4 Loop over the clients again now looking just to see how many
    # intermediate steps to store. Skip mit_mot outputs as their
    # store_steps is always 0 (all intermediate values are needed).
    for i, out in enumerate(
        node.outputs[op_info.n_mit_mot : c_outs], start=op_info.n_mit_mot
    ):
        # look at all its clients
        for cl, _ in fgraph.clients[out]:
            if isinstance(cl.op, Output):
                store_steps[i] = 0
                break
            elif not isinstance(cl.op, Subtensor):
                store_steps[i] = 0
                break
            else:
                this_slice = get_idx_list(cl.inputs, cl.op.idx_list)

                if isinstance(this_slice[0], slice):
                    start = this_slice[0].start
                    if isinstance(start, Constant):
                        start = start.data
                    # Don't do anything if the subtensor is starting from the beginning of the buffer
                    # Or just skipping the initial values (default output returned to the user).
                    # Trimming the initial values would require a roll to align the buffer once scan is done
                    # As it always starts writing at position [0+max(taps)], and ends up at position [:max(taps)]
                    # It's cheaper to just keep the initial values in the buffer and slice them away (default output)
                    if start in (0, None, init_l[i]):
                        store_steps[i] = 0
                        break

                length = node.inputs[0] + init_l[i]
                cf_slice = get_canonical_form_slice(this_slice[0], length)

                if isinstance(cf_slice[0], slice):
                    start = pt.get_scalar_constant_value(
                        cf_slice[0].start, raise_not_constant=False
                    )
                else:
                    start = pt.get_scalar_constant_value(
                        cf_slice[0], raise_not_constant=False
                    )

                if start == 0 or store_steps[i] == 0:
                    store_steps[i] = 0
                else:
                    # The "+ 1" is because of the memory pre-allocation
                    # mechanism used to in the Scan op to reduce overhead.
                    # To prevent aliasing between the inputs and outputs
                    # of recurrent states, it requires that the buffer be
                    # large enough to that, the new state and the oldest
                    # tap needed don't occupy the sample place in the
                    # circular buffer. For now, this only needs to be done
                    # for mitsots and sitsots (because mitmots are not
                    # currently supported by the mechanism) and only if
                    # the pre-allocation mechanism is activated.
                    prealloc_outs = (
                        backend_supports_output_pre_allocation
                        and config.scan__allow_output_prealloc
                    )

                    first_mitsot_idx = op_info.n_mit_mot
                    last_sitsot_idx = (
                        op_info.n_mit_mot + op_info.n_mit_sot + op_info.n_sit_sot - 1
                    )
                    preallocable_output = first_mitsot_idx <= i <= last_sitsot_idx

                    if prealloc_outs and preallocable_output:
                        # TODO: If there's only one output or other outputs do not depend
                        #  on the same input, we could reduce the buffer size to the minimum
                        # The extra entry to prevent aliasing between the new
                        # state and the oldest tap is only needed when the
                        # scan actually runs (nw_steps >= 1).
                        pval = select_max(
                            nw_steps - start + init_l[i],
                            init_l[i] + minimum(nw_steps, 1),
                        )
                    else:
                        pval = select_max(nw_steps - start + init_l[i], init_l[i])

                    if store_steps[i] != -1:
                        pval = select_max(pval, store_steps[i])

                    store_steps[i] = pval
                    flag_store = True

    # A clientless mit_sot / sit_sot may still be read by the inner
    # recurrence; keep the minimum its taps need (plus one slot under prealloc).
    prealloc_outs = (
        backend_supports_output_pre_allocation and config.scan__allow_output_prealloc
    )
    for i in range(
        op_info.n_mit_mot,
        op_info.n_mit_mot + op_info.n_mit_sot + op_info.n_sit_sot,
    ):
        if store_steps[i] == -1:
            store_steps[i] = init_l[i] + (1 if prealloc_outs else 0)
            flag_store = True
    # Remaining -1s are unused nit_sots; leave their buffers untouched (0 =
    # keep all) -- scan_remove_unused will drop them entirely.
    store_steps = [0 if x == -1 else x for x in store_steps]

    # 3. is there anything to change ?
    if flag_store or global_nsteps is not None:
        # 3.1 initialize inputs for the new scan
        old_outputs = []
        nw_inputs = list(node.inputs)
        nw_inputs[0] = nw_steps

        # 3.2. compose replace pairs for those nodes that need not store everything in memory
        replaced_outs = []
        offset = 1 + op_info.n_seqs + op_info.n_mit_mot
        for idx, val in enumerate(store_steps[op_info.n_mit_mot :]):
            i = idx + op_info.n_mit_mot
            if not (isinstance(val, int) and val <= 0):
                # If the memory for this output has been pre-allocated
                # before going into the scan op (by an alloc node)
                if idx < op_info.n_mit_sot + op_info.n_sit_sot:
                    taps = init_l[i]
                    nw_input = nw_inputs[offset + idx]

                    # Recreate default buffers with new size
                    if _is_default_scan_buffer(nw_input, taps):
                        extra_size = val - taps
                        nw_input = expand_empty(nw_input.owner.inputs[1], extra_size)
                    # Otherwise, just trim with a slice
                    else:
                        nw_input = nw_input[:val]

                    nw_inputs[offset + idx] = nw_input
                    replaced_outs.append(op_info.n_mit_mot + idx)
                    odx = op_info.n_mit_mot + idx
                    old_outputs += [
                        (
                            odx,
                            [
                                x[0].outputs[0]
                                for x in fgraph.clients[node.outputs[odx]]
                            ],
                        )
                    ]
                # If there is no memory pre-allocated for this output
                elif idx < op_info.n_mit_sot + op_info.n_sit_sot + op_info.n_nit_sot:
                    pos = (
                        op_info.n_mit_mot
                        + idx
                        + op_info.n_seqs
                        + 1
                        + op_info.n_untraced_sit_sot
                    )
                    if nw_inputs[pos] == node.inputs[0]:
                        nw_inputs[pos] = val
                    odx = op_info.n_mit_mot + idx
                    replaced_outs.append(odx)
                    old_outputs += [
                        (
                            odx,
                            [
                                x[0].outputs[0]
                                for x in fgraph.clients[node.outputs[odx]]
                            ],
                        )
                    ]
        # 3.3. Recompute inputs for everything else based on the new number of steps
        if global_nsteps is not None:
            for idx, val in enumerate(store_steps[op_info.n_mit_mot :]):
                if val == 0:
                    # val == 0 means that we want to keep all intermediate
                    # results for that state, including the initial values.
                    if idx < op_info.n_mit_sot + op_info.n_sit_sot:
                        taps = init_l[op_info.n_mit_mot + idx]
                        in_idx = offset + idx
                        nw_input = nw_inputs[in_idx]
                        if _is_default_scan_buffer(nw_input, taps):
                            nw_input = expand_empty(nw_input.owner.inputs[1], nw_steps)
                        else:
                            nw_input = nw_input[: (taps + nw_steps)]
                        nw_inputs[in_idx] = nw_input

                    elif (
                        idx < op_info.n_mit_sot + op_info.n_sit_sot + op_info.n_nit_sot
                    ):
                        in_idx = offset + idx + op_info.n_untraced_sit_sot
                        if nw_inputs[in_idx] == node.inputs[0]:
                            nw_inputs[in_idx] = nw_steps

        # 3.4. Recreate the same scan with new outer inputs.
        new_outs = cast(list[TensorVariable], op(*nw_inputs, return_list=True))

        old_new = []
        # 3.5 Get replace pairs for those outputs that do not change
        # the number of intermediate steps stored
        for idx, sl in enumerate(slices):
            if global_nsteps and sl is not None and store_steps[idx] == 0:
                for hdx, cl in enumerate(fgraph.clients[node.outputs[idx]]):
                    cnf_slice, old_slices = sl[hdx]
                    # Sanitize the nw_slice by converting ints back into
                    # constants :) I only need to do this for the first
                    # slice since that is the only slice

                    if isinstance(cnf_slice[0], slice):
                        fslice = slice(
                            sanitize(cnf_slice[0].start),
                            sanitize(cnf_slice[0].stop),
                            sanitize(cnf_slice[0].step),
                        )
                    else:
                        fslice = sanitize(cnf_slice[0])

                    new_o = basic_subtensor(new_outs[idx], fslice, *old_slices[1:])
                    if new_o.ndim > 0:
                        new_o = new_o[:: cnf_slice[1]]
                    replaced_outs.append(idx)
                    old_new += [(cl[0].outputs[0], new_o)]
        # 3.6. Get replace pairs for those outputs that change
        # the number of stored intermediate steps
        for pos, old_outs in old_outputs:
            if len(old_outs) > 0:
                for k, old in enumerate(old_outs):
                    # Get the correct slice
                    cnf_slice, old_slices = slices[pos][k]
                    if isinstance(cnf_slice[0], slice):
                        start = (
                            cnf_slice[0].start
                            - nw_steps
                            - init_l[pos]
                            + store_steps[pos]
                        )
                        if cnf_slice[0].stop is not None:
                            stop = (
                                cnf_slice[0].stop
                                - nw_steps
                                - init_l[pos]
                                + store_steps[pos]
                            )
                        else:
                            stop = None
                        nw_slice = (
                            slice(
                                sanitize(start),
                                sanitize(stop),
                                sanitize(cnf_slice[0].step),
                            ),
                            *old_slices[1:],
                        )

                    else:
                        # Special case when only last value is requested
                        if (
                            isinstance(old_slices[0], ScalarConstant)
                            and old_slices[0].value == -1
                        ):
                            position = old_slices[0]
                        else:
                            position = (
                                cnf_slice[0] - nw_steps - init_l[pos] + store_steps[pos]
                            )

                        nw_slice = (sanitize(position), *old_slices[1:])
                    new_o = basic_subtensor(new_outs[pos], *nw_slice)
                    if new_o.ndim > 0:
                        new_o = new_o[:: cnf_slice[1]]
                    old_new += [(old, new_o)]

        # 3.7. Get replace pairs for all other nodes
        for idx, o in enumerate(node.outputs):
            if idx not in replaced_outs:
                old_new += [(o, new_outs[idx])]
        # Check if the new outputs depend on the old scan node
        old_scan_is_used = [apply_depends_on(new.owner, node) for old, new in old_new]
        if any(old_scan_is_used):
            return False

        replacements = dict(old_new)
        replacements["remove"] = [node]
        return replacements

    return False


@node_rewriter([Scan])
def scan_save_mem_prealloc(fgraph, node):
    return scan_save_mem_rewrite(
        fgraph, node, backend_supports_output_pre_allocation=True
    )


@node_rewriter([Scan])
def scan_save_mem_no_prealloc(fgraph, node):
    return scan_save_mem_rewrite(
        fgraph, node, backend_supports_output_pre_allocation=False
    )


@node_rewriter([Scan])
def scan_sit_sot_to_untraced(fgraph, node):
    """Convert sit_sot with buffer size=1 to untraced_sit_sot.

    After scan_save_mem has reduced buffer sizes, sit_sot outputs that only
    need one state stored (buffer size=1) can be converted to untraced_sit_sot,
    which avoids the overhead of reading/writing circular buffers each iteration.
    """
    op = node.op
    info = op.info

    if info.n_sit_sot == 0:
        return False

    outer_sitsot = op.outer_sitsot(node.inputs)
    convertible = [
        idx for idx in range(info.n_sit_sot) if outer_sitsot[idx].type.shape[0] == 1
    ]

    if not convertible:
        return False

    convertible_set = set(convertible)

    # Gather current inner inputs/outputs by category
    inner_inputs = list(op.inner_inputs)
    inner_outputs = list(op.inner_outputs)

    inner_sitsot_ins = op.inner_sitsot(inner_inputs)
    inner_sitsot_outs = op.inner_sitsot_outs(inner_outputs)
    inner_untraced_ins = op.inner_untraced_sit_sot(inner_inputs)
    inner_untraced_outs = op.inner_untraced_sit_sot_outs(inner_outputs)

    # Split sit_sot into remaining and converted
    new_sit_sot_in_slices = []
    remaining_inner_sitsot_ins = []
    remaining_inner_sitsot_outs = []
    remaining_outer_sitsot = []
    converted_inner_untraced_ins = []
    converted_inner_untraced_outs = []
    converted_outer_untraced = []

    for idx in range(info.n_sit_sot):
        if idx in convertible_set:
            converted_inner_untraced_ins.append(inner_sitsot_ins[idx])
            converted_inner_untraced_outs.append(inner_sitsot_outs[idx])
            converted_outer_untraced.append(outer_sitsot[idx][0])
        else:
            new_sit_sot_in_slices.append(info.sit_sot_in_slices[idx])
            remaining_inner_sitsot_ins.append(inner_sitsot_ins[idx])
            remaining_inner_sitsot_outs.append(inner_sitsot_outs[idx])
            remaining_outer_sitsot.append(outer_sitsot[idx])

    # Rebuild inner inputs:
    # seqs | mit_mot_taps | mit_sot_taps | sit_sot | untraced_sit_sot | non_seqs
    n_taps_before_sitsot = sum(
        len(x) for x in chain(info.mit_mot_in_slices, info.mit_sot_in_slices)
    )
    pre_sitsot_inner = inner_inputs[: info.n_seqs + n_taps_before_sitsot]
    inner_non_seqs = op.inner_non_seqs(inner_inputs)

    new_inner_inputs = (
        pre_sitsot_inner
        + remaining_inner_sitsot_ins
        + list(inner_untraced_ins)
        + converted_inner_untraced_ins
        + inner_non_seqs
    )

    # Rebuild inner outputs:
    # mit_mot_outs | mit_sot | sit_sot | nit_sot | untraced_sit_sot [| while_cond]
    n_mit_mot_outs = sum(len(x) for x in info.mit_mot_out_slices)
    pre_sitsot_inner_outs = inner_outputs[: n_mit_mot_outs + info.n_mit_sot]
    nitsot_outs = op.inner_nitsot_outs(inner_outputs)

    new_inner_outputs = (
        pre_sitsot_inner_outs
        + remaining_inner_sitsot_outs
        + nitsot_outs
        + list(inner_untraced_outs)
        + converted_inner_untraced_outs
    )
    if info.as_while:
        new_inner_outputs.append(inner_outputs[-1])

    # Rebuild outer inputs:
    # n_steps | seqs | mit_mot | mit_sot | sit_sot | untraced_sit_sot | nit_sot | non_seqs
    pre_sitsot_outer = list(
        node.inputs[: 1 + info.n_seqs + info.n_mit_mot + info.n_mit_sot]
    )
    outer_untraced = list(op.outer_untraced_sit_sot(node.inputs))
    outer_nitsot = list(op.outer_nitsot(node.inputs))
    outer_non_seqs = list(op.outer_non_seqs(node.inputs))

    new_outer_inputs = (
        pre_sitsot_outer
        + remaining_outer_sitsot
        + outer_untraced
        + converted_outer_untraced
        + outer_nitsot
        + outer_non_seqs
    )

    # Build new ScanInfo
    new_info = dataclasses.replace(
        info,
        sit_sot_in_slices=tuple(new_sit_sot_in_slices),
        n_untraced_sit_sot=info.n_untraced_sit_sot + len(convertible),
    )

    new_op = Scan(
        new_inner_inputs,
        new_inner_outputs,
        new_info,
        mode=op.mode,
        profile=op.profile,
        truncate_gradient=op.truncate_gradient,
        name=op.name,
        allow_gc=op.allow_gc,
    )
    new_outs = cast(list[TensorVariable], new_op(*new_outer_inputs, return_list=True))

    # Build replacement mapping
    # Old outer outputs: mit_mot | mit_sot | sit_sot | nit_sot | untraced_sit_sot
    # New outer outputs: mit_mot | mit_sot | remaining_sit_sot | nit_sot | old_untraced | converted_untraced
    old_outputs = node.outputs
    replacements: dict = {}

    # mit_mot + mit_sot: same relative positions
    n_pre = info.n_mit_mot + info.n_mit_sot
    for i in range(n_pre):
        replacements[old_outputs[i]] = new_outs[i]

    # sit_sot: remaining keep position, converted become untraced
    old_sitsot_offset = n_pre
    new_remaining_offset = new_info.n_mit_mot + new_info.n_mit_sot
    new_converted_offset = (
        new_info.n_mit_mot
        + new_info.n_mit_sot
        + new_info.n_sit_sot
        + new_info.n_nit_sot
        + info.n_untraced_sit_sot
    )
    remaining_count = 0
    converted_count = 0
    for idx in range(info.n_sit_sot):
        old_out = old_outputs[old_sitsot_offset + idx]
        if idx in convertible_set:
            new_untraced = new_outs[new_converted_offset + converted_count]
            replacements[old_out] = pt.expand_dims(new_untraced, 0)
            converted_count += 1
        else:
            replacements[old_out] = new_outs[new_remaining_offset + remaining_count]
            remaining_count += 1

    # nit_sot
    old_nitsot_offset = n_pre + info.n_sit_sot
    new_nitsot_offset = new_info.n_mit_mot + new_info.n_mit_sot + new_info.n_sit_sot
    for i in range(info.n_nit_sot):
        replacements[old_outputs[old_nitsot_offset + i]] = new_outs[
            new_nitsot_offset + i
        ]

    # Original untraced_sit_sot
    old_untraced_offset = n_pre + info.n_sit_sot + info.n_nit_sot
    new_untraced_offset = (
        new_info.n_mit_mot
        + new_info.n_mit_sot
        + new_info.n_sit_sot
        + new_info.n_nit_sot
    )
    for i in range(info.n_untraced_sit_sot):
        replacements[old_outputs[old_untraced_offset + i]] = new_outs[
            new_untraced_offset + i
        ]

    replacements["remove"] = [node]
    return replacements

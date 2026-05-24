"""Fuse indexed reads and updates into Elemwise iteration loops.

Introduces ``IndexedElemwise``, an ``OpFromGraph`` that wraps
``AdvancedSubtensor1`` + ``Elemwise`` + ``AdvancedIncSubtensor1`` subgraphs
so the Numba backend can generate a single loop with indirect indexing,
eliminating materialised intermediate arrays.
"""

from pytensor.compile import optdb
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import GraphRewriter, dfs_rewriter
from pytensor.graph.rewriting.db import SequenceDB
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.printing import op_debug_information
from pytensor.scalar.basic import Composite
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.rewriting.elemwise import InplaceElemwiseOptimizer
from pytensor.tensor.shape import Reshape, shape_padright
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    indices_from_subtensor,
)
from pytensor.tensor.variable import TensorVariable


def _unwrap_axis_swapped_subtensor1(fgraph, var):
    """Unwrap ``AdvancedSubtensor1`` with optional ``DimShuffle`` axis-swap.

    Detects two patterns (all intermediates must be single-client):

    - ``AdvancedSubtensor1(source, idx)`` → ``(source, idx, 0)``
    - ``DimShuffle{swap}(AdvancedSubtensor1(DimShuffle{swap}(source), idx))``
      → ``(source, idx, axis)`` where *axis* is the non-zero swapped axis.
    """
    if var.owner is None:
        return None

    # Bare AdvancedSubtensor1 on axis 0
    if isinstance(var.owner.op, AdvancedSubtensor1):
        if len(fgraph.clients[var]) != 1:
            return None
        return var.owner.inputs[0], var.owner.inputs[1], 0

    # Check for axis-swap DimShuffle wrapping AdvancedSubtensor1
    if not isinstance(var.owner.op, DimShuffle) or not var.owner.op.is_transpose:
        return None

    # Find the swapped axis: exactly two positions differ from identity
    order = var.owner.op.new_order
    swapped = [i for i, o in enumerate(order) if o != i]
    if len(swapped) != 2:
        return None
    ax_a, ax_b = swapped
    if order[ax_a] != ax_b or order[ax_b] != ax_a:
        return None
    axis = max(ax_a, ax_b)  # the non-zero axis (0 was swapped to axis)

    # Inner must be a single-client AdvancedSubtensor1
    asub1_out = var.owner.inputs[0]
    if len(fgraph.clients[asub1_out]) != 1:
        return None
    match asub1_out.owner_op_and_inputs:
        case AdvancedSubtensor1(), inner_ds_var, idx_var:
            pass
        case _:
            return None

    # AdvancedSubtensor1's input must be a single-client inverse DimShuffle (same swap)
    if len(fgraph.clients[inner_ds_var]) != 1:
        return None
    match inner_ds_var.owner_op_and_inputs:
        case DimShuffle(is_transpose=True, new_order=new_order), source:
            if new_order != tuple(order):
                return None
        case _:
            return None

    return source, idx_var, axis


@node_rewriter([OpPattern(DimShuffle, is_transpose=True)])
def undo_take_dimshuffle_for_fusion(fgraph, node):
    """Undo ``DimShuffle(AdvancedSubtensor1(DimShuffle(x), idx))`` -> ``AdvancedSubtensor(x, :, ..., idx, :, ...)``.

    The ``local_replace_AdvancedSubtensor`` specialize rewrite converts
    ``x[:, idx]`` into ``x.T[idx].T`` (axis-swap + AdvancedSubtensor1 +
    axis-swap).  This rewrite undoes that when the result feeds a single
    Elemwise, so ``FuseIndexedElemwise`` can absorb the indexing directly
    on the correct axis.

    See also ``undo_take_reshape_for_fusion`` which handles the analogous
    Reshape+flatten pattern for ND indices.
    """
    # Outer DimShuffle must be consumed only by a single Elemwise
    clients = fgraph.clients[node.outputs[0]]
    if len(clients) != 1:
        return None
    client_node, _client_idx = clients[0]
    if not isinstance(client_node.op, Elemwise):
        return None

    result = _unwrap_axis_swapped_subtensor1(fgraph, node.outputs[0])
    if result is None:
        return None
    source, idx_var, axis = result

    # Build AdvancedSubtensor: x[:, ..., idx, :, ...]
    idx_list = [slice(None)] * (axis + 1)
    idx_list[axis] = 0  # pointer to the single index variable
    new_out = AdvancedSubtensor(idx_list=idx_list)(source, idx_var)
    return [new_out]


@node_rewriter([Reshape])
def undo_take_reshape_for_fusion(fgraph, node):
    """Undo ``Reshape(AdvancedSubtensor1(x, flatten(idx)), shape)`` for ND indices.

    ``transform_take`` rewrites ``x[mat_idx]`` (ND integer index) into
    ``AdvancedSubtensor1(x, mat_idx.ravel()).reshape(mat_idx.shape + ...)``,
    possibly with DimShuffle axis-swaps for non-zero axes.  This rewrite
    undoes that so ``FuseIndexedElemwise`` can absorb the ND index directly.
    """
    [reshape_out] = node.outputs

    # Must feed a single Elemwise (or chain to one via another pre-fusion rewrite)
    clients = fgraph.clients[reshape_out]
    if len(clients) != 1:
        return None
    client_node, _ = clients[0]
    if not isinstance(client_node.op, Elemwise):
        return None

    result = _unwrap_axis_swapped_subtensor1(fgraph, node.inputs[0])
    if result is None:
        return None
    source, flat_idx, axis = result

    # The index input to AdvancedSubtensor1 must be Reshape{1}(mat_idx, [-1]) (flatten)
    if flat_idx.owner is None or not isinstance(flat_idx.owner.op, Reshape):
        return None
    if flat_idx.owner.op.ndim != 1:
        return None
    mat_idx = flat_idx.owner.inputs[0]
    if mat_idx.ndim < 2:
        return None

    # Build AdvancedSubtensor: source[:, ..., mat_idx, :, ...]
    src_ndim = source.type.ndim
    idx_list = [slice(None)] * src_ndim
    idx_list[axis] = 0  # pointer to the single index variable
    new_out = AdvancedSubtensor(idx_list=idx_list)(source, mat_idx)
    return [new_out]


indexed_elemwise_optdb = SequenceDB()
optdb.register(
    "fuse_indexed_into_elemwise",
    indexed_elemwise_optdb,
    "numba",
    # After inplace_elemwise (position=50.5) so we see final inplace patterns,
    # same position as other numba-specific rewrites (BlockwiseWithCoreShape).
    position=100,
)

indexed_elemwise_optdb.register(
    "undo_take_dimshuffle_for_fusion",
    dfs_rewriter(undo_take_dimshuffle_for_fusion),
    "numba",
    position=0,
)

indexed_elemwise_optdb.register(
    "undo_take_reshape_for_fusion",
    dfs_rewriter(undo_take_reshape_for_fusion),
    "numba",
    position=0.5,
)


class IndexedElemwise(OpFromGraph):
    """Fuse indexed reads and updates into a single Elemwise iteration loop.

    Absorbs ``AdvancedSubtensor1`` (indexed reads on inputs) and
    ``AdvancedIncSubtensor1`` (indexed updates on outputs) into one loop,
    avoiding materialisation of intermediate arrays.

    Inner fgraph contains the unfused subgraph.
    Non-Numba backends run it as-is via ``OpFromGraph.perform``.
    The Numba backend generates a single loop with indirect indexing.

    Outer inputs are ordered as::

        [elemwise_inputs..., idx_0, idx_1, ..., update_target_0, ...]

    Parameters
    ----------
    indexed_inputs : tuple of ((tuple[int, ...], int) | None)
        One entry per index array k (at outer input position n_elemwise + k).
        ``None`` if index k has no read role (write-only).
        Otherwise ``(sources, source_axis)``:

        - ``sources``: which elemwise input positions read through this
          index.  Inputs that share the same ``(idx, axis)`` are grouped
          so the codegen loads the index once and reuses the indirect
          lookup for all of them.
          E.g. ``x[idx] + y[idx]`` sharing the same ``idx`` → ``(0, 1)``.
        - ``source_axis``: which axis of the source array is indexed
          (not the index array's own axes).
          E.g. ``x[:, idx]`` on a 3-D array → ``source_axis=1``.

        The grouping key is ``(idx_var, source_axis)``, so the same index variable
        on different axes produces separate entries.

        Examples::

            z = x[idx] + y[idx]          → [((0,1), 0)]
            z = x[idx_a] + y[idx_b]      → [((0,), 0), ((1,), 0)]
            write-only inc(tgt, v, idx)  → [None]

    indexed_outputs : tuple of ((tuple[int, ...], int, str) | None)
        One entry per index array k, parallel to ``indexed_inputs``.
        ``None`` if index k has no write role.
        Otherwise ``(sources, source_axis, mode)``:

        - ``sources``: which Elemwise output positions are written
          through this index into the update target buffer.
        - ``source_axis``: which target-array axis is indexed.
        - ``mode``: ``"inc"`` (accumulate) or ``"set"`` (overwrite).

        Examples::

            tgt[idx] += exp(x)   → indexed_outputs=[((0,), 0, "inc")]
    """

    def __init__(self, *args, indexed_inputs=(), indexed_outputs=(), **kwargs):
        self.indexed_inputs = indexed_inputs
        self.indexed_outputs = indexed_outputs
        super().__init__(*args, accept_inplace=True, **kwargs)

    def __str__(self):
        for node in self.fgraph.apply_nodes:
            if isinstance(node.op, Elemwise):
                return f"IndexedElemwise{{{node.op!s}}}"
        return "IndexedElemwise"


@op_debug_information.register(IndexedElemwise)
def _op_debug_information_IndexedElemwise(op, node):
    info = {}

    n_idx = len(op.indexed_inputs)
    n_update_targets = sum(1 for e in op.indexed_outputs if e is not None)
    n_elemwise = len(node.inputs) - n_idx - n_update_targets

    # Annotate indexed-read inputs
    for k, entry in enumerate(op.indexed_inputs):
        if entry is None:
            continue
        sources, _source_axis = entry
        idx_label = f"idx_{k}"
        for src in sources:
            if src < len(node.inputs):
                info[node.inputs[src]] = f"indexed read ({idx_label})"

    # Annotate index arrays (after elemwise inputs)
    for k in range(n_idx):
        idx_pos = n_elemwise + k
        if idx_pos < len(node.inputs):
            info[node.inputs[idx_pos]] = f"idx_{k}"

    # Annotate update targets and outputs
    buf_counter = 0
    target_start = n_elemwise + n_idx
    target_offset = 0
    for k, entry in enumerate(op.indexed_outputs):
        if entry is None:
            continue
        sources, _source_axis, mode = entry
        buf_label = f"buf_{buf_counter}"
        buf_counter += 1
        idx_label = f"idx_{k}"

        target_pos = target_start + target_offset
        target_offset += 1
        if target_pos < len(node.inputs):
            info[node.inputs[target_pos]] = buf_label

        for out_idx in sources:
            if out_idx < len(node.outputs):
                info[node.outputs[out_idx]] = (
                    f"indexed {mode} ({buf_label}, {idx_label})"
                )

    return {node: info}


class FuseIndexedElemwise(GraphRewriter):
    """Fuse indexed reads and indexed updates into Elemwise loops.

    Absorbs single-client ``AdvancedSubtensor1`` on inputs (indexed reads)
    and single-client ``AdvancedIncSubtensor1`` on outputs (indexed updates)
    into the Elemwise iteration, avoiding intermediate arrays.

    Supports multiple index arrays: e.g. ``x[idx_a] + y[idx_b]`` produces
    two index groups.  Index arrays are shared between reads and updates
    when they refer to the same variable.
    """

    @staticmethod
    def _extract_idx_axis_pairs(node):
        """Extract ``(idx_var, axis)`` pairs from an Advanced(Inc)Subtensor node.

        Returns a list of pairs, or ``None`` if the node uses non-consecutive
        advanced indexing, boolean indices, or mixed slice/integer patterns
        that we can't fuse.
        """
        op = node.op
        if isinstance(op, AdvancedSubtensor1):
            return [(node.inputs[1], 0)]
        if isinstance(op, AdvancedIncSubtensor1):
            return [(node.inputs[2], 0)]
        if isinstance(op, AdvancedSubtensor | AdvancedIncSubtensor):
            if op.non_consecutive_adv_indexing(node):
                return None
            n_skip = 2 if isinstance(op, AdvancedIncSubtensor) else 1
            idx_vars = node.inputs[n_skip:]
            indices = indices_from_subtensor(idx_vars, op.idx_list)
            pairs = [
                (idx, j)
                for j, idx in enumerate(indices)
                if isinstance(idx, TensorVariable) and idx.type.dtype != "bool"
            ]
            if len(pairs) != sum(idx != slice(None) for idx in indices):
                return None
            return pairs
        return None

    @staticmethod
    def _duplicate_multi_client_outputs(node, multi_client_outs):
        """Add duplicate outputs for Elemwise results that have both write and non-write consumers.

        Returns ``(new_node, dup_map)`` where *dup_map* maps each original
        output index to its duplicate position.
        """
        scalar_op = node.op.scalar_op
        if isinstance(scalar_op, Composite):
            s_inputs = list(scalar_op.inputs)
            s_outputs = list(scalar_op.outputs)
        else:
            scalar_node = scalar_op.make_node(
                *[inp.type.to_scalar_type()() for inp in node.inputs]
            )
            s_inputs = list(scalar_node.inputs)
            s_outputs = list(scalar_node.outputs)

        dup_map = {}
        for out_idx in sorted(multi_client_outs):
            dup_map[out_idx] = len(s_outputs)
            s_outputs.append(s_outputs[out_idx])

        new_scalar_op = Composite(s_inputs, s_outputs)
        new_node = Elemwise(new_scalar_op).make_node(*node.inputs)
        return new_node, dup_map

    @staticmethod
    def transpose_non_indexed_write_axes(node, write_targets):
        """Move excess leading non-indexed dims to the right of the write target.

        Only the leftmost non-indexed dims that aren't covered by the Elemwise
        loop are moved. Loop dims and indexed axes keep their relative order so
        the val's dim layout stays aligned with the write slice.

        Returns a list of ``(old_out, new_out)`` replacement pairs, or an
        empty list if no write target needed transposing.
        """
        replacements = []
        elemwise_batch_ndim = len(node.outputs[0].type.broadcastable)
        for update_node in write_targets.values():
            op = update_node.op
            target, val, *idx_vars = update_node.inputs

            idx_axes = [i for i, e in enumerate(op.idx_list) if e != slice(None)]
            n_indexed_axes = len(idx_axes)
            n_idx_dims = max(v.ndim for v in idx_vars)
            source_batch = elemwise_batch_ndim + n_indexed_axes - n_idx_dims
            if max(idx_axes) < source_batch:
                # Indexed axes already within batch dims, no transpose needed
                continue

            # Move excess leading non-indexed axes to the right
            non_idx_axes = [a for a in range(target.type.ndim) if a not in idx_axes]
            excess = target.type.ndim - source_batch
            excess_axes = non_idx_axes[:excess]
            non_excess_axes = [
                a for a in range(target.type.ndim) if a not in excess_axes
            ]
            perm = non_excess_axes + excess_axes
            target_t = target.dimshuffle(perm)

            # Pad val so it broadcasts with excess dims moved to the right.
            # Non-indexed axes can't be between indexed axes (_extract_idx_axis_pairs rejects non-consecutive indexing).
            val = shape_padright(val, excess)
            new_idx_list = [op.idx_list[perm[i]] for i in range(len(perm))]

            # Create new write node
            props = op._props_dict()
            props["idx_list"] = tuple(new_idx_list)
            new_inc = type(op)(**props)(target_t, val, *idx_vars)

            # Permute updated node so it behaves like original one
            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            new_update = new_inc.dimshuffle(inv_perm)

            replacements.append((update_node.outputs[0], new_update))
        return replacements

    def apply(self, fgraph):
        worklist = list(reversed(fgraph.toposort()))
        while worklist:
            node = worklist.pop()
            if not isinstance(node.op, Elemwise):
                continue
            if node not in fgraph.apply_nodes:
                continue

            idx_groups = {}  # (idx_var, axis) -> (reads: list[int], writes: list[int])

            # Find indexed reads to fuse: single client AdvancedSubtensor(1)
            for i, inp in enumerate(node.inputs):
                inp_node = inp.owner
                if inp_node is None or any(
                    c is not node for c, _ in fgraph.clients[inp]
                ):
                    continue
                idx_axis_pairs = self._extract_idx_axis_pairs(inp_node)
                if idx_axis_pairs is None:
                    continue
                for idx_axis_pair in idx_axis_pairs:
                    if idx_axis_pair not in idx_groups:
                        idx_groups[idx_axis_pair] = ([], [])
                    idx_groups[idx_axis_pair][0].append(i)

            # For indexed writes to fuse: single client AdvancedIncSubtensor(1)
            # The write may be separated from the output by a right expand_dims DimShuffle
            # All indexed write axes have to overlap and not broadcast the core Elemwise loop
            # Our current vectorize codegen can't produce write only loops that don't force
            # the recomputation of the core function in every step.
            write_targets = {}  # out_idx -> update_node
            must_transpose_write_axes = False
            for out_idx, out in enumerate(node.outputs):
                clients = fgraph.clients[out]
                # Allow right expand_dims between elemwise and write
                # buffer[idx].set(f(x)[..., None, None])
                # These will be absorbed by broadcasting the result at ecah iteration of the loop
                # into the sliced non-scalar indexed buffer, making use of vectorize_codegen with fake "core output dims"
                # We actually move left broadcasted dims to the right below
                right_pad = 0
                if (
                    len(clients) == 1
                    and isinstance((ds_op := clients[0][0].op), DimShuffle)
                    and ds_op.is_right_expand_dims
                    and len(fgraph.clients[(ds_out := clients[0][0].outputs[0])]) == 1
                ):
                    right_pad = len(ds_op.augment)
                    clients = fgraph.clients[ds_out]
                inc_clients = [
                    (c, ci)
                    for c, ci in clients
                    if ci == 1
                    and isinstance(c.op, AdvancedIncSubtensor1 | AdvancedIncSubtensor)
                ]
                if len(inc_clients) != 1:
                    # TODO: support multiple writes from the same Elemwise output via Composite duplication
                    continue
                [(client_node, _)] = inc_clients
                idx_axis_pairs = self._extract_idx_axis_pairs(client_node)
                if idx_axis_pairs is None:
                    continue

                target, _, *idx_vars = client_node.inputs
                write_bcast = AdvancedSubtensor(idx_list=client_node.op.idx_list)(
                    target, *idx_vars
                ).type.broadcastable
                indexed_write_bcast = (
                    write_bcast[: len(write_bcast) - right_pad]
                    if right_pad
                    else write_bcast
                )
                left_pad = min(a for _, a in idx_axis_pairs)
                if out.type.ndim + left_pad < len(indexed_write_bcast):
                    # out does not cover all indexed write dims
                    continue

                if any(
                    ob and not iwb
                    for ob, iwb in zip(
                        reversed(out.type.broadcastable), reversed(indexed_write_bcast)
                    )
                ):
                    # TODO: support broadcast on non-indexed dims by squeezing them out of the Elemwise first
                    continue

                if len(indexed_write_bcast) > out.type.ndim:
                    must_transpose_write_axes = True

                for idx_axis_pair in idx_axis_pairs:
                    if idx_axis_pair not in idx_groups:
                        idx_groups[idx_axis_pair] = ([], [])
                    idx_groups[idx_axis_pair][1].append(out_idx)
                write_targets[out_idx] = client_node

            if not idx_groups:
                continue

            if must_transpose_write_axes:
                replacements = self.transpose_non_indexed_write_axes(
                    node, write_targets
                )
                assert replacements
                fgraph.replace_all(
                    replacements,
                    reason="fuse_indexed_elemwise_move_write_axes",
                )
                worklist.append(node)
                continue

            indexed_reads = {i for reads, _ in idx_groups.values() for i in reads}

            # If any inplace targets an indexed-read input,
            # strip and re-run inplace with those inputs protected
            if any(
                inp_idx in indexed_reads for inp_idx in node.op.inplace_pattern.values()
            ):
                stripped_node = Elemwise(node.op.scalar_op).make_node(*node.inputs)
                fgraph.replace_all(
                    zip(node.outputs, stripped_node.outputs),
                    reason="fuse_indexed_elemwise_strip_inplace",
                )
                protected = frozenset(stripped_node.inputs[i] for i in indexed_reads)
                # try_inplace_on_node does its own fgraph.replace_all internally,
                # so the returned node is already in the fgraph
                new_inplace_node = InplaceElemwiseOptimizer().try_inplace_on_node(
                    fgraph,
                    stripped_node,
                    reason="fuse_indexed_elemwise_inplace_read_buffers",
                    extra_protected_inputs=protected,
                )
                worklist.append(new_inplace_node)
                continue

            # If any indexed-write output also has other consumers,
            # duplicate it via Composite so the write replaces the duplicate
            # while the original stays available for non-write consumers.
            # We still avoid one extra write loop,
            # even if we can't skip the output materialization altogether
            def _has_non_write_clients(out_idx):
                update = write_targets[out_idx]
                for c, _ in fgraph.clients[node.outputs[out_idx]]:
                    if c is update:
                        continue
                    # Look through shape_padright from write normalization
                    if (
                        isinstance(c.op, DimShuffle)
                        and c.op.is_right_expand_dims
                        and len(ds_clients := fgraph.clients[c.outputs[0]]) == 1
                        and ds_clients[0][0] is update
                    ):
                        continue
                    return True
                return False

            if write_and_direct_use_outs := {
                out_idx for out_idx in write_targets if _has_non_write_clients(out_idx)
            }:
                new_node, dup_map = self._duplicate_multi_client_outputs(
                    node, write_and_direct_use_outs
                )
                replacements = list(
                    zip(node.outputs, new_node.outputs[: len(node.outputs)])
                )
                for out_idx, dup_idx in dup_map.items():
                    update_node = write_targets[out_idx]
                    new_update_out = update_node.op(
                        update_node.inputs[0],
                        new_node.outputs[dup_idx],
                        *update_node.inputs[2:],
                    )
                    replacements.append((update_node.outputs[0], new_update_out))
                fgraph.replace_all(
                    replacements,
                    reason="fuse_indexed_elemwise_write_and_direct_outputs",
                )
                worklist.append(new_node)
                continue

            idx_vars = [idx for idx, _axis in idx_groups]

            fgraph_destroy_map = {
                out_idx: [inp_idx]
                for out_idx, inp_idx in node.op.inplace_pattern.items()
                if out_idx not in write_targets
            }

            # Fgraph inputs: substitute indexed sources back to their
            # pre-subtensor arrays, append index arrays and update targets.
            fgraph_inputs = [
                inp.owner.inputs[0] if i in indexed_reads else inp
                for i, inp in enumerate(node.inputs)
            ] + idx_vars

            # Non-inplace write targets need a copy so the original isn't destroyed
            # Elemwise will always destroy the write buffers inplace afterwards.
            copy_positions = set()

            # Inner fgraph outputs: Elemwise outputs, with write targets
            # replaced by their AdvancedIncSubtensor result
            fgraph_outputs = list(node.outputs)
            for out_idx, update_node in sorted(write_targets.items()):
                target = update_node.inputs[0]

                fgraph_inputs.append(target)
                target_pos = len(fgraph_inputs) - 1

                if not update_node.op.inplace:
                    copy_positions.add(target_pos)

                # Build the indexed write for the inner fgraph
                if update_node.op.inplace:
                    write_out = update_node.outputs[0]
                else:
                    props = update_node.op._props_dict()
                    props["inplace"] = True
                    inplace_op = type(update_node.op)(**props)
                    write_out = inplace_op(
                        target, node.outputs[out_idx], *update_node.inputs[2:]
                    )

                fgraph_outputs[out_idx] = write_out
                fgraph_destroy_map[out_idx] = [target_pos]

            # indexed_inputs_spec: ((read_positions, axis) | None, ...)
            # indexed_outputs_spec: ((write_positions, axis, "inc"|"set") | None, ...)
            indexed_inputs_spec = tuple(
                (tuple(reads), axis) if reads else None
                for (_, axis), (reads, _) in idx_groups.items()
            )
            indexed_outputs_spec = tuple(
                (
                    tuple(writes),
                    key[1],
                    "set" if write_targets[writes[0]].op.set_instead_of_inc else "inc",
                )
                if writes
                else None
                for key, (_, writes) in idx_groups.items()
            )

            outer_inputs = [
                inp.copy() if i in copy_positions else inp
                for i, inp in enumerate(fgraph_inputs)
            ]

            new_outs = IndexedElemwise(
                fgraph_inputs,
                fgraph_outputs,
                destroy_map=fgraph_destroy_map,
                indexed_inputs=indexed_inputs_spec,
                indexed_outputs=indexed_outputs_spec,
            )(*outer_inputs, return_list=True)

            replacements = []
            for out_idx in range(len(node.outputs)):
                if out_idx in write_targets:
                    replacements.append(
                        (write_targets[out_idx].outputs[0], new_outs[out_idx])
                    )
                else:
                    replacements.append((node.outputs[out_idx], new_outs[out_idx]))

            fgraph.replace_all_validate(
                replacements,
                reason="fuse_indexed_into_elemwise",
            )


indexed_elemwise_optdb.register(
    "fuse_indexed_elemwise",
    FuseIndexedElemwise(),
    "numba",
    position=1,
)

import abc
import itertools
import operator
import sys
from collections import defaultdict, deque
from collections.abc import Generator, Sequence
from functools import cache, reduce
from operator import or_
from warnings import warn

import pytensor.scalar.basic as ps
from pytensor import clone_replace, compile
from pytensor.compile.function.types import Supervisor
from pytensor.compile.mode import get_target_language
from pytensor.configdefaults import config
from pytensor.graph import FunctionGraph, Op
from pytensor.graph.basic import Apply, Variable, io_toposort
from pytensor.graph.destroyhandler import DestroyHandler, inplace_candidates
from pytensor.graph.features import ReplaceValidate
from pytensor.graph.fg import Output
from pytensor.graph.rewriting.basic import (
    GraphRewriter,
    copy_stack_trace,
    in2out,
    node_rewriter,
    out2in,
)
from pytensor.graph.rewriting.db import SequenceDB
from pytensor.graph.utils import InconsistencyError, MethodNotDefined
from pytensor.misc.ordered_set import OrderedSet
from pytensor.scalar.math import Grad2F1Loop, _grad_2f1_loop
from pytensor.tensor.basic import (
    MakeVector,
    constant,
)
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import add, exp, mul
from pytensor.tensor.rewriting.basic import (
    alloc_like,
    broadcasted_by,
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.variable import TensorConstant, TensorVariable


class InplaceGraphOptimizer(GraphRewriter):
    op: type[Op]

    def add_requirements(self, fgraph):
        fgraph.attach_feature(DestroyHandler())

    @abc.abstractmethod
    def filter_candidate_pairs(
        self, fgraph: FunctionGraph, node: Apply, protected_inputs: Sequence[Variable]
    ) -> Sequence[tuple[tuple[int, Variable], tuple[int, Variable]]]:
        pass

    @abc.abstractmethod
    def create_inplace_node(
        self, node: Apply, inplace_pattern: dict[int, Sequence[int]]
    ) -> Apply:
        pass

    def apply(self, fgraph):
        r"""

        Attempts to replace all `Op`\s by versions of them that operate
        inplace. It operates greedily: for each `Op` that is encountered,
        it tries to inplace all the valid inputs at once (if the Op supports it),
        if that fails, it tries to inplace one input at a time.

        Examples
        --------

            x + y + z -> x += y += z
            (x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)

        """
        # We should not validate too often as this takes too much time to execute!
        # It is the _dfs_toposort() fct in pytensor/graph/destroyhandler.py
        # that takes so much time.
        # Should we try to use another lib that does toposort?
        #   igraph: http://igraph.sourceforge.net/
        #   networkx: https://networkx.lanl.gov/
        # Should we try to use cython?
        #   Compiling only that fct is not enough, should we try to add the
        #   deque class too?
        #   And init the deque and other list to an upper bound number of
        #   elements?
        # Maybe PyTensor should do online toposort as in
        #   http://code.google.com/p/acyclic
        #
        # The next longest rewriter is the canonizer phase.
        # Then I think it is the [io_?]toposort (need to validate) so check if
        # the solution is also applicable there.

        # 2025: The above comment is not specific to Elemwise, if we have concerns about this approach, we should
        # tackle them in a more general way. The whole try/except approach is probably suboptimal.
        # We can consider restricting inputs with static shapes that are large enough.

        if config.tensor__insert_inplace_optimizer_validate_nb != -1:
            warn(
                "tensor__insert_inplace_optimizer_validate_nb config is deprecated. Setting it will fail in a future release.",
                FutureWarning,
            )

        reason = f"{self.op}_inplace_optimizer"
        prof = {
            "opt": self,
            "node_before": len(fgraph.apply_nodes),
            "nb_eager_inconsistent": 0,
            "nb_inconsistent": 0,
            "nb_replaced": 0,
        }
        large_graph = len(fgraph.apply_nodes) > 500

        protected_inputs = set(
            itertools.chain.from_iterable(
                f.protected for f in fgraph._features if isinstance(f, Supervisor)
            )
        )
        protected_inputs.update(fgraph.outputs)
        root_destroyer = fgraph.destroy_handler.root_destroyer

        self_op = self.op
        update_mapping = fgraph.update_mapping or {}
        op_updates: dict[TensorVariable, TensorVariable] = {
            out: fgraph.inputs[update_mapping[out_idx]]
            for out_idx, out in enumerate(fgraph.outputs)
            if (
                out_idx in update_mapping
                and out.owner
                and isinstance(out.owner.op, self_op)
            )
        }
        set_op_updates = set(op_updates.keys())

        for node in fgraph.toposort():
            if not isinstance(node.op, self_op) or node.op.destroy_map:
                continue

            # If big graph and the outputs are scalar, do not make it inplace.
            if large_graph and all(node.outputs[0].type.broadcastable):
                continue

            candidate_pairs = self.filter_candidate_pairs(
                fgraph, node, protected_inputs
            )

            if not candidate_pairs:
                continue

            sorted_candidate_pairs = candidate_pairs
            if op_updates and (node_updates := set(node.outputs) & set_op_updates):
                # If the fgraph has updates, we try to prioritize in-placing on the pairs that correspond to the update
                direct_update_pairs = []
                indirect_update_pairs = []
                other_update_pairs = []
                for pair in candidate_pairs:
                    ((o, out), (i, inp)) = pair
                    if out in node_updates:
                        direct_update_inp = op_updates[out]
                        if direct_update_inp is inp:
                            # This pair is the whole graph update
                            direct_update_pairs.append(pair)
                            continue
                        elif (inp_node := inp.owner) is not None and any(
                            root_destroyer.get(up_inp, None) is inp_node
                            for up_inp in op_updates.values()
                        ):
                            # This pair connects to an updated input
                            indirect_update_pairs.append(pair)
                            continue
                    other_update_pairs.append(pair)

                sorted_candidate_pairs = (
                    direct_update_pairs + indirect_update_pairs + other_update_pairs
                )

            # Try in-placing all outputs at once
            tried_inputs = set()
            inplace_pattern = {}
            for (o, _), (i, _) in sorted_candidate_pairs:
                if o not in inplace_pattern and i not in tried_inputs:
                    inplace_pattern[o] = [i]
                    tried_inputs.add(i)

            inplace_node = self.create_inplace_node(node, inplace_pattern)
            if inplace_node.op.destroy_map == inplace_pattern:
                replacements = tuple(zip(node.outputs, inplace_node.outputs))
                try:
                    fgraph.replace_all_validate(replacements, reason=reason)
                except InconsistencyError:
                    prof["nb_eager_inconsistent"] += 1
                else:
                    prof["nb_replaced"] += 1
                    copy_stack_trace(node.outputs, inplace_node.outputs)
                    continue

            # If it fails or doesn't match the desired inplace pattern, try one output/input at a time
            tried_inputs = set()
            inplace_pattern = {}
            replaced = False
            original_node = node
            for (o, _), (i, _) in sorted_candidate_pairs:
                if o not in inplace_pattern and i not in tried_inputs:
                    inplace_pattern[o] = [i]
                    tried_inputs.add(i)

                    inplace_node = self.create_inplace_node(node, inplace_pattern)
                    if inplace_node.op.destroy_map != inplace_pattern:
                        # This Op can't respect this partial inplace pattern,
                        # We assume it can't support any other cases
                        break
                    else:
                        replacements = tuple(zip(node.outputs, inplace_node.outputs))
                        try:
                            fgraph.replace_all_validate(replacements, reason=reason)
                            node = inplace_node
                            replaced = True
                        except InconsistencyError:
                            prof["nb_inconsistent"] += 1
                            # The input, not the output caused inconsistencies
                            inplace_pattern.pop(o)
            if replaced:
                copy_stack_trace(original_node.outputs, node.outputs)
                prof["nb_replaced"] += replaced

        return prof

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        blanc = "    " * level
        print(blanc, cls.__name__, file=stream)
        for k in [
            "node_before",
            "nb_eager_inconsistent",
            "nb_inconsistent",
            "nb_replaced",
        ]:
            print(blanc, k, prof[k], file=stream)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(
            f"{' ' * level}{self.__class__.__name__}",
            file=stream,
        )


class InplaceElemwiseOptimizer(InplaceGraphOptimizer):
    op = Elemwise

    def filter_candidate_pairs(self, fgraph, node, protected_inputs):
        candidate_inputs = [
            (node.inputs.index(inp), inp)
            for inp in inplace_candidates(
                fgraph,
                node.inputs,
                protected_inputs=protected_inputs,
            )
        ]
        if not candidate_inputs:
            return []

        return [
            ((o, out), (i, inp))
            for o, out in enumerate(node.outputs)
            for i, inp in candidate_inputs
            if inp.type == out.type
        ]

    def create_inplace_node(self, node, inplace_pattern):
        op = node.op
        scalar_op = op.scalar_op
        inplace_pattern = {i: o for i, [o] in inplace_pattern.items()}
        if hasattr(scalar_op, "make_new_inplace"):
            new_scalar_op = scalar_op.make_new_inplace(
                ps.transfer_type(
                    *[
                        inplace_pattern.get(i, o.dtype)
                        for i, o in enumerate(node.outputs)
                    ]
                )
            )
        else:
            new_scalar_op = type(scalar_op)(
                ps.transfer_type(
                    *[inplace_pattern.get(i, None) for i in range(len(node.outputs))]
                )
            )
        return type(op)(new_scalar_op, inplace_pattern).make_node(*node.inputs)


compile.optdb.register(
    "inplace_elemwise",
    InplaceElemwiseOptimizer(),
    "inplace_elemwise_opt",  # for historic reason
    "inplace_elemwise_optimizer",
    "fast_run",
    "inplace",
    position=50.5,
)


def apply_local_dimshuffle_lift(fgraph, var):
    """
    lift recursively
    """
    if var.owner is None:
        return var
    new = local_dimshuffle_lift.transform(fgraph, var.owner)
    if new:
        return new[0]
    return var


def is_dimshuffle_useless(new_order, input):
    """
    Checks for two types of useless dimshuffles:
      1 - dimshuffle all dimensions in order.
      2 - dimshuffle a broadcastable dimension.
    """
    is_useless = True
    if len(new_order) == input.type.ndim:
        all_broadcastable_dims = [
            i
            for (i, is_broadcastable) in enumerate(input.type.broadcastable)
            if is_broadcastable
        ] + ["x"]
        for i in range(input.type.ndim):
            if new_order[i] == i or (
                i in all_broadcastable_dims and new_order[i] in all_broadcastable_dims
            ):
                is_useless = True
            else:
                is_useless = False
                break
    else:
        is_useless = False
    return is_useless


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([DimShuffle])
def local_dimshuffle_lift(fgraph, node):
    """
    "Lifts" DimShuffle through Elemwise operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Elemwise(x, y)) => Elemwise(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)
    DimShuffle{0,1,...}(x) => x (when the dimshuffle do nothing)

    After this transform, clusters of Elemwise operations are
    void of DimShuffle operations.

    """
    op = node.op

    inp = node.inputs[0]
    inode = inp.owner
    new_order = op.new_order
    if (
        inode
        and isinstance(inode.op, Elemwise)
        and len(inode.outputs) == 1
        and (len(fgraph.clients[inp]) == 1)
    ):
        # Don't use make_node to have tag.test_value set.
        new_inputs = []
        for inp in inode.inputs:
            new_inp = inp.dimshuffle(op.new_order)
            new_inputs.append(apply_local_dimshuffle_lift(fgraph, new_inp))
        copy_stack_trace(node.outputs[0], new_inputs)
        ret = inode.op(*new_inputs, return_list=True)
        return ret
    if inode and isinstance(inode.op, DimShuffle):
        new_order = [x == "x" and "x" or inode.op.new_order[x] for x in new_order]
        inp = inode.inputs[0]

    if is_dimshuffle_useless(new_order, inp):
        return [inp]
    elif inode and isinstance(inode.op, DimShuffle):
        ret = inp.dimshuffle(new_order)
        ret = apply_local_dimshuffle_lift(fgraph, ret)
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


@register_canonicalize
@register_specialize
@node_rewriter([DimShuffle])
def local_useless_dimshuffle_makevector(fgraph, node):
    r"""Remove `DimShuffle`\s that drop one dimensional broadcastable `MakeVector`s.

    This rewrite is needed in order to clean up after
    `local_subtensor_remove_broadcastable_index`, which produces a
    not-so-intuitive canonical form for `x[0]` when `x.shape == (1,)`
    (i.e. one broadcastable dimension): i.e. `x.dimshuffle(())`.
    """

    # The `DimShuffle` should be removing the single broadcastable dimension
    if node.op.new_order != ():
        return

    makevector_out = node.inputs[0]

    if not (
        makevector_out.owner
        and isinstance(makevector_out.owner.op, MakeVector)
        and makevector_out.broadcastable == (True,)
    ):
        return

    assert len(makevector_out.owner.inputs) == 1

    return [makevector_out.owner.inputs[0]]


@register_canonicalize
@node_rewriter([Elemwise])
def local_upcast_elemwise_constant_inputs(fgraph, node):
    """This explicitly upcasts constant inputs to elemwise Ops, when
    those Ops do implicit upcasting anyway.

    Rationale: it helps merge things like (1-x) and (1.0 - x).

    """
    if len(node.outputs) > 1:
        return None

    if getattr(node.op.scalar_op, "output_types_preference", None) not in (
        ps.upgrade_to_float,
        ps.upcast_out,
    ):
        return None

    # this is the kind of op that we can screw with the input
    # dtypes by upcasting explicitly
    [old_out] = node.outputs
    output_dtype = old_out.type.dtype
    new_inputs = list(node.inputs)
    changed = False
    for i, inp in enumerate(node.inputs):
        if inp.type.dtype != output_dtype and isinstance(inp, TensorConstant):
            new_inputs[i] = constant(inp.data.astype(output_dtype))
            changed = True

    if not changed:
        return None

    rval = node.op(*new_inputs)
    if not old_out.type.is_super(rval.type):
        # This can happen for example when floatX=float32
        # and we do the true division between and int64
        # and a constant that will get typed as int8.
        # As this is just to allow merging more case, if
        # the upcast don't work, we can just skip it.
        return None

    # Copy over output stacktrace from before upcasting
    copy_stack_trace(old_out, rval)
    return [rval]


@node_rewriter([add, mul])
def flatten_nested_add_mul(fgraph, node):
    """Fuse consecutive add or mul in one such node with more inputs.

    It is better to fuse add/mul that way then in a Composite node as
    this make the inner graph of the Composite smaller. This allows to
    put more computation in a Composite before hitting the max
    recursion limit when pickling Composite.

    This rewrite is almost useless after the AlgebraicCanonizer is used,
    but it catches a few edge cases that are not canonicalized by it
    """
    s_op = node.op.scalar_op
    new_inp = []
    fused = False
    for inp in node.inputs:
        if (
            inp.owner
            and isinstance(inp.owner.op, Elemwise)
            and inp.owner.op.scalar_op == s_op
            # Do not duplicate the operation.
            and len(fgraph.clients[inp]) == 1
        ):
            new_inp.extend(inp.owner.inputs)
            fused = True
        else:
            new_inp.append(inp)

    # We can not compare the number of inputs as Mul and Add could have
    # 0 or 1 inputs in some corner cases.
    if fused:
        output = node.op(*new_inp)
        copy_stack_trace(node.outputs[0], output)

        # Do the recursion here to help lower the number of
        # FusionOptimizer iteration.
        if output.owner:
            output2 = flatten_nested_add_mul.transform(fgraph, output.owner)
            if output2:
                return output2
        return [output]


def elemwise_max_operands_fct(node) -> int:
    # `Elemwise.perform` uses NumPy ufuncs and they are limited to 32 operands (inputs and outputs)
    if not config.cxx:
        return 32
    return 1024


class FusionOptimizer(GraphRewriter):
    """Graph optimizer that fuses consecutive Elemwise operations."""

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())

    @staticmethod
    def elemwise_to_scalar(inputs, outputs):
        replacement = {
            inp: ps.get_scalar_type(inp.type.dtype).make_variable() for inp in inputs
        }
        for node in io_toposort(inputs, outputs):
            scalar_inputs = [replacement[inp] for inp in node.inputs]
            replacement.update(
                dict(
                    zip(
                        node.outputs,
                        node.op.scalar_op.make_node(*scalar_inputs).outputs,
                    )
                )
            )

        return (
            [replacement[inp] for inp in inputs],
            [replacement[out] for out in outputs],
        )

    def apply(self, fgraph):
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callbacks_before = fgraph.execute_callbacks_times.copy()
            callback_before = fgraph.execute_callbacks_time

        max_operands = elemwise_max_operands_fct(None)

        def find_fuseable_subgraphs(
            fg: FunctionGraph,
        ) -> Generator[tuple[set[Variable], list[Variable]], None, None]:
            @cache
            def elemwise_scalar_op_has_c_code(node: Apply) -> bool:
                # TODO: This should not play a role in non-c backends!
                if node.op.scalar_op.supports_c_code(node.inputs, node.outputs):
                    return True
                else:
                    if config.optimizer_verbose:
                        warn(
                            f"Loop fusion interrupted because {node.op.scalar_op} does not provide a C implementation."
                        )
                    return False

            fuseable_inputs: defaultdict[Variable, list[Variable]] = defaultdict(list)
            fuseable_clients: defaultdict[Variable, list[Variable]] = defaultdict(list)
            for out, clients in fg.clients.items():
                if not (
                    out.owner is not None
                    and isinstance(out.owner.op, Elemwise)
                    # and not isinstance(out.owner.op.scalar_op, ps.Composite)
                    and len(out.owner.outputs) == 1
                    and elemwise_scalar_op_has_c_code(out.owner)
                ):
                    continue

                out_bcast = out.type.broadcastable
                for client, _ in clients:
                    if (
                        isinstance(client.op, Elemwise)
                        # and not isinstance(client.op.scalar_op, ps.Composite)
                        and len(client.outputs) == 1
                        and out_bcast == client.outputs[0].type.broadcastable
                        and elemwise_scalar_op_has_c_code(client)
                    ):
                        client_out = client.outputs[0]
                        fuseable_clients[out].append(client_out)
                        fuseable_inputs[client_out].append(out)

            if not fuseable_clients:
                return None

            toposort_index = {
                out: 1 << i
                for i, node in enumerate(fgraph.toposort())
                for out in node.outputs
            }

            # Create a bitset of ancestors for each node
            # Each variable is represented by the bit of its owner node in the toposort
            # With two variables {a, b, c} owned by nodes {A, B, C}, where a is an input of b, and b an input of c,
            # the ancestors bitset is {A: 0b001, B: 0b011, C: 0b111}
            ancestors_bitset = {}
            ancestors_bitset_with_self = {}
            for var, var_bit in toposort_index.items():
                ancestors_bitset[var] = var_ancestors_bitset = reduce(
                    or_,
                    (
                        ancestors_bitset_with_self.get(inp, 0)
                        for inp in var.owner.inputs
                    ),
                )
                ancestors_bitset_with_self[var] = var_ancestors_bitset | var_bit
            del ancestors_bitset_with_self

            subgraphs: list[tuple[OrderedSet[Variable], list[Variable]]] = []
            all_subgraphs_bitset = 0
            # Start exploring from sink nodes
            for starting_var, starting_bit in reversed(toposort_index.items()):
                if starting_bit & all_subgraphs_bitset:
                    # Already part of a previous subgraph
                    continue

                if fuseable_clients.get(starting_var):
                    # Not a sink node
                    continue

                if not fuseable_inputs.get(starting_var):
                    # Not fuseable with any of its inputs
                    continue

                fuseable_variables_to_visit_next = deque(fuseable_inputs[starting_var])

                subgraph_inputs = OrderedSet(starting_var.owner.inputs)
                subgraph_inputs_ancestors_bitset = reduce(
                    or_,
                    (ancestors_bitset.get(inp, 0) for inp in subgraph_inputs),
                )
                subgraph_variables = [starting_var]
                subgraph_variables_bitset = starting_bit
                all_subgraphs_bitset |= starting_bit

                # We now try to expand as much as possible towards the potentially
                # fuseable ancestors and clients to detect the largest possible
                # subgraph that can be Composed together into a single `Op`. The
                # largest issue to watch out is for "non-convexity", where
                # some nodes in the set may be connected to each other via a path
                # that cannot be made part of the set (due to unfuseable nodes in between)
                # TODO: Keep two ordered quesues, one for ancestors, one for clients
                # For ancestors we want to visit the larger toposort index firsts (have more dependencies)
                # whereas for clients we want to visit the smaller toposort index first (have less dependencies)
                # As this should result in the fastest convergence
                while fuseable_variables_to_visit_next:
                    next_var = fuseable_variables_to_visit_next.popleft()
                    next_var_bit = toposort_index[next_var]

                    if next_var_bit & all_subgraphs_bitset:
                        # Already part of a previous subgraph
                        continue

                    if next_var in subgraph_inputs:
                        # It's an input of S
                        # assert not (ancestors_bitset[next_var] & subgraph_variables_bitset)

                        # Could next_out become part of S?
                        # For that, other inputs of S must not depend on it
                        # Since a variable does not depend on itself,
                        # the presence in the inputs_ancestors must mean another input depends on it
                        if next_var_bit & subgraph_inputs_ancestors_bitset:
                            continue
                        else:
                            # If so move it from inputs of S to variables of S
                            subgraph_variables.append(next_var)
                            subgraph_variables_bitset |= next_var_bit
                            all_subgraphs_bitset |= next_var_bit
                            subgraph_inputs.remove(next_var)
                            subgraph_inputs.update(next_var.owner.inputs)
                            # I don't think there's a way to recompute this incrementally faster
                            # without having some counting representation
                            subgraph_inputs_ancestors_bitset = reduce(
                                or_,
                                (
                                    ancestors_bitset.get(inp, 0)
                                    for inp in subgraph_inputs
                                ),
                            )

                    else:
                        # It must be a fuseable client
                        # assert not (next_var_bit & subgraph_inputs_ancestors_bitset)

                        # Could it become part of S?
                        # For that none of the inputs not in S should depend on S
                        new_inputs = []
                        ancestors_of_new_required_inputs_bitset = 0
                        for inp in next_var.owner.inputs:
                            if not (
                                toposort_index.get(inp, 0) & subgraph_variables_bitset
                            ):
                                new_inputs.append(inp)
                                ancestors_of_new_required_inputs_bitset |= (
                                    ancestors_bitset.get(inp, 0)
                                )
                        if (
                            ancestors_of_new_required_inputs_bitset
                            & subgraph_variables_bitset
                        ):
                            continue
                        else:
                            # If so, add it to the subgraph variables
                            subgraph_variables.append(next_var)
                            subgraph_variables_bitset |= next_var_bit
                            all_subgraphs_bitset |= next_var_bit
                            subgraph_inputs.update(new_inputs)
                            subgraph_inputs_ancestors_bitset |= (
                                ancestors_of_new_required_inputs_bitset
                            )

                    # Explore inputs and clients of new node next
                    fuseable_variables_to_visit_next.extendleft(
                        fuseable_inputs.get(next_var, ())
                    )
                    fuseable_variables_to_visit_next.extend(
                        fuseable_clients.get(next_var, ())
                    )

                if len(subgraph_variables) > 1:
                    # We found a subgraph of at least two nodes

                    # Find out which variables must become outputs
                    subgraph_outputs = [
                        var
                        for var in subgraph_variables
                        if any(
                            isinstance(client.op, Output)
                            or client.outputs[0] not in subgraph_variables
                            for client, _ in fg.clients[var]
                        )
                    ]
                    # We use the min toposort_index for sorting the subgraphs later
                    min_toposort_index = min(
                        toposort_index[var] for var in subgraph_outputs
                    )
                    subgraphs.append(
                        (min_toposort_index, (subgraph_inputs, subgraph_outputs))
                    )

                # Update fuseable clients, inputs can no longer be fused with graph variables
                # And graph nodes can no longer be fused with anything
                for inp in subgraph_inputs:
                    if inp_fuseable_clients := fuseable_clients.get(inp):
                        fuseable_clients[inp] = [
                            i
                            for i in inp_fuseable_clients
                            if i not in subgraph_variables
                        ]

            # We need to replace in reverse topological order
            yield from (io for _, io in sorted(subgraphs, reverse=True))

        nb_fused = 0
        nb_replacement = 0
        for inputs, outputs in find_fuseable_subgraphs(fgraph):
            # TODO: This has to be done in reversed topological order
            if (len(inputs) + len(outputs)) > max_operands:
                warn(
                    "Loop fusion failed because the resulting node would exceed "
                    "the kernel argument limit."
                )
                continue

            scalar_inputs, scalar_outputs = self.elemwise_to_scalar(inputs, outputs)
            composite_outputs = Elemwise(ps.Composite(scalar_inputs, scalar_outputs))(
                *inputs, return_list=True
            )
            assert len(outputs) == len(
                composite_outputs
            )  # Should be guaranteed by elemwise_to_scalar
            for old_out, composite_out in zip(outputs, composite_outputs):
                # Preserve any names on the original outputs
                if old_out.name:
                    composite_out.name = old_out.name

            starting_nodes = len(fgraph.apply_nodes)
            fgraph.replace_all_validate(
                tuple(zip(outputs, composite_outputs)),
                reason=self.__class__.__name__,
            )
            nb_fused += 1
            nb_replacement += (starting_nodes - len(fgraph.apply_nodes)) + 1

        if fgraph.profile:
            validate_time = fgraph.profile.validate_time - validate_before
            callback_time = fgraph.execute_callbacks_time - callback_before
            callbacks_time = {}
            for k, v in fgraph.execute_callbacks_times.items():
                if k in callbacks_before:
                    callbacks_time[k] = v - callbacks_before[k]
                else:
                    callbacks_time[k] = v
        else:
            validate_time = None
            callback_time = None
            callbacks_time = {}

        return (
            self,
            nb_fused,
            nb_replacement,
            0,  # nb_inconsintency_replace
            validate_time,
            callback_time,
            callbacks_time,
            -1,  # toposort_time
        )

    @staticmethod
    def print_profile(stream, prof, level=0):
        blanc = "    " * level
        print(blanc, "FusionOptimizer", file=stream)
        print(blanc, " nb_fused", prof[1], file=stream)
        print(blanc, " nb_replacement", prof[2], file=stream)
        print(blanc, " nb_inconsistency_replace", prof[3], file=stream)
        print(blanc, " validate_time", prof[4], file=stream)
        print(blanc, " callback_time", prof[5], file=stream)
        if prof[5] is not None and prof[5] > 1:
            print(blanc, " callbacks_time", file=stream)
            for i in sorted(prof[6].items(), key=lambda a: a[1])[::-1]:
                if i[1] > 0:
                    print(blanc, "     ", i)  # noqa: T201
        print(blanc, " time_toposort", prof[7], file=stream)


@register_canonicalize
@register_specialize
@node_rewriter([Elemwise])
def local_useless_composite_outputs(fgraph, node):
    """Remove inputs and outputs of Composite Ops that are not used anywhere."""
    if not (
        isinstance(node.op, Elemwise) and isinstance(node.op.scalar_op, ps.Composite)
    ):
        return
    comp = node.op.scalar_op
    used_outputs_idxs = [
        i for i, o_extern in enumerate(node.outputs) if fgraph.clients[o_extern]
    ]
    used_inner_outputs = [comp.outputs[i] for i in used_outputs_idxs]
    comp_fgraph = FunctionGraph(
        inputs=comp.inputs, outputs=used_inner_outputs, clone=False
    )
    used_inputs_idxs = [
        i
        for i, i_intern in enumerate(comp_fgraph.inputs)
        if comp_fgraph.clients[i_intern]
    ]
    used_inner_inputs = [comp.inputs[i] for i in used_inputs_idxs]
    if len(used_inner_inputs) < len(node.inputs) or len(used_inner_outputs) < len(
        node.outputs
    ):
        used_inputs = [node.inputs[i] for i in used_inputs_idxs]
        c = ps.Composite(inputs=used_inner_inputs, outputs=used_inner_outputs)
        e = Elemwise(scalar_op=c)(*used_inputs, return_list=True)
        return dict(zip([node.outputs[i] for i in used_outputs_idxs], e, strict=True))


@node_rewriter([CAReduce])
def local_careduce_fusion(fgraph, node):
    """Fuse a `CAReduce` applied to an `Elemwise`."""

    (car_input,) = node.inputs
    car_scalar_op = node.op.scalar_op

    # FIXME: This check is needed because of the faulty logic in the FIXME below!
    # Right now, rewrite only works for `Sum`/`Prod`
    if not isinstance(car_scalar_op, ps.Add | ps.Mul):
        return None

    elm_node = car_input.owner

    if not (elm_node and isinstance(elm_node.op, Elemwise)):
        return False

    elm_scalar_op = elm_node.op.scalar_op

    elm_inputs = elm_node.inputs
    elm_outputs = elm_node.outputs

    if len(elm_inputs) > 1 or len(elm_outputs) > 1:
        # TODO: Implement the multiple inputs case
        return False

    if len(fgraph.clients[elm_outputs[0]]) > 1:
        return False

    # Don't form the fusion when the target language is Python
    if get_target_language() == ("py",):
        return False

    if not elm_scalar_op.supports_c_code(elm_inputs, elm_outputs):
        return None

    # FIXME: This fails with Ops like `Max` whose `c_code` always expects two inputs!
    #  Should implement a `CAReduce.supports_c_code`?
    try:
        car_scalar_op.c_code(
            node,
            "test_presence_of_c_code",
            ["x" for x in node.inputs],
            ["z" for z in node.outputs],
            {"fail": "%(fail)s"},
        )
    except (NotImplementedError, MethodNotDefined):
        return False

    car_op = node.op
    car_acc_dtype = node.op.acc_dtype

    scalar_elm_inputs = [
        ps.get_scalar_type(inp.type.dtype).make_variable() for inp in elm_inputs
    ]

    elm_output = elm_scalar_op(*scalar_elm_inputs)

    # This input represents the previous value in the `CAReduce` binary reduction
    carried_car_input = ps.get_scalar_type(car_acc_dtype).make_variable()

    scalar_fused_output = car_scalar_op(carried_car_input, elm_output)
    if scalar_fused_output.type.dtype != car_acc_dtype:
        scalar_fused_output = ps.cast(scalar_fused_output, car_acc_dtype)

    fused_scalar_op = ps.Composite(
        inputs=[carried_car_input, *scalar_elm_inputs], outputs=[scalar_fused_output]
    )

    # The fused `Op` needs to look and behave like a `BinaryScalarOp`
    # TODO: Generate a new `type` and make this relationship official?
    fused_scalar_op.identity = car_scalar_op.identity
    fused_scalar_op.nin = 2
    fused_scalar_op.nout = 1

    new_car_op = CAReduce(
        scalar_op=fused_scalar_op,
        axis=car_op.axis,
        acc_dtype=car_acc_dtype,
        dtype=car_op.dtype,
        upcast_discrete_output=car_op.upcast_discrete_output,
    )

    return [new_car_op(*elm_inputs)]


@node_rewriter([Elemwise])
def local_inline_composite_constants(fgraph, node):
    """Inline scalar constants in Composite graphs."""
    composite_op = node.op.scalar_op

    if not isinstance(composite_op, ps.Composite):
        return None

    new_outer_inputs = []
    new_inner_inputs = []
    inner_replacements = {}
    for outer_inp, inner_inp in zip(
        node.inputs, composite_op.fgraph.inputs, strict=True
    ):
        # Complex variables don't have a `c_literal` that can be inlined
        if (
            isinstance(outer_inp, TensorConstant)
            and "complex" not in outer_inp.type.dtype
        ):
            if outer_inp.unique_value is not None:
                inner_replacements[inner_inp] = ps.constant(
                    outer_inp.unique_value, dtype=inner_inp.dtype
                )
                continue
        new_outer_inputs.append(outer_inp)
        new_inner_inputs.append(inner_inp)

    if not inner_replacements:
        return None

    new_inner_outs = clone_replace(
        composite_op.fgraph.outputs, replace=inner_replacements
    )
    new_composite_op = ps.Composite(new_inner_inputs, new_inner_outs)
    new_outputs = Elemwise(new_composite_op).make_node(*new_outer_inputs).outputs

    # Some of the inlined constants were broadcasting the output shape
    if node.outputs[0].type.broadcastable != new_outputs[0].type.broadcastable:
        new_outputs = [
            alloc_like(new_out, template=node.outputs[0], fgraph=fgraph)
            for new_out in new_outputs
        ]

    copy_stack_trace(node.outputs, new_outputs)
    return new_outputs


@node_rewriter(tracks=[add, mul])
def constant_fold_branches_of_add_mul(fgraph, node):
    old_constants = [inp for inp in node.inputs if isinstance(inp, TensorConstant)]

    if len(old_constants) <= 1:
        return None

    new_constants = old_constants.copy()

    # Multiply constants if it doesn't result in higher intermediate memory
    while True:
        n_constants = len(new_constants)
        if n_constants <= 1:
            break

        for i in range(n_constants):
            reference_inp = new_constants[i]
            other_inps = []
            for j in range(n_constants):
                if i == j:
                    continue
                other_inp = new_constants[j]
                if not broadcasted_by(reference_inp, other_inp):
                    other_inps.append(other_inp)
            if other_inps:
                python_op = operator.mul if node.op == mul else operator.add
                folded_inputs = [reference_inp, *other_inps]
                new_inp = constant(
                    reduce(python_op, (const.data for const in folded_inputs))
                )
                new_constants = [
                    new_inp,
                    *(inp for inp in new_constants if inp not in folded_inputs),
                ]
                break
        else:  # no-break
            break

    if len(new_constants) == len(old_constants):
        return None

    non_constants = [inp for inp in node.inputs if not isinstance(inp, TensorConstant)]
    new_out = node.op(
        *new_constants,
        *non_constants,
    )
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


add_mul_fusion_seqopt = SequenceDB()
compile.optdb.register(
    "add_mul_fusion",
    add_mul_fusion_seqopt,
    "fast_run",
    position=48,  # Before Elemwise fusion
)
add_mul_fusion_seqopt.register(
    flatten_nested_add_mul.__name__,
    out2in(flatten_nested_add_mul, ignore_newtrees=False),
    "fast_run",
    position=0,
)
add_mul_fusion_seqopt.register(
    constant_fold_branches_of_add_mul.__name__,
    in2out(constant_fold_branches_of_add_mul, ignore_newtrees=True),
    "fast_run",
    position=1,
)

# Register fusion database just before AddDestroyHandler(49.5) (inplace rewrites)
fuse_seqopt = SequenceDB()
compile.optdb.register(
    "elemwise_fusion",
    fuse_seqopt,
    "fast_run",
    "fusion",
    "local_elemwise_fusion",
    "FusionOptimizer",
    position=49,
)
fuse_seqopt.register(
    "composite_elemwise_fusion",
    FusionOptimizer(),
    "fast_run",
    "fusion",
    position=1,
)
fuse_seqopt.register(
    "local_useless_composite_outputs",
    in2out(local_useless_composite_outputs),
    "fast_run",
    "fusion",
    position=2,
)
fuse_seqopt.register(
    "local_careduce_fusion",
    in2out(local_careduce_fusion),
    "fast_run",
    "fusion",
    position=10,
)
fuse_seqopt.register(
    "local_inline_composite_constants",
    in2out(local_inline_composite_constants, ignore_newtrees=True),
    "fast_run",
    "fusion",
    position=20,
)


def _rebuild_partial_2f1grad_loop(node, wrt):
    a, b, c, log_z, sign_z = node.inputs[-5:]
    z = exp(log_z) * sign_z

    # Reconstruct scalar loop with relevant outputs
    a_, b_, c_, z_ = (x.type.to_scalar_type()() for x in (a, b, c, z))
    new_loop_op = _grad_2f1_loop(
        a_, b_, c_, z_, skip_loop=False, wrt=wrt, dtype=a_.type.dtype
    )[0].owner.op

    # Reconstruct elemwise loop
    new_elemwise_op = Elemwise(scalar_op=new_loop_op)
    n_steps = node.inputs[0]
    init_grad_vars = node.inputs[1:10]
    other_inputs = node.inputs[10:]

    init_grads = init_grad_vars[: len(wrt)]
    init_gs = init_grad_vars[3 : 3 + len(wrt)]
    init_gs_signs = init_grad_vars[6 : 6 + len(wrt)]
    subset_init_grad_vars = init_grads + init_gs + init_gs_signs

    return new_elemwise_op(n_steps, *subset_init_grad_vars, *other_inputs)


@register_specialize
@node_rewriter([Elemwise])
def local_useless_2f1grad_loop(fgraph, node):
    # Remove unused terms from the hyp2f1 grad loop

    loop_op = node.op.scalar_op
    if not isinstance(loop_op, Grad2F1Loop):
        return

    grad_related_vars = node.outputs[:-4]
    # Rewrite was already applied
    if len(grad_related_vars) // 3 != 3:
        return None

    grad_vars = grad_related_vars[:3]
    grad_var_is_used = [bool(fgraph.clients.get(v)) for v in grad_vars]

    # Nothing to do here
    if sum(grad_var_is_used) == 3:
        return None

    *other_vars, converges = node.outputs[3:]

    # Check that None of the remaining vars (except the converge flag) is used anywhere
    if any(bool(fgraph.clients.get(v)) for v in other_vars):
        return None

    wrt = [i for i, used in enumerate(grad_var_is_used) if used]
    *new_outs, new_converges = _rebuild_partial_2f1grad_loop(node, wrt=wrt)

    replacements = {converges: new_converges}
    i = 0
    for grad_var, is_used in zip(grad_vars, grad_var_is_used, strict=True):
        if not is_used:
            continue
        replacements[grad_var] = new_outs[i]
        i += 1
    return replacements


@node_rewriter([Elemwise])
def split_2f1grad_loop(fgraph, node):
    """
    2f1grad loop has too many operands for Numpy frompyfunc code used by Elemwise nodes on python mode.

    This rewrite splits it across 3 different operations. It is not needed if `local_useless_2f1grad_loop` was applied
    """
    loop_op = node.op.scalar_op

    if not isinstance(loop_op, Grad2F1Loop):
        return None

    grad_related_vars = node.outputs[:-4]
    # local_useless_2f1grad_loop was used, we should be safe
    if len(grad_related_vars) // 3 != 3:
        return None

    grad_vars = grad_related_vars[:3]
    *other_vars, converges = node.outputs[3:]

    # Check that None of the remaining vars is used anywhere
    if any(bool(fgraph.clients.get(v)) for v in other_vars):
        return None

    new_grad0, new_grad1, *_, new_converges01 = _rebuild_partial_2f1grad_loop(
        node, wrt=[0, 1]
    )
    new_grad2, *_, new_converges2 = _rebuild_partial_2f1grad_loop(node, wrt=[2])

    replacements = {
        converges: new_converges01 & new_converges2,
        grad_vars[0]: new_grad0,
        grad_vars[1]: new_grad1,
        grad_vars[2]: new_grad2,
    }
    return replacements


compile.optdb["py_only"].register(
    "split_2f1grad_loop",
    split_2f1grad_loop,
    "fast_compile",
)

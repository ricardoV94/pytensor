from typing import cast

from pytensor.compile import Supervisor, get_default_mode, optdb
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function.types import add_supervisor_to_fgraph
from pytensor.compile.io import In
from pytensor.graph import Apply, Constant, Variable, clone_replace, node_rewriter
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.graph.rewriting.basic import copy_stack_trace, in2out
from pytensor.tensor.basic import AllocDiag
from pytensor.tensor.rewriting.basic import register_specialize


def inline_ofg_node(node: Apply) -> list[Variable]:
    op = node.op
    assert isinstance(op, OpFromGraph)
    inlined_outs = clone_replace(
        op.inner_outputs, dict(zip(op.inner_inputs, node.inputs, strict=True))
    )
    copy_stack_trace(op.inner_outputs, inlined_outs)
    return cast(list[Variable], inlined_outs)


@node_rewriter([OpFromGraph])
def inline_ofg_expansion(fgraph, node):
    """
    This optimization expands internal graph of OpFromGraph.
    Only performed if node.op.is_inline == True
    Doing so can improve optimization at the cost of compilation speed.
    """
    op = node.op
    if not op.is_inline:
        return False

    return inline_ofg_node(node)


# We want to run this before the first merge optimizer
# and before the first scan optimizer.
optdb.register(
    "inline_ofg_expansion",
    in2out(inline_ofg_expansion),
    "fast_compile",
    "fast_run",
    position=-0.01,
)


@register_specialize("inline_ofg")
@node_rewriter([AllocDiag])
def late_inline_OpFromGraph(fgraph, node):
    """
    Inline `OpFromGraph` nodes.

    OpFromGraph nodes are used to compactly represent the output of a function graph. Certain `Ops`, like, einsum,
    diag, and kron, are implemented using pytensor `Op`s. As a result, their outputs are not a single `Op`, but a
    graph. To allow rewrites to easily spot and manipulate these "composite functions", we use the `OpFromGraph` node.
    This node is a thin wrapper around the output graph. It is not, however, meant to be included in the final
    program, because it hides the inner graph from certain optimizations.

    This rewrite specifies that all `OpFromGraph` nodes should be replaced by their inner graphs by setting the
    `inplace=True` flag.

    Parameters
    ----------
    fgraph: FunctionGraph
        The function graph being rewritten
    node: Apply
        Node of the function graph to be optimized

    Returns
    -------

    """
    return inline_ofg_node(node)


@node_rewriter(tracks=[OpFromGraph], inplace=True)
def inplace_ofg(fgraph, node):
    # TODO: This should be a graph rewriter, that looks for a subset of destroyiable inputs across all applications of the Op
    #  To avoid duplicating OFG Ops
    op: OpFromGraph = node.op

    if op.destroy_map:
        return

    supervisor: Supervisor = fgraph._supervisor
    destroy_handler: DestroyHandler = fgraph.destroy_handler

    candidate_inputs = [
        inp
        for inp in node.inputs
        if not (
            # Constants can't be destroyed
            isinstance(inp, Constant)
            # Nor protected variables (or views of protected variables)
            or destroy_handler.droot.get(inp, None) in supervisor.protected
            # Or variables that are already destroyed
            or fgraph.has_destroyers([inp])
        )
    ]
    if not candidate_inputs:
        return

    while len(set(candidate_inputs)) < len(candidate_inputs):
        # Don't try to inplace on duplicate inputs
        # TODO: Duplicate inputs could be merged in the inner graph, but this may mess up with other rewrites that operate on specific OFGs
        for duplicated_inp in candidate_inputs:
            if candidate_inputs.count(duplicated_inp) > 1:
                # Found it
                candidate_inputs = [
                    inp for inp in candidate_inputs if inp is not duplicated_inp
                ]

    # We will have to eagerly rewrite the inner fgraph, to see what it can inplace on (if anything)
    # FIXME: This should definitely only be done once per Op!
    inner_fgraph = op.fgraph.clone()
    rewriter = get_default_mode().optimizer
    add_supervisor_to_fgraph(
        inner_fgraph,
        input_specs=[
            In(inner_inp, mutable=outer_inp in candidate_inputs)
            for outer_inp, inner_inp in zip(node.inputs, fgraph.inputs, strict=True)
        ],
        accept_inplace=True,
    )
    rewriter(inner_fgraph)
    if not inner_fgraph.has_destroyers(inner_fgraph.inputs):
        # Nothing came out of it
        return None

    destroyed_input_idxs = [
        i
        for i, inner_inp in enumerate(inner_fgraph.inputs)
        if inner_fgraph.has_destroyers([inner_inp])
    ]
    # This is arbitrary, the destroyers may not even be outputs, they could be intermediate variables
    # I'm also not sure why PyTensor needs this
    destroy_map = {0: destroyed_input_idxs}

    if hasattr(op, "_props"):
        props = op._props_dict()
    else:
        props = {}
    try:
        inplace_op = type(node.op)(
            inputs=inner_fgraph.inputs,
            outputs=inner_fgraph.outputs,
            destroy_map=destroy_map,
            **props,
        )
        new_outs = inplace_op(*node.inputs, return_list=True)
    except Exception:
        # OpFromGraph is a bit messy, so here is a messy safety valve
        return None
    copy_stack_trace(node.outputs, new_outs)
    return new_outs


optdb.register(
    "inplace_ofg",
    in2out(inplace_ofg),
    "fast_run",
    "inplace",
    # Run this before other inplaces, because if Op can be used many times inplace the benefits compound
    position=50.0,
)

.. _prng:

===========================================
Pseudo random number generation in PyTensor
===========================================

PyTensor has native support for `pseudo random number generation (PRNG) <https://en.wikipedia.org/wiki/Pseudorandom_number_generator>`_.

This document describes the details of how PRNGs are implemented in PyTensor, via the RandomVariable Operator.
For a more applied example see :ref:`using_random_numbers`

We also discuss how initial seeding and seeding updates are implemented, and some harder cases such as using RandomVariables inside Scan, or with other backends like JAX.

We will use PRNG and RNG interchangeably, keeping in mind we are always talking about PRNGs.

The basics
==========

NumPy
-----

To start off, let's recall how PRNGs works in NumPy

>>> import numpy as np
>>> rng = np.random.default_rng(seed=123)
>>> print(rng.uniform(size=2), rng.uniform(size=2))
[0.68235186 0.05382102] [0.22035987 0.18437181]

In the first line np.random.default_rng(seed) creates a random Generator.

>>> rng # doctest: +SKIP
Generator(PCG64) at 0x7F6C04535820

Every NumPy Generator holds a BitGenerator, which is able to generate high-quality sequences of pseudo random bits.
NumPy generators' methods convert these sequences of bits into sequences of numbers that follow a specific statistical distribution.
For more details, you can read `NumPy random sampling documentation <https://numpy.org/doc/stable/reference/random>`_.

>>> rng.bit_generator # doctest: +SKIP
<numpy.random._pcg64.PCG64 at 0x7f6c045030f0>

>>> rng.bit_generator.state # doctest: +SKIP
{'bit_generator': 'PCG64',
 'state': {'state': 143289216567205249174526524509312027761,
  'inc': 17686443629577124697969402389330893883},
 'has_uint32': 0,
 'uinteger': 0}

When we call rng.uniform(size=2), the Generator class requested a new array of pseudo random bits (state) from the BitGenerator,
and used a deterministic mapping function to convert those into a float64 numbers.

It did this twice, because we requested two draws via the size argument.
In the long-run this deterministic mapping function should produce draws that are statistically indistinguishable from a true uniform distribution.

For illustration we implement a very bad mapping function from a bit generator to uniform draws.

.. code:: python

    def bad_uniform_rng(rng, size):
        bit_generator = rng.bit_generator

        uniform_draws = np.empty(size)
        for i in range(size):
            bit_generator.advance(1)
            state = rng.bit_generator.state["state"]["state"]
            last_3_digits = state % 1_000
            uniform_draws[i] = (last_3_digits + 1) / 1_000
        return uniform_draws

    bad_uniform_rng(rng, size=5)
    # array([0.033, 0.972, 0.459, 0.71 , 0.765])

SciPy
-----

SciPy wraps these NumPy routines in a slightly different API.

>>> import scipy.stats as st
>>> rng = np.random.default_rng(seed=123)
>>> print(st.uniform.rvs(size=2, random_state=rng), st.uniform.rvs(size=2, random_state=rng))
[0.68235186 0.05382102] [0.22035987 0.18437181]

PyTensor
--------

PyTensor does not implement its own bit/generators methods.
Just like SciPy, it borrows NumPy routines directly.

PyTensor's RNG API uses explicit RNG state threading: every random draw returns a ``(next_rng, draw)`` tuple,
where ``next_rng`` is the updated generator state that should be used for subsequent draws.

>>> import pytensor
>>> import pytensor.tensor as pt

>>> rng = pt.random.rng("rng")
>>> next_rng, x = rng.uniform(size=2)
>>> f = pytensor.function([rng], [x])

We created a function that takes a NumPy RandomGenerator and returns two uniform draws. Let's evaluate it

>>> rng_val = np.random.default_rng(123)
>>> print(f(rng_val), f(rng_val))
[array([0.68235186, 0.05382102])] [array([0.68235186, 0.05382102])]

The first numbers were exactly the same as the NumPy and SciPy calls, because we are using the very same routines.

Perhaps surprisingly, we got the same results when we called the function the second time!
This is because PyTensor functions do not hold an internal state and do not modify inputs inplace unless requested to.

We made sure that the rng_val was not modified when calling our PyTensor function, by copying it before using it.
This may feel inefficient (and it is), but PyTensor is built on a pure functional approach, which is not allowed to have side-effects by default.

We will later see how we can get around this issue by making the inputs mutable or using shared variables with explicit update rules.

Before that, let's convince ourselves we can actually get different draws, when we modify the bit generator of our input RNG.

>>> _ = rng_val.bit_generator.advance(1)
>>> print(f(rng_val), f(rng_val))
[array([0.05382102, 0.22035987])] [array([0.05382102, 0.22035987])]

>>> _ = rng_val.bit_generator.advance(1)
>>> print(f(rng_val), f(rng_val))
[array([0.22035987, 0.18437181])] [array([0.22035987, 0.18437181])]

Updating the bit generator manually is not a good practice.
For starters, it may be unclear how much we have to advance it!

In this case we had to advance it twice to get two completely new draws, because the inner function uses two states.
But other distributions could need more states for a single draw, or they could be clever and reuse the same state for multiple draws.

That is why the RNG variable methods always return a ``(next_rng, draw)`` tuple.
The ``next_rng`` contains the bit generator that was already modified when taking draws, and can be safely used again.

We can compile a function that returns the next_rng explicitly, so that we can use it as the input of the function in subsequent calls.

>>> f = pytensor.function([rng], [next_rng, x])

>>> rng_val = np.random.default_rng(123)
>>> next_rng_val, x_val = f(rng_val)
>>> print(x_val)
[0.68235186 0.05382102]

>>> next_rng_val, x_val = f(next_rng_val)
>>> print(x_val)
[0.22035987 0.18437181]

>>> next_rng_val, x_val = f(next_rng_val)
>>> print(x_val)
[0.1759059  0.81209451]

Shared variables
================

At this point we can make use of PyTensor shared variables.
Shared variables are global variables that don't need (and can't) be passed as explicit inputs to the functions where they are used.

The ``pt.random.shared_rng()`` helper creates a shared RNG variable:

>>> rng = pt.random.shared_rng(value=np.random.default_rng(123))
>>> next_rng, x = rng.uniform()
>>>
>>> f = pytensor.function([], [next_rng, x])
>>>
>>> next_rng_val, x_val = f()
>>> print(x_val)
0.6823518632481435

We can update the value of shared variables across calls.

>>> rng.set_value(next_rng_val)
>>> next_rng_val, x_val = f()
>>> print(x_val)
0.053821018802222675

>>> rng.set_value(next_rng_val)
>>> next_rng_val, x_val = f()
>>> print(x_val)
0.22035987277261138

The real benefit of using shared variables is that we can automate this updating via the aptly named updates kwarg of PyTensor functions.

In this case it makes sense to simply replace the original value by the next_rng_val (there is not really any other operation we can do with PyTensor RNGs)

>>> rng = pt.random.shared_rng(value=np.random.default_rng(123))
>>> next_rng, x = rng.uniform()
>>>
>>> f = pytensor.function([], x, updates={rng: next_rng})
>>>
>>> f(), f(), f()
(array(0.68235186), array(0.05382102), array(0.22035987))

Reseeding
---------

Shared RNG variables can be "reseeded" by setting them to a new RNG with the desired seed

>>> rng = pt.random.shared_rng(value=np.random.default_rng(123))
>>> next_rng, x = rng.normal()
>>>
>>> f = pytensor.function([], x, updates={rng: next_rng})
>>>
>>> print(f(), f())
-0.9891213503478509 -0.3677866514678832
>>> rng.set_value(np.random.default_rng(123))
>>> print(f(), f())
-0.9891213503478509 -0.3677866514678832

.. _rng_reuse_warning:

RNG reuse warning
-----------------

A common mistake is to use the same RNG variable for multiple draws without threading the updated state.
This would produce correlated (identical) draws, since both operations see the same input state.

PyTensor warns when it detects this pattern:

.. code:: python

    rng = pt.random.rng()
    _, x = rng.normal()
    _, y = rng.normal()  # WARNING: rng already used!

The correct pattern threads the ``next_rng`` returned by each draw:

.. code:: python

    rng = pt.random.rng()
    next_rng, x = rng.normal()
    next_rng, y = next_rng.normal()  # Correct: uses the updated state

Or, more commonly, use separate RNG variables for independent draws (see :ref:`multiple_random_variables`).

Inplace optimization
====================

As mentioned, RandomVariable Ops default to making a copy of the input RNG before using it, which can be quite slow.

>>> rng = pt.random.shared_rng(value=np.random.default_rng(123), name="rng")
>>> next_rng, x = rng.uniform()
>>> f = pytensor.function([], x)
>>> pytensor.dprint(f, print_destroy_map=True) # doctest: +SKIP
uniform_rv{"(),()->()"}.1 [id A] 'x' 0
 ├─ rng [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]


>>> %timeit f()  # doctest: +SKIP
81.8 µs ± 15.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

>>> rng_np = np.random.default_rng(123)
>>> %timeit rng_np.uniform()  # doctest: +SKIP
2.15 µs ± 63.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

Like other PyTensor operators, RandomVariable's can be given permission to modify inputs inplace during their operation.

In this case, there is a `inplace` flag that when `true` tells the RandomVariable Op that it is safe to modify the RNG input inplace.
If the flag is set, the RNG will not be copied before taking random draws.

The `random_make_inplace <https://github.com/pymc-devs/pytensor/blob/3fcf6369d013c597a9c964b2400a3c5e20aa8dce/pytensor/tensor/random/rewriting/basic.py#L42-L52>`_
rewrite automatically replaces RandomVariable Ops by their inplace counterparts, when such operation is deemed safe. This happens when:

#. An input RNG is flagged as `mutable` and is not used anywhere else.
#. A RNG is created intermediately and not used anywhere else.

The first case is true when a user uses the `mutable` `kwarg` directly.

>>> from pytensor.compile.io import In
>>> rng = pt.random.rng("rng")
>>> next_rng, x = rng.uniform()
>>> with pytensor.config.change_flags(optimizer_verbose=True):  # doctest: +SKIP
...     inplace_f = pytensor.function([In(rng, mutable=True)], [x])
>>> pytensor.dprint(inplace_f, print_destroy_map=True) # doctest: +SKIP
rewriting: rewrite random_make_inplace replaces uniform_rv{"(),()->()"}.out of uniform_rv{"(),()->()"}(rng, NoneConst{None}, 0.0, 1.0) with uniform_rv{"(),()->()"}.out of uniform_rv{"(),()->()"}(rng, NoneConst{None}, 0.0, 1.0)
uniform_rv{"(),()->()"}.1 [id A] d={0: [0]} 0
 ├─ rng [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]

Or, much more commonly, when a shared RNG is used and a (default or manual) update expression is given.
In this case, a RandomVariable is allowed to modify the RNG because the shared variable holding it will be rewritten anyway.

>>> rng = pt.random.shared_rng(name="rng")
>>> next_rng, x = rng.uniform()
>>>
>>> inplace_f = pytensor.function([], [x], updates={rng: next_rng})
>>> pytensor.dprint(inplace_f, print_destroy_map=True) # doctest: +SKIP
uniform_rv{"(),()->()"}.1 [id A] d={0: [0]} 0
 ├─ rng [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]
uniform_rv{"(),()->()"}.0 [id A] d={0: [0]} 0
 └─ ···

The second case is not very common, because RNGs are not usually chained across multiple RandomVariable Ops.
See more details in the next section.

Unused RNG consumer optimization (``random_unsafe``)
-----------------------------------------------------

When only the RNG output of a RandomVariable is used (not the draw), PyTensor can bypass the entire node
by connecting each RNG output directly to its corresponding RNG input. This is done by the
``sidestep_unused_rng_consumer`` rewrite.

This can happen when chaining multiple RNGs but only keeping some of the draws,
or when only the shape is needed and is eventually lifted away from the RV.

Because this optimization alters the RNG state sequence, it is tagged as ``random_unsafe`` and can be excluded
if exact reproducibility of the RNG stream is needed:

.. code:: python

    mode = pytensor.Mode(optimizer="fast_run").excluding("random_unsafe")
    f = pytensor.function([], [x], mode=mode)

.. _multiple_random_variables:

Multiple random variables
=========================

It's common practice to use separate RNG variables for each RandomVariable in PyTensor.

>>> rng_x = pt.random.shared_rng(value=np.random.default_rng(123), name="rng_x")
>>> rng_y = pt.random.shared_rng(value=np.random.default_rng(456), name="rng_y")
>>>
>>> next_rng_x, x = rng_x.normal(loc=0, scale=10)
>>> next_rng_y, y = rng_y.normal(loc=x, scale=0.1)
>>>
>>> f = pytensor.function([], [x, y], updates={rng_x: next_rng_x, rng_y: next_rng_y})
>>> pytensor.dprint(f, print_type=True) # doctest: +SKIP
normal_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 ├─ rng_x [id B] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ 0 [id D] <Scalar(int8, shape=())>
 └─ 10 [id E] <Scalar(int8, shape=())>
normal_rv{"(),()->()"}.1 [id F] <Scalar(float64, shape=())> 1
 ├─ rng_y [id G] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ normal_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 │  └─ ···
 └─ 0.1 [id H] <Scalar(float64, shape=())>
normal_rv{"(),()->()"}.0 [id A] <RandomGeneratorType> 0
 └─ ···
normal_rv{"(),()->()"}.0 [id F] <RandomGeneratorType> 1
 └─ ···

>>> f(), f(), f()
([array(-9.8912135), array(-9.80160951)], [array(-3.67786651), array(-3.89026137)], [array(12.87925261), array(13.04327299)])

We could have used a single rng by threading the state.

>>> rng = pt.random.shared_rng(value=np.random.default_rng(seed=123), name="rng")
>>> next_rng, x = rng.normal(loc=0, scale=1)
>>> final_rng, y = next_rng.normal(loc=100, scale=1)
>>>
>>> f = pytensor.function([], [x, y], updates={rng: final_rng})
>>> pytensor.dprint(f, print_type=True) # doctest: +SKIP
normal_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 ├─ rng [id B] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ 0 [id D] <Scalar(int8, shape=())>
 └─ 1 [id E] <Scalar(int8, shape=())>
normal_rv{"(),()->()"}.1 [id F] <Scalar(float64, shape=())> 1
 ├─ normal_rv{"(),()->()"}.0 [id A] <RandomGeneratorType> 0
 │  └─ ···
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ 100 [id G] <Scalar(int8, shape=())>
 └─ 1 [id E] <Scalar(int8, shape=())>
normal_rv{"(),()->()"}.0 [id F] <RandomGeneratorType> 1
 └─ ···

>>> f(), f()
([array(-0.98912135), array(99.63221335)], [array(1.28792526), array(100.19397442)])

It works, but that graph is slightly unorthodox in PyTensor.

One practical reason why, is that it is more difficult to define the correct update expression for the shared RNG variable.

One technical reason why, is that it makes rewrites more challenging in cases where RandomVariables could otherwise be manipulated independently.

Creating multiple RNG variables
-------------------------------

When creating multiple RNG variables, follow the NumPy best practices for
`parallel random number generation <https://numpy.org/doc/stable/reference/random/parallel.html#parallel-random-number-generation>`_
to ensure high quality, independent sequences.

Random variables in inner graphs
================================

Scan
----

Scan works very similar to a function (that is called repeatedly inside an outer scope).

This means that random variables will always return the same output unless the RNG state is threaded across iterations.
If we use an RNG as a non-sequence (i.e. the same value is passed to every iteration), every step sees the same state
and produces the same draw:

>>> rng = pt.random.shared_rng(value=np.random.default_rng(123), name="rng")

>>> def constant_step(rng):
...     _, x = rng.normal()
...     return x

>>> draws = pytensor.scan(
...     fn=constant_step,
...     outputs_info=[None],
...     non_sequences=[rng],
...     n_steps=5,
...     strict=True,
...     return_updates=False,
... )

>>> f = pytensor.function([], draws)
>>> f(), f()
(array([-0.98912135, -0.98912135, -0.98912135, -0.98912135, -0.98912135]), array([-0.98912135, -0.98912135, -0.98912135, -0.98912135, -0.98912135]))

To get different draws at each step, the RNG should be passed as a recurrent state via ``outputs_info``.
The step function receives the current RNG, uses it for draws, and returns the updated RNG alongside the output.
Scan only returns the last RNG state (not all intermediate states),
because it doesn't know how to trace this kind of variable. The final RNG state can then be used to build
the update dictionary for the outer function.

>>> rng = pt.random.shared_rng(value=np.random.default_rng(123))

>>> def random_step(rng):
...     next_rng, x = rng.normal()
...     return x, next_rng

>>> draws, final_rng = pytensor.scan(
...     fn=random_step,
...     outputs_info=[None, rng],
...     n_steps=5,
...     return_updates=False,
... )

>>> f = pytensor.function([], draws, updates={rng: final_rng})
>>> f(), f()
(array([-0.98912135, -0.36778665,  1.28792526,  0.19397442,  0.9202309 ]), array([ 0.57710379, -0.63646365,  0.54195222, -0.31659545, -0.32238912]))

OpFromGraph
-----------

In contrast to Scan, non-shared RNG variables can be used directly in OpFromGraph

>>> from pytensor.compile.builders import OpFromGraph

>>> rng = pt.random.rng("rng")

>>> def lognormal(rng):
...     next_rng, x = rng.normal()
...     return [next_rng, pt.exp(x)]

>>> lognormal_ofg = OpFromGraph([rng], lognormal(rng))

>>> rng_x = pt.random.shared_rng(value=np.random.default_rng(1), name="rng_x")
>>> rng_y = pt.random.shared_rng(value=np.random.default_rng(2), name="rng_y")

>>> next_rng_x, x = lognormal_ofg(rng_x)
>>> next_rng_y, y = lognormal_ofg(rng_y)

>>> f = pytensor.function([], [x, y], updates={rng_x: next_rng_x, rng_y: next_rng_y})

>>> f(), f(), f()
([array(1.41281503), array(1.20810544)], [array(2.27417681), array(0.59288879)], [array(1.39157622), array(0.66162024)])

Also in contrast to Scan, there is no special treatment of updates for shared variables used in the inner graphs of OpFromGraph.

Any "updates" must be modeled as explicit outputs and used in the outer graph directly as in the example above.

This is arguably more clean.

Other backends (and their limitations)
======================================

Numba
-----

NumPy random generators can be natively used with the Numba backend.

>>> rng = pt.random.shared_rng(value=np.random.default_rng(123), name="randomstate_rng")
>>> next_rng, x = rng.normal()
>>> numba_fn = pytensor.function([], x, updates={rng: next_rng}, mode="NUMBA")
>>> pytensor.dprint(numba_fn, print_type=True) # doctest: +SKIP
[normal_rv{"(),()->()"}].1 [id A] <Scalar(float64, shape=())> 0
 ├─ [] [id B] <Vector(int64, shape=(0,))>
 ├─ randomstate_rng [id C] <RandomGeneratorType>
 ├─ NoneConst{None} [id D] <NoneTypeT>
 ├─ 0.0 [id E] <Scalar(float32, shape=())>
 └─ 1.0 [id F] <Scalar(float32, shape=())>
Inner graphs:
[normal_rv{"(),()->()"}] [id A]
 ← normal_rv{"(),()->()"}.0 [id G] <RandomGeneratorType>
    ├─ *1-<RandomGeneratorType> [id H] <RandomGeneratorType>
    ├─ *2-<NoneTypeT> [id I] <NoneTypeT>
    ├─ *3-<Scalar(float32, shape=())> [id J] <Scalar(float32, shape=())>
    └─ *4-<Scalar(float32, shape=())> [id K] <Scalar(float32, shape=())>
 ← normal_rv{"(),()->()"}.1 [id G] <Scalar(float64, shape=())>
    └─ ···

>>> print(numba_fn(), numba_fn())
-0.9891213503478509 -0.3677866514678832

JAX
---

JAX uses a different type of PRNG than those of NumPy. This means that the standard shared RNGs cannot be used directly in graphs transpiled to JAX.

Instead a copy of the Shared RNG variable is made, and its bit generator state is expanded with a jax_state entry. This is what's actually used by the JAX random variables.

In general, update rules are still respected, but they won't update/rely on the original shared variable.

>>> import jax  # doctest: +SKIP
>>> rng = pt.random.shared_rng(value=np.random.default_rng(123), name="rng")  # doctest: +SKIP
>>> next_rng, x = rng.uniform()  # doctest: +SKIP
>>> jax_fn = pytensor.function([], [x], updates={rng: next_rng}, mode="JAX")  # doctest: +SKIP
>>> pytensor.dprint(jax_fn, print_type=True) # doctest: +SKIP
uniform_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 ├─ RNG(<Generator(PCG64) at 0x7FA448D68200>) [id B] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ 0.0 [id D] <Scalar(float32, shape=())>
 └─ 1.0 [id E] <Scalar(float32, shape=())>
uniform_rv{"(),()->()"}.0 [id A] <RandomGeneratorType> 0
 └─ ···

>>> print(jax_fn(), jax_fn())  # doctest: +SKIP
[Array(0.07577298, dtype=float64)] [Array(0.09217023, dtype=float64)]

>>> rng.set_value(np.random.default_rng(123))  # No effect on the jax evaluation  # doctest: +SKIP
>>> print(jax_fn(), jax_fn())  # doctest: +SKIP
[Array(0.13929162, dtype=float64)] [Array(0.45162648, dtype=float64)]

>>> [jax_rng] = jax_fn.input_storage[0].storage  # doctest: +SKIP
>>> jax_rng  # doctest: +SKIP
{'bit_generator': Array(1, dtype=int64, weak_type=True),
 'has_uint32': Array(0, dtype=int64, weak_type=True),
 'jax_state': Array([2647707238, 2709433097], dtype=uint32),
 'state': {'inc': Array(-9061352147377205305, dtype=int64),
  'state': Array(-6044258077699604239, dtype=int64)},
 'uinteger': Array(0, dtype=int64, weak_type=True)}

>>> [jax_rng] = jax_fn.input_storage[0].storage  # doctest: +SKIP
>>> jax_rng["jax_state"] = jax.random.PRNGKey(0)  # doctest: +SKIP
>>> print(jax_fn(), jax_fn())  # doctest: +SKIP
[Array(0.57655083, dtype=float64)] [Array(0.50347362, dtype=float64)]

>>> [jax_rng] = jax_fn.input_storage[0].storage  # doctest: +SKIP
>>> jax_rng["jax_state"] = jax.random.PRNGKey(0)  # doctest: +SKIP
>>> print(jax_fn(), jax_fn())  # doctest: +SKIP
[Array(0.57655083, dtype=float64)] [Array(0.50347362, dtype=float64)]

PyTensor could provide shared JAX-like RNGs and allow RandomVariables to accept them,
but that would break the spirit of one graph `->` multiple backends.

Alternatively, PyTensor could try to use a more general type for RNGs that can be used across different backends,
either directly or after some conversion operation (if such operations can be implemented in the different backends).

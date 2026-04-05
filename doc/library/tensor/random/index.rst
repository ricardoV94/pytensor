
.. _libdoc_tensor_random_basic:

=============================================
:mod:`random` -- Random number functionality
=============================================

.. module:: pytensor.tensor.random
   :synopsis: symbolic random variables


The :mod:`pytensor.tensor.random` module provides random-number drawing functionality
that closely resembles the :mod:`numpy.random` module.


High-level API
==============

PyTensor assigns NumPy RNG states (i.e. `Generator` objects) to
each `RandomVariable`.  The combination of an RNG state, a specific
`RandomVariable` type (e.g. `NormalRV`), and a set of distribution parameters
uniquely defines the `RandomVariable` instances in a graph.

Creating RNG variables
----------------------

There are two ways to create RNG variables:

.. function:: rng(name=None)

   Create a symbolic random number generator variable, suitable for use as a
   function input.

   :param str name: Optional name for the variable.
   :returns: A symbolic :class:`RandomGeneratorVariable`.

   .. testcode:: constructors

      import numpy as np
      import pytensor
      import pytensor.tensor as pt

      rng = pt.random.rng("rng")
      next_rng, x = rng.normal(0, 1, size=(2, 2))

      fn = pytensor.function([rng], [x])
      print(fn(np.random.default_rng(123)))

.. function:: shared_rng(value=None, *, name=None, borrow=False)

   Create a shared random number generator variable. Shared RNG variables
   persist across function calls and can be automatically updated.

   :param numpy.random.Generator value: Initial RNG state. If ``None``, a new
       ``numpy.random.default_rng()`` is used.
   :param str name: Optional name for the shared variable.
   :param bool borrow: If ``True``, use the provided value directly without copying.
   :returns: A :class:`RandomGeneratorSharedVariable`.

   .. testcode:: constructors

      rng = pt.random.shared_rng(value=np.random.default_rng(123))
      next_rng, x = rng.normal(0, 1, size=(2, 2))

      fn = pytensor.function([], [x], updates={rng: next_rng})
      print(fn())  # different numbers on each call due to updates

Using RNG variables
-------------------

:class:`RandomGeneratorVariable` objects have distribution methods that mirror
:mod:`pytensor.tensor.random` functions. Each method returns a tuple of
``(next_rng, draw)``:

.. testcode:: constructors

   rng = pt.random.rng()
   next_rng, x = rng.uniform(0, 1, size=(3,))
   next_rng, y = next_rng.normal(0, 1, size=(3,))

The ``next_rng`` must be used for subsequent draws to ensure correct state
threading. Reusing the same RNG variable for multiple draws will trigger
a warning. See :ref:`rng_reuse_warning` for details.

For an example of how to use random numbers, see :ref:`Using Random Numbers <using_random_numbers>`.
For a technical explanation of how PyTensor implements random variables see :ref:`prng`.


Distributions
=============

See :ref:`Available distributions <_libdoc_tensor_random_distributions>` for the full
list of distributions available as both module-level functions and
:class:`RandomGeneratorVariable` methods.


Low-level objects
=================

.. automodule:: pytensor.tensor.random.op
   :members: RandomVariable, default_rng

.. automodule:: pytensor.tensor.random.type
   :members: RandomType, RandomGeneratorType, random_generator_type

.. automodule:: pytensor.tensor.random.var
    :members: RandomGeneratorSharedVariable


.. _views_and_inplace:

============================
Views and inplace operations
============================

PyTensor allows the definition of :class:`Op`\s which return a :term:`view` on one
of their inputs or operate :term:`inplace` on one or several
inputs. This allows more efficient operations on NumPy's :class:`ndarray`
data type than would be possible otherwise.
However, in order to work correctly, these :class:`Op`\s need to
implement an additional interface.

PyTensor recognizes views and inplace operations specially. It ensures
that they are used in a consistent manner and it ensures that
operations will be carried in a compatible order.

.. _views:

Views
=====

A "view" on an object ``x`` is an object ``y`` which shares memory
with ``x`` in some way. In other words, changing ``x`` might also
change ``y`` and vice versa. For example, imagine a `vector` structure
which contains two fields: an integer length and a pointer to a memory
buffer. Suppose we have:

::

   x = vector {length: 256,
               address: 0xDEADBEEF}

   y = vector {length: 224,
               address: 0xDEADBEEF + 0x10}

   z = vector {length: 256,
               address: 0xCAFEBABE}


So ``x`` uses the memory range ``0xDEADBEEF - 0xDEADBFEF``, ``y`` the
range ``0xDEADBEFF - 0xDEADBFDF`` and z the range ``0xCAFEBABE -
0xCAFEBBBE``. Since the ranges for ``x`` and ``y`` overlap, ``y`` is
considered to be a view of ``x`` and vice versa.

Suppose you had an :class:`Op` which took ``x`` as input and returned
``y``. You would need to tell PyTensor that ``y`` is a view of ``x``. For this
purpose, you would set the :class:`Op.view_map` field as follows:


.. testsetup::

   from pytensor.graph.op import Op
   myop = Op()

.. testcode::

   myop.view_map = {0: [0]}


What this means is that the first output (position 0) is a view of the
first input (position 0). Even though the interface allows a list of
inputs that are viewed by a given output, this feature is currently
unsupported. Here are more examples:


.. testcode::

   myop.view_map = {0: [0]} # first output is a view of first input
   myop.view_map = {0: [1]} # first output is a view of second input
   myop.view_map = {1: [0]} # second output is a view of first input

   myop.view_map = {0: [0], # first output is a view of first input
                    1: [1]} # *AND* second output is a view of second input

   myop.view_map = {0: [0], # first output is a view of first input
                    1: [0]} # *AND* second output is *ALSO* a view of first input

   myop.view_map = {0: [0, 1]} # THIS IS NOT SUPPORTED YET! Only put a single input number in the list!


.. _inplace:


Inplace operations
==================

An inplace operation is one that modifies one or more of its
inputs. For example, the expression ``x += y`` where ``x`` and ``y``
are :class:`numpy.ndarray` instances would normally represent an inplace
operation on ``x``.

.. note::

   Inplace operations in PyTensor still work in a functional setting:
   they need to return the modified input. Symbolically, PyTensor
   requires one :class:`Variable` standing for the input before being modified
   and another :class:`Variable` representing the input after being
   modified. Therefore, code using inplace operations would look like
   this:

   .. testcode::

      from pytensor.tensor import dscalars, log
      from pytensor.tensor.inplace import add_inplace

      x, y = dscalars('x', 'y')
      r1 = log(x)

      # r2 is x AFTER the add_inplace - x still represents the value before adding y
      r2 = add_inplace(x, y)

      # r3 is log(x) using the x from BEFORE the add_inplace
      # r3 is the SAME as r1, even if we wrote this line after the add_inplace line
      # PyTensor is actually going to compute r3 BEFORE r2
      r3 = log(x)

      # this is log(x) using the x from AFTER the add_inplace (so it's like log(x + y))
      r4 = log(r2)

   Needless to say, this goes for user-defined inplace operations as
   well; the modified input must figure in the list of outputs you
   give to :class:`Apply` in the definition of :meth:`Apply.make_node`.

   Also, for technical reasons but also because they are slightly
   confusing to use as evidenced by the previous code, PyTensor does not
   allow the end user to use inplace operations by default. However,
   it does allow rewrites to substitute them in in a later
   phase. Therefore, typically, if you define an inplace operation,
   you will define a pure equivalent and a rewrite which
   substitutes one for the other. PyTensor will automatically verify if
   it is possible to do so and will refuse the substitution if it
   introduces inconsistencies.


Take the previous definitions of ``x``, ``y`` and ``z`` and suppose an :class:`Op` which
adds one to every byte of its input. If we give ``x`` as an input to
that :class:`Op`, it can either allocate a new buffer of the same size as ``x``
(that could be ``z``) and set that new buffer's bytes to the variable of
the addition. That would be a normal, :term:`pure`\ :class:`Op`. Alternatively,
it could add one to each byte in the buffer ``x``, therefore
changing it. That would be an inplace :class:`Op`.

PyTensor needs to be notified of this fact. The syntax is similar to
that of :attr:`Op.view_map`:


.. testcode::

   myop.destroy_map = {0: [0]}


What this means is that the first output (position 0) operates inplace on the
first input (position 0).


.. testcode::

   myop.destroy_map = {0: [0]} # first output operates inplace on first input
   myop.destroy_map = {0: [1]} # first output operates inplace on second input
   myop.destroy_map = {1: [0]} # second output operates inplace on first input

   myop.destroy_map = {0: [0], # first output operates inplace on first input
                       1: [1]} # *AND* second output operates inplace on second input

   myop.destroy_map = {0: [0], # first output operates inplace on first input
                       1: [0]} # *AND* second output *ALSO* operates inplace on first input

   myop.destroy_map = {0: [0, 1]} # first output operates inplace on both the first and second input
   # unlike for views, the previous line is legal and supported

.. note::
   :class:`DestroyHandler` provides a hackish means of specifying that a variable cannot be
   "destroyed" by an in-place operation: ``var.tag.indestructible = True``.

Destructive Operations
======================

While some operations will operate inplace on their inputs, some might
simply destroy or corrupt them. For example, an :class:`Op` could do temporary
calculations right in its inputs. If that is the case, PyTensor also
needs to be notified. The way to notify PyTensor is to assume that some
output operated inplace on whatever inputs are changed or corrupted by
the :class:`Op` (even if the output does not technically reuse any of the
input(s)'s memory). From there, go to the previous section.


.. warning::
   Failure to correctly mark down views and inplace operations using
   :attr:`Op.view_map` and :attr:`Op.destroy_map` can lead to nasty bugs. In the
   absence of this information, PyTensor might assume that it is safe to
   execute an inplace operation on some inputs before doing other
   calculations on the previous values of the inputs. For example,
   in the code: ``y = log(x); x2 = add_inplace(x, z)`` it is
   imperative to do the logarithm before the addition (because after
   the addition, the original x that we wanted to take the logarithm
   of is gone). If PyTensor does not know that ``add_inplace`` changes
   the value of ``x`` it might invert the order and that will
   certainly lead to erroneous computations.

   You can often identify an incorrect `Op.view_map` or :attr:`Op.destroy_map`
   by using :ref:`DebugMode <debugmode>`.

.. note::
   Consider using :class:`DebugMode` when developing
   a new :class:`Op` that uses :attr:`Op.view_map` and/or :attr:`Op.destroy_map`.

Inplace Rewriting and `DebugMode`
=================================

It is recommended that during the graph construction, all :class:`Op`\s are not inplace.
Then a rewrite replaces them with inplace ones. Currently :class:`DebugMode` checks
all rewrites that were tried even if they got rejected. One reason an inplace
rewrite can get rejected is when there is another :class:`Op` that is already being applied
inplace on the same input. Another reason to reject an inplace rewrite is
if it would introduce a cycle into the graph.

The problem with `DebugMode` is that it will trigger a useless error when
checking a rejected inplace rewrite, since it will lead to wrong results.
In order to be able to use `DebugMode` in more situations, your inplace
rewrite can pre-check whether it will get rejected by using the
:func:`pytensor.graph.destroyhandler.fast_inplace_check` function, that will tell
which :class:`Op`\s can be performed inplace. You may then skip the rewrite if it is
incompatible with this check. Note, however, that this check does not cover all
cases where a rewrite may be rejected (it will not detect cycles).

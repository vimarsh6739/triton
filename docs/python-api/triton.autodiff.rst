triton.experimental.autodiff
===========================

.. currentmodule:: triton.experimental

.. warning::

   This API is experimental. It supports forward and combined reverse modes
   through the Enzyme-JAX Python bindings. The available operations depend on
   the Enzyme-JAX build.

``fwddiff`` adapts a ``triton.jit`` kernel at launch time. It returns a
callable kernel object that uses Triton's normal ``kernel[grid](...)`` launch
syntax, but compiles the kernel through Enzyme-JAX's forward-mode TTIR pass. The
original ``triton.jit`` kernel is unchanged and remains callable as the primal
kernel.

``fwddiff`` follows the same argument activity style as Enzyme:

* ``Duplicated(primal, tangent)`` marks an argument as active and supplies its
  tangent value.
* ``Const(value)`` marks an argument as constant.

The generated derivative kernel receives active arguments as primal/tangent
pairs. For a vector-add kernel, the derivative launch computes both the primal
output and the tangent output:

.. code-block:: python

   from triton.experimental import Const, Duplicated, fwddiff

   @triton.jit
   def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(axis=0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)
       tl.store(output_ptr + offsets, x + y, mask=mask)

   output = torch.empty_like(x)
   doutput = torch.empty_like(output)
   grid = lambda meta: (triton.cdiv(output.numel(), meta["BLOCK_SIZE"]), )

   fwddiff(add_kernel)[grid](
       Duplicated(x, dx),
       Duplicated(y, dy),
       Duplicated(output, doutput),
       Const(output.numel()),
       BLOCK_SIZE=1024,
   )

   # output  == x + y
   # doutput == dx + dy

To launch the original primal kernel, call it directly:

.. code-block:: python

   add_kernel[grid](x, y, output, output.numel(), BLOCK_SIZE=1024)

Reverse mode
------------

``autodiff(kernel)`` uses ``"ReverseModeCombined"`` by default.
It creates a kernel that computes the primal result and its reverse derivative
in one launch. It uses the same launch syntax as ``fwddiff``.
The only accepted modes are ``"ForwardMode"`` and ``"ReverseModeCombined"``.
``fwddiff(kernel)`` is equivalent to ``autodiff(kernel, mode="ForwardMode")``.

Both functions also support decorator syntax. Put ``@autodiff`` or ``@fwddiff``
above ``@triton.jit``. To supply options, use parentheses, for example
``@fwddiff(tensor_shape=(1024,))``. The decorated object exposes the original
JIT kernel through its ``primal`` property.

Use ``Duplicated(primal, gradient)`` for each active pointer argument.
The derivative kernel adds input gradients to the supplied buffers.
Initialize these buffers to zero unless you want to add to existing gradients.
Set each output gradient buffer to the required seed before each launch.
The derivative kernel reads these seeds and clears them at stored addresses.
Masked stores leave the other seed values unchanged.

Use ``Const(value)`` for inactive pointers and scalar arguments.
Reverse mode currently supports gradients in pointer buffers only.
It does not return scalar gradients or a separate backward kernel.
Use separate storage for primal values and gradient buffers.

For the vector-add kernel above:

.. code-block:: python

   from triton.experimental import autodiff

   dx = torch.zeros_like(x)
   dy = torch.zeros_like(y)
   doutput = torch.ones_like(output)

   autodiff(add_kernel)[grid](
       Duplicated(x, dx),
       Duplicated(y, dy),
       Duplicated(output, doutput),
       Const(output.numel()),
       BLOCK_SIZE=1024,
   )

   # The kernel sets output to x + y.
   # The kernel adds one to dx and dy, then clears doutput.

Reverse mode differentiates the TTIR entry function directly.
It does not need ``tensor_shape``. The argument remains available for forward mode.
The Enzyme-JAX build must include the Triton reverse rules and the
``remove-unnecessary-enzyme-ops`` pass. Unsupported operations cause a compilation error.

Installation
------------

Install an Enzyme-JAX wheel built with Triton support in the same Python
environment as Triton:

.. code-block:: bash

   python -m pip install /path/to/enzyme_ad-*.whl

Triton calls ``enzyme_ad.jax.enzyme_call.run_pass_pipeline`` to transform the
TTIR. Enzyme-JAX is loaded when you first use autodiff. The wheel's Python
package also imports JAX, but the transformation takes MLIR text, not JAX arrays.
The ``enzyme_opt`` argument has been removed. ``TRITON_ENZYME_OPT`` has no effect.

When Triton cannot infer the tensor shape used by Enzyme's wrapper module, pass
``tensor_shape`` to the ``fwddiff`` adapter:

.. code-block:: python

   diff_kernel = fwddiff(kernel, tensor_shape=(1024,))
   diff_kernel[grid](...)

API reference
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   autodiff
   fwddiff
   Duplicated
   Const

Scheduling TTIR around AD
-------------------------

Both decorators accept ``pre_ad`` and ``post_ad``. The first schedule sees the
primal TTIR after Triton's TTIR stage. The second sees the generated derivative
kernel before Triton chooses GPU layouts. A string runs a registered Enzyme-JAX
pass pipeline. For example, hoist a shared projection before reverse AD and
eliminate common subexpressions afterward.

.. code-block:: python

   gradient = autodiff(kernel, pre_ad="loop-invariant-code-motion", post_ad="cse")

To target particular operations, pass an MLIR Transform module.

.. code-block:: python

   from triton.experimental import TTIRTransform

   hoist = TTIRTransform("""
     module attributes {transform.with_named_sequence} {
       transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
         %loops = transform.structured.match ops{["scf.for"]} in %root
             : (!transform.any_op) -> !transform.any_op
         transform.apply_licm to %loops : !transform.any_op
         transform.yield
       }
     }
   """)
   gradient = autodiff(kernel, pre_ad=hoist, post_ad="cse")

The kernel cache includes the schedule text and its placement relative to AD.
Use self-contained Transform modules so the cache key describes the schedule.

triton.autodiff
===============

.. currentmodule:: triton

.. warning::

   This API is experimental. It currently supports forward-mode differentiation
   of simple Triton kernels by shelling out to Enzyme-JAX's
   ``enzymexlamlir-opt`` and differentiating Triton TTIR.

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

   triton.fwddiff(add_kernel)[grid](
       triton.Duplicated(x, dx),
       triton.Duplicated(y, dy),
       triton.Duplicated(output, doutput),
       triton.Const(output.numel()),
       BLOCK_SIZE=1024,
   )

   # output  == x + y
   # doutput == dx + dy

To launch the original primal kernel, call it directly:

.. code-block:: python

   add_kernel[grid](x, y, output, output.numel(), BLOCK_SIZE=1024)

By default Triton searches for ``enzymexlamlir-opt`` on ``PATH`` and in the
local Enzyme-JAX checkout paths used by the development environment. Set
``TRITON_ENZYME_OPT`` to point at the optimizer binary if it lives elsewhere.

When Triton cannot infer the tensor shape used by Enzyme's wrapper module, pass
``tensor_shape`` to the decorator:

.. code-block:: python

   diff_kernel = triton.fwddiff(kernel, tensor_shape=(1024,))
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

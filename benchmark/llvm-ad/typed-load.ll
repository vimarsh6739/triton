; Each version of @control reads x when mask is true and substitutes zero otherwise.
define void @control(ptr addrspace(1) %x, ptr addrspace(1) %y, i1 %mask) {
  br i1 %mask, label %read, label %done
read:
  %value = load float, ptr addrspace(1) %x
  br label %done
done:
  %selected = phi float [ %value, %read ], [ 0.0, %0 ]
  %squared = fmul float %selected, %selected
  store float %squared, ptr addrspace(1) %y
  ret void
}

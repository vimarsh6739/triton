declare void @__enzyme_fwddiff(...)

define ptx_kernel void @attempt(ptr addrspace(1) %x, ptr addrspace(1) %dx,
                                ptr addrspace(1) %y, ptr addrspace(1) %dy, i1 %mask) {
  call void (...) @__enzyme_fwddiff(
    ptr @control,
    metadata !"enzyme_dup", ptr addrspace(1) %x, ptr addrspace(1) %dx,
    metadata !"enzyme_dup", ptr addrspace(1) %y, ptr addrspace(1) %dy,
    metadata !"enzyme_const", i1 %mask)
  ret void
}

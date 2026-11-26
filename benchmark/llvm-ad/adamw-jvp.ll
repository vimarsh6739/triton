declare void @__enzyme_fwddiff(...)

define ptx_kernel void @attempt(ptr addrspace(1) %p0, ptr addrspace(1) %d0, ptr addrspace(1) %p1, ptr addrspace(1) %d1, ptr addrspace(1) %p2, ptr addrspace(1) %d2, ptr addrspace(1) %p3, ptr addrspace(1) %d3, ptr addrspace(1) %p4, ptr addrspace(1) %d4, ptr addrspace(1) %p5, ptr addrspace(1) %d5, ptr addrspace(1) %p6, ptr addrspace(1) %d6, i32 %n, ptr addrspace(1) %scratch, ptr addrspace(1) %profile) {
  call void (...) @__enzyme_fwddiff(
    ptr @step,
    metadata !"enzyme_dup", ptr addrspace(1) %p0, ptr addrspace(1) %d0,
    metadata !"enzyme_dup", ptr addrspace(1) %p1, ptr addrspace(1) %d1,
    metadata !"enzyme_dup", ptr addrspace(1) %p2, ptr addrspace(1) %d2,
    metadata !"enzyme_dup", ptr addrspace(1) %p3, ptr addrspace(1) %d3,
    metadata !"enzyme_dup", ptr addrspace(1) %p4, ptr addrspace(1) %d4,
    metadata !"enzyme_dup", ptr addrspace(1) %p5, ptr addrspace(1) %d5,
    metadata !"enzyme_dup", ptr addrspace(1) %p6, ptr addrspace(1) %d6,
    metadata !"enzyme_const", i32 %n,
    metadata !"enzyme_const", ptr addrspace(1) %scratch,
    metadata !"enzyme_const", ptr addrspace(1) %profile)
  ret void
}

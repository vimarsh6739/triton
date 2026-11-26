; The predicated load sequence and constraints come from Triton's lowered AdamW.
define void @control(ptr addrspace(1) %x, ptr addrspace(1) %y, i1 %mask) {
  %value = call float asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 0 ];", "=r,r,l,b"(float 0.0, ptr addrspace(1) %x, i1 %mask)
  %squared = fmul float %value, %value
  store float %squared, ptr addrspace(1) %y
  ret void
}

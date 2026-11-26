; AdamW primal, 1023 parameters, block 256, four warps, sm_120
source_filename = "LLVMDialectModule"
target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"

@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1

; Function Attrs: nounwind
define ptx_kernel void @step(ptr addrspace(1) align 16 %0, ptr addrspace(1) align 16 %1, ptr addrspace(1) align 16 %2, ptr addrspace(1) align 16 %3, ptr addrspace(1) align 16 %4, ptr addrspace(1) align 16 %5, ptr addrspace(1) align 16 %6, i32 %7, ptr addrspace(1) nofree readnone captures(none) %8, ptr addrspace(1) nofree readnone captures(none) %9) local_unnamed_addr #0 {
  %11 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %12 = shl i32 %11, 8
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %14 = shl nuw nsw i32 %13, 1
  %15 = and i32 %14, 254
  %16 = or disjoint i32 %15, %12
  %17 = or disjoint i32 %16, 1
  %18 = icmp slt i32 %16, %7
  %19 = icmp slt i32 %17, %7
  %20 = sext i32 %16 to i64
  %21 = getelementptr [4 x i8], ptr addrspace(1) %0, i64 %20
  %22 = tail call float asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 0 ];", "=r,r,l,b"(float 0.000000e+00, ptr addrspace(1) %21, i1 %18) #4
  %23 = tail call float asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 4 ];", "=r,r,l,b"(float 0.000000e+00, ptr addrspace(1) %21, i1 %19) #4
  %24 = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %20
  %25 = tail call float asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 0 ];", "=r,r,l,b"(float 0.000000e+00, ptr addrspace(1) %24, i1 %18) #4
  %26 = tail call float asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 4 ];", "=r,r,l,b"(float 0.000000e+00, ptr addrspace(1) %24, i1 %19) #4
  %27 = getelementptr [4 x i8], ptr addrspace(1) %2, i64 %20
  %28 = tail call float asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 0 ];", "=r,r,l,b"(float 0.000000e+00, ptr addrspace(1) %27, i1 %18) #4
  %29 = tail call float asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 4 ];", "=r,r,l,b"(float 0.000000e+00, ptr addrspace(1) %27, i1 %19) #4
  %30 = getelementptr [4 x i8], ptr addrspace(1) %3, i64 %20
  %31 = tail call float asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 0 ];", "=r,r,l,b"(float 1.000000e+00, ptr addrspace(1) %30, i1 %18) #4
  %32 = tail call float asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 4 ];", "=r,r,l,b"(float 1.000000e+00, ptr addrspace(1) %30, i1 %19) #4
  %nvptx.pack9 = insertelement <2 x float> poison, float %31, i64 0
  %nvptx.pack10 = insertelement <2 x float> %nvptx.pack9, float %32, i64 1
  %nvptx.packed11 = fmul <2 x float> %nvptx.pack10, splat (float 9.990000e-01)
  %nvptx.pack = insertelement <2 x float> poison, float %25, i64 0
  %nvptx.pack6 = insertelement <2 x float> %nvptx.pack, float %26, i64 1
  %nvptx.packed = fmul <2 x float> %nvptx.pack6, splat (float 1.000000e-03)
  %nvptx.packed7 = fmul <2 x float> %nvptx.pack6, %nvptx.packed
  %nvptx.packed12 = fadd <2 x float> %nvptx.packed7, %nvptx.packed11
  %nvptx.packed15 = fmul <2 x float> %nvptx.packed12, splat (float f0x41FDE576)
  %nvptx.extract = extractelement <2 x float> %nvptx.packed15, i64 0
  %nvptx.extract16 = extractelement <2 x float> %nvptx.packed15, i64 1
  %33 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #4
  %.not.i = icmp eq i32 %33, 0
  %34 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #4
  %.not1.i = icmp eq i32 %34, 0
  br i1 %.not.i, label %40, label %35

35:                                               ; preds = %10
  br i1 %.not1.i, label %38, label %36

36:                                               ; preds = %35
  %37 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %nvptx.extract) #4
  br label %__nv_sqrtf.exit

38:                                               ; preds = %35
  %39 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %nvptx.extract) #4
  br label %__nv_sqrtf.exit

40:                                               ; preds = %10
  br i1 %.not1.i, label %43, label %41

41:                                               ; preds = %40
  %42 = tail call float @llvm.nvvm.sqrt.rn.f(float %nvptx.extract) #4
  br label %__nv_sqrtf.exit

43:                                               ; preds = %40
  %44 = tail call float @llvm.nvvm.sqrt.approx.f(float %nvptx.extract) #4
  br label %__nv_sqrtf.exit

__nv_sqrtf.exit:                                  ; preds = %43, %41, %38, %36
  %.0.i = phi float [ %37, %36 ], [ %39, %38 ], [ %42, %41 ], [ %44, %43 ]
  %45 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #4
  %.not.i1 = icmp eq i32 %45, 0
  %46 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #4
  %.not1.i4 = icmp eq i32 %46, 0
  br i1 %.not.i1, label %52, label %47

47:                                               ; preds = %__nv_sqrtf.exit
  br i1 %.not1.i4, label %50, label %48

48:                                               ; preds = %47
  %49 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %nvptx.extract16) #4
  br label %__nv_sqrtf.exit5

50:                                               ; preds = %47
  %51 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %nvptx.extract16) #4
  br label %__nv_sqrtf.exit5

52:                                               ; preds = %__nv_sqrtf.exit
  br i1 %.not1.i4, label %55, label %53

53:                                               ; preds = %52
  %54 = tail call float @llvm.nvvm.sqrt.rn.f(float %nvptx.extract16) #4
  br label %__nv_sqrtf.exit5

55:                                               ; preds = %52
  %56 = tail call float @llvm.nvvm.sqrt.approx.f(float %nvptx.extract16) #4
  br label %__nv_sqrtf.exit5

__nv_sqrtf.exit5:                                 ; preds = %55, %53, %50, %48
  %.0.i3 = phi float [ %49, %48 ], [ %51, %50 ], [ %54, %53 ], [ %56, %55 ]
  %nvptx.pack20 = insertelement <2 x float> poison, float %29, i64 0
  %nvptx.pack21 = insertelement <2 x float> %nvptx.pack20, float %28, i64 1
  %nvptx.packed22 = fmul <2 x float> %nvptx.pack21, splat (float f0x3F666666)
  %nvptx.pack17 = insertelement <2 x float> poison, float %26, i64 0
  %nvptx.pack18 = insertelement <2 x float> %nvptx.pack17, float %25, i64 1
  %nvptx.packed19 = fmul <2 x float> %nvptx.pack18, splat (float 1.000000e-01)
  %nvptx.packed23 = fadd <2 x float> %nvptx.packed19, %nvptx.packed22
  %nvptx.extract24 = extractelement <2 x float> %nvptx.packed23, i64 0
  %nvptx.extract25 = extractelement <2 x float> %nvptx.packed23, i64 1
  %57 = fadd float %.0.i, f0x322BCC77
  %58 = fadd float %.0.i3, f0x322BCC77
  %59 = fmul float %22, 9.999700e-01
  %60 = fmul float %23, 9.999700e-01
  %61 = fmul float %nvptx.extract25, f0x3B4B9954
  %62 = fmul float %nvptx.extract24, f0x3B4B9954
  %63 = tail call float @llvm.nvvm.div.full(float %61, float %57)
  %64 = tail call float @llvm.nvvm.div.full(float %62, float %58)
  %65 = fsub float %59, %63
  %66 = fsub float %60, %64
  %67 = getelementptr [4 x i8], ptr addrspace(1) %4, i64 %20
  %68 = bitcast float %65 to i32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %68, ptr addrspace(1) %67, i1 %18) #4
  %69 = bitcast float %66 to i32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 4 ], { $0 };", "r,l,b"(i32 %69, ptr addrspace(1) %67, i1 %19) #4
  %70 = getelementptr [4 x i8], ptr addrspace(1) %5, i64 %20
  %71 = bitcast float %nvptx.extract25 to i32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %71, ptr addrspace(1) %70, i1 %18) #4
  %72 = bitcast float %nvptx.extract24 to i32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 4 ], { $0 };", "r,l,b"(i32 %72, ptr addrspace(1) %70, i1 %19) #4
  %73 = getelementptr [4 x i8], ptr addrspace(1) %6, i64 %20
  %bc = bitcast <2 x float> %nvptx.packed12 to <2 x i32>
  %74 = extractelement <2 x i32> %bc, i64 0
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %74, ptr addrspace(1) %73, i1 %18) #4
  %bc26 = bitcast <2 x float> %nvptx.packed12 to <2 x i32>
  %75 = extractelement <2 x i32> %bc26, i64 1
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 4 ], { $0 };", "r,l,b"(i32 %75, ptr addrspace(1) %73, i1 %19) #4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.div.full(float, float) #2

declare i32 @__nvvm_reflect(ptr) local_unnamed_addr #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.ftz.f(float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.ftz.f(float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.f(float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.f(float) #2

attributes #0 = { nounwind "nvvm.reqntid"="128" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}

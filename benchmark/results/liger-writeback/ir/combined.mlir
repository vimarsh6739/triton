module {
  tt.func public @revdiffe_swiglu_forward_kernel_tiled(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: f32, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024xbf16>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_program_id y : i32
    %3 = arith.extsi %arg6 : i32 to i64
    %4 = arith.muli %1, %3 : i64
    %5 = tt.addptr %arg1, %4 : !tt.ptr<bf16>, i64
    %6 = tt.addptr %arg0, %4 : !tt.ptr<bf16>, i64
    %7 = tt.addptr %arg3, %4 : !tt.ptr<bf16>, i64
    %8 = tt.addptr %arg2, %4 : !tt.ptr<bf16>, i64
    %9 = tt.addptr %arg5, %4 : !tt.ptr<bf16>, i64
    %10 = tt.addptr %arg4, %4 : !tt.ptr<bf16>, i64
    %11 = arith.muli %2, %c1024_i32 : i32
    %12 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %13 = tt.splat %11 : i32 -> tensor<1024xi32>
    %14 = arith.addi %13, %12 : tensor<1024xi32>
    %15 = tt.splat %arg8 : i32 -> tensor<1024xi32>
    %16 = arith.cmpi slt, %14, %15 : tensor<1024xi32>
    %17 = tt.splat %5 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %18 = tt.splat %6 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %19 = tt.addptr %17, %14 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %20 = tt.addptr %18, %14 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %21 = tt.load %20, %16, %cst_1 : tensor<1024x!tt.ptr<bf16>>
    %22 = arith.extf %21 : tensor<1024xbf16> to tensor<1024xf32>
    %23 = tt.splat %arg7 : f32 -> tensor<1024xf32>
    %24 = arith.mulf %22, %23 : tensor<1024xf32>
    %25 = tt.splat %7 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %26 = tt.splat %8 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %27 = tt.addptr %25, %14 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %28 = tt.addptr %26, %14 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %29 = tt.load %28, %16, %cst_1 : tensor<1024x!tt.ptr<bf16>>
    %30 = arith.negf %24 : tensor<1024xf32>
    %31 = math.exp %30 : tensor<1024xf32>
    %32 = arith.addf %31, %cst : tensor<1024xf32>
    %33 = arith.divf %cst, %32 : tensor<1024xf32>
    %34 = arith.mulf %24, %33 : tensor<1024xf32>
    %35 = arith.truncf %34 : tensor<1024xf32> to tensor<1024xbf16>
    %36 = arith.mulf %35, %29 : tensor<1024xbf16>
    %37 = tt.splat %9 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %38 = tt.splat %10 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %39 = tt.addptr %37, %14 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %40 = tt.addptr %38, %14 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    tt.store %40, %36, %16 : tensor<1024x!tt.ptr<bf16>>
    %41 = tt.load %39, %16, %cst_1 : tensor<1024x!tt.ptr<bf16>>
    gpu.barrier
    tt.store %39, %cst_1, %16 : tensor<1024x!tt.ptr<bf16>>
    %42 = arith.mulf %41, %29 fastmath<fast> : tensor<1024xbf16>
    %43 = arith.mulf %41, %35 fastmath<fast> : tensor<1024xbf16>
    %44 = arith.extf %42 fastmath<fast> : tensor<1024xbf16> to tensor<1024xf32>
    %45 = arith.mulf %44, %33 fastmath<fast> : tensor<1024xf32>
    %46 = arith.mulf %44, %24 fastmath<fast> : tensor<1024xf32>
    %47 = arith.divf %46, %32 fastmath<fast> : tensor<1024xf32>
    %48 = arith.divf %cst, %32 fastmath<fast> : tensor<1024xf32>
    %49 = arith.mulf %47, %48 fastmath<fast> : tensor<1024xf32>
    %50 = arith.subf %cst_0, %49 fastmath<fast> : tensor<1024xf32>
    %51 = math.exp %30 fastmath<fast> : tensor<1024xf32>
    %52 = arith.mulf %50, %51 fastmath<fast> : tensor<1024xf32>
    %53 = arith.subf %45, %52 fastmath<fast> : tensor<1024xf32>
    %54 = tt.atomic_rmw fadd, relaxed, gpu, %27, %43, %16 : (tensor<1024x!tt.ptr<bf16>>, tensor<1024xbf16>, tensor<1024xi1>) -> tensor<1024xbf16>
    %55 = arith.mulf %53, %23 fastmath<fast> : tensor<1024xf32>
    %56 = arith.truncf %55 fastmath<fast> : tensor<1024xf32> to tensor<1024xbf16>
    %57 = tt.atomic_rmw fadd, relaxed, gpu, %19, %56, %16 : (tensor<1024x!tt.ptr<bf16>>, tensor<1024xbf16>, tensor<1024xi1>) -> tensor<1024xbf16>
    tt.return
  }
}

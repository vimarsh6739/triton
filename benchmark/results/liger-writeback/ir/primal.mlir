module {
  tt.func public @_swiglu_forward_kernel_tiled(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: f32, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1024xbf16>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_program_id y : i32
    %3 = arith.extsi %arg3 : i32 to i64
    %4 = arith.muli %1, %3 : i64
    %5 = tt.addptr %arg0, %4 : !tt.ptr<bf16>, i64
    %6 = tt.addptr %arg1, %4 : !tt.ptr<bf16>, i64
    %7 = tt.addptr %arg2, %4 : !tt.ptr<bf16>, i64
    %8 = arith.muli %2, %c1024_i32 : i32
    %9 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %10 = tt.splat %8 : i32 -> tensor<1024xi32>
    %11 = arith.addi %10, %9 : tensor<1024xi32>
    %12 = tt.splat %arg5 : i32 -> tensor<1024xi32>
    %13 = arith.cmpi slt, %11, %12 : tensor<1024xi32>
    %14 = tt.splat %5 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %15 = tt.addptr %14, %11 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %16 = tt.load %15, %13, %cst_0 : tensor<1024x!tt.ptr<bf16>>
    %17 = arith.extf %16 : tensor<1024xbf16> to tensor<1024xf32>
    %18 = tt.splat %arg4 : f32 -> tensor<1024xf32>
    %19 = arith.mulf %17, %18 : tensor<1024xf32>
    %20 = tt.splat %6 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %21 = tt.addptr %20, %11 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %22 = tt.load %21, %13, %cst_0 : tensor<1024x!tt.ptr<bf16>>
    %23 = arith.negf %19 : tensor<1024xf32>
    %24 = math.exp %23 : tensor<1024xf32>
    %25 = arith.addf %24, %cst : tensor<1024xf32>
    %26 = arith.divf %cst, %25 : tensor<1024xf32>
    %27 = arith.mulf %19, %26 : tensor<1024xf32>
    %28 = arith.truncf %27 : tensor<1024xf32> to tensor<1024xbf16>
    %29 = arith.mulf %28, %22 : tensor<1024xbf16>
    %30 = tt.splat %7 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %31 = tt.addptr %30, %11 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    tt.store %31, %29, %13 : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}

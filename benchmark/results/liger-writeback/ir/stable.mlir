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
    %10 = arith.muli %2, %c1024_i32 : i32
    %11 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %12 = tt.splat %10 : i32 -> tensor<1024xi32>
    %13 = arith.addi %12, %11 : tensor<1024xi32>
    %14 = tt.splat %arg8 : i32 -> tensor<1024xi32>
    %15 = arith.cmpi slt, %13, %14 : tensor<1024xi32>
    %16 = tt.splat %5 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %17 = tt.splat %6 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %18 = tt.addptr %16, %13 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %19 = tt.addptr %17, %13 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %20 = tt.load %19, %15, %cst_1 : tensor<1024x!tt.ptr<bf16>>
    %21 = arith.extf %20 : tensor<1024xbf16> to tensor<1024xf32>
    %22 = tt.splat %arg7 : f32 -> tensor<1024xf32>
    %23 = arith.mulf %21, %22 : tensor<1024xf32>
    %24 = tt.splat %7 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %25 = tt.splat %8 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %26 = tt.addptr %24, %13 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %27 = tt.addptr %25, %13 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %28 = tt.load %27, %15, %cst_1 : tensor<1024x!tt.ptr<bf16>>
    %29 = arith.negf %23 : tensor<1024xf32>
    %30 = arith.cmpf oge, %23, %cst_0 : tensor<1024xf32>
    %31 = arith.select %30, %29, %23 : tensor<1024xi1>, tensor<1024xf32>
    %32 = math.exp %31 : tensor<1024xf32>
    %33 = arith.addf %32, %cst : tensor<1024xf32>
    %34 = arith.select %30, %cst, %32 : tensor<1024xi1>, tensor<1024xf32>
    %35 = arith.divf %34, %33 : tensor<1024xf32>
    %36 = arith.mulf %23, %35 : tensor<1024xf32>
    %37 = arith.truncf %36 : tensor<1024xf32> to tensor<1024xbf16>
    %38 = tt.splat %9 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %39 = tt.addptr %38, %13 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %40 = tt.load %39, %15, %cst_1 : tensor<1024x!tt.ptr<bf16>>
    %41 = arith.mulf %40, %28 fastmath<fast> : tensor<1024xbf16>
    %42 = arith.mulf %40, %37 fastmath<fast> : tensor<1024xbf16>
    %43 = arith.extf %41 fastmath<fast> : tensor<1024xbf16> to tensor<1024xf32>
    %44 = arith.mulf %43, %35 fastmath<fast> : tensor<1024xf32>
    %45 = arith.mulf %43, %23 fastmath<fast> : tensor<1024xf32>
    %46 = arith.divf %45, %33 fastmath<fast> : tensor<1024xf32>
    %47 = arith.divf %34, %33 fastmath<fast> : tensor<1024xf32>
    %48 = arith.mulf %46, %47 fastmath<fast> : tensor<1024xf32>
    %49 = arith.subf %cst_0, %48 fastmath<fast> : tensor<1024xf32>
    %50 = arith.select %30, %cst_0, %46 : tensor<1024xi1>, tensor<1024xf32>
    %51 = arith.addf %50, %49 fastmath<fast> : tensor<1024xf32>
    %52 = math.exp %31 fastmath<fast> : tensor<1024xf32>
    %53 = arith.mulf %51, %52 fastmath<fast> : tensor<1024xf32>
    %54 = arith.select %30, %53, %cst_0 : tensor<1024xi1>, tensor<1024xf32>
    %55 = arith.select %30, %cst_0, %53 : tensor<1024xi1>, tensor<1024xf32>
    %56 = arith.addf %44, %55 fastmath<fast> : tensor<1024xf32>
    %57 = arith.subf %56, %54 fastmath<fast> : tensor<1024xf32>
    tt.store %26, %42, %15 : tensor<1024x!tt.ptr<bf16>>
    %58 = arith.mulf %57, %22 fastmath<fast> : tensor<1024xf32>
    %59 = arith.truncf %58 fastmath<fast> : tensor<1024xf32> to tensor<1024xbf16>
    tt.store %18, %59, %15 : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}

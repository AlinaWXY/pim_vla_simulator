# VLM calibration audit

Passed: 106.893161 ms versus 104.950496 ms; error 1.851%.
Candidate 1: `{"alpha": 1.0, "window": 1024, "write_high": 768, "write_low": 256}`.

Kernel error median 0.343%; p90 8.648%.
Unchanged measured duration sum 105.220578 ms; native excess 1.672583 ms.
GPU read/write bytes: 11,804,375,072 / 8,028,931,008; row hit fraction 98.070%.
Mean external CA/DQ utilization: 36.501% / 67.984%.

All 2357 units, bytes, DQ cycles, timestamps and frozen assessment checks passed.
CA includes refresh on both ranks. Native traffic is simulated; alpha is a fitted writeback fraction, not a measured DRAM counter.
Total error is the acceptance gate. Kernel median/p90 and the following rankings are diagnostics.

| Kernel name | Count | Measured ms | Native ms | Excess ms |
|---|---:|---:|---:|---:|
| nvjet_sm110_tst_128x128_64x6_1x2_h_bz_bias_TNT | 324 | 5.250438 | 5.723322 | 0.472884 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(int, T2, T3) | 36 | 0.985075 | 1.258160 | 0.273085 |
| nvjet_sm110_tst_224x128_64x7_1x2_2cta_h_bz_bias_TNN | 81 | 4.469530 | 4.613650 | 0.144120 |
| nvjet_sm110_tst_256x200_64x3_2x1_v_bz_TNT | 90 | 19.354368 | 19.486122 | 0.131754 |
| nvjet_sm110_tst_144x128_64x6_4x1_v_bz_bias_TNN | 81 | 4.788269 | 4.916965 | 0.128696 |
| nvjet_sm110_tst_128x248_64x4_2x1_v_bz_TNT | 18 | 1.101933 | 1.168940 | 0.067007 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase &, at::native::GeluType)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 4)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3) | 99 | 4.975052 | 5.040593 | 0.065541 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) | 72 | 3.006758 | 3.067965 | 0.061207 |
| nvjet_sm110_tst_256x248_64x3_1x2_h_bz_TNT | 36 | 15.547462 | 15.606022 | 0.058559 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3) | 60 | 3.451006 | 3.499213 | 0.048206 |
| void <unnamed>::softmax_warp_forward<float, float, float, (int)10, (bool)0, (bool)0>(T2 *, const T1 *, int, int, int, const bool *, int, bool) | 18 | 4.376384 | 4.408881 | 0.032497 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3) | 18 | 6.404570 | 6.436796 | 0.032227 |
| nvjet_sm110_tst_128x200_64x5_2x1_v_bz_NNT | 18 | 0.686554 | 0.713869 | 0.027315 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) | 18 | 5.384864 | 5.407322 | 0.022458 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3) | 19 | 1.033971 | 1.050053 | 0.016081 |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3) | 37 | 1.214029 | 1.229451 | 0.015422 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 12)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) | 507 | 5.091936 | 5.103899 | 0.011963 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7) | 39 | 1.443539 | 1.454983 | 0.011444 |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) | 37 | 1.418067 | 1.429137 | 0.011070 |
| void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueType<(unsigned int)2>, unsigned int, (int)2, (int)128, (int)1, (int)16, (int)8>(char *, at::native::<unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, at::native::<unnamed>::TensorSizeStride<T2, (unsigned int)4>, int, T2) | 1 | 0.012941 | 0.023383 | 0.010443 |

Full per-ordinal errors, grouped errors and channel details are in the adjacent CSV files.

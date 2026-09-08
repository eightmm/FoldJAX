// Development-only native linear dispatch. Not part of the runtime wheel.
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <limits>
#include <string>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {
struct Resources {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  ~Resources() {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (d) cublasLtMatrixLayoutDestroy(d);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
    if (handle) cublasLtDestroy(handle);
  }
};

#define CHECK_BLAS(call) do { \
  cublasStatus_t status = (call); \
  if (status != CUBLAS_STATUS_SUCCESS) \
    return ffi::Error::Internal(std::string(#call) + ": " + std::to_string(status)); \
} while (false)

template <ffi::DataType Dtype, cudaDataType_t CudaDtype>
ffi::Error NativeLinear(cudaStream_t stream,
                        ffi::BufferR2<Dtype> x,
                        ffi::BufferR2<Dtype> weight,
                        ffi::ResultBufferR2<Dtype> output,
                        ffi::ResultBufferR1<ffi::U8> scratch) {
  const int64_t m = x.dimensions()[0], k = x.dimensions()[1];
  const int64_t n = weight.dimensions()[0];
  if (m <= 0 || n <= 0 || k <= 0 ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || weight.dimensions()[1] != k ||
      output->dimensions()[0] != m || output->dimensions()[1] != n ||
      scratch->dimensions()[0] != 32 * 1024 * 1024) {
    return ffi::Error::InvalidArgument("invalid positive native linear dimensions");
  }
  Resources r;
  CHECK_BLAS(cublasLtCreate(&r.handle));
  CHECK_BLAS(cublasLtMatmulDescCreate(&r.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  // Native row-major x @ weight.T is column-major weight @ x.T. A singleton
  // output channel has native contiguous A=[1,K], avoiding a redundant transpose.
  cublasOperation_t transa = n == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
  CHECK_BLAS(cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_TRANSA,
                                           &transa, sizeof(transa)));
  CHECK_BLAS(cublasLtMatrixLayoutCreate(&r.a, CudaDtype,
                                       n == 1 ? 1 : k, n == 1 ? k : n,
                                       n == 1 ? 1 : k));
  CHECK_BLAS(cublasLtMatrixLayoutCreate(&r.b, CudaDtype, k, m, k));
  CHECK_BLAS(cublasLtMatrixLayoutCreate(&r.d, CudaDtype, n, m, n));
  CHECK_BLAS(cublasLtMatmulPreferenceCreate(&r.preference));
  const size_t workspace_limit = 32 * 1024 * 1024;
  CHECK_BLAS(cublasLtMatmulPreferenceSetAttribute(r.preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit)));
  const uint32_t alignment = 16;
  for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
    CHECK_BLAS(cublasLtMatmulPreferenceSetAttribute(r.preference, attr,
                                                   &alignment, sizeof(alignment)));
  }
  cublasLtMatmulHeuristicResult_t heuristic{};
  int count = 0;
  CHECK_BLAS(cublasLtMatmulAlgoGetHeuristic(r.handle, r.op, r.a, r.b, r.d, r.d,
                                          r.preference, 1, &heuristic, &count));
  if (count != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS ||
      heuristic.workspaceSize > workspace_limit) {
    return ffi::Error::Internal("no supported first native cuBLASLt heuristic");
  }
  // XLA owns this scratch result's lifetime, just as for a library GEMM custom
  // call. It is never returned by the Python wrapper or read as model data.
  void* workspace = heuristic.workspaceSize ? scratch->typed_data() : nullptr;
  const float alpha = 1, beta = 0;
  CHECK_BLAS(cublasLtMatmul(r.handle, r.op, &alpha,
      weight.typed_data(), r.a, x.typed_data(), r.b, &beta,
      output->typed_data(), r.d, output->typed_data(), r.d,
      &heuristic.algo, workspace, heuristic.workspaceSize, stream));
  return ffi::Error::Success();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(FoldjaxBenchNativeLinear,
  (NativeLinear<ffi::BF16, CUDA_R_16BF>),
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::BufferR2<ffi::BF16>>().Arg<ffi::BufferR2<ffi::BF16>>()
      .Ret<ffi::BufferR2<ffi::BF16>>().Ret<ffi::BufferR1<ffi::U8>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(FoldjaxBenchNativeLinearF32,
  (NativeLinear<ffi::F32, CUDA_R_32F>),
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::BufferR2<ffi::F32>>().Arg<ffi::BufferR2<ffi::F32>>()
      .Ret<ffi::BufferR2<ffi::F32>>().Ret<ffi::BufferR1<ffi::U8>>());

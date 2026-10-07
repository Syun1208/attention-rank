#pragma once

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string>

#include "attnrank/core.hpp"

namespace attnrank {

void cuda_check(cudaError_t status, const char* expression, const char* file, int line);
void cublas_check(cublasStatus_t status, const char* expression, const char* file, int line);

#define ATTNRANK_CUDA_CHECK(expr) ::attnrank::cuda_check((expr), #expr, __FILE__, __LINE__)
#define ATTNRANK_CUBLAS_CHECK(expr) ::attnrank::cublas_check((expr), #expr, __FILE__, __LINE__)

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t byte_size);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void allocate(std::size_t byte_size);
  void ensure_capacity(std::size_t byte_size);
  void release();

  void* raw() { return pointer_; }
  const void* raw() const { return pointer_; }
  float* as_float() { return static_cast<float*>(pointer_); }
  const float* as_float() const { return static_cast<const float*>(pointer_); }
  int* as_int() { return static_cast<int*>(pointer_); }
  const int* as_int() const { return static_cast<const int*>(pointer_); }
  double* as_double() { return static_cast<double*>(pointer_); }
  std::size_t byte_size() const { return byte_size_; }
  bool empty() const { return pointer_ == nullptr; }

  void upload(const void* host, std::size_t bytes, std::size_t byte_offset = 0);
  void download(void* host, std::size_t bytes, std::size_t byte_offset = 0) const;
  void zero();

 private:
  void* pointer_ = nullptr;
  std::size_t byte_size_ = 0;
};

struct CudaContextConfig {
  int device_index = -1;
};

class CudaContext {
 public:
  explicit CudaContext(const CudaContextConfig& config);
  ~CudaContext();

  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  cublasHandle_t blas() const { return blas_; }
  int device_index() const { return config_.device_index; }
  std::string device_name() const;
  std::size_t free_memory() const;
  std::size_t total_memory() const;
  void synchronize() const;

 private:
  CudaContextConfig config_;
  cublasHandle_t blas_ = nullptr;
};

struct RotaryShape {
  int token_count = 0;
  int query_heads = 0;
  int key_heads = 0;
  int head_dim = 0;
  int position_offset = 0;
  float theta = 0.0f;
};

struct KvCacheLayout {
  int token_count = 0;
  int position_offset = 0;
  int head_count = 0;
  int head_dim = 0;
  int max_sequence_length = 0;
};

struct SoftmaxShape {
  int head_count = 0;
  int query_count = 0;
  int key_count = 0;
  int position_offset = 0;
  int sliding_window = 0;
};

struct DocumentAttentionReduction {
  int head_count = 0;
  int query_count = 0;
  int key_count = 0;
  int query_position_offset = 0;
  int query_begin = 0;
  int query_end = 0;
  const int* span_begins = nullptr;
  const int* span_ends = nullptr;
  int span_count = 0;
  bool normalize_by_length = false;
  double* accumulator = nullptr;
};

namespace kernels {

void convert_to_float(float* destination, const void* source, DType source_dtype, std::size_t count);

void convert_to_half(void* destination, const void* source, DType source_dtype, std::size_t count);

void cast_float_to_half(void* destination, const float* source, std::size_t count);

void embedding_lookup(float* destination, const void* table, const int* token_ids, int token_count,
                      int hidden_size);

void rms_norm(float* destination, const float* source, const float* weight, int token_count,
              int hidden_size, float epsilon);

void rotary_embedding(float* queries, float* keys, const RotaryShape& shape);

void softmax_causal(float* scores, const SoftmaxShape& shape);

void silu_multiply(float* destination, const float* gate, const float* up, std::size_t count);

void add_in_place(float* destination, const float* source, std::size_t count);

void add_row_bias(float* destination, const float* bias, int rows, int columns);

void scatter_kv_cache(float* key_cache, float* value_cache, const float* keys, const float* values,
                      const KvCacheLayout& layout);

void reduce_document_attention(const float* scores, const DocumentAttentionReduction& reduction);

}  // namespace kernels

}  // namespace attnrank

#include "cuda_runtime.hpp"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <utility>

namespace attnrank {

namespace {

constexpr int kReductionThreads = 256;
constexpr std::size_t kElementwiseThreads = 256;
constexpr int kEmbeddingThreads = 256;
constexpr int kHeadThreads = 64;
constexpr float kSoftmaxDenominatorFloor = 1e-20f;

__global__ void convert_half_kernel(float* destination, const __half* source, std::size_t count) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < count) destination[index] = __half2float(source[index]);
}

__global__ void convert_bfloat16_kernel(float* destination, const __nv_bfloat16* source,
                                        std::size_t count) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < count) destination[index] = __bfloat162float(source[index]);
}

__global__ void copy_float_kernel(float* destination, const float* source, std::size_t count) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < count) destination[index] = source[index];
}

__global__ void half_from_half_kernel(__half* destination, const __half* source, std::size_t count) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < count) destination[index] = source[index];
}

__global__ void half_from_bfloat16_kernel(__half* destination, const __nv_bfloat16* source,
                                          std::size_t count) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < count) destination[index] = __float2half(__bfloat162float(source[index]));
}

__global__ void half_from_float_kernel(__half* destination, const float* source, std::size_t count) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < count) destination[index] = __float2half(source[index]);
}

__global__ void embedding_lookup_kernel(float* destination, const __half* table, const int* token_ids,
                                        int hidden_size) {
  const int token = blockIdx.x;
  const int source_row = token_ids[token];
  for (int channel = threadIdx.x; channel < hidden_size; channel += blockDim.x) {
    destination[static_cast<std::size_t>(token) * hidden_size + channel] =
        __half2float(table[static_cast<std::size_t>(source_row) * hidden_size + channel]);
  }
}

__global__ void rms_norm_kernel(float* destination, const float* source, const float* weight,
                                int hidden_size, float epsilon) {
  extern __shared__ float shared[];
  const int token = blockIdx.x;
  const float* row = source + static_cast<std::size_t>(token) * hidden_size;
  float* output_row = destination + static_cast<std::size_t>(token) * hidden_size;

  float partial = 0.0f;
  for (int channel = threadIdx.x; channel < hidden_size; channel += blockDim.x) {
    partial += row[channel] * row[channel];
  }
  shared[threadIdx.x] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }

  const float scale = rsqrtf(shared[0] / hidden_size + epsilon);
  for (int channel = threadIdx.x; channel < hidden_size; channel += blockDim.x) {
    output_row[channel] = row[channel] * scale * weight[channel];
  }
}

__global__ void rotary_embedding_kernel(float* queries, float* keys, int query_heads, int key_heads,
                                        int head_dim, int position_offset, float theta) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int half_dim = head_dim / 2;
  const int position = position_offset + token;
  const int total_heads = max(query_heads, key_heads);
  if (head >= total_heads) return;

  for (int index = threadIdx.x; index < half_dim; index += blockDim.x) {
    const float inverse_frequency = powf(theta, -2.0f * index / static_cast<float>(head_dim));
    const float angle = position * inverse_frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);

    if (head < query_heads) {
      float* row = queries + (static_cast<std::size_t>(token) * query_heads + head) * head_dim;
      const float low = row[index];
      const float high = row[index + half_dim];
      row[index] = low * cosine - high * sine;
      row[index + half_dim] = high * cosine + low * sine;
    }
    if (head < key_heads) {
      float* row = keys + (static_cast<std::size_t>(token) * key_heads + head) * head_dim;
      const float low = row[index];
      const float high = row[index + half_dim];
      row[index] = low * cosine - high * sine;
      row[index + half_dim] = high * cosine + low * sine;
    }
  }
}

__global__ void softmax_causal_kernel(float* scores, int query_count, int key_count,
                                      int position_offset, int sliding_window) {
  extern __shared__ float shared[];
  const int head = blockIdx.y;
  const int query = blockIdx.x;
  if (query >= query_count) return;

  float* row = scores + (static_cast<std::size_t>(head) * query_count + query) * key_count;
  const int limit = position_offset + query + 1;
  const int floor = sliding_window > 0 ? limit - sliding_window : 0;

  float local_max = -INFINITY;
  for (int key = threadIdx.x; key < key_count; key += blockDim.x) {
    if (key < limit && key >= floor) local_max = fmaxf(local_max, row[key]);
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  const float maximum = shared[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int key = threadIdx.x; key < key_count; key += blockDim.x) {
    if (key < limit && key >= floor) {
      const float value = __expf(row[key] - maximum);
      row[key] = value;
      local_sum += value;
    } else {
      row[key] = 0.0f;
    }
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }

  const float inverse_sum = 1.0f / fmaxf(shared[0], kSoftmaxDenominatorFloor);
  for (int key = threadIdx.x; key < key_count; key += blockDim.x) {
    row[key] *= inverse_sum;
  }
}

__global__ void silu_multiply_kernel(float* destination, const float* gate, const float* up,
                                     std::size_t count) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= count) return;
  const float value = gate[index];
  destination[index] = (value / (1.0f + __expf(-value))) * up[index];
}

__global__ void add_in_place_kernel(float* destination, const float* source, std::size_t count) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < count) destination[index] += source[index];
}

__global__ void add_row_bias_kernel(float* destination, const float* bias, int rows, int columns) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= columns) return;
  const float value = bias[column];
  for (int row = 0; row < rows; ++row) {
    destination[static_cast<std::size_t>(row) * columns + column] += value;
  }
}

__global__ void scatter_kv_cache_kernel(float* key_cache, float* value_cache, const float* keys,
                                        const float* values, int position_offset, int head_count,
                                        int head_dim, int max_sequence_length) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const std::size_t source_base =
      (static_cast<std::size_t>(token) * head_count + head) * head_dim;
  const std::size_t destination_base =
      (static_cast<std::size_t>(head) * max_sequence_length + position_offset + token) * head_dim;

  for (int channel = threadIdx.x; channel < head_dim; channel += blockDim.x) {
    key_cache[destination_base + channel] = keys[source_base + channel];
    value_cache[destination_base + channel] = values[source_base + channel];
  }
}

__global__ void reduce_document_attention_kernel(const float* scores, int head_count,
                                                 int query_count, int key_count,
                                                 int query_position_offset, int query_begin,
                                                 int query_end, const int* span_begins,
                                                 const int* span_ends, bool normalize_by_length,
                                                 double* accumulator) {
  extern __shared__ double shared_double[];
  const int span = blockIdx.x;
  const int begin = span_begins[span];
  const int end = min(span_ends[span], key_count);

  double partial = 0.0;

  for (int flat = threadIdx.x; flat < head_count * query_count; flat += blockDim.x) {
    const int head = flat / query_count;
    const int query = flat % query_count;
    const int absolute = query_position_offset + query;
    if (absolute < query_begin || absolute >= query_end) continue;

    const float* row = scores + (static_cast<std::size_t>(head) * query_count + query) * key_count;
    double sum = 0.0;
    for (int key = begin; key < end; ++key) sum += row[key];
    if (normalize_by_length && end > begin) sum /= static_cast<double>(end - begin);
    partial += sum;
  }

  shared_double[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared_double[threadIdx.x] += shared_double[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) accumulator[span] += shared_double[0];
}

std::size_t grid_for(std::size_t count, std::size_t block) { return (count + block - 1) / block; }

std::size_t elementwise_grid(std::size_t count) { return grid_for(count, kElementwiseThreads); }

}  // namespace

void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) return;
  throw Error(std::string("CUDA error at ") + file + ":" + std::to_string(line) + " (" +
                      expression + "): " + cudaGetErrorString(status));
}

void cublas_check(cublasStatus_t status, const char* expression, const char* file, int line) {
  if (status == CUBLAS_STATUS_SUCCESS) return;
  throw Error(std::string("cuBLAS error ") + std::to_string(static_cast<int>(status)) +
                      " at " + file + ":" + std::to_string(line) + " (" + expression + ")");
}

DeviceBuffer::DeviceBuffer(std::size_t byte_size) { allocate(byte_size); }

DeviceBuffer::~DeviceBuffer() { release(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : pointer_(other.pointer_), byte_size_(other.byte_size_) {
  other.pointer_ = nullptr;
  other.byte_size_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    release();
    pointer_ = other.pointer_;
    byte_size_ = other.byte_size_;
    other.pointer_ = nullptr;
    other.byte_size_ = 0;
  }
  return *this;
}

void DeviceBuffer::allocate(std::size_t byte_size) {
  release();
  if (byte_size == 0) return;
  ATTNRANK_CUDA_CHECK(cudaMalloc(&pointer_, byte_size));
  byte_size_ = byte_size;
}

void DeviceBuffer::ensure_capacity(std::size_t byte_size) {
  if (byte_size_ < byte_size) allocate(byte_size);
}

void DeviceBuffer::release() {
  if (pointer_ != nullptr) {
    cudaFree(pointer_);
    pointer_ = nullptr;
    byte_size_ = 0;
  }
}

void DeviceBuffer::upload(const void* host, std::size_t bytes, std::size_t byte_offset) {
  if (byte_offset + bytes > byte_size_) throw Error("device upload out of range");
  ATTNRANK_CUDA_CHECK(cudaMemcpy(static_cast<std::uint8_t*>(pointer_) + byte_offset, host, bytes,
                                 cudaMemcpyHostToDevice));
}

void DeviceBuffer::download(void* host, std::size_t bytes, std::size_t byte_offset) const {
  if (byte_offset + bytes > byte_size_) throw Error("device download out of range");
  ATTNRANK_CUDA_CHECK(cudaMemcpy(host, static_cast<const std::uint8_t*>(pointer_) + byte_offset,
                                 bytes, cudaMemcpyDeviceToHost));
}

void DeviceBuffer::zero() {
  if (pointer_ != nullptr) ATTNRANK_CUDA_CHECK(cudaMemset(pointer_, 0, byte_size_));
}

CudaContext::CudaContext(const CudaContextConfig& config) : config_(config) {
  int device_count = 0;
  ATTNRANK_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) throw Error("no CUDA device available");

  if (config_.device_index < 0) {
    std::size_t best_free = 0;
    int best_device = 0;
    for (int candidate = 0; candidate < device_count; ++candidate) {
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      if (cudaSetDevice(candidate) != cudaSuccess) continue;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) continue;
      if (free_bytes > best_free) {
        best_free = free_bytes;
        best_device = candidate;
      }
    }
    config_.device_index = best_device;
  }

  if (config_.device_index >= device_count) {
    throw Error("requested CUDA device " + std::to_string(config_.device_index) +
                        " but only " + std::to_string(device_count) + " are present");
  }
  ATTNRANK_CUDA_CHECK(cudaSetDevice(config_.device_index));
  ATTNRANK_CUBLAS_CHECK(cublasCreate(&blas_));
  ATTNRANK_CUBLAS_CHECK(cublasSetMathMode(blas_, CUBLAS_DEFAULT_MATH));
}

CudaContext::~CudaContext() {
  if (blas_ != nullptr) cublasDestroy(blas_);
}

std::string CudaContext::device_name() const {
  cudaDeviceProp properties{};
  ATTNRANK_CUDA_CHECK(cudaGetDeviceProperties(&properties, config_.device_index));
  return properties.name;
}

std::size_t CudaContext::free_memory() const {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  ATTNRANK_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return free_bytes;
}

std::size_t CudaContext::total_memory() const {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  ATTNRANK_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return total_bytes;
}

void CudaContext::synchronize() const { ATTNRANK_CUDA_CHECK(cudaDeviceSynchronize()); }

namespace kernels {

void convert_to_float(float* destination, const void* source, DType source_dtype,
                      std::size_t count) {
  if (count == 0) return;
  const std::size_t block = kElementwiseThreads;
  const std::size_t grid = elementwise_grid(count);
  switch (source_dtype) {
    case DType::F16:
      convert_half_kernel<<<grid, block>>>(destination, static_cast<const __half*>(source), count);
      break;
    case DType::BF16:
      convert_bfloat16_kernel<<<grid, block>>>(destination,
                                               static_cast<const __nv_bfloat16*>(source), count);
      break;
    case DType::F32:
      copy_float_kernel<<<grid, block>>>(destination, static_cast<const float*>(source), count);
      break;
    default:
      throw Error(std::string("cannot convert dtype ") + dtype_name(source_dtype) +
                          " to float");
  }
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void convert_to_half(void* destination, const void* source, DType source_dtype,
                     std::size_t count) {
  if (count == 0) return;
  const std::size_t block = kElementwiseThreads;
  const std::size_t grid = elementwise_grid(count);
  __half* typed = static_cast<__half*>(destination);
  switch (source_dtype) {
    case DType::F16:
      half_from_half_kernel<<<grid, block>>>(typed, static_cast<const __half*>(source), count);
      break;
    case DType::BF16:
      half_from_bfloat16_kernel<<<grid, block>>>(typed, static_cast<const __nv_bfloat16*>(source),
                                                 count);
      break;
    case DType::F32:
      half_from_float_kernel<<<grid, block>>>(typed, static_cast<const float*>(source), count);
      break;
    default:
      throw Error(std::string("cannot convert dtype ") + dtype_name(source_dtype) +
                          " to half");
  }
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void cast_float_to_half(void* destination, const float* source, std::size_t count) {
  if (count == 0) return;
  half_from_float_kernel<<<elementwise_grid(count), kElementwiseThreads>>>(
      static_cast<__half*>(destination), source, count);
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void embedding_lookup(float* destination, const void* table, const int* token_ids, int token_count,
                      int hidden_size) {
  if (token_count == 0) return;
  embedding_lookup_kernel<<<token_count, kEmbeddingThreads>>>(
      destination, static_cast<const __half*>(table), token_ids, hidden_size);
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void rms_norm(float* destination, const float* source, const float* weight, int token_count,
              int hidden_size, float epsilon) {
  if (token_count == 0) return;
  const std::size_t shared = kReductionThreads * sizeof(float);
  rms_norm_kernel<<<token_count, kReductionThreads, shared>>>(destination, source, weight,
                                                              hidden_size, epsilon);
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void rotary_embedding(float* queries, float* keys, const RotaryShape& shape) {
  if (shape.token_count == 0) return;
  const dim3 grid(shape.token_count, std::max(shape.query_heads, shape.key_heads));
  rotary_embedding_kernel<<<grid, kHeadThreads>>>(queries, keys, shape.query_heads, shape.key_heads,
                                                  shape.head_dim, shape.position_offset, shape.theta);
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void softmax_causal(float* scores, const SoftmaxShape& shape) {
  if (shape.query_count == 0 || shape.head_count == 0) return;
  const dim3 grid(shape.query_count, shape.head_count);
  const std::size_t shared = kReductionThreads * sizeof(float);
  softmax_causal_kernel<<<grid, kReductionThreads, shared>>>(
      scores, shape.query_count, shape.key_count, shape.position_offset, shape.sliding_window);
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void silu_multiply(float* destination, const float* gate, const float* up, std::size_t count) {
  if (count == 0) return;
  silu_multiply_kernel<<<elementwise_grid(count), kElementwiseThreads>>>(destination, gate, up, count);
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void add_in_place(float* destination, const float* source, std::size_t count) {
  if (count == 0) return;
  add_in_place_kernel<<<elementwise_grid(count), kElementwiseThreads>>>(destination, source, count);
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void add_row_bias(float* destination, const float* bias, int rows, int columns) {
  if (rows == 0 || columns == 0) return;
  add_row_bias_kernel<<<elementwise_grid(static_cast<std::size_t>(columns)), kElementwiseThreads>>>(
      destination, bias, rows, columns);
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void scatter_kv_cache(float* key_cache, float* value_cache, const float* keys, const float* values,
                      const KvCacheLayout& layout) {
  if (layout.token_count == 0) return;
  const dim3 grid(layout.token_count, layout.head_count);
  scatter_kv_cache_kernel<<<grid, kHeadThreads>>>(key_cache, value_cache, keys, values,
                                                  layout.position_offset, layout.head_count,
                                                  layout.head_dim, layout.max_sequence_length);
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

void reduce_document_attention(const float* scores, const DocumentAttentionReduction& reduction) {
  if (reduction.span_count == 0 || reduction.query_count == 0) return;
  const std::size_t shared = kReductionThreads * sizeof(double);
  reduce_document_attention_kernel<<<reduction.span_count, kReductionThreads, shared>>>(
      scores, reduction.head_count, reduction.query_count, reduction.key_count,
      reduction.query_position_offset, reduction.query_begin, reduction.query_end,
      reduction.span_begins, reduction.span_ends, reduction.normalize_by_length,
      reduction.accumulator);
  ATTNRANK_CUDA_CHECK(cudaGetLastError());
}

}  // namespace kernels

}  // namespace attnrank

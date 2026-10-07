#include "attnrank/model.hpp"
#include "cuda_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace attnrank {

namespace {

constexpr int kLayerUploadLogInterval = 8;

std::string layer_tensor_name(int layer_index, const char* suffix) {
  return "model.layers." + std::to_string(layer_index) + "." + suffix;
}

bool ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct GemmShape {
  int rows = 0;
  int output_features = 0;
  int input_features = 0;
};

struct TensorShape {
  std::int64_t rows = 0;
  std::int64_t columns = 0;
};

class TensorUploader {
 public:
  explicit TensorUploader(const WeightRegistry& weights) : weights_(weights) {}

  DeviceBuffer half(const std::string& name, const TensorShape& shape) { return half(require(name), shape); }

  DeviceBuffer half(const TensorView& tensor, const TensorShape& shape) {
    const std::size_t count = stage(tensor, shape);
    DeviceBuffer buffer(count * sizeof(__half));
    kernels::convert_to_half(buffer.raw(), staging_.raw(), tensor.dtype, count);
    return buffer;
  }

  DeviceBuffer floating(const std::string& name, const TensorShape& shape) {
    const TensorView& tensor = require(name);
    const std::size_t count = stage(tensor, shape);
    DeviceBuffer buffer(count * sizeof(float));
    kernels::convert_to_float(buffer.as_float(), staging_.raw(), tensor.dtype, count);
    return buffer;
  }

  DeviceBuffer optional_floating(const std::string& name, const TensorShape& shape) {
    return weights_.find(name) == nullptr ? DeviceBuffer() : floating(name, shape);
  }

  const TensorView* find(const std::string& name) const { return weights_.find(name); }

  const TensorView& require(const std::string& name) const {
    const TensorView* found = weights_.find(name);
    if (found != nullptr) return *found;
    for (const std::string& candidate : weights_.names()) {
      if (ends_with(candidate, name)) return weights_.require(candidate);
    }
    throw Error("checkpoint is missing tensor '" + name + "'");
  }

  void finish() { staging_.release(); }

 private:
  static void validate_shape(const TensorView& tensor, const TensorShape& shape) {
    if (tensor.rank() == 2 && (tensor.dim(0) != shape.rows || tensor.dim(1) != shape.columns)) {
      throw Error("tensor '" + tensor.name + "' has shape [" + std::to_string(tensor.dim(0)) + "," +
                  std::to_string(tensor.dim(1)) + "] but [" + std::to_string(shape.rows) + "," +
                  std::to_string(shape.columns) + "] was expected");
    }
    if (tensor.rank() == 1 && tensor.dim(0) != shape.rows * shape.columns) {
      throw Error("tensor '" + tensor.name + "' has unexpected length");
    }
  }

  std::size_t stage(const TensorView& tensor, const TensorShape& shape) {
    validate_shape(tensor, shape);
    staging_.ensure_capacity(tensor.byte_size);
    staging_.upload(tensor.data, tensor.byte_size);
    return static_cast<std::size_t>(tensor.element_count());
  }

  const WeightRegistry& weights_;
  DeviceBuffer staging_;
};

struct LayerWeights {
  DeviceBuffer input_norm;
  DeviceBuffer query_projection;
  DeviceBuffer key_projection;
  DeviceBuffer value_projection;
  DeviceBuffer output_projection;
  DeviceBuffer query_bias;
  DeviceBuffer key_bias;
  DeviceBuffer value_bias;
  DeviceBuffer post_attention_norm;
  DeviceBuffer gate_projection;
  DeviceBuffer up_projection;
  DeviceBuffer down_projection;
};

class DeviceWeights {
 public:
  DeviceWeights(const LlamaConfig& config, const WeightRegistry& weights) {
    TensorUploader uploader(weights);
    const TensorShape vector_shape{config.hidden_size, 1};
    const TensorShape vocabulary_shape{config.vocab_size, config.hidden_size};

    embedding_ = uploader.half("model.embed_tokens.weight", vocabulary_shape);
    final_norm_ = uploader.floating("model.norm.weight", vector_shape);
    output_head_ = upload_output_head(uploader, config, vocabulary_shape);

    layers_.reserve(static_cast<std::size_t>(config.num_hidden_layers));
    for (int index = 0; index < config.num_hidden_layers; ++index) {
      layers_.push_back(upload_layer(uploader, config, index));
      if ((index + 1) % kLayerUploadLogInterval == 0) {
        Log::debug("uploaded %d/%d layers", index + 1, config.num_hidden_layers);
      }
    }
    uploader.finish();
  }

  const DeviceBuffer& embedding() const { return embedding_; }
  const DeviceBuffer& final_norm() const { return final_norm_; }
  const DeviceBuffer& output_head() const { return output_head_; }
  const LayerWeights& layer(int index) const { return layers_[static_cast<std::size_t>(index)]; }

 private:
  static DeviceBuffer upload_output_head(TensorUploader& uploader, const LlamaConfig& config,
                                         const TensorShape& shape) {
    const TensorView* head = uploader.find("lm_head.weight");
    if (head != nullptr) return uploader.half(*head, shape);
    if (config.tie_word_embeddings) return uploader.half("model.embed_tokens.weight", shape);
    throw Error("checkpoint is missing 'lm_head.weight'");
  }

  static LayerWeights upload_layer(TensorUploader& uploader, const LlamaConfig& config, int index) {
    const std::int64_t hidden = config.hidden_size;
    const std::int64_t intermediate = config.intermediate_size;
    const std::int64_t kv_hidden = config.kv_hidden_size();
    const auto name = [index](const char* suffix) { return layer_tensor_name(index, suffix); };

    LayerWeights layer;
    layer.input_norm = uploader.floating(name("input_layernorm.weight"), {hidden, 1});
    layer.query_projection = uploader.half(name("self_attn.q_proj.weight"), {hidden, hidden});
    layer.key_projection = uploader.half(name("self_attn.k_proj.weight"), {kv_hidden, hidden});
    layer.value_projection = uploader.half(name("self_attn.v_proj.weight"), {kv_hidden, hidden});
    layer.output_projection = uploader.half(name("self_attn.o_proj.weight"), {hidden, hidden});
    layer.query_bias = uploader.optional_floating(name("self_attn.q_proj.bias"), {hidden, 1});
    layer.key_bias = uploader.optional_floating(name("self_attn.k_proj.bias"), {kv_hidden, 1});
    layer.value_bias = uploader.optional_floating(name("self_attn.v_proj.bias"), {kv_hidden, 1});
    layer.post_attention_norm = uploader.floating(name("post_attention_layernorm.weight"), {hidden, 1});
    layer.gate_projection = uploader.half(name("mlp.gate_proj.weight"), {intermediate, hidden});
    layer.up_projection = uploader.half(name("mlp.up_proj.weight"), {intermediate, hidden});
    layer.down_projection = uploader.half(name("mlp.down_proj.weight"), {hidden, intermediate});
    return layer;
  }

  DeviceBuffer embedding_;
  DeviceBuffer final_norm_;
  DeviceBuffer output_head_;
  std::vector<LayerWeights> layers_;
};

class KvCache {
 public:
  KvCache(const LlamaConfig& config, int max_sequence_length)
      : layer_stride_(static_cast<long long>(max_sequence_length) * config.head_dim()) {
    const std::size_t elements = static_cast<std::size_t>(config.num_key_value_heads) *
                                 static_cast<std::size_t>(max_sequence_length) *
                                 static_cast<std::size_t>(config.head_dim());
    keys_.reserve(static_cast<std::size_t>(config.num_hidden_layers));
    values_.reserve(static_cast<std::size_t>(config.num_hidden_layers));
    for (int layer = 0; layer < config.num_hidden_layers; ++layer) {
      keys_.emplace_back(elements * sizeof(float));
      values_.emplace_back(elements * sizeof(float));
    }
  }

  float* keys(int layer) { return keys_[static_cast<std::size_t>(layer)].as_float(); }
  float* values(int layer) { return values_[static_cast<std::size_t>(layer)].as_float(); }
  long long head_stride() const { return layer_stride_; }

 private:
  long long layer_stride_;
  std::vector<DeviceBuffer> keys_;
  std::vector<DeviceBuffer> values_;
};

struct ActivationWorkspace {
  ActivationWorkspace(const LlamaConfig& config, const LlamaRuntimeOptions& options) {
    const std::size_t chunk = static_cast<std::size_t>(options.prefill_chunk_tokens);
    const std::size_t hidden = static_cast<std::size_t>(config.hidden_size);
    const std::size_t intermediate = static_cast<std::size_t>(config.intermediate_size);
    const std::size_t kv_hidden = static_cast<std::size_t>(config.kv_hidden_size());

    token_ids.allocate(chunk * sizeof(int));
    hidden_states.allocate(chunk * hidden * sizeof(float));
    normalized.allocate(chunk * hidden * sizeof(float));
    queries.allocate(chunk * hidden * sizeof(float));
    keys.allocate(chunk * kv_hidden * sizeof(float));
    values.allocate(chunk * kv_hidden * sizeof(float));
    attention_output.allocate(chunk * hidden * sizeof(float));
    projection.allocate(chunk * hidden * sizeof(float));
    gate.allocate(chunk * intermediate * sizeof(float));
    up.allocate(chunk * intermediate * sizeof(float));
    scores.allocate(static_cast<std::size_t>(config.num_attention_heads) * chunk *
                    static_cast<std::size_t>(options.max_sequence_length) * sizeof(float));
    logits.allocate(static_cast<std::size_t>(config.vocab_size) * sizeof(float));
    gemm_input.allocate(chunk * std::max(hidden, intermediate) * sizeof(__half));
  }

  DeviceBuffer token_ids;
  DeviceBuffer hidden_states;
  DeviceBuffer normalized;
  DeviceBuffer queries;
  DeviceBuffer keys;
  DeviceBuffer values;
  DeviceBuffer attention_output;
  DeviceBuffer projection;
  DeviceBuffer gate;
  DeviceBuffer up;
  DeviceBuffer scores;
  DeviceBuffer logits;
  DeviceBuffer gemm_input;
};

struct ProbeWindow {
  LayerRange layers;
  TokenSpan query_span;
  std::vector<TokenSpan> document_spans;
  bool normalize_by_length = false;
};

class DocumentAttentionProbe {
 public:
  DocumentAttentionProbe(const ProbeWindow& window, int token_count)
      : layers_(window.layers),
        query_begin_(window.query_span.begin),
        query_end_(std::min(window.query_span.end, token_count)),
        span_count_(static_cast<int>(window.document_spans.size())),
        normalize_by_length_(window.normalize_by_length) {
    const std::size_t count = window.document_spans.size();
    std::vector<int> begins(count);
    std::vector<int> ends(count);
    for (std::size_t index = 0; index < count; ++index) {
      begins[index] = window.document_spans[index].begin;
      ends[index] = window.document_spans[index].end;
    }
    span_begins_.allocate(sizeof(int) * count);
    span_ends_.allocate(sizeof(int) * count);
    span_begins_.upload(begins.data(), sizeof(int) * count);
    span_ends_.upload(ends.data(), sizeof(int) * count);
    accumulator_.allocate(sizeof(double) * cell_count());
    accumulator_.zero();
  }

  bool covers(int layer_index) const { return layers_.contains(layer_index); }
  int last_layer() const { return layers_.end - 1; }
  int query_rows() const { return std::max(0, query_end_ - query_begin_); }

  DocumentAttentionReduction reduction(int layer_index, const SoftmaxShape& scores) {
    DocumentAttentionReduction reduction;
    reduction.head_count = scores.head_count;
    reduction.query_count = scores.query_count;
    reduction.key_count = scores.key_count;
    reduction.query_position_offset = scores.position_offset;
    reduction.query_begin = query_begin_;
    reduction.query_end = query_end_;
    reduction.span_begins = span_begins_.as_int();
    reduction.span_ends = span_ends_.as_int();
    reduction.span_count = span_count_;
    reduction.normalize_by_length = normalize_by_length_;
    reduction.accumulator = accumulator_.as_double() +
                            static_cast<std::size_t>(layer_index - layers_.begin) *
                                static_cast<std::size_t>(span_count_);
    return reduction;
  }

  std::vector<std::vector<double>> totals_by_layer() const {
    std::vector<double> flat(cell_count(), 0.0);
    accumulator_.download(flat.data(), sizeof(double) * flat.size());
    std::vector<std::vector<double>> rows(static_cast<std::size_t>(layers_.count()));
    for (std::size_t layer = 0; layer < rows.size(); ++layer) {
      const auto first = flat.begin() + static_cast<long>(layer * static_cast<std::size_t>(span_count_));
      rows[layer].assign(first, first + span_count_);
    }
    return rows;
  }

 private:
  std::size_t cell_count() const {
    return static_cast<std::size_t>(layers_.count()) * static_cast<std::size_t>(span_count_);
  }

  LayerRange layers_;
  int query_begin_;
  int query_end_;
  int span_count_;
  bool normalize_by_length_;
  DeviceBuffer span_begins_;
  DeviceBuffer span_ends_;
  DeviceBuffer accumulator_;
};

class LlamaForwardPass {
 public:
  LlamaForwardPass(const LlamaConfig& config, const LlamaRuntimeOptions& options, CudaContext& context,
                   const DeviceWeights& weights, KvCache& cache, ActivationWorkspace& workspace)
      : config_(config), options_(options), context_(context), weights_(weights), cache_(cache),
        workspace_(workspace) {}

  void forward_chunk(const TokenSequence& tokens, int position_offset, int last_layer,
                     DocumentAttentionProbe* probe) {
    const int token_count = static_cast<int>(tokens.size());
    if (token_count == 0) return;
    if (token_count > options_.prefill_chunk_tokens) throw Error("chunk larger than the configured prefill window");

    workspace_.token_ids.upload(tokens.data(), sizeof(int) * tokens.size());
    kernels::embedding_lookup(workspace_.hidden_states.as_float(), weights_.embedding().raw(),
                              workspace_.token_ids.as_int(), token_count, config_.hidden_size);

    for (int layer_index = 0; layer_index <= last_layer; ++layer_index) {
      const LayerWeights& layer = weights_.layer(layer_index);
      attention_block(layer, layer_index, token_count, position_offset, probe);
      feed_forward_block(layer, token_count);
    }
  }

  void read_logits(int row_index, int token_count, std::vector<float>& destination) {
    kernels::rms_norm(workspace_.normalized.as_float(), workspace_.hidden_states.as_float(),
                      weights_.final_norm().as_float(), token_count, config_.hidden_size, config_.rms_norm_eps);
    matmul(workspace_.logits.as_float(),
           workspace_.normalized.as_float() + static_cast<long long>(row_index) * config_.hidden_size,
           weights_.output_head().raw(), GemmShape{1, config_.vocab_size, config_.hidden_size});

    destination.resize(static_cast<std::size_t>(config_.vocab_size));
    workspace_.logits.download(destination.data(), destination.size() * sizeof(float));
  }

 private:
  void matmul(float* destination, const float* source, const void* weight, const GemmShape& shape) {
    const std::size_t count = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.input_features);
    workspace_.gemm_input.ensure_capacity(count * sizeof(__half));
    kernels::cast_float_to_half(workspace_.gemm_input.raw(), source, count);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    ATTNRANK_CUBLAS_CHECK(cublasGemmEx(context_.blas(), CUBLAS_OP_T, CUBLAS_OP_N, shape.output_features, shape.rows,
                                       shape.input_features, &alpha, weight, CUDA_R_16F, shape.input_features,
                                       workspace_.gemm_input.raw(), CUDA_R_16F, shape.input_features, &beta,
                                       destination, CUDA_R_32F, shape.output_features, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT));
  }

  void attention_block(const LayerWeights& layer, int layer_index, int token_count, int position_offset,
                       DocumentAttentionProbe* probe) {
    const int hidden = config_.hidden_size;
    const int kv_hidden = config_.kv_hidden_size();
    const std::size_t hidden_elements = static_cast<std::size_t>(token_count) * static_cast<std::size_t>(hidden);

    kernels::rms_norm(workspace_.normalized.as_float(), workspace_.hidden_states.as_float(),
                      layer.input_norm.as_float(), token_count, hidden, config_.rms_norm_eps);
    matmul(workspace_.queries.as_float(), workspace_.normalized.as_float(), layer.query_projection.raw(),
           GemmShape{token_count, hidden, hidden});
    matmul(workspace_.keys.as_float(), workspace_.normalized.as_float(), layer.key_projection.raw(),
           GemmShape{token_count, kv_hidden, hidden});
    matmul(workspace_.values.as_float(), workspace_.normalized.as_float(), layer.value_projection.raw(),
           GemmShape{token_count, kv_hidden, hidden});
    add_bias(workspace_.queries, layer.query_bias, token_count, hidden);
    add_bias(workspace_.keys, layer.key_bias, token_count, kv_hidden);
    add_bias(workspace_.values, layer.value_bias, token_count, kv_hidden);

    RotaryShape rotary;
    rotary.token_count = token_count;
    rotary.query_heads = config_.num_attention_heads;
    rotary.key_heads = config_.num_key_value_heads;
    rotary.head_dim = config_.head_dim();
    rotary.position_offset = position_offset;
    rotary.theta = config_.rope_theta;
    kernels::rotary_embedding(workspace_.queries.as_float(), workspace_.keys.as_float(), rotary);

    attend(layer_index, token_count, position_offset, probe);

    matmul(workspace_.projection.as_float(), workspace_.attention_output.as_float(), layer.output_projection.raw(),
           GemmShape{token_count, hidden, hidden});
    kernels::add_in_place(workspace_.hidden_states.as_float(), workspace_.projection.as_float(), hidden_elements);
  }

  void feed_forward_block(const LayerWeights& layer, int token_count) {
    const int hidden = config_.hidden_size;
    const int intermediate = config_.intermediate_size;
    const std::size_t hidden_elements = static_cast<std::size_t>(token_count) * static_cast<std::size_t>(hidden);
    const std::size_t intermediate_elements =
        static_cast<std::size_t>(token_count) * static_cast<std::size_t>(intermediate);

    kernels::rms_norm(workspace_.normalized.as_float(), workspace_.hidden_states.as_float(),
                      layer.post_attention_norm.as_float(), token_count, hidden, config_.rms_norm_eps);
    matmul(workspace_.gate.as_float(), workspace_.normalized.as_float(), layer.gate_projection.raw(),
           GemmShape{token_count, intermediate, hidden});
    matmul(workspace_.up.as_float(), workspace_.normalized.as_float(), layer.up_projection.raw(),
           GemmShape{token_count, intermediate, hidden});
    kernels::silu_multiply(workspace_.gate.as_float(), workspace_.gate.as_float(), workspace_.up.as_float(),
                           intermediate_elements);
    matmul(workspace_.projection.as_float(), workspace_.gate.as_float(), layer.down_projection.raw(),
           GemmShape{token_count, hidden, intermediate});
    kernels::add_in_place(workspace_.hidden_states.as_float(), workspace_.projection.as_float(), hidden_elements);
  }

  static void add_bias(DeviceBuffer& activations, const DeviceBuffer& bias, int rows, int columns) {
    if (bias.empty()) return;
    kernels::add_row_bias(activations.as_float(), bias.as_float(), rows, columns);
  }

  void attend(int layer_index, int token_count, int position_offset, DocumentAttentionProbe* probe) {
    const int head_dim = config_.head_dim();
    const int query_heads = config_.num_attention_heads;
    const int key_heads = config_.num_key_value_heads;
    const int group = config_.heads_per_kv_group();
    const int key_count = position_offset + token_count;
    float* key_cache = cache_.keys(layer_index);
    float* value_cache = cache_.values(layer_index);

    KvCacheLayout layout;
    layout.token_count = token_count;
    layout.position_offset = position_offset;
    layout.head_count = key_heads;
    layout.head_dim = head_dim;
    layout.max_sequence_length = options_.max_sequence_length;
    kernels::scatter_kv_cache(key_cache, value_cache, workspace_.keys.as_float(), workspace_.values.as_float(),
                              layout);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float zero = 0.0f;
    const float one = 1.0f;
    const long long cache_stride = cache_.head_stride();
    const long long score_stride = static_cast<long long>(token_count) * key_count;
    const long long group_stride = static_cast<long long>(group) * head_dim;
    float* scores = workspace_.scores.as_float();

    for (int offset = 0; offset < group; ++offset) {
      ATTNRANK_CUBLAS_CHECK(cublasSgemmStridedBatched(
          context_.blas(), CUBLAS_OP_T, CUBLAS_OP_N, key_count, token_count, head_dim, &scale, key_cache, head_dim,
          cache_stride, workspace_.queries.as_float() + static_cast<long long>(offset) * head_dim,
          query_heads * head_dim, group_stride, &zero, scores + static_cast<long long>(offset) * score_stride,
          key_count, static_cast<long long>(group) * score_stride, key_heads));
    }

    SoftmaxShape softmax;
    softmax.head_count = query_heads;
    softmax.query_count = token_count;
    softmax.key_count = key_count;
    softmax.position_offset = position_offset;
    softmax.sliding_window = config_.sliding_window;
    kernels::softmax_causal(scores, softmax);

    if (probe != nullptr && probe->covers(layer_index)) {
      kernels::reduce_document_attention(scores, probe->reduction(layer_index, softmax));
    }

    for (int offset = 0; offset < group; ++offset) {
      ATTNRANK_CUBLAS_CHECK(cublasSgemmStridedBatched(
          context_.blas(), CUBLAS_OP_N, CUBLAS_OP_N, head_dim, token_count, key_count, &one, value_cache, head_dim,
          cache_stride, scores + static_cast<long long>(offset) * score_stride, key_count,
          static_cast<long long>(group) * score_stride, &zero,
          workspace_.attention_output.as_float() + static_cast<long long>(offset) * head_dim,
          query_heads * head_dim, group_stride, key_heads));
    }
  }

  const LlamaConfig& config_;
  const LlamaRuntimeOptions& options_;
  CudaContext& context_;
  const DeviceWeights& weights_;
  KvCache& cache_;
  ActivationWorkspace& workspace_;
};

bool apply_stop_sequence(const std::string& decoded, const std::vector<std::string>& stop_sequences,
                         GenerationResult& result) {
  for (const std::string& stop : stop_sequences) {
    if (stop.empty()) continue;
    const std::size_t found = decoded.find(stop);
    if (found == std::string::npos) continue;
    result.hit_stop_sequence = true;
    result.text = decoded.substr(0, found);
    return true;
  }
  return false;
}

void normalize_to_unit_sum(std::vector<double>& values) {
  double sum = 0.0;
  for (const double value : values) sum += value;
  if (sum <= 0.0) return;
  for (double& value : values) value /= sum;
}

}  // namespace

class CudaLlamaModel::Impl {
 public:
  Impl(const LlamaConfig& config, const WeightRegistry& weights, std::shared_ptr<const ITokenizer> tokenizer,
       const LlamaRuntimeOptions& options)
      : config_(config),
        tokenizer_(validated_tokenizer(std::move(tokenizer))),
        options_(validated_options(options)),
        context_(CudaContextConfig{options.device_index}),
        free_memory_before_upload_(report_device(context_)),
        weights_(config_, weights),
        cache_(config_, options_.max_sequence_length),
        workspace_(config_, options_),
        forward_(config_, options_, context_, weights_, cache_, workspace_) {
    const std::size_t free_after = context_.free_memory();
    Log::info("model resident: %.2f GiB allocated, %.1f GiB device memory still free",
              gibibytes(free_memory_before_upload_ - free_after), gibibytes(free_after));
  }

  const LlamaConfig& config() const { return config_; }

  GenerationResult generate(const GenerationRequest& request, ISampler& sampler) {
    if (request.prompt_tokens.empty()) throw Error("generation requires a non-empty prompt");
    const int budget = options_.max_sequence_length - static_cast<int>(request.prompt_tokens.size());
    if (budget <= 0) {
      throw Error("prompt of " + std::to_string(request.prompt_tokens.size()) + " tokens exceeds the " +
                  std::to_string(options_.max_sequence_length) + " token window");
    }

    GenerationResult result;
    result.prompt_token_count = static_cast<int>(request.prompt_tokens.size());

    std::vector<float> logits;
    const int prompt_rows = prefill(request.prompt_tokens, config_.num_hidden_layers - 1, nullptr);
    forward_.read_logits(prompt_rows - 1, prompt_rows, logits);

    const int limit = std::min(request.max_new_tokens, budget);
    std::string decoded;
    for (int step = 0; step < limit; ++step) {
      const int token = sampler.select(logits);
      if (token == config_.eos_token_id) break;

      result.generated_tokens.push_back(token);
      decoded = tokenizer_->decode(result.generated_tokens);
      if (apply_stop_sequence(decoded, request.stop_sequences, result)) break;
      if (position_ >= options_.max_sequence_length) break;

      forward_.forward_chunk(TokenSequence{token}, position_, config_.num_hidden_layers - 1, nullptr);
      position_ += 1;
      forward_.read_logits(0, 1, logits);
    }

    if (!result.hit_stop_sequence) result.text = decoded;
    return result;
  }

  AttentionProbeResult measure_attention(const TokenSequence& tokens, const AttentionProbeRequest& request) {
    ProbeWindow window;
    window.layers = request.resolve_layers(config_.num_hidden_layers);
    window.query_span = request.query_span;
    window.document_spans = request.document_spans;
    window.normalize_by_length = request.normalize_by_document_length;

    DocumentAttentionProbe probe = run_probe(tokens, window);
    const std::vector<std::vector<double>> totals = probe.totals_by_layer();
    const double observations = static_cast<double>(probe.query_rows()) *
                                static_cast<double>(config_.num_attention_heads) *
                                static_cast<double>(window.layers.count());

    AttentionProbeResult result;
    result.document_attention.assign(window.document_spans.size(), 0.0);
    if (observations <= 0.0) return result;

    for (const std::vector<double>& row : totals) {
      for (std::size_t span = 0; span < row.size(); ++span) {
        result.document_attention[span] += row[span] / observations;
      }
    }
    normalize_to_unit_sum(result.document_attention);
    return result;
  }

  AttentionLayerProbeResult measure_attention_by_layer(const TokenSequence& tokens,
                                                       const AttentionLayerProbeRequest& request) {
    ProbeWindow window;
    window.layers = request.resolve_layers(config_.num_hidden_layers);
    window.query_span = request.query_span;
    window.document_spans = request.document_spans;
    window.normalize_by_length = request.normalize_by_document_length;

    DocumentAttentionProbe probe = run_probe(tokens, window);
    const double observations =
        static_cast<double>(probe.query_rows()) * static_cast<double>(config_.num_attention_heads);

    AttentionLayerProbeResult result;
    result.layer_begin = window.layers.begin;
    result.attention_by_layer = probe.totals_by_layer();
    for (std::vector<double>& row : result.attention_by_layer) {
      if (observations <= 0.0) {
        std::fill(row.begin(), row.end(), 0.0);
        continue;
      }
      for (double& value : row) value /= observations;
      normalize_to_unit_sum(row);
    }
    return result;
  }

 private:
  static std::shared_ptr<const ITokenizer> validated_tokenizer(std::shared_ptr<const ITokenizer> tokenizer) {
    if (tokenizer == nullptr) throw Error("CudaLlamaModel requires a tokenizer");
    return tokenizer;
  }

  static LlamaRuntimeOptions validated_options(const LlamaRuntimeOptions& options) {
    if (options.max_sequence_length <= 0) throw Error("max_sequence_length must be positive");
    if (options.prefill_chunk_tokens <= 0) throw Error("prefill_chunk_tokens must be positive");
    return options;
  }

  static std::size_t report_device(const CudaContext& context) {
    const std::size_t free_bytes = context.free_memory();
    Log::info("cuda device %d: %s, %.1f GiB free", context.device_index(), context.device_name().c_str(),
              gibibytes(free_bytes));
    return free_bytes;
  }

  DocumentAttentionProbe run_probe(const TokenSequence& tokens, const ProbeWindow& window) {
    if (window.document_spans.empty()) throw Error("attention probe requires document spans");
    if (window.layers.begin < 0 || window.layers.end > config_.num_hidden_layers || window.layers.count() <= 0) {
      throw Error("probe layers [" + std::to_string(window.layers.begin) + "," +
                  std::to_string(window.layers.end) + ") are out of range");
    }
    if (static_cast<int>(tokens.size()) > options_.max_sequence_length) {
      throw Error("probe input of " + std::to_string(tokens.size()) + " tokens exceeds the context window");
    }

    DocumentAttentionProbe probe(window, static_cast<int>(tokens.size()));
    prefill(tokens, probe.last_layer(), &probe);
    return probe;
  }

  int prefill(const TokenSequence& tokens, int last_layer, DocumentAttentionProbe* probe) {
    position_ = 0;
    int rows_in_last_chunk = 0;
    for (std::size_t offset = 0; offset < tokens.size();) {
      const std::size_t count =
          std::min<std::size_t>(static_cast<std::size_t>(options_.prefill_chunk_tokens), tokens.size() - offset);
      const TokenSequence chunk(tokens.begin() + static_cast<long>(offset),
                                tokens.begin() + static_cast<long>(offset + count));
      forward_.forward_chunk(chunk, position_, last_layer, probe);
      position_ += static_cast<int>(count);
      offset += count;
      rows_in_last_chunk = static_cast<int>(count);
    }
    return rows_in_last_chunk;
  }

  LlamaConfig config_;
  std::shared_ptr<const ITokenizer> tokenizer_;
  LlamaRuntimeOptions options_;
  CudaContext context_;
  std::size_t free_memory_before_upload_;
  DeviceWeights weights_;
  KvCache cache_;
  ActivationWorkspace workspace_;
  LlamaForwardPass forward_;
  int position_ = 0;
};

CudaLlamaModel::CudaLlamaModel(const LlamaConfig& config, const WeightRegistry& weights,
                               std::shared_ptr<const ITokenizer> tokenizer, const LlamaRuntimeOptions& options)
    : impl_(std::make_unique<Impl>(config, weights, std::move(tokenizer), options)) {}

CudaLlamaModel::~CudaLlamaModel() = default;

const LlamaConfig& CudaLlamaModel::config() const { return impl_->config(); }

GenerationResult CudaLlamaModel::generate(const GenerationRequest& request, ISampler& sampler) {
  return impl_->generate(request, sampler);
}

AttentionProbeResult CudaLlamaModel::measure_attention(const TokenSequence& tokens,
                                                       const AttentionProbeRequest& request) {
  return impl_->measure_attention(tokens, request);
}

AttentionLayerProbeResult CudaLlamaModel::measure_attention_by_layer(const TokenSequence& tokens,
                                                                     const AttentionLayerProbeRequest& request) {
  return impl_->measure_attention_by_layer(tokens, request);
}

}  // namespace attnrank

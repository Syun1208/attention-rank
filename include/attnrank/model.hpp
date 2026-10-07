#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "attnrank/checkpoint.hpp"
#include "attnrank/core.hpp"
#include "attnrank/tokenizer.hpp"

namespace attnrank {

constexpr const char* kModelConfigFileName = "config.json";

struct LlamaConfig {
  int hidden_size = 4096;
  int intermediate_size = 11008;
  int num_hidden_layers = 32;
  int num_attention_heads = 32;
  int num_key_value_heads = 32;
  int vocab_size = 32000;
  int max_position_embeddings = 4096;
  float rms_norm_eps = 1e-5f;
  float rope_theta = 10000.0f;
  int bos_token_id = 1;
  int eos_token_id = 2;
  int sliding_window = 0;
  bool tie_word_embeddings = false;

  int head_dim() const { return hidden_size / num_attention_heads; }
  int kv_hidden_size() const { return num_key_value_heads * head_dim(); }
  int heads_per_kv_group() const { return num_attention_heads / num_key_value_heads; }

  static LlamaConfig from_json(const Json& root);
  static LlamaConfig from_file(const std::string& path);
  static LlamaConfig from_directory(const std::string& model_directory);
};

struct LayerRange {
  int begin = 0;
  int end = 0;

  int count() const { return end - begin; }
  bool contains(int layer_index) const { return layer_index >= begin && layer_index < end; }
};

struct AttentionProbeRequest {
  int layer_index = 0;
  bool average_all_layers = false;
  int layer_span_begin = -1;
  int layer_span_end = -1;
  TokenSpan query_span;
  std::vector<TokenSpan> document_spans;
  bool normalize_by_document_length = false;

  LayerRange resolve_layers(int num_hidden_layers) const;
};

struct AttentionProbeResult {
  std::vector<double> document_attention;
};

struct AttentionLayerProbeRequest {
  int layer_begin = 0;
  int layer_end = -1;
  TokenSpan query_span;
  std::vector<TokenSpan> document_spans;
  bool normalize_by_document_length = false;

  LayerRange resolve_layers(int num_hidden_layers) const;
};

struct AttentionLayerProbeResult {
  int layer_begin = 0;
  std::vector<std::vector<double>> attention_by_layer;
};

class IAttentionProvider {
 public:
  virtual ~IAttentionProvider() = default;
  virtual AttentionProbeResult measure_attention(const TokenSequence& tokens,
                                                 const AttentionProbeRequest& request) = 0;
  virtual AttentionLayerProbeResult measure_attention_by_layer(
      const TokenSequence& tokens, const AttentionLayerProbeRequest& request) = 0;
};

class ISampler {
 public:
  virtual ~ISampler() = default;
  virtual int select(const std::vector<float>& logits) = 0;
};

class GreedySampler : public ISampler {
 public:
  int select(const std::vector<float>& logits) override;
};

struct NucleusSamplerConfig {
  float temperature = 0.7f;
  float top_p = 0.9f;
  int top_k = 0;
  std::uint64_t seed = 1234;
};

class NucleusSampler : public ISampler {
 public:
  explicit NucleusSampler(const NucleusSamplerConfig& config);
  int select(const std::vector<float>& logits) override;

 private:
  NucleusSamplerConfig config_;
  std::mt19937_64 engine_;
};

std::unique_ptr<ISampler> make_sampler(const NucleusSamplerConfig& config);

struct GenerationRequest {
  TokenSequence prompt_tokens;
  int max_new_tokens = 256;
  std::vector<std::string> stop_sequences;
};

struct GenerationResult {
  TokenSequence generated_tokens;
  std::string text;
  int prompt_token_count = 0;
  bool hit_stop_sequence = false;
};

class ILanguageModel {
 public:
  virtual ~ILanguageModel() = default;
  virtual const LlamaConfig& config() const = 0;
  virtual GenerationResult generate(const GenerationRequest& request, ISampler& sampler) = 0;
};

struct LlamaRuntimeOptions {
  int device_index = -1;
  int max_sequence_length = 2048;
  int prefill_chunk_tokens = 512;
};

class CudaLlamaModel : public ILanguageModel, public IAttentionProvider {
 public:
  CudaLlamaModel(const LlamaConfig& config,
                 const WeightRegistry& weights,
                 std::shared_ptr<const ITokenizer> tokenizer,
                 const LlamaRuntimeOptions& options);
  ~CudaLlamaModel() override;

  CudaLlamaModel(const CudaLlamaModel&) = delete;
  CudaLlamaModel& operator=(const CudaLlamaModel&) = delete;

  const LlamaConfig& config() const override;
  GenerationResult generate(const GenerationRequest& request, ISampler& sampler) override;
  AttentionProbeResult measure_attention(const TokenSequence& tokens,
                                         const AttentionProbeRequest& request) override;
  AttentionLayerProbeResult measure_attention_by_layer(
      const TokenSequence& tokens, const AttentionLayerProbeRequest& request) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace attnrank

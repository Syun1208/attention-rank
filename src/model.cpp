#include "attnrank/model.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>

namespace attnrank {

namespace {

int read_int(const Json& root, const char* key, int fallback) {
  const Json* found = root.find(key);
  return found == nullptr || found->is_null() ? fallback : static_cast<int>(found->as_int64(fallback));
}

float read_float(const Json& root, const char* key, float fallback) {
  const Json* found = root.find(key);
  return found == nullptr || found->is_null() ? fallback : static_cast<float>(found->as_double(fallback));
}

}  // namespace

LlamaConfig LlamaConfig::from_json(const Json& root) {
  LlamaConfig config;
  config.hidden_size = read_int(root, "hidden_size", config.hidden_size);
  config.intermediate_size = read_int(root, "intermediate_size", config.intermediate_size);
  config.num_hidden_layers = read_int(root, "num_hidden_layers", config.num_hidden_layers);
  config.num_attention_heads = read_int(root, "num_attention_heads", config.num_attention_heads);
  config.num_key_value_heads = read_int(root, "num_key_value_heads", config.num_attention_heads);
  config.vocab_size = read_int(root, "vocab_size", config.vocab_size);
  config.max_position_embeddings = read_int(root, "max_position_embeddings", config.max_position_embeddings);
  config.rms_norm_eps = read_float(root, "rms_norm_eps", config.rms_norm_eps);
  config.rope_theta = read_float(root, "rope_theta", config.rope_theta);
  config.bos_token_id = read_int(root, "bos_token_id", config.bos_token_id);
  config.eos_token_id = read_int(root, "eos_token_id", config.eos_token_id);

  const Json* window = root.find("sliding_window");
  if (window != nullptr && window->is_number()) config.sliding_window = static_cast<int>(window->as_int64(0));

  const Json* tied = root.find("tie_word_embeddings");
  config.tie_word_embeddings = tied != nullptr && tied->as_bool(false);

  if (config.num_attention_heads <= 0 || config.hidden_size % config.num_attention_heads != 0) {
    throw Error("invalid attention head configuration");
  }
  if (config.num_key_value_heads <= 0 || config.num_attention_heads % config.num_key_value_heads != 0) {
    throw Error("invalid key/value head configuration");
  }
  return config;
}

LlamaConfig LlamaConfig::from_file(const std::string& path) { return from_json(Json::parse_file(path)); }

LlamaConfig LlamaConfig::from_directory(const std::string& model_directory) {
  return from_file((std::filesystem::path(model_directory) / kModelConfigFileName).string());
}

LayerRange AttentionProbeRequest::resolve_layers(int num_hidden_layers) const {
  const bool has_span = layer_span_begin >= 0 && layer_span_end > layer_span_begin;
  if (has_span) return LayerRange{layer_span_begin, layer_span_end};
  if (average_all_layers) return LayerRange{0, num_hidden_layers};
  return LayerRange{layer_index, layer_index + 1};
}

LayerRange AttentionLayerProbeRequest::resolve_layers(int num_hidden_layers) const {
  return LayerRange{layer_begin < 0 ? 0 : layer_begin, layer_end < 0 ? num_hidden_layers : layer_end};
}

int GreedySampler::select(const std::vector<float>& logits) {
  if (logits.empty()) throw Error("cannot sample from empty logits");
  return static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

NucleusSampler::NucleusSampler(const NucleusSamplerConfig& config) : config_(config), engine_(config.seed) {
  if (config_.temperature <= 0.0f) throw Error("nucleus sampling needs a positive temperature");
}

int NucleusSampler::select(const std::vector<float>& logits) {
  if (logits.empty()) throw Error("cannot sample from empty logits");

  std::vector<int> candidates(logits.size());
  std::iota(candidates.begin(), candidates.end(), 0);
  const std::size_t retained =
      config_.top_k > 0 ? std::min<std::size_t>(static_cast<std::size_t>(config_.top_k), logits.size())
                        : logits.size();
  std::partial_sort(candidates.begin(), candidates.begin() + static_cast<long>(retained), candidates.end(),
                    [&logits](int left, int right) {
                      return logits[static_cast<std::size_t>(left)] > logits[static_cast<std::size_t>(right)];
                    });
  candidates.resize(retained);

  const double maximum = logits[static_cast<std::size_t>(candidates.front())];
  std::vector<double> probabilities(retained);
  double total = 0.0;
  for (std::size_t index = 0; index < retained; ++index) {
    const double logit = logits[static_cast<std::size_t>(candidates[index])];
    probabilities[index] = std::exp((logit - maximum) / static_cast<double>(config_.temperature));
    total += probabilities[index];
  }

  double cumulative = 0.0;
  std::size_t nucleus = retained;
  for (std::size_t index = 0; index < retained; ++index) {
    cumulative += probabilities[index] / total;
    if (cumulative >= static_cast<double>(config_.top_p)) {
      nucleus = index + 1;
      break;
    }
  }

  double nucleus_mass = 0.0;
  for (std::size_t index = 0; index < nucleus; ++index) nucleus_mass += probabilities[index];

  std::uniform_real_distribution<double> distribution(0.0, nucleus_mass);
  double target = distribution(engine_);
  for (std::size_t index = 0; index < nucleus; ++index) {
    target -= probabilities[index];
    if (target <= 0.0) return candidates[index];
  }
  return candidates[nucleus - 1];
}

std::unique_ptr<ISampler> make_sampler(const NucleusSamplerConfig& config) {
  if (config.temperature <= 0.0f) return std::make_unique<GreedySampler>();
  return std::make_unique<NucleusSampler>(config);
}

}  // namespace attnrank

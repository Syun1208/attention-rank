#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "attnrank/attnrank.hpp"
#include "attnrank/core.hpp"
#include "attnrank/model.hpp"
#include "attnrank/prompt.hpp"
#include "attnrank/tokenizer.hpp"

namespace attnrank {

struct PromptOptions {
  std::string chat_format;
  std::string system_prompt;
  DocumentPromptTemplate prompt_template;
};

struct EngineOptions {
  LlamaRuntimeOptions runtime;
  PromptOptions prompt;
};

struct GenerationOptions {
  int max_new_tokens = 128;
  std::vector<std::string> stop_sequences;
  float temperature = 0.0f;
  float top_p = 0.9f;
  int top_k = 0;
  std::uint64_t seed = kDefaultRerankSeed;

  NucleusSamplerConfig sampler_config() const;
};

class Engine {
 public:
  Engine(const std::string& model_directory, const EngineOptions& options);

  const std::string& model_directory() const { return model_directory_; }
  const LlamaConfig& config() const { return config_; }
  int num_layers() const { return config_.num_hidden_layers; }
  const std::string& chat_format() const { return chat_format_; }
  const DocumentPromptTemplate& prompt_template() const { return prompt_template_; }
  void set_prompt_template(const DocumentPromptTemplate& prompt_template);

  std::shared_ptr<const ITokenizer> tokenizer() const { return tokenizer_; }
  std::shared_ptr<CudaLlamaModel> model() const { return model_; }
  std::shared_ptr<const IPromptBuilder> prompt_builder() const { return prompt_builder_; }

  TokenSequence encode(const std::string& text, bool add_bos = true) const;
  std::string decode(const TokenSequence& tokens) const;

  StructuredPrompt build_prompt(const std::string& question,
                                const std::vector<Document>& documents) const;
  std::vector<std::string> stop_sequences() const;

  std::vector<double> measure_attention(const StructuredPrompt& prompt,
                                        const AttentionProfilerConfig& layers) const;
  AttentionLayerProbeResult measure_attention_by_layer(const StructuredPrompt& prompt,
                                                       const LayerScanConfig& layers) const;

  GenerationResult generate(const TokenSequence& prompt_tokens,
                            const GenerationOptions& options) const;
  GenerationResult answer(const StructuredPrompt& prompt, const GenerationOptions& options) const;

  AttentionProfile build_profile(const IProbeSampleSource& samples,
                                 const AttentionProfilerConfig& config,
                                 std::shared_ptr<IProgressObserver> progress = nullptr) const;
  std::vector<AttentionProfile> scan_layers(const IProbeSampleSource& samples,
                                            const LayerScanConfig& config,
                                            std::shared_ptr<IProgressObserver> progress = nullptr) const;

 private:
  ProbeDependencies probe_dependencies(int log_interval,
                                       std::shared_ptr<IProgressObserver> progress) const;

  std::string model_directory_;
  LlamaConfig config_;
  std::shared_ptr<ITokenizer> tokenizer_;
  std::shared_ptr<CudaLlamaModel> model_;
  std::string chat_format_;
  std::shared_ptr<const IChatFormatter> formatter_;
  DocumentPromptTemplate prompt_template_;
  std::shared_ptr<DocumentPromptBuilder> prompt_builder_;
};

}  // namespace attnrank

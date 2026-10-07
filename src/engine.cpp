#include "attnrank/engine.hpp"

#include <utility>

#include "attnrank/checkpoint.hpp"

namespace attnrank {

NucleusSamplerConfig GenerationOptions::sampler_config() const {
  NucleusSamplerConfig config;
  config.temperature = temperature;
  config.top_p = top_p;
  config.top_k = top_k;
  config.seed = seed;
  return config;
}

Engine::Engine(const std::string& model_directory, const EngineOptions& options)
    : model_directory_(model_directory),
      config_(LlamaConfig::from_directory(model_directory)),
      tokenizer_(TokenizerFactory::create(model_directory, config_)),
      chat_format_(options.prompt.chat_format.empty() ? ChatFormatterFactory::detect(model_directory)
                                                      : options.prompt.chat_format),
      formatter_(ChatFormatterFactory::create(chat_format_, options.prompt.system_prompt)),
      prompt_template_(options.prompt.prompt_template) {
  const CheckpointLoader loader = CheckpointLoader::with_default_factories();
  const WeightRegistry weights = loader.load_directory(model_directory);
  model_ = std::make_shared<CudaLlamaModel>(config_, weights, tokenizer_, options.runtime);
  prompt_builder_ = std::make_shared<DocumentPromptBuilder>(tokenizer_, formatter_, prompt_template_);
}

void Engine::set_prompt_template(const DocumentPromptTemplate& prompt_template) {
  prompt_template_ = prompt_template;
  prompt_builder_ = std::make_shared<DocumentPromptBuilder>(tokenizer_, formatter_, prompt_template_);
}

TokenSequence Engine::encode(const std::string& text, bool add_bos) const {
  EncodeOptions options;
  options.add_bos = add_bos;
  options.add_eos = false;
  options.add_dummy_prefix = true;
  return tokenizer_->encode(text, options);
}

std::string Engine::decode(const TokenSequence& tokens) const { return tokenizer_->decode(tokens); }

StructuredPrompt Engine::build_prompt(const std::string& question, const std::vector<Document>& documents) const {
  PromptRequest request;
  request.question = question;
  request.documents = documents;
  return prompt_builder_->build(request);
}

std::vector<std::string> Engine::stop_sequences() const { return prompt_builder_->stop_sequences(); }

std::vector<double> Engine::measure_attention(const StructuredPrompt& prompt,
                                              const AttentionProfilerConfig& layers) const {
  return model_->measure_attention(prompt.tokens, layers.probe_request(prompt)).document_attention;
}

AttentionLayerProbeResult Engine::measure_attention_by_layer(const StructuredPrompt& prompt,
                                                             const LayerScanConfig& layers) const {
  return model_->measure_attention_by_layer(prompt.tokens, layers.probe_request(prompt));
}

GenerationResult Engine::generate(const TokenSequence& prompt_tokens, const GenerationOptions& options) const {
  const std::unique_ptr<ISampler> sampler = make_sampler(options.sampler_config());

  GenerationRequest request;
  request.prompt_tokens = prompt_tokens;
  request.max_new_tokens = options.max_new_tokens;
  request.stop_sequences = options.stop_sequences;
  return model_->generate(request, *sampler);
}

GenerationResult Engine::answer(const StructuredPrompt& prompt, const GenerationOptions& options) const {
  GenerationOptions merged = options;
  const std::vector<std::string> defaults = stop_sequences();
  merged.stop_sequences.insert(merged.stop_sequences.begin(), defaults.begin(), defaults.end());
  return generate(prompt.tokens, merged);
}

ProbeDependencies Engine::probe_dependencies(int log_interval, std::shared_ptr<IProgressObserver> progress) const {
  ProbeDependencies dependencies;
  dependencies.attention_provider = model_;
  dependencies.prompt_builder = prompt_builder_;
  dependencies.progress = progress != nullptr ? std::move(progress) : make_progress_observer(log_interval);
  return dependencies;
}

AttentionProfile Engine::build_profile(const IProbeSampleSource& samples, const AttentionProfilerConfig& config,
                                       std::shared_ptr<IProgressObserver> progress) const {
  AttentionProfilerConfig resolved = config;
  if (resolved.model_id.empty()) resolved.model_id = model_directory_;
  AttentionProfiler profiler(probe_dependencies(config.progress_interval, std::move(progress)), resolved);
  return profiler.build_profile(samples);
}

std::vector<AttentionProfile> Engine::scan_layers(const IProbeSampleSource& samples, const LayerScanConfig& config,
                                                  std::shared_ptr<IProgressObserver> progress) const {
  LayerScanConfig resolved = config;
  if (resolved.model_id.empty()) resolved.model_id = model_directory_;
  AttentionLayerScanner scanner(probe_dependencies(config.progress_interval, std::move(progress)), resolved);
  return scanner.scan(samples);
}

}  // namespace attnrank

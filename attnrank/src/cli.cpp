#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "attnrank/checkpoint.hpp"
#include "attnrank/engine.hpp"

namespace attnrank::cli {

constexpr int kExitSuccess = 0;
constexpr int kExitFailure = 1;
constexpr int kExitNoBasinLayer = 2;
constexpr int kReportIndent = 2;
constexpr int kDefaultSyntheticSlots = 10;
constexpr int kDefaultSyntheticSamples = 400;
constexpr int kDefaultWordsPerDocument = 60;

const LlamaRuntimeOptions kRuntimeDefaults{};
const AttentionProfilerConfig kProfilerDefaults{};
const BasinCriterion kBasinDefaults{};
const DocumentPromptTemplate kTemplateDefaults{};

class Arguments {
 public:
  Arguments(int argc, char** argv, int first_index) {
    for (int index = first_index; index < argc; ++index) {
      std::string token = argv[index];
      if (token.rfind("--", 0) != 0) continue;
      token = token.substr(2);
      const std::size_t separator = token.find('=');
      if (separator != std::string::npos) {
        values_[token.substr(0, separator)] = token.substr(separator + 1);
      } else if (index + 1 < argc && std::string(argv[index + 1]).rfind("--", 0) != 0) {
        values_[token] = argv[++index];
      } else {
        values_[token] = "1";
      }
    }
  }

  bool has(const std::string& name) const { return values_.count(name) != 0; }

  bool flag(const std::string& name) const {
    const auto found = values_.find(name);
    return found != values_.end() && found->second != "0" && found->second != "false";
  }

  std::string text(const std::string& name, const std::string& fallback = {}) const {
    const auto found = values_.find(name);
    return found == values_.end() ? fallback : found->second;
  }

  std::string required_text(const std::string& name) const {
    const auto found = values_.find(name);
    if (found == values_.end()) throw Error("missing required option --" + name);
    return found->second;
  }

  int integer(const std::string& name, int fallback) const {
    const auto found = values_.find(name);
    if (found == values_.end()) return fallback;
    try {
      return std::stoi(found->second);
    } catch (const std::exception&) {
      throw Error("option --" + name + " expects an integer, got '" + found->second + "'");
    }
  }

  double real(const std::string& name, double fallback) const {
    const auto found = values_.find(name);
    if (found == values_.end()) return fallback;
    try {
      return std::stod(found->second);
    } catch (const std::exception&) {
      throw Error("option --" + name + " expects a number, got '" + found->second + "'");
    }
  }

  std::uint64_t seed(std::uint64_t fallback) const {
    const auto found = values_.find("seed");
    return found == values_.end() ? fallback : static_cast<std::uint64_t>(std::stoull(found->second));
  }

 private:
  std::map<std::string, std::string> values_;
};

class ICommand {
 public:
  virtual ~ICommand() = default;
  virtual const char* name() const = 0;
  virtual const char* usage() const = 0;
  virtual int run(const Arguments& arguments) const = 0;
};

namespace {

std::vector<std::string> split_list(const std::string& value) {
  std::vector<std::string> parts;
  std::string current;
  for (const char character : value) {
    if (character == ',') {
      if (!current.empty()) parts.push_back(current);
      current.clear();
    } else {
      current.push_back(character);
    }
  }
  if (!current.empty()) parts.push_back(current);
  return parts;
}

std::string unescape(const std::string& value) {
  std::string result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (value[index] != '\\' || index + 1 == value.size()) {
      result.push_back(value[index]);
      continue;
    }
    const char next = value[++index];
    result.push_back(next == 'n' ? '\n' : next == 't' ? '\t' : next);
  }
  return result;
}

struct InclusiveLayerRange {
  int first = 0;
  int last = 0;
};

bool parse_layer_range(const std::string& value, const char* option, InclusiveLayerRange& range) {
  if (value.empty()) return false;
  const std::size_t separator = value.find(':');
  if (separator == std::string::npos) throw Error(std::string(option) + " expects FIRST:LAST");
  range.first = std::stoi(value.substr(0, separator));
  range.last = std::stoi(value.substr(separator + 1));
  return true;
}

void write_text(const std::string& path, const std::string& text) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) throw Error("cannot write to '" + path + "'");
  stream << text << "\n";
}

class EngineOptionsFactory {
 public:
  static EngineOptions from_arguments(const Arguments& arguments) {
    EngineOptions options;
    options.runtime.device_index = arguments.integer("device", kRuntimeDefaults.device_index);
    options.runtime.max_sequence_length = arguments.integer("max-sequence", kRuntimeDefaults.max_sequence_length);
    options.runtime.prefill_chunk_tokens = arguments.integer("chunk", kRuntimeDefaults.prefill_chunk_tokens);
    options.prompt.chat_format = arguments.text("chat-format");
    options.prompt.system_prompt = arguments.text("system-prompt");
    options.prompt.prompt_template = prompt_template(arguments);
    return options;
  }

 private:
  static DocumentPromptTemplate prompt_template(const Arguments& arguments) {
    DocumentPromptTemplate prompt_template;
    prompt_template.max_document_tokens =
        arguments.integer("max-document-tokens", kTemplateDefaults.max_document_tokens);
    prompt_template.include_titles = arguments.flag("document-titles");
    for (const std::string& stop : split_list(arguments.text("stop"))) {
      prompt_template.extra_stop_sequences.push_back(unescape(stop));
    }
    if (arguments.has("document-separator")) {
      prompt_template.document_separator = unescape(arguments.text("document-separator"));
    }
    if (arguments.has("instruction")) prompt_template.instruction = arguments.text("instruction");
    if (arguments.has("example")) prompt_template.example = unescape(arguments.text("example"));
    return prompt_template;
  }
};

class ProbeSourceFactory {
 public:
  static std::unique_ptr<IProbeSampleSource> from_arguments(const Arguments& arguments) {
    if (arguments.has("samples-file")) return from_file(arguments);
    SyntheticProbeConfig config;
    config.document_slots = arguments.integer("slots", kDefaultSyntheticSlots);
    config.sample_count = arguments.integer("samples", kDefaultSyntheticSamples);
    config.words_per_document = arguments.integer("words-per-document", kDefaultWordsPerDocument);
    config.seed = arguments.seed(kDefaultSyntheticSeed);
    return std::make_unique<SyntheticProbeSampleSource>(config);
  }

 private:
  static std::unique_ptr<IProbeSampleSource> from_file(const Arguments& arguments) {
    std::vector<ProbeSample> samples = load_probe_samples(arguments.text("samples-file"));
    const int limit = arguments.integer("samples", 0);
    if (limit > 0 && static_cast<std::size_t>(limit) < samples.size()) {
      samples.resize(static_cast<std::size_t>(limit));
    }
    return std::make_unique<StaticProbeSampleSource>(std::move(samples));
  }
};

class ProfilerConfigFactory {
 public:
  static AttentionProfilerConfig from_arguments(const Arguments& arguments, const std::string& model_id) {
    AttentionProfilerConfig config;
    config.model_id = model_id;
    config.layer_index = arguments.integer("layer", kProfilerDefaults.layer_index);
    config.normalize_by_document_length = arguments.flag("normalize-by-length");
    config.progress_interval = arguments.integer("progress", kProfilerDefaults.progress_interval);
    InclusiveLayerRange span;
    if (parse_layer_range(arguments.text("layers"), "--layers", span)) {
      config.layer_span_begin = span.first;
      config.layer_span_end = span.last + 1;
    }
    return config;
  }
};

class LayerScanConfigFactory {
 public:
  static LayerScanConfig from_arguments(const Arguments& arguments, const std::string& model_id, int num_layers) {
    LayerScanConfig config;
    config.model_id = model_id;
    config.normalize_by_document_length = arguments.flag("normalize-by-length");
    config.progress_interval = arguments.integer("progress", kProfilerDefaults.progress_interval);
    InclusiveLayerRange range{0, num_layers - 1};
    parse_layer_range(arguments.text("layer-scan"), "--layer-scan", range);
    config.layer_begin = range.first;
    config.layer_end = range.last + 1;
    return config;
  }
};

class ModelSummaryReport {
 public:
  static std::string render(const LlamaConfig& config, const WeightRegistry& weights) {
    JsonWriter writer(kReportIndent);
    writer.begin_object();
    write_int(writer, "hidden_size", config.hidden_size);
    write_int(writer, "num_hidden_layers", config.num_hidden_layers);
    write_int(writer, "num_attention_heads", config.num_attention_heads);
    write_int(writer, "num_key_value_heads", config.num_key_value_heads);
    write_int(writer, "intermediate_size", config.intermediate_size);
    write_int(writer, "vocab_size", config.vocab_size);
    write_int(writer, "max_position_embeddings", config.max_position_embeddings);
    writer.key("rope_theta");
    writer.value_number(static_cast<double>(config.rope_theta));
    write_int(writer, "tensor_count", static_cast<std::int64_t>(weights.tensor_count()));
    write_int(writer, "checkpoint_bytes", static_cast<std::int64_t>(weights.total_bytes()));
    writer.end_object();
    return writer.str();
  }

  static void print_tensors(const WeightRegistry& weights, std::ostream& out) {
    for (const std::string& tensor_name : weights.names()) {
      const TensorView& tensor = weights.require(tensor_name);
      out << tensor_name << "\t" << dtype_name(tensor.dtype) << "\t[";
      for (std::size_t axis = 0; axis < tensor.shape.size(); ++axis) {
        if (axis != 0) out << ",";
        out << tensor.shape[axis];
      }
      out << "]\n";
    }
  }

 private:
  static void write_int(JsonWriter& writer, const char* key, std::int64_t value) {
    writer.key(key);
    writer.value_int(value);
  }
};

class LayerScanReport {
 public:
  LayerScanReport(const std::vector<AttentionProfile>& profiles, const BasinCriterion& criterion)
      : profiles_(profiles),
        criterion_(criterion),
        selected_layer_(select_shallowest_basin_layer(profiles, criterion)) {}

  int selected_layer() const { return selected_layer_; }

  const AttentionProfile& selected_profile() const {
    for (const AttentionProfile& profile : profiles_) {
      if (profile.layer_index == selected_layer_) return profile;
    }
    throw Error("no attention-basin layer was selected");
  }

  void log_rows() const {
    for (const AttentionProfile& profile : profiles_) {
      const BasinStatistics statistics = basin_statistics(profile.attention, criterion_);
      Log::info("layer %3d  first %.4f  interior %.4f  last %.4f  edge-ratio %6.2f  %s", profile.layer_index,
                statistics.first, statistics.interior_mean, statistics.last, statistics.edge_ratio,
                statistics.is_basin ? "basin" : "-");
    }
  }

  std::string render(const std::string& model_id) const {
    JsonWriter writer(kReportIndent);
    writer.begin_object();
    writer.key("model_id");
    writer.value_string(model_id);
    writer.key("sample_count");
    writer.value_int(profiles_.empty() ? 0 : profiles_.front().sample_count);
    writer.key("min_edge_ratio");
    writer.value_number(criterion_.min_edge_ratio);
    writer.key("selected_layer");
    writer.value_int(selected_layer_);
    writer.key("profiles");
    writer.begin_array();
    for (const AttentionProfile& profile : profiles_) {
      writer.begin_object();
      write_profile(writer, profile);
      write_basin(writer, basin_statistics(profile.attention, criterion_));
      writer.end_object();
    }
    writer.end_array();
    writer.end_object();
    return writer.str();
  }

 private:
  static void write_profile(JsonWriter& writer, const AttentionProfile& profile) {
    writer.key("layer_index");
    writer.value_int(profile.layer_index);
    writer.key("attention");
    writer.begin_array();
    for (const double value : profile.attention) writer.value_number(value);
    writer.end_array();
    writer.key("position_order");
    writer.begin_array();
    for (const int slot : profile.position_order) writer.value_int(slot);
    writer.end_array();
  }

  static void write_basin(JsonWriter& writer, const BasinStatistics& statistics) {
    writer.key("basin");
    writer.begin_object();
    writer.key("first");
    writer.value_number(statistics.first);
    writer.key("last");
    writer.value_number(statistics.last);
    writer.key("interior_mean");
    writer.value_number(statistics.interior_mean);
    writer.key("edge_ratio");
    writer.value_number(statistics.edge_ratio);
    writer.key("is_basin");
    writer.value_bool(statistics.is_basin);
    writer.end_object();
  }

  const std::vector<AttentionProfile>& profiles_;
  BasinCriterion criterion_;
  int selected_layer_;
};

}  // namespace

class InspectCommand : public ICommand {
 public:
  const char* name() const override { return "inspect"; }
  const char* usage() const override { return "  inspect   --model DIR [--list-tensors]\n"; }

  int run(const Arguments& arguments) const override {
    const std::string model_directory = arguments.required_text("model");
    const LlamaConfig config = LlamaConfig::from_directory(model_directory);
    const WeightRegistry weights = CheckpointLoader::with_default_factories().load_directory(model_directory);

    std::cout << ModelSummaryReport::render(config, weights) << "\n";
    if (arguments.flag("list-tensors")) ModelSummaryReport::print_tensors(weights, std::cout);
    return kExitSuccess;
  }
};

class ProfileCommand : public ICommand {
 public:
  const char* name() const override { return "profile"; }
  const char* usage() const override {
    return "  profile   --model DIR [--out FILE]\n"
           "            probe samples: --samples-file FILE.jsonl [--samples N]\n"
           "                        or --slots N --samples N [--words-per-document N] [--seed N]\n"
           "            single layer: [--layer N=5] [--layers FIRST:LAST]\n"
           "            layer scan:   --layer-scan FIRST:LAST | --auto-layer [--layer-scan FIRST:LAST]\n"
           "                          [--min-edge-ratio F=1.5]\n"
           "            --auto-layer picks the shallowest layer whose first and last slot both\n"
           "            exceed the interior mean by --min-edge-ratio and writes that layer's\n"
           "            profile to --out\n";
  }

  int run(const Arguments& arguments) const override {
    const Engine engine(arguments.required_text("model"), EngineOptionsFactory::from_arguments(arguments));
    const std::unique_ptr<IProbeSampleSource> samples = ProbeSourceFactory::from_arguments(arguments);
    const std::string model_id = arguments.text("model-id", engine.model_directory());

    if (arguments.has("layer-scan") || arguments.flag("auto-layer")) {
      return scan(arguments, engine, *samples, model_id);
    }
    return profile_single_layer(arguments, engine, *samples, model_id);
  }

 private:
  static int profile_single_layer(const Arguments& arguments, const Engine& engine,
                                  const IProbeSampleSource& samples, const std::string& model_id) {
    const AttentionProfilerConfig config = ProfilerConfigFactory::from_arguments(arguments, model_id);
    const AttentionProfile profile = engine.build_profile(samples, config);

    const std::string output = arguments.text("out");
    if (!output.empty()) {
      profile.save(output);
      Log::info("attention profile written to %s", output.c_str());
    }
    std::cout << profile.to_json_string() << "\n";
    return kExitSuccess;
  }

  static int scan(const Arguments& arguments, const Engine& engine, const IProbeSampleSource& samples,
                  const std::string& model_id) {
    const LayerScanConfig config = LayerScanConfigFactory::from_arguments(arguments, model_id, engine.num_layers());
    const std::vector<AttentionProfile> profiles = engine.scan_layers(samples, config);

    BasinCriterion criterion;
    criterion.min_edge_ratio = arguments.real("min-edge-ratio", kBasinDefaults.min_edge_ratio);
    const LayerScanReport report(profiles, criterion);
    report.log_rows();

    const bool auto_layer = arguments.flag("auto-layer");
    if (report.selected_layer() >= 0) {
      Log::info("shallowest attention-basin layer: %d", report.selected_layer());
    } else {
      Log::warn("no layer in [%d,%d] satisfies edge-ratio >= %.2f", config.layer_begin, config.layer_end - 1,
                criterion.min_edge_ratio);
    }

    const std::string rendered = report.render(model_id);
    const std::string output = arguments.text("out");
    if (!output.empty()) {
      if (auto_layer) {
        if (report.selected_layer() < 0) throw Error("--auto-layer found no attention-basin layer");
        report.selected_profile().save(output);
        Log::info("attention profile for layer %d written to %s", report.selected_layer(), output.c_str());
      } else {
        write_text(output, rendered);
        Log::info("layer scan written to %s", output.c_str());
      }
    }
    std::cout << rendered << "\n";
    return report.selected_layer() >= 0 || !auto_layer ? kExitSuccess : kExitNoBasinLayer;
  }
};

class CommandRegistry {
 public:
  CommandRegistry() {
    commands_.push_back(std::make_unique<InspectCommand>());
    commands_.push_back(std::make_unique<ProfileCommand>());
  }

  const ICommand* find(const std::string& name) const {
    for (const auto& command : commands_) {
      if (name == command->name()) return command.get();
    }
    return nullptr;
  }

  void print_usage() const {
    std::cout << "attnrank <command> [options]\n\ncommands:\n";
    for (const auto& command : commands_) std::cout << command->usage();
    std::cout
        << "\nsamples-file lines: {\"question\": \"...\", \"documents\": [\"text\", ...]} or\n"
        << "                    {\"question\": \"...\", \"documents\": "
        << "[{\"title\": \"...\", \"text\": \"...\"}, ...]}\n\n"
        << "shared options:\n"
        << "  --device N            cuda device index (default: most free memory)\n"
        << "  --max-sequence N      context window in tokens (default 2048)\n"
        << "  --chunk N             prefill chunk size in tokens (default 512)\n"
        << "  --max-document-tokens truncate each document to N tokens (default 0 = no truncation)\n"
        << "  --chat-format NAME    vicuna, chatml, llama2/mistral or plain (auto-detected)\n"
        << "  --system-prompt TEXT  <<SYS>> block for the llama2/mistral format\n"
        << "  --document-titles     render each document as 'Title: text'\n"
        << "  --document-separator  text between documents (default \\n)\n"
        << "  --instruction TEXT    replace the default task instruction\n"
        << "  --example TEXT        one-shot example block placed between instruction and documents\n"
        << "  --normalize-by-length divide each document's attention mass by its token count\n"
        << "  --model-id TEXT       identifier stored in the profile (default: --model)\n"
        << "  --progress N          log every N probe samples (default 25)\n"
        << "  --verbose             enable debug logging\n";
  }

 private:
  std::vector<std::unique_ptr<ICommand>> commands_;
};

}  // namespace attnrank::cli

int main(int argc, char** argv) {
  using namespace attnrank;
  using namespace attnrank::cli;

  const CommandRegistry registry;
  if (argc < 2) {
    registry.print_usage();
    return kExitFailure;
  }
  const std::string name = argv[1];
  if (name == "--help" || name == "-h" || name == "help") {
    registry.print_usage();
    return kExitSuccess;
  }
  const ICommand* command = registry.find(name);
  if (command == nullptr) {
    std::cerr << "unknown command '" << name << "'\n\n";
    registry.print_usage();
    return kExitFailure;
  }
  try {
    const Arguments arguments(argc, argv, 2);
    if (arguments.flag("verbose")) Log::set_threshold(LogLevel::Debug);
    if (arguments.flag("quiet")) Log::set_threshold(LogLevel::Warn);
    return command->run(arguments);
  } catch (const std::exception& error) {
    std::cerr << "attnrank: " << error.what() << "\n";
    return kExitFailure;
  }
}

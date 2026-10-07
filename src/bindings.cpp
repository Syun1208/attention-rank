#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "attnrank/engine.hpp"

namespace py = pybind11;
using namespace attnrank;

namespace {

using ProgressCallback = std::function<void(std::size_t, std::size_t)>;

class PythonProgressObserver : public IProgressObserver {
 public:
  explicit PythonProgressObserver(ProgressCallback callback) : callback_(std::move(callback)) {}

  void on_progress(std::size_t completed, std::size_t total) override {
    py::gil_scoped_acquire gil;
    callback_(completed, total);
  }

 private:
  ProgressCallback callback_;
};

std::shared_ptr<IProgressObserver> make_observer(const std::optional<ProgressCallback>& callback) {
  if (!callback.has_value()) return nullptr;
  return std::make_shared<PythonProgressObserver>(*callback);
}

std::shared_ptr<Engine> make_engine(const std::string& model_directory, const LlamaRuntimeOptions& runtime,
                                    const std::string& chat_format, const std::string& system_prompt,
                                    const DocumentPromptTemplate& prompt_template) {
  EngineOptions options;
  options.runtime = runtime;
  options.prompt.chat_format = chat_format;
  options.prompt.system_prompt = system_prompt;
  options.prompt.prompt_template = prompt_template;
  return std::make_shared<Engine>(model_directory, options);
}

std::vector<double> measure_attention(const Engine& engine, const StructuredPrompt& prompt, int layer_index,
                                      int layer_span_begin, int layer_span_end, bool average_all_layers,
                                      bool normalize_by_length) {
  AttentionProfilerConfig layers;
  layers.layer_index = layer_index;
  layers.layer_span_begin = layer_span_begin;
  layers.layer_span_end = layer_span_end;
  layers.average_all_layers = average_all_layers;
  layers.normalize_by_document_length = normalize_by_length;
  return engine.measure_attention(prompt, layers);
}

AttentionLayerProbeResult measure_attention_by_layer(const Engine& engine, const StructuredPrompt& prompt,
                                                     int layer_begin, int layer_end,
                                                     bool normalize_by_length) {
  LayerScanConfig layers;
  layers.layer_begin = layer_begin;
  layers.layer_end = layer_end;
  layers.normalize_by_document_length = normalize_by_length;
  return engine.measure_attention_by_layer(prompt, layers);
}

GenerationResult generate(const Engine& engine, const TokenSequence& prompt_tokens, int max_new_tokens,
                          const std::vector<std::string>& stop_sequences, float temperature, float top_p,
                          int top_k, std::uint64_t seed) {
  GenerationOptions options;
  options.max_new_tokens = max_new_tokens;
  options.stop_sequences = stop_sequences;
  options.temperature = temperature;
  options.top_p = top_p;
  options.top_k = top_k;
  options.seed = seed;
  return engine.generate(prompt_tokens, options);
}

AttentionProfile build_attention_profile(const Engine& engine, const std::vector<ProbeSample>& samples,
                                         const AttentionProfilerConfig& config,
                                         const std::optional<ProgressCallback>& progress) {
  return engine.build_profile(StaticProbeSampleSource(samples), config, make_observer(progress));
}

std::vector<AttentionProfile> scan_attention_layers(const Engine& engine, const std::vector<ProbeSample>& samples,
                                                    const LayerScanConfig& config,
                                                    const std::optional<ProgressCallback>& progress) {
  return engine.scan_layers(StaticProbeSampleSource(samples), config, make_observer(progress));
}

std::vector<ProbeSample> synthetic_probe_samples(const SyntheticProbeConfig& config) {
  const SyntheticProbeSampleSource source(config);
  std::vector<ProbeSample> samples;
  samples.reserve(source.sample_count());
  for (std::size_t index = 0; index < source.sample_count(); ++index) samples.push_back(source.sample_at(index));
  return samples;
}

std::shared_ptr<IReranker> make_reranker(const std::string& name, const AttentionProfile* profile,
                                         std::uint64_t seed, bool lim_start_first) {
  RerankerOptions options;
  options.profile = profile;
  options.seed = seed;
  options.lim_start_first = lim_start_first;
  return RerankerFactory::create(name, options);
}

py::str lenient_text(const std::string& text) {
  PyObject* decoded = PyUnicode_DecodeUTF8(text.data(), static_cast<Py_ssize_t>(text.size()), "replace");
  if (decoded == nullptr) throw py::error_already_set();
  return py::reinterpret_steal<py::str>(decoded);
}

void set_log_level(const std::string& level) {
  if (level == "debug") return Log::set_threshold(LogLevel::Debug);
  if (level == "info") return Log::set_threshold(LogLevel::Info);
  if (level == "warn") return Log::set_threshold(LogLevel::Warn);
  if (level == "error") return Log::set_threshold(LogLevel::Error);
  throw Error("unknown log level '" + level + "'");
}

}  // namespace

PYBIND11_MODULE(_core, module) {
  module.doc() = "AttnRank core: CUDA Llama forward pass, attention probing, profiling, reranking";

  py::register_exception<Error>(module, "AttnRankError", PyExc_RuntimeError);

  py::class_<Document>(module, "Document")
      .def(py::init([](std::string id, std::string title, std::string text) {
             Document document;
             document.id = std::move(id);
             document.title = std::move(title);
             document.text = std::move(text);
             return document;
           }),
           py::arg("id") = "", py::arg("title") = "", py::arg("text") = "")
      .def_readwrite("id", &Document::id)
      .def_readwrite("title", &Document::title)
      .def_readwrite("text", &Document::text)
      .def("__repr__", [](const Document& document) {
        return "Document(id=" + document.id + ", title=" + document.title +
               ", chars=" + std::to_string(document.text.size()) + ")";
      });

  py::class_<ProbeSample>(module, "ProbeSample")
      .def(py::init([](std::string question, std::vector<Document> documents) {
             ProbeSample sample;
             sample.question = std::move(question);
             sample.documents = std::move(documents);
             return sample;
           }),
           py::arg("question"), py::arg("documents"))
      .def_readwrite("question", &ProbeSample::question)
      .def_readwrite("documents", &ProbeSample::documents);

  py::class_<TokenSpan>(module, "TokenSpan")
      .def(py::init<>())
      .def_readwrite("begin", &TokenSpan::begin)
      .def_readwrite("end", &TokenSpan::end)
      .def("length", &TokenSpan::length)
      .def("__repr__", [](const TokenSpan& span) {
        return "TokenSpan(" + std::to_string(span.begin) + ", " + std::to_string(span.end) + ")";
      });

  py::class_<StructuredPrompt>(module, "StructuredPrompt")
      .def_readonly("tokens", &StructuredPrompt::tokens)
      .def_readonly("document_spans", &StructuredPrompt::document_spans)
      .def_readonly("query_span", &StructuredPrompt::query_span)
      .def_property_readonly("text", [](const StructuredPrompt& prompt) { return lenient_text(prompt.text); });

  py::class_<LlamaRuntimeOptions>(module, "RuntimeOptions")
      .def(py::init([](int device_index, int max_sequence_length, int prefill_chunk_tokens) {
             LlamaRuntimeOptions options;
             options.device_index = device_index;
             options.max_sequence_length = max_sequence_length;
             options.prefill_chunk_tokens = prefill_chunk_tokens;
             return options;
           }),
           py::arg("device_index") = -1, py::arg("max_sequence_length") = 2048,
           py::arg("prefill_chunk_tokens") = 512)
      .def_readwrite("device_index", &LlamaRuntimeOptions::device_index)
      .def_readwrite("max_sequence_length", &LlamaRuntimeOptions::max_sequence_length)
      .def_readwrite("prefill_chunk_tokens", &LlamaRuntimeOptions::prefill_chunk_tokens);

  py::class_<LlamaConfig>(module, "LlamaConfig")
      .def_readonly("hidden_size", &LlamaConfig::hidden_size)
      .def_readonly("intermediate_size", &LlamaConfig::intermediate_size)
      .def_readonly("num_hidden_layers", &LlamaConfig::num_hidden_layers)
      .def_readonly("num_attention_heads", &LlamaConfig::num_attention_heads)
      .def_readonly("num_key_value_heads", &LlamaConfig::num_key_value_heads)
      .def_readonly("vocab_size", &LlamaConfig::vocab_size)
      .def_readonly("max_position_embeddings", &LlamaConfig::max_position_embeddings)
      .def_readonly("rms_norm_eps", &LlamaConfig::rms_norm_eps)
      .def_readonly("rope_theta", &LlamaConfig::rope_theta)
      .def_readonly("bos_token_id", &LlamaConfig::bos_token_id)
      .def_readonly("eos_token_id", &LlamaConfig::eos_token_id)
      .def_readonly("tie_word_embeddings", &LlamaConfig::tie_word_embeddings);

  py::class_<DocumentPromptTemplate>(module, "PromptTemplate")
      .def(py::init<>())
      .def_readwrite("instruction", &DocumentPromptTemplate::instruction)
      .def_readwrite("example", &DocumentPromptTemplate::example)
      .def_readwrite("documents_header", &DocumentPromptTemplate::documents_header)
      .def_readwrite("document_label", &DocumentPromptTemplate::document_label)
      .def_readwrite("document_index_placeholder", &DocumentPromptTemplate::document_index_placeholder)
      .def_readwrite("document_separator", &DocumentPromptTemplate::document_separator)
      .def_readwrite("question_prefix", &DocumentPromptTemplate::question_prefix)
      .def_readwrite("answer_prefix", &DocumentPromptTemplate::answer_prefix)
      .def_readwrite("max_document_tokens", &DocumentPromptTemplate::max_document_tokens)
      .def_readwrite("include_titles", &DocumentPromptTemplate::include_titles)
      .def_readwrite("extra_stop_sequences", &DocumentPromptTemplate::extra_stop_sequences);

  py::class_<AttentionProfilerConfig>(module, "ProfilerConfig")
      .def(py::init<>())
      .def_readwrite("model_id", &AttentionProfilerConfig::model_id)
      .def_readwrite("layer_index", &AttentionProfilerConfig::layer_index)
      .def_readwrite("average_all_layers", &AttentionProfilerConfig::average_all_layers)
      .def_readwrite("layer_span_begin", &AttentionProfilerConfig::layer_span_begin)
      .def_readwrite("layer_span_end", &AttentionProfilerConfig::layer_span_end)
      .def_readwrite("normalize_by_document_length", &AttentionProfilerConfig::normalize_by_document_length)
      .def_readwrite("progress_interval", &AttentionProfilerConfig::progress_interval);

  py::class_<LayerScanConfig>(module, "LayerScanConfig")
      .def(py::init<>())
      .def_readwrite("model_id", &LayerScanConfig::model_id)
      .def_readwrite("layer_begin", &LayerScanConfig::layer_begin)
      .def_readwrite("layer_end", &LayerScanConfig::layer_end)
      .def_readwrite("normalize_by_document_length", &LayerScanConfig::normalize_by_document_length)
      .def_readwrite("progress_interval", &LayerScanConfig::progress_interval);

  py::class_<SyntheticProbeConfig>(module, "SyntheticProbeConfig")
      .def(py::init<>())
      .def_readwrite("document_slots", &SyntheticProbeConfig::document_slots)
      .def_readwrite("sample_count", &SyntheticProbeConfig::sample_count)
      .def_readwrite("words_per_document", &SyntheticProbeConfig::words_per_document)
      .def_readwrite("seed", &SyntheticProbeConfig::seed);

  py::class_<AttentionProfile>(module, "AttentionProfile")
      .def(py::init<>())
      .def_readwrite("model_id", &AttentionProfile::model_id)
      .def_readwrite("layer_index", &AttentionProfile::layer_index)
      .def_readwrite("sample_count", &AttentionProfile::sample_count)
      .def_readwrite("attention", &AttentionProfile::attention)
      .def_readwrite("position_order", &AttentionProfile::position_order)
      .def("recompute_position_order", &AttentionProfile::recompute_position_order)
      .def("to_json_string", &AttentionProfile::to_json_string)
      .def("save", &AttentionProfile::save, py::arg("path"))
      .def_static("load", &AttentionProfile::load, py::arg("path"))
      .def("__repr__", [](const AttentionProfile& profile) { return profile.to_json_string(); });

  py::class_<BasinCriterion>(module, "BasinCriterion")
      .def(py::init([](double min_edge_ratio) {
             BasinCriterion criterion;
             criterion.min_edge_ratio = min_edge_ratio;
             return criterion;
           }),
           py::arg("min_edge_ratio") = 1.5)
      .def_readwrite("min_edge_ratio", &BasinCriterion::min_edge_ratio);

  py::class_<BasinStatistics>(module, "BasinStatistics")
      .def_readonly("first", &BasinStatistics::first)
      .def_readonly("last", &BasinStatistics::last)
      .def_readonly("interior_mean", &BasinStatistics::interior_mean)
      .def_readonly("edge_ratio", &BasinStatistics::edge_ratio)
      .def_readonly("is_basin", &BasinStatistics::is_basin)
      .def("__repr__", [](const BasinStatistics& statistics) {
        return "BasinStatistics(first=" + std::to_string(statistics.first) +
               ", interior_mean=" + std::to_string(statistics.interior_mean) +
               ", last=" + std::to_string(statistics.last) +
               ", edge_ratio=" + std::to_string(statistics.edge_ratio) +
               ", is_basin=" + (statistics.is_basin ? "True" : "False") + ")";
      });

  py::class_<AttentionLayerProbeResult>(module, "LayerProbeResult")
      .def_readonly("layer_begin", &AttentionLayerProbeResult::layer_begin)
      .def_readonly("attention_by_layer", &AttentionLayerProbeResult::attention_by_layer);

  py::class_<GenerationResult>(module, "GenerationResult")
      .def_readonly("generated_tokens", &GenerationResult::generated_tokens)
      .def_property_readonly("text", [](const GenerationResult& result) { return lenient_text(result.text); })
      .def_property_readonly("raw_bytes", [](const GenerationResult& result) { return py::bytes(result.text); })
      .def_readonly("prompt_token_count", &GenerationResult::prompt_token_count)
      .def_readonly("hit_stop_sequence", &GenerationResult::hit_stop_sequence);

  py::class_<ScoredDocument>(module, "ScoredDocument")
      .def(py::init([](std::size_t index, double score) {
             ScoredDocument scored;
             scored.index = index;
             scored.score = score;
             return scored;
           }),
           py::arg("index"), py::arg("score") = 0.0)
      .def_readwrite("index", &ScoredDocument::index)
      .def_readwrite("score", &ScoredDocument::score)
      .def("__repr__", [](const ScoredDocument& scored) {
        return "ScoredDocument(index=" + std::to_string(scored.index) +
               ", score=" + std::to_string(scored.score) + ")";
      });

  py::class_<IReranker, std::shared_ptr<IReranker>>(module, "Reranker")
      .def_property_readonly("name", &IReranker::name)
      .def("rerank", &IReranker::rerank, py::arg("ranked"));
  py::class_<DescendingReranker, IReranker, std::shared_ptr<DescendingReranker>>(module, "DescendingReranker")
      .def(py::init<>());
  py::class_<AscendingReranker, IReranker, std::shared_ptr<AscendingReranker>>(module, "AscendingReranker")
      .def(py::init<>());
  py::class_<RandomReranker, IReranker, std::shared_ptr<RandomReranker>>(module, "RandomReranker")
      .def(py::init<std::uint64_t>(), py::arg("seed") = 1234);
  py::class_<LostInTheMiddleReranker, IReranker, std::shared_ptr<LostInTheMiddleReranker>>(
      module, "LostInTheMiddleReranker")
      .def(py::init<bool>(), py::arg("start_first") = false);
  py::class_<AttentionReranker, IReranker, std::shared_ptr<AttentionReranker>>(module, "AttentionReranker")
      .def(py::init<std::vector<int>>(), py::arg("position_order"))
      .def_property_readonly("position_order", &AttentionReranker::position_order);

  module.def("reranker_names", [] { return RerankerFactory::names(); });
  module.def("make_reranker", &make_reranker, py::arg("name"), py::arg("profile") = nullptr,
             py::arg("seed") = 1234, py::arg("lim_start_first") = false);
  module.def("resolve_position_order", &resolve_position_order, py::arg("preferred"), py::arg("count"));
  module.def("basin_statistics", &basin_statistics, py::arg("attention"), py::arg("criterion"));
  module.def("select_shallowest_basin_layer", &select_shallowest_basin_layer, py::arg("profiles"),
             py::arg("criterion"));
  module.def("load_probe_samples", &load_probe_samples, py::arg("jsonl_path"));
  module.def("synthetic_probe_samples", &synthetic_probe_samples, py::arg("config"));
  module.def("chat_format_names", [] { return ChatFormatterFactory::names(); });
  module.def("detect_chat_format", &ChatFormatterFactory::detect, py::arg("model_directory"));
  module.def("set_log_level", &set_log_level, py::arg("level"));

  py::class_<Engine, std::shared_ptr<Engine>>(module, "Engine")
      .def(py::init(&make_engine), py::arg("model_directory"), py::arg("options") = LlamaRuntimeOptions{},
           py::arg("chat_format") = "", py::arg("system_prompt") = "",
           py::arg("prompt_template") = DocumentPromptTemplate{}, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("config", &Engine::config)
      .def_property_readonly("num_layers", &Engine::num_layers)
      .def_property_readonly("model_directory", &Engine::model_directory)
      .def_property_readonly("chat_format", &Engine::chat_format)
      .def_property("prompt_template", &Engine::prompt_template, &Engine::set_prompt_template)
      .def("stop_sequences", &Engine::stop_sequences)
      .def("build_prompt", &Engine::build_prompt, py::arg("question"), py::arg("documents"))
      .def("encode", &Engine::encode, py::arg("text"), py::arg("add_bos") = true)
      .def(
          "decode",
          [](const Engine& engine, const TokenSequence& tokens) { return lenient_text(engine.decode(tokens)); },
          py::arg("tokens"))
      .def("measure_attention", &measure_attention, py::arg("prompt"), py::arg("layer_index") = 0,
           py::arg("layer_span_begin") = -1, py::arg("layer_span_end") = -1,
           py::arg("average_all_layers") = false, py::arg("normalize_by_length") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("measure_attention_by_layer", &measure_attention_by_layer, py::arg("prompt"),
           py::arg("layer_begin") = 0, py::arg("layer_end") = -1, py::arg("normalize_by_length") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("generate", &generate, py::arg("prompt_tokens"), py::arg("max_new_tokens") = 128,
           py::arg("stop_sequences") = std::vector<std::string>{}, py::arg("temperature") = 0.0f,
           py::arg("top_p") = 0.9f, py::arg("top_k") = 0, py::arg("seed") = 1234,
           py::call_guard<py::gil_scoped_release>());

  module.def("build_attention_profile", &build_attention_profile, py::arg("engine"), py::arg("samples"),
             py::arg("config"), py::arg("progress") = py::none(), py::call_guard<py::gil_scoped_release>());
  module.def("scan_attention_layers", &scan_attention_layers, py::arg("engine"), py::arg("samples"),
             py::arg("config"), py::arg("progress") = py::none(), py::call_guard<py::gil_scoped_release>());
}

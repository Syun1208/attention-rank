#include "attnrank/attnrank.hpp"

#include <algorithm>
#include <fstream>
#include <memory>
#include <limits>
#include <numeric>
#include <utility>

namespace attnrank {
namespace {

constexpr std::uint64_t kSlotStride = 1315423911ULL;
constexpr std::uint64_t kSampleStride = 2654435761ULL;
constexpr std::uint64_t kGoldenRatioHash = 0x9E3779B97F4A7C15ull;
constexpr std::uint64_t kMixMultiplierA = 0xBF58476D1CE4E5B9ull;
constexpr std::uint64_t kMixMultiplierB = 0x94D049BB133111EBull;
constexpr std::size_t kAverageWordBytes = 8;
constexpr std::size_t kMinimumBasinSlots = 3;

std::uint64_t mix(std::uint64_t value) {
  value += kGoldenRatioHash;
  value = (value ^ (value >> 30)) * kMixMultiplierA;
  value = (value ^ (value >> 27)) * kMixMultiplierB;
  return value ^ (value >> 31);
}

const std::vector<std::string>& filler_vocabulary() {
  static const std::vector<std::string> words = {
      "policy",   "capital",  "reserve",   "clause",   "report",   "market",   "auditor",
      "ledger",   "issuer",   "mandate",   "exposure", "treasury", "guidance", "compliance",
      "buffer",   "ratio",    "circular",  "annex",    "review",   "quarter",  "threshold",
      "framework","liquidity","settlement","custodian","valuation","provision","disclosure",
      "counterpart","schedule","registry", "supervisor","directive","committee","statement",
      "portfolio","benchmark","tolerance", "governance","procedure","aggregate","instrument",
      "obligation","assurance","allocation","resolution","monitoring","submission","classification",
      "requirement","limitation","notification","designation","verification","adjustment","calculation"};
  return words;
}

Document document_from_json(const Json& item, std::size_t sample_index, std::size_t slot) {
  Document document;
  document.id = "sample-" + std::to_string(sample_index) + "-" + std::to_string(slot);
  if (item.is_string()) {
    document.text = item.as_string();
    return document;
  }
  if (!item.is_object()) {
    throw Error("probe sample " + std::to_string(sample_index) + " document " +
                std::to_string(slot) + " must be a string or an object");
  }
  const Json* id = item.find("id");
  if (id != nullptr && !id->as_string_or("").empty()) document.id = id->as_string_or("");
  const Json* title = item.find("title");
  document.title = title != nullptr ? title->as_string_or("") : "";
  const Json* text = item.find("text");
  if (text == nullptr) {
    throw Error("probe sample " + std::to_string(sample_index) + " document " +
                std::to_string(slot) + " has no 'text'");
  }
  document.text = text->as_string_or("");
  return document;
}

std::vector<ScoredDocument> place(const std::vector<ScoredDocument>& ranked,
                                  const std::vector<int>& order) {
  std::vector<ScoredDocument> result(ranked.size());
  for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
    result[static_cast<std::size_t>(order[rank])] = ranked[rank];
  }
  return result;
}

}  // namespace

StaticProbeSampleSource::StaticProbeSampleSource(std::vector<ProbeSample> samples)
    : samples_(std::move(samples)) {
  if (samples_.empty()) throw Error("probe needs at least one sample");
  document_slots_ = static_cast<int>(samples_.front().documents.size());
  if (document_slots_ <= 0) throw Error("probe samples need at least one document");
  for (std::size_t index = 0; index < samples_.size(); ++index) {
    if (static_cast<int>(samples_[index].documents.size()) != document_slots_) {
      throw Error("probe sample " + std::to_string(index) + " has " +
                  std::to_string(samples_[index].documents.size()) +
                  " documents but the first sample has " + std::to_string(document_slots_));
    }
  }
}

SyntheticProbeSampleSource::SyntheticProbeSampleSource(const SyntheticProbeConfig& config)
    : config_(config) {
  if (config_.document_slots <= 0) throw Error("probe needs at least one document slot");
  if (config_.sample_count <= 0) throw Error("probe needs at least one sample");
}

ProbeSample SyntheticProbeSampleSource::sample_at(std::size_t index) const {
  const std::vector<std::string>& vocabulary = filler_vocabulary();
  ProbeSample sample;
  sample.documents.reserve(static_cast<std::size_t>(config_.document_slots));

  for (int slot = 0; slot < config_.document_slots; ++slot) {
    std::uint64_t state = mix(config_.seed + static_cast<std::uint64_t>(index) * kSlotStride +
                              static_cast<std::uint64_t>(slot));

    Document document;
    document.id = "probe-" + std::to_string(index) + "-" + std::to_string(slot);
    document.title = "probe";
    document.text.reserve(static_cast<std::size_t>(config_.words_per_document) * kAverageWordBytes);

    for (int word = 0; word < config_.words_per_document; ++word) {
      state = mix(state);
      if (word != 0) document.text.push_back(' ');
      document.text += vocabulary[state % vocabulary.size()];
    }
    document.text.push_back('.');
    sample.documents.push_back(std::move(document));
  }

  const std::uint64_t question_state =
      mix(config_.seed ^ (static_cast<std::uint64_t>(index) * kSampleStride));
  sample.question = "Which section of the " + vocabulary[question_state % vocabulary.size()] +
                    " document states the applicable " +
                    vocabulary[mix(question_state) % vocabulary.size()] + "?";
  return sample;
}

std::vector<ProbeSample> load_probe_samples(const std::string& jsonl_path) {
  std::vector<ProbeSample> samples;
  const std::vector<Json> lines = Json::parse_lines_file(jsonl_path);
  samples.reserve(lines.size());
  for (std::size_t index = 0; index < lines.size(); ++index) {
    const Json& root = lines[index];
    if (!root.is_object()) throw Error("probe sample " + std::to_string(index) + " is not a JSON object");
    const Json* question = root.find("question");
    if (question == nullptr) throw Error("probe sample " + std::to_string(index) + " has no 'question'");
    const Json* documents = root.find("documents");
    if (documents == nullptr || !documents->is_array()) {
      throw Error("probe sample " + std::to_string(index) + " has no 'documents' array");
    }
    ProbeSample sample;
    sample.question = question->as_string_or("");
    sample.documents.reserve(documents->size());
    for (std::size_t slot = 0; slot < documents->size(); ++slot) {
      sample.documents.push_back(document_from_json(documents->at(slot), index, slot));
    }
    samples.push_back(std::move(sample));
  }
  return samples;
}

AttentionProfile AttentionProfile::from_accumulated(std::string model_id, int layer_index,
                                                    std::size_t sample_count,
                                                    const std::vector<double>& accumulated) {
  AttentionProfile profile;
  profile.model_id = std::move(model_id);
  profile.layer_index = layer_index;
  profile.sample_count = static_cast<int>(sample_count);
  profile.attention.resize(accumulated.size());
  double total = 0.0;
  for (std::size_t slot = 0; slot < accumulated.size(); ++slot) {
    profile.attention[slot] = accumulated[slot] / static_cast<double>(sample_count);
    total += profile.attention[slot];
  }
  if (total > 0.0) {
    for (double& value : profile.attention) value /= total;
  }
  profile.recompute_position_order();
  return profile;
}

void AttentionProfile::recompute_position_order() {
  position_order.resize(attention.size());
  std::iota(position_order.begin(), position_order.end(), 0);
  std::stable_sort(position_order.begin(), position_order.end(), [this](int left, int right) {
    return attention[static_cast<std::size_t>(left)] > attention[static_cast<std::size_t>(right)];
  });
}

std::string AttentionProfile::to_json_string() const {
  JsonWriter writer(2);
  writer.begin_object();
  writer.key("model_id");
  writer.value_string(model_id);
  writer.key("layer_index");
  writer.value_int(layer_index);
  writer.key("sample_count");
  writer.value_int(sample_count);
  writer.key("document_slots");
  writer.value_int(static_cast<std::int64_t>(attention.size()));
  writer.key("attention");
  writer.begin_array();
  for (const double value : attention) writer.value_number(value);
  writer.end_array();
  writer.key("position_order");
  writer.begin_array();
  for (const int slot : position_order) writer.value_int(slot);
  writer.end_array();
  writer.end_object();
  return writer.str();
}

void AttentionProfile::save(const std::string& path) const {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) throw Error("cannot write attention profile to '" + path + "'");
  stream << to_json_string() << "\n";
}

AttentionProfile AttentionProfile::from_json(const Json& root) {
  AttentionProfile profile;
  profile.model_id = root.find("model_id") != nullptr ? root.at("model_id").as_string_or("") : "";
  profile.layer_index = static_cast<int>(root.at("layer_index").as_int64(0));
  profile.sample_count = static_cast<int>(root.at("sample_count").as_int64(0));

  const Json& attention = root.at("attention");
  profile.attention.reserve(attention.size());
  for (std::size_t index = 0; index < attention.size(); ++index) {
    profile.attention.push_back(attention.at(index).as_double(0.0));
  }

  const Json* order = root.find("position_order");
  if (order != nullptr && order->size() == profile.attention.size()) {
    profile.position_order.reserve(order->size());
    for (std::size_t index = 0; index < order->size(); ++index) {
      profile.position_order.push_back(static_cast<int>(order->at(index).as_int64(0)));
    }
  } else {
    profile.recompute_position_order();
  }
  return profile;
}

AttentionProfile AttentionProfile::load(const std::string& path) {
  return from_json(Json::parse_file(path));
}

BasinStatistics basin_statistics(const std::vector<double>& attention,
                                 const BasinCriterion& criterion) {
  BasinStatistics statistics;
  if (attention.size() < kMinimumBasinSlots) return statistics;

  statistics.first = attention.front();
  statistics.last = attention.back();
  double interior = 0.0;
  for (std::size_t slot = 1; slot + 1 < attention.size(); ++slot) interior += attention[slot];
  statistics.interior_mean = interior / static_cast<double>(attention.size() - 2);

  const double weakest_edge = std::min(statistics.first, statistics.last);
  if (statistics.interior_mean > 0.0) {
    statistics.edge_ratio = weakest_edge / statistics.interior_mean;
  } else {
    statistics.edge_ratio = weakest_edge > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
  }
  statistics.is_basin = statistics.edge_ratio >= criterion.min_edge_ratio;
  return statistics;
}

int select_shallowest_basin_layer(const std::vector<AttentionProfile>& profiles,
                                  const BasinCriterion& criterion) {
  int selected = -1;
  for (const AttentionProfile& profile : profiles) {
    if (!basin_statistics(profile.attention, criterion).is_basin) continue;
    if (selected < 0 || profile.layer_index < selected) selected = profile.layer_index;
  }
  return selected;
}

void LogProgressObserver::on_progress(std::size_t completed, std::size_t total) {
  if (interval_ <= 0 || completed % static_cast<std::size_t>(interval_) != 0) return;
  Log::info("probed %zu/%zu samples", completed, total);
}

std::shared_ptr<IProgressObserver> make_progress_observer(int log_interval) {
  if (log_interval <= 0) return std::make_shared<NullProgressObserver>();
  return std::make_shared<LogProgressObserver>(log_interval);
}

AttentionProbeRun::AttentionProbeRun(ProbeDependencies dependencies) : dependencies_(std::move(dependencies)) {
  if (dependencies_.attention_provider == nullptr) throw Error("attention probe requires a provider");
  if (dependencies_.prompt_builder == nullptr) throw Error("attention probe requires a prompt builder");
  if (dependencies_.progress == nullptr) dependencies_.progress = std::make_shared<NullProgressObserver>();
}

std::size_t AttentionProbeRun::run(const IProbeSampleSource& samples) {
  const std::size_t slots = static_cast<std::size_t>(samples.document_slots());
  const std::size_t total = samples.sample_count();
  std::size_t accepted = 0;

  for (std::size_t index = 0; index < total; ++index) {
    const ProbeSample sample = samples.sample_at(index);

    PromptRequest request;
    request.question = sample.question;
    request.documents = sample.documents;
    const StructuredPrompt prompt = dependencies_.prompt_builder->build(request);
    if (prompt.document_spans.size() != slots) {
      throw Error("probe sample " + std::to_string(index) + " produced " +
                  std::to_string(prompt.document_spans.size()) + " spans, expected " + std::to_string(slots));
    }

    observe(prompt);
    ++accepted;
    dependencies_.progress->on_progress(accepted, total);
  }

  if (accepted == 0) throw Error("attention probe produced no samples");
  return accepted;
}

AttentionProbeRequest AttentionProfilerConfig::probe_request(const StructuredPrompt& prompt) const {
  AttentionProbeRequest request;
  request.layer_index = layer_index;
  request.average_all_layers = average_all_layers;
  request.layer_span_begin = layer_span_begin;
  request.layer_span_end = layer_span_end;
  request.query_span = prompt.query_span;
  request.document_spans = prompt.document_spans;
  request.normalize_by_document_length = normalize_by_document_length;
  return request;
}

AttentionProfiler::AttentionProfiler(ProbeDependencies dependencies, const AttentionProfilerConfig& config)
    : AttentionProbeRun(std::move(dependencies)), config_(config) {}

void AttentionProfiler::observe(const StructuredPrompt& prompt) {
  const AttentionProbeResult measured = provider().measure_attention(prompt.tokens, config_.probe_request(prompt));
  for (std::size_t slot = 0; slot < accumulated_.size(); ++slot) {
    accumulated_[slot] += measured.document_attention[slot];
  }
}

AttentionProfile AttentionProfiler::build_profile(const IProbeSampleSource& samples) {
  accumulated_.assign(static_cast<std::size_t>(samples.document_slots()), 0.0);
  const std::size_t accepted = run(samples);
  return AttentionProfile::from_accumulated(config_.model_id, config_.layer_index, accepted, accumulated_);
}

AttentionLayerProbeRequest LayerScanConfig::probe_request(const StructuredPrompt& prompt) const {
  AttentionLayerProbeRequest request;
  request.layer_begin = layer_begin;
  request.layer_end = layer_end;
  request.query_span = prompt.query_span;
  request.document_spans = prompt.document_spans;
  request.normalize_by_document_length = normalize_by_document_length;
  return request;
}

AttentionLayerScanner::AttentionLayerScanner(ProbeDependencies dependencies, const LayerScanConfig& config)
    : AttentionProbeRun(std::move(dependencies)), config_(config) {}

void AttentionLayerScanner::observe(const StructuredPrompt& prompt) {
  const AttentionLayerProbeResult measured =
      provider().measure_attention_by_layer(prompt.tokens, config_.probe_request(prompt));
  if (accumulated_.empty()) {
    layer_begin_ = measured.layer_begin;
    accumulated_.assign(measured.attention_by_layer.size(), std::vector<double>(prompt.document_spans.size(), 0.0));
  }
  for (std::size_t layer = 0; layer < accumulated_.size(); ++layer) {
    for (std::size_t slot = 0; slot < accumulated_[layer].size(); ++slot) {
      accumulated_[layer][slot] += measured.attention_by_layer[layer][slot];
    }
  }
}

std::vector<AttentionProfile> AttentionLayerScanner::scan(const IProbeSampleSource& samples) {
  accumulated_.clear();
  const std::size_t accepted = run(samples);

  std::vector<AttentionProfile> profiles;
  profiles.reserve(accumulated_.size());
  for (std::size_t layer = 0; layer < accumulated_.size(); ++layer) {
    profiles.push_back(AttentionProfile::from_accumulated(config_.model_id, layer_begin_ + static_cast<int>(layer),
                                                          accepted, accumulated_[layer]));
  }
  return profiles;
}

std::vector<int> resolve_position_order(const std::vector<int>& preferred, std::size_t count) {
  std::vector<int> order;
  order.reserve(count);
  std::vector<bool> taken(count, false);

  for (const int slot : preferred) {
    if (slot < 0 || static_cast<std::size_t>(slot) >= count) continue;
    if (taken[static_cast<std::size_t>(slot)]) continue;
    taken[static_cast<std::size_t>(slot)] = true;
    order.push_back(slot);
  }
  for (std::size_t slot = 0; slot < count; ++slot) {
    if (!taken[slot]) order.push_back(static_cast<int>(slot));
  }
  return order;
}

std::vector<ScoredDocument> DescendingReranker::rerank(const std::vector<ScoredDocument>& ranked) const {
  return ranked;
}

std::vector<ScoredDocument> AscendingReranker::rerank(const std::vector<ScoredDocument>& ranked) const {
  return std::vector<ScoredDocument>(ranked.rbegin(), ranked.rend());
}

RandomReranker::RandomReranker(std::uint64_t seed) : state_(seed == 0 ? kGoldenRatioHash : seed) {}

std::vector<ScoredDocument> RandomReranker::rerank(const std::vector<ScoredDocument>& ranked) const {
  std::vector<ScoredDocument> result = ranked;
  for (std::size_t index = result.size(); index > 1; --index) {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    std::swap(result[index - 1], result[state_ % index]);
  }
  return result;
}

std::vector<ScoredDocument> PositionPriorityReranker::rerank(
    const std::vector<ScoredDocument>& ranked) const {
  return place(ranked, resolve_position_order(priority(ranked.size()), ranked.size()));
}

std::vector<int> LostInTheMiddleReranker::priority(std::size_t count) const {
  std::vector<int> order;
  order.reserve(count);
  int front = 0;
  int back = static_cast<int>(count) - 1;
  bool take_front = start_first_;
  while (front <= back) {
    order.push_back(take_front ? front++ : back--);
    take_front = !take_front;
  }
  return order;
}

AttentionReranker::AttentionReranker(std::vector<int> position_order)
    : position_order_(std::move(position_order)) {
  if (position_order_.empty()) throw Error("AttentionReranker requires a position order");
}

std::vector<int> AttentionReranker::priority(std::size_t) const { return position_order_; }

const std::vector<std::string>& RerankerFactory::names() {
  static const std::vector<std::string> values = {"random", "descending", "ascending", "lim", "attnrank"};
  return values;
}

std::shared_ptr<IReranker> RerankerFactory::create(const std::string& name,
                                                   const RerankerOptions& options) {
  if (name == "descending") return std::make_shared<DescendingReranker>();
  if (name == "ascending") return std::make_shared<AscendingReranker>();
  if (name == "random") return std::make_shared<RandomReranker>(options.seed);
  if (name == "lim") return std::make_shared<LostInTheMiddleReranker>(options.lim_start_first);
  if (name == "attnrank") {
    if (options.profile == nullptr) throw Error("strategy 'attnrank' needs an attention profile");
    return std::make_shared<AttentionReranker>(options.profile->position_order);
  }
  throw Error("unknown reranking strategy '" + name + "'");
}

}  // namespace attnrank

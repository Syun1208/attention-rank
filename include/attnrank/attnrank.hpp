#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "attnrank/core.hpp"
#include "attnrank/model.hpp"
#include "attnrank/prompt.hpp"

namespace attnrank {

struct ProbeSample {
  std::string question;
  std::vector<Document> documents;
};

class IProbeSampleSource {
 public:
  virtual ~IProbeSampleSource() = default;
  virtual std::size_t sample_count() const = 0;
  virtual int document_slots() const = 0;
  virtual ProbeSample sample_at(std::size_t index) const = 0;
};

class StaticProbeSampleSource : public IProbeSampleSource {
 public:
  explicit StaticProbeSampleSource(std::vector<ProbeSample> samples);

  std::size_t sample_count() const override { return samples_.size(); }
  int document_slots() const override { return document_slots_; }
  ProbeSample sample_at(std::size_t index) const override { return samples_.at(index); }

 private:
  std::vector<ProbeSample> samples_;
  int document_slots_ = 0;
};

constexpr std::uint64_t kDefaultSyntheticSeed = 20250807;
constexpr std::uint64_t kDefaultRerankSeed = 1234;

struct SyntheticProbeConfig {
  int document_slots = 10;
  int sample_count = 400;
  int words_per_document = 60;
  std::uint64_t seed = kDefaultSyntheticSeed;
};

class SyntheticProbeSampleSource : public IProbeSampleSource {
 public:
  explicit SyntheticProbeSampleSource(const SyntheticProbeConfig& config);

  std::size_t sample_count() const override { return static_cast<std::size_t>(config_.sample_count); }
  int document_slots() const override { return config_.document_slots; }
  ProbeSample sample_at(std::size_t index) const override;

 private:
  SyntheticProbeConfig config_;
};

std::vector<ProbeSample> load_probe_samples(const std::string& jsonl_path);

struct AttentionProfile {
  std::string model_id;
  int layer_index = 0;
  int sample_count = 0;
  std::vector<double> attention;
  std::vector<int> position_order;

  static AttentionProfile from_accumulated(std::string model_id, int layer_index,
                                           std::size_t sample_count,
                                           const std::vector<double>& accumulated);

  void recompute_position_order();
  std::string to_json_string() const;
  void save(const std::string& path) const;

  static AttentionProfile from_json(const Json& root);
  static AttentionProfile load(const std::string& path);
};

struct BasinCriterion {
  double min_edge_ratio = 1.5;
};

struct BasinStatistics {
  double first = 0.0;
  double last = 0.0;
  double interior_mean = 0.0;
  double edge_ratio = 0.0;
  bool is_basin = false;
};

BasinStatistics basin_statistics(const std::vector<double>& attention,
                                 const BasinCriterion& criterion);

int select_shallowest_basin_layer(const std::vector<AttentionProfile>& profiles,
                                  const BasinCriterion& criterion);

class IProgressObserver {
 public:
  virtual ~IProgressObserver() = default;
  virtual void on_progress(std::size_t completed, std::size_t total) = 0;
};

class NullProgressObserver : public IProgressObserver {
 public:
  void on_progress(std::size_t, std::size_t) override {}
};

class LogProgressObserver : public IProgressObserver {
 public:
  explicit LogProgressObserver(int interval) : interval_(interval) {}
  void on_progress(std::size_t completed, std::size_t total) override;

 private:
  int interval_;
};

std::shared_ptr<IProgressObserver> make_progress_observer(int log_interval);

struct ProbeDependencies {
  std::shared_ptr<IAttentionProvider> attention_provider;
  std::shared_ptr<const IPromptBuilder> prompt_builder;
  std::shared_ptr<IProgressObserver> progress;
};

class AttentionProbeRun {
 public:
  virtual ~AttentionProbeRun() = default;

 protected:
  explicit AttentionProbeRun(ProbeDependencies dependencies);

  std::size_t run(const IProbeSampleSource& samples);
  virtual void observe(const StructuredPrompt& prompt) = 0;
  IAttentionProvider& provider() { return *dependencies_.attention_provider; }

 private:
  ProbeDependencies dependencies_;
};

struct AttentionProfilerConfig {
  std::string model_id;
  int layer_index = 5;
  bool average_all_layers = false;
  int layer_span_begin = -1;
  int layer_span_end = -1;
  bool normalize_by_document_length = false;
  int progress_interval = 25;

  AttentionProbeRequest probe_request(const StructuredPrompt& prompt) const;
};

class AttentionProfiler : public AttentionProbeRun {
 public:
  AttentionProfiler(ProbeDependencies dependencies, const AttentionProfilerConfig& config);

  AttentionProfile build_profile(const IProbeSampleSource& samples);

 protected:
  void observe(const StructuredPrompt& prompt) override;

 private:
  AttentionProfilerConfig config_;
  std::vector<double> accumulated_;
};

struct LayerScanConfig {
  std::string model_id;
  int layer_begin = 0;
  int layer_end = -1;
  bool normalize_by_document_length = false;
  int progress_interval = 25;

  AttentionLayerProbeRequest probe_request(const StructuredPrompt& prompt) const;
};

class AttentionLayerScanner : public AttentionProbeRun {
 public:
  AttentionLayerScanner(ProbeDependencies dependencies, const LayerScanConfig& config);

  std::vector<AttentionProfile> scan(const IProbeSampleSource& samples);

 protected:
  void observe(const StructuredPrompt& prompt) override;

 private:
  LayerScanConfig config_;
  int layer_begin_ = 0;
  std::vector<std::vector<double>> accumulated_;
};

std::vector<int> resolve_position_order(const std::vector<int>& preferred, std::size_t count);

class IReranker {
 public:
  virtual ~IReranker() = default;
  virtual const char* name() const = 0;
  virtual std::vector<ScoredDocument> rerank(const std::vector<ScoredDocument>& ranked) const = 0;
};

class DescendingReranker : public IReranker {
 public:
  const char* name() const override { return "descending"; }
  std::vector<ScoredDocument> rerank(const std::vector<ScoredDocument>& ranked) const override;
};

class AscendingReranker : public IReranker {
 public:
  const char* name() const override { return "ascending"; }
  std::vector<ScoredDocument> rerank(const std::vector<ScoredDocument>& ranked) const override;
};

class RandomReranker : public IReranker {
 public:
  explicit RandomReranker(std::uint64_t seed);
  const char* name() const override { return "random"; }
  std::vector<ScoredDocument> rerank(const std::vector<ScoredDocument>& ranked) const override;

 private:
  mutable std::uint64_t state_;
};

class PositionPriorityReranker : public IReranker {
 public:
  std::vector<ScoredDocument> rerank(const std::vector<ScoredDocument>& ranked) const override;

 protected:
  virtual std::vector<int> priority(std::size_t count) const = 0;
};

class LostInTheMiddleReranker : public PositionPriorityReranker {
 public:
  explicit LostInTheMiddleReranker(bool start_first = false) : start_first_(start_first) {}
  const char* name() const override { return "lim"; }

 protected:
  std::vector<int> priority(std::size_t count) const override;

 private:
  bool start_first_ = false;
};

class AttentionReranker : public PositionPriorityReranker {
 public:
  explicit AttentionReranker(std::vector<int> position_order);
  const char* name() const override { return "attnrank"; }
  const std::vector<int>& position_order() const { return position_order_; }

 protected:
  std::vector<int> priority(std::size_t count) const override;

 private:
  std::vector<int> position_order_;
};

struct RerankerOptions {
  const AttentionProfile* profile = nullptr;
  std::uint64_t seed = kDefaultRerankSeed;
  bool lim_start_first = false;
};

class RerankerFactory {
 public:
  static const std::vector<std::string>& names();
  static std::shared_ptr<IReranker> create(const std::string& name, const RerankerOptions& options);
};

}  // namespace attnrank

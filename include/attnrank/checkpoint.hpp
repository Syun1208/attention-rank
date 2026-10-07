#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "attnrank/core.hpp"

namespace attnrank {

class IWeightSource {
 public:
  virtual ~IWeightSource() = default;
  virtual const std::string& origin() const = 0;
  virtual const std::vector<TensorView>& tensors() const = 0;
};

class IWeightSourceFactory {
 public:
  virtual ~IWeightSourceFactory() = default;
  virtual const char* format_name() const = 0;
  virtual bool accepts(const std::string& file_path) const = 0;
  virtual std::unique_ptr<IWeightSource> create(const std::string& file_path) const = 0;
};

class WeightRegistry {
 public:
  void add_source(std::unique_ptr<IWeightSource> source);
  const TensorView* find(const std::string& name) const;
  const TensorView& require(const std::string& name) const;
  std::vector<std::string> names() const;
  std::size_t tensor_count() const { return index_.size(); }
  std::size_t total_bytes() const;

 private:
  std::vector<std::unique_ptr<IWeightSource>> sources_;
  std::unordered_map<std::string, const TensorView*> index_;
};

class SafetensorsWeightSource : public IWeightSource {
 public:
  explicit SafetensorsWeightSource(const std::string& file_path);

  const std::string& origin() const override { return origin_; }
  const std::vector<TensorView>& tensors() const override { return tensors_; }

 private:
  std::string origin_;
  MappedFile file_;
  std::vector<TensorView> tensors_;
};

class TorchArchiveWeightSource : public IWeightSource {
 public:
  explicit TorchArchiveWeightSource(const std::string& file_path);

  const std::string& origin() const override { return origin_; }
  const std::vector<TensorView>& tensors() const override { return tensors_; }

 private:
  std::string origin_;
  MappedFile file_;
  std::vector<TensorView> tensors_;
};

class SafetensorsWeightSourceFactory : public IWeightSourceFactory {
 public:
  const char* format_name() const override { return "safetensors"; }
  bool accepts(const std::string& file_path) const override;
  std::unique_ptr<IWeightSource> create(const std::string& file_path) const override;
};

class TorchArchiveWeightSourceFactory : public IWeightSourceFactory {
 public:
  const char* format_name() const override { return "torch"; }
  bool accepts(const std::string& file_path) const override;
  std::unique_ptr<IWeightSource> create(const std::string& file_path) const override;
};

class CheckpointLoader {
 public:
  explicit CheckpointLoader(std::vector<std::shared_ptr<IWeightSourceFactory>> factories);

  static CheckpointLoader with_default_factories();

  WeightRegistry load_directory(const std::string& directory) const;

 private:
  std::vector<std::string> select_shards(const std::string& directory) const;

  std::vector<std::shared_ptr<IWeightSourceFactory>> factories_;
};

}  // namespace attnrank

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "attnrank/core.hpp"

namespace attnrank {

struct EncodeOptions {
  bool add_bos = false;
  bool add_eos = false;
  bool add_dummy_prefix = true;
};

class ITokenizer {
 public:
  virtual ~ITokenizer() = default;
  virtual TokenSequence encode(std::string_view text, const EncodeOptions& options) const = 0;
  virtual std::string decode(const TokenSequence& tokens) const = 0;
  virtual std::string token_text(int token_id) const = 0;
  virtual int vocab_size() const = 0;
  virtual int bos_id() const = 0;
  virtual int eos_id() const = 0;
  virtual int unk_id() const = 0;
};

struct SentencePieceTokenizerConfig {
  std::string model_path;
  int bos_id = 1;
  int eos_id = 2;
  int unk_id = 0;
};

class SentencePieceTokenizer : public ITokenizer {
 public:
  explicit SentencePieceTokenizer(const SentencePieceTokenizerConfig& config);

  TokenSequence encode(std::string_view text, const EncodeOptions& options) const override;
  std::string decode(const TokenSequence& tokens) const override;
  std::string token_text(int token_id) const override;
  int vocab_size() const override { return static_cast<int>(pieces_.size()); }
  int bos_id() const override { return config_.bos_id; }
  int eos_id() const override { return config_.eos_id; }
  int unk_id() const override { return config_.unk_id; }

 private:
  struct Piece {
    std::string text;
    float score = 0.0f;
    int type = 1;
  };

  void load_model(const std::string& model_path);
  int lookup(std::string_view piece) const;
  void append_byte_fallback(std::string_view symbol, TokenSequence& out) const;

  SentencePieceTokenizerConfig config_;
  std::vector<Piece> pieces_;
  std::unordered_map<std::string, int> vocab_;
};

struct ByteBpeTokenizerConfig {
  std::string vocab_path;
  std::string merges_path;
  std::string tokenizer_json_path;
  int bos_id = -1;
  int eos_id = -1;
  int unk_id = -1;
};

class ByteBpeTokenizer : public ITokenizer {
 public:
  explicit ByteBpeTokenizer(const ByteBpeTokenizerConfig& config);

  TokenSequence encode(std::string_view text, const EncodeOptions& options) const override;
  std::string decode(const TokenSequence& tokens) const override;
  std::string token_text(int token_id) const override;
  int vocab_size() const override { return static_cast<int>(id_to_token_.size()); }
  int bos_id() const override { return config_.bos_id; }
  int eos_id() const override { return config_.eos_id; }
  int unk_id() const override { return config_.unk_id; }

 private:
  void load_from_tokenizer_json(const std::string& path);
  void load_from_vocab_and_merges(const std::string& vocab_path, const std::string& merges_path);
  void finalise();

  std::vector<std::string> split_words(std::string_view text) const;
  std::vector<std::string> apply_merges(const std::string& word) const;
  std::string bytes_to_symbols(std::string_view raw) const;

  ByteBpeTokenizerConfig config_;
  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merge_ranks_;
  std::unordered_map<std::string, int> special_to_id_;
  std::vector<std::string> byte_to_symbol_;
  std::unordered_map<std::string, int> symbol_to_byte_;
};

class TokenizerFactory {
 public:
  static std::shared_ptr<ITokenizer> create(const std::string& model_directory, int bos_token_id,
                                            int eos_token_id);

  template <typename Config>
  static std::shared_ptr<ITokenizer> create(const std::string& model_directory, const Config& config) {
    return create(model_directory, config.bos_token_id, config.eos_token_id);
  }
};

}  // namespace attnrank

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "attnrank/core.hpp"
#include "attnrank/tokenizer.hpp"

namespace attnrank {

struct StructuredPrompt {
  TokenSequence tokens;
  std::vector<TokenSpan> document_spans;
  TokenSpan query_span;
  std::string text;
};

struct PromptRequest {
  std::string question;
  std::vector<Document> documents;
};

class IPromptBuilder {
 public:
  virtual ~IPromptBuilder() = default;
  virtual StructuredPrompt build(const PromptRequest& request) const = 0;
  virtual std::vector<std::string> stop_sequences() const = 0;
};

class IChatFormatter {
 public:
  virtual ~IChatFormatter() = default;
  virtual std::string system_message() const = 0;
  virtual std::string user_turn_prefix() const = 0;
  virtual std::string assistant_turn_prefix() const = 0;
  virtual std::vector<std::string> stop_sequences() const = 0;
};

class VicunaChatFormatter : public IChatFormatter {
 public:
  std::string system_message() const override;
  std::string user_turn_prefix() const override;
  std::string assistant_turn_prefix() const override;
  std::vector<std::string> stop_sequences() const override;
};

class ChatMlFormatter : public IChatFormatter {
 public:
  std::string system_message() const override;
  std::string user_turn_prefix() const override;
  std::string assistant_turn_prefix() const override;
  std::vector<std::string> stop_sequences() const override;
};

class Llama2ChatFormatter : public IChatFormatter {
 public:
  explicit Llama2ChatFormatter(std::string system_prompt = {})
      : system_prompt_(std::move(system_prompt)) {}

  std::string system_message() const override;
  std::string user_turn_prefix() const override;
  std::string assistant_turn_prefix() const override;
  std::vector<std::string> stop_sequences() const override;

 private:
  std::string system_prompt_;
};

class PlainChatFormatter : public IChatFormatter {
 public:
  std::string system_message() const override { return {}; }
  std::string user_turn_prefix() const override { return {}; }
  std::string assistant_turn_prefix() const override { return {}; }
  std::vector<std::string> stop_sequences() const override { return {}; }
};

struct DocumentPromptTemplate {
  std::string instruction =
      "Write a high-quality answer for the given question using only the provided search "
      "results (some of which might be irrelevant).";
  std::string example;
  std::string documents_header;
  std::string document_label = "Document [{index}] ";
  std::string document_index_placeholder = "{index}";
  std::string document_separator = "\n";
  std::string question_prefix = "Question: ";
  std::string answer_prefix = "Answer:";
  int max_document_tokens = 0;
  bool include_titles = false;
  std::vector<std::string> extra_stop_sequences;
};

class DocumentPromptBuilder : public IPromptBuilder {
 public:
  DocumentPromptBuilder(std::shared_ptr<const ITokenizer> tokenizer,
                   std::shared_ptr<const IChatFormatter> formatter,
                   DocumentPromptTemplate prompt_template);

  StructuredPrompt build(const PromptRequest& request) const override;
  std::vector<std::string> stop_sequences() const override;

 private:
  void append_segment(const std::string& text, bool dummy_prefix, StructuredPrompt& prompt) const;
  std::string format_document_label(std::size_t position) const;
  std::string render_document(const Document& document) const;
  std::string truncate_to_tokens(const std::string& text, int max_tokens) const;

  std::shared_ptr<const ITokenizer> tokenizer_;
  std::shared_ptr<const IChatFormatter> formatter_;
  DocumentPromptTemplate template_;
};

class ChatFormatterFactory {
 public:
  static const std::vector<std::string>& names();
  static std::string detect(const std::string& model_directory);
  static std::shared_ptr<const IChatFormatter> create(const std::string& name,
                                                      const std::string& system_prompt = {});
};

}  // namespace attnrank

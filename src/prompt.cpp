#include "attnrank/prompt.hpp"

#include <filesystem>
#include <utility>

#include "attnrank/model.hpp"

namespace attnrank {

namespace {

constexpr const char* kDocumentsTerminator = "\n\n";
constexpr const char* kParagraphBreak = "\n\n";
constexpr const char* kLineBreak = "\n";
constexpr const char* kQwenModelTypePrefix = "qwen";
constexpr const char* kMistralModelType = "mistral";
constexpr const char* kSentencePieceFileName = "tokenizer.model";

std::string model_type_of(const Json& config) {
  const Json* model_type = config.find("model_type");
  return model_type == nullptr ? std::string() : model_type->as_string_or("");
}

}  // namespace

std::string VicunaChatFormatter::system_message() const {
  return "A chat between a curious user and an artificial intelligence assistant. "
         "The assistant gives helpful, detailed, and polite answers to the user's questions.";
}

std::string VicunaChatFormatter::user_turn_prefix() const { return " USER: "; }

std::string VicunaChatFormatter::assistant_turn_prefix() const { return " ASSISTANT:"; }

std::vector<std::string> VicunaChatFormatter::stop_sequences() const {
  return {"</s>", "\nUSER:", " USER:"};
}

std::string ChatMlFormatter::system_message() const {
  return "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
         "assistant.<|im_end|>\n";
}

std::string ChatMlFormatter::user_turn_prefix() const { return "<|im_start|>user\n"; }

std::string ChatMlFormatter::assistant_turn_prefix() const {
  return "<|im_end|>\n<|im_start|>assistant\n";
}

std::vector<std::string> ChatMlFormatter::stop_sequences() const {
  return {"<|im_end|>", "<|endoftext|>"};
}

std::string Llama2ChatFormatter::system_message() const { return {}; }

std::string Llama2ChatFormatter::user_turn_prefix() const {
  std::string prefix = "[INST] ";
  if (!system_prompt_.empty()) {
    prefix += "<<SYS>>\n";
    prefix += system_prompt_;
    prefix += "\n<</SYS>>\n\n";
  }
  return prefix;
}

std::string Llama2ChatFormatter::assistant_turn_prefix() const { return " [/INST]"; }

std::vector<std::string> Llama2ChatFormatter::stop_sequences() const {
  return {"</s>", "[INST]"};
}

DocumentPromptBuilder::DocumentPromptBuilder(std::shared_ptr<const ITokenizer> tokenizer,
                                   std::shared_ptr<const IChatFormatter> formatter,
                                   DocumentPromptTemplate prompt_template)
    : tokenizer_(std::move(tokenizer)),
      formatter_(std::move(formatter)),
      template_(std::move(prompt_template)) {
  if (tokenizer_ == nullptr) throw Error("DocumentPromptBuilder requires a tokenizer");
  if (formatter_ == nullptr) throw Error("DocumentPromptBuilder requires a chat formatter");
}

std::string DocumentPromptBuilder::format_document_label(std::size_t position) const {
  std::string label = template_.document_label;
  const std::string& placeholder = template_.document_index_placeholder;
  if (placeholder.empty()) return label;

  const std::string value = std::to_string(position);
  for (std::size_t found = label.find(placeholder); found != std::string::npos;
       found = label.find(placeholder, found + value.size())) {
    label.replace(found, placeholder.size(), value);
  }
  return label;
}

std::string DocumentPromptBuilder::render_document(const Document& document) const {
  std::string rendered;
  if (template_.include_titles && !document.title.empty()) {
    rendered = document.title;
    rendered += ": ";
  }
  rendered += document.text;
  return truncate_to_tokens(rendered, template_.max_document_tokens);
}

std::string DocumentPromptBuilder::truncate_to_tokens(const std::string& text, int max_tokens) const {
  if (max_tokens <= 0) return text;
  EncodeOptions options;
  options.add_bos = false;
  options.add_dummy_prefix = false;
  const TokenSequence tokens = tokenizer_->encode(text, options);
  if (static_cast<int>(tokens.size()) <= max_tokens) return text;
  const TokenSequence truncated(tokens.begin(), tokens.begin() + max_tokens);
  return tokenizer_->decode(truncated);
}

void DocumentPromptBuilder::append_segment(const std::string& text, bool dummy_prefix,
                                      StructuredPrompt& prompt) const {
  EncodeOptions options;
  options.add_bos = false;
  options.add_eos = false;
  options.add_dummy_prefix = dummy_prefix;
  const TokenSequence tokens = tokenizer_->encode(text, options);
  prompt.tokens.insert(prompt.tokens.end(), tokens.begin(), tokens.end());
  prompt.text += text;
}

std::vector<std::string> DocumentPromptBuilder::stop_sequences() const {
  std::vector<std::string> sequences = formatter_->stop_sequences();
  sequences.insert(sequences.end(), template_.extra_stop_sequences.begin(),
                   template_.extra_stop_sequences.end());
  return sequences;
}

StructuredPrompt DocumentPromptBuilder::build(const PromptRequest& request) const {
  StructuredPrompt prompt;
  if (tokenizer_->bos_id() >= 0) prompt.tokens.push_back(tokenizer_->bos_id());

  std::string head = formatter_->system_message();
  head += formatter_->user_turn_prefix();
  head += template_.instruction;
  head += kParagraphBreak;
  if (!template_.example.empty()) {
    head += template_.example;
    head += kParagraphBreak;
  }
  if (!template_.documents_header.empty()) {
    head += template_.documents_header;
    head += kLineBreak;
  }
  append_segment(head, true, prompt);

  prompt.document_spans.reserve(request.documents.size());
  for (std::size_t index = 0; index < request.documents.size(); ++index) {
    std::string segment = format_document_label(index + 1);
    segment += render_document(request.documents[index]);
    segment += index + 1 < request.documents.size() ? template_.document_separator : kDocumentsTerminator;

    TokenSpan span;
    span.begin = static_cast<int>(prompt.tokens.size());
    append_segment(segment, false, prompt);
    span.end = static_cast<int>(prompt.tokens.size());
    prompt.document_spans.push_back(span);
  }

  std::string query = template_.question_prefix;
  query += request.question;
  query += kLineBreak;

  prompt.query_span.begin = static_cast<int>(prompt.tokens.size());
  append_segment(query, false, prompt);
  prompt.query_span.end = static_cast<int>(prompt.tokens.size());

  std::string tail = template_.answer_prefix;
  tail += formatter_->assistant_turn_prefix();
  append_segment(tail, false, prompt);

  return prompt;
}

const std::vector<std::string>& ChatFormatterFactory::names() {
  static const std::vector<std::string> values = {"plain", "chatml", "vicuna", "llama2", "mistral"};
  return values;
}

std::string ChatFormatterFactory::detect(const std::string& model_directory) {
  namespace fs = std::filesystem;
  const fs::path config_path = fs::path(model_directory) / kModelConfigFileName;
  if (fs::exists(config_path)) {
    try {
      const Json config = Json::parse_file(config_path.string());
      const std::string model_type = model_type_of(config);
      if (model_type.rfind(kQwenModelTypePrefix, 0) == 0) return "chatml";
      if (model_type == kMistralModelType) return "mistral";
      const Json* window = config.find("sliding_window");
      if (window != nullptr && window->is_number()) return "llama2";
    } catch (const std::exception&) {
    }
  }
  return fs::exists(fs::path(model_directory) / kSentencePieceFileName) ? "vicuna" : "chatml";
}

std::shared_ptr<const IChatFormatter> ChatFormatterFactory::create(const std::string& name,
                                                                   const std::string& system_prompt) {
  if (name == "plain") return std::make_shared<PlainChatFormatter>();
  if (name == "chatml") return std::make_shared<ChatMlFormatter>();
  if (name == "vicuna") return std::make_shared<VicunaChatFormatter>();
  if (name == "llama2" || name == "mistral") return std::make_shared<Llama2ChatFormatter>(system_prompt);
  throw Error("unknown chat format '" + name + "' (expected plain, chatml, vicuna, llama2 or mistral)");
}

}  // namespace attnrank

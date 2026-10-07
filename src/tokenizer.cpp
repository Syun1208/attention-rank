#include "attnrank/tokenizer.hpp"

#include <filesystem>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <utility>
#include <vector>

namespace attnrank {

namespace {

constexpr const char* kSpaceMarker = "\xe2\x96\x81";
constexpr std::size_t kSpaceMarkerLength = 3;

std::size_t utf8_sequence_length(unsigned char lead) {
  if ((lead & 0x80) == 0x00) return 1;
  if ((lead & 0xE0) == 0xC0) return 2;
  if ((lead & 0xF0) == 0xE0) return 3;
  if ((lead & 0xF8) == 0xF0) return 4;
  return 1;
}

class ProtobufReader {
 public:
  ProtobufReader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

  bool done() const { return position_ >= size_; }

  std::uint64_t read_varint() {
    std::uint64_t value = 0;
    int shift = 0;
    while (position_ < size_) {
      const std::uint8_t byte = data_[position_++];
      value |= static_cast<std::uint64_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) return value;
      shift += 7;
      if (shift > 63) break;
    }
    throw Error("malformed varint in sentencepiece model");
  }

  void skip(std::size_t count) {
    if (position_ + count > size_) throw Error("truncated sentencepiece model");
    position_ += count;
  }

  const std::uint8_t* take(std::size_t count) {
    if (position_ + count > size_) throw Error("truncated sentencepiece model");
    const std::uint8_t* pointer = data_ + position_;
    position_ += count;
    return pointer;
  }

  void skip_field(std::uint32_t wire_type) {
    switch (wire_type) {
      case 0: read_varint(); return;
      case 1: skip(8); return;
      case 2: skip(read_varint()); return;
      case 5: skip(4); return;
      default: throw Error("unsupported protobuf wire type in sentencepiece model");
    }
  }

 private:
  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t position_ = 0;
};

struct Symbol {
  int previous = -1;
  int next = -1;
  const char* text = nullptr;
  std::size_t length = 0;
};

struct Bigram {
  int left = -1;
  int right = -1;
  float score = 0.0f;
  std::size_t length = 0;

  bool operator<(const Bigram& other) const {
    if (score != other.score) return score < other.score;
    return left > other.left;
  }
};

}  // namespace

SentencePieceTokenizer::SentencePieceTokenizer(const SentencePieceTokenizerConfig& config)
    : config_(config) {
  load_model(config.model_path);
  Log::info("tokenizer loaded: %zu pieces from %s", pieces_.size(), config.model_path.c_str());
}

void SentencePieceTokenizer::load_model(const std::string& model_path) {
  const MappedFile file(model_path);
  ProtobufReader reader(file.data(), file.size());

  while (!reader.done()) {
    const std::uint64_t tag = reader.read_varint();
    const std::uint32_t field_number = static_cast<std::uint32_t>(tag >> 3);
    const std::uint32_t wire_type = static_cast<std::uint32_t>(tag & 0x7);

    if (field_number != 1 || wire_type != 2) {
      reader.skip_field(wire_type);
      continue;
    }

    const std::size_t message_length = reader.read_varint();
    ProtobufReader piece_reader(reader.take(message_length), message_length);

    Piece piece;
    while (!piece_reader.done()) {
      const std::uint64_t piece_tag = piece_reader.read_varint();
      const std::uint32_t piece_field = static_cast<std::uint32_t>(piece_tag >> 3);
      const std::uint32_t piece_wire = static_cast<std::uint32_t>(piece_tag & 0x7);

      if (piece_field == 1 && piece_wire == 2) {
        const std::size_t length = piece_reader.read_varint();
        piece.text.assign(reinterpret_cast<const char*>(piece_reader.take(length)), length);
      } else if (piece_field == 2 && piece_wire == 5) {
        std::memcpy(&piece.score, piece_reader.take(4), 4);
      } else if (piece_field == 3 && piece_wire == 0) {
        piece.type = static_cast<int>(piece_reader.read_varint());
      } else {
        piece_reader.skip_field(piece_wire);
      }
    }

    vocab_.emplace(piece.text, static_cast<int>(pieces_.size()));
    pieces_.push_back(std::move(piece));
  }

  if (pieces_.empty()) throw Error("no pieces found in '" + model_path + "'");
}

int SentencePieceTokenizer::lookup(std::string_view piece) const {
  const auto found = vocab_.find(std::string(piece));
  return found == vocab_.end() ? -1 : found->second;
}

void SentencePieceTokenizer::append_byte_fallback(std::string_view symbol, TokenSequence& out) const {
  static const char* kHexDigits = "0123456789ABCDEF";
  for (const char raw : symbol) {
    const unsigned char byte = static_cast<unsigned char>(raw);
    char buffer[7] = {'<', '0', 'x', kHexDigits[byte >> 4], kHexDigits[byte & 0x0F], '>', '\0'};
    const int token = lookup(buffer);
    out.push_back(token >= 0 ? token : config_.unk_id);
  }
}

TokenSequence SentencePieceTokenizer::encode(std::string_view text,
                                             const EncodeOptions& options) const {
  TokenSequence result;
  if (options.add_bos) result.push_back(config_.bos_id);

  std::string normalized;
  normalized.reserve(text.size() + kSpaceMarkerLength);
  if (options.add_dummy_prefix) normalized.append(kSpaceMarker, kSpaceMarkerLength);
  for (const char character : text) {
    if (character == ' ') {
      normalized.append(kSpaceMarker, kSpaceMarkerLength);
    } else {
      normalized.push_back(character);
    }
  }

  if (normalized.empty()) {
    if (options.add_eos) result.push_back(config_.eos_id);
    return result;
  }

  std::vector<Symbol> symbols;
  symbols.reserve(normalized.size());
  for (std::size_t offset = 0; offset < normalized.size();) {
    const std::size_t length =
        std::min(utf8_sequence_length(static_cast<unsigned char>(normalized[offset])),
                 normalized.size() - offset);
    Symbol symbol;
    symbol.text = normalized.data() + offset;
    symbol.length = length;
    symbol.previous = static_cast<int>(symbols.size()) - 1;
    symbol.next = static_cast<int>(symbols.size()) + 1;
    symbols.push_back(symbol);
    offset += length;
  }
  if (!symbols.empty()) symbols.back().next = -1;

  std::priority_queue<Bigram> queue;
  const auto try_push = [&](int left, int right) {
    if (left < 0 || right < 0) return;
    const Symbol& left_symbol = symbols[static_cast<std::size_t>(left)];
    const Symbol& right_symbol = symbols[static_cast<std::size_t>(right)];
    const std::size_t length = left_symbol.length + right_symbol.length;
    const int token = lookup(std::string_view(left_symbol.text, length));
    if (token < 0) return;
    Bigram bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.score = pieces_[static_cast<std::size_t>(token)].score;
    bigram.length = length;
    queue.push(bigram);
  };

  for (std::size_t index = 1; index < symbols.size(); ++index) {
    try_push(static_cast<int>(index) - 1, static_cast<int>(index));
  }

  while (!queue.empty()) {
    const Bigram bigram = queue.top();
    queue.pop();

    Symbol& left = symbols[static_cast<std::size_t>(bigram.left)];
    Symbol& right = symbols[static_cast<std::size_t>(bigram.right)];
    if (left.length == 0 || right.length == 0 || left.length + right.length != bigram.length) {
      continue;
    }

    left.length += right.length;
    right.length = 0;
    left.next = right.next;
    if (right.next >= 0) symbols[static_cast<std::size_t>(right.next)].previous = bigram.left;

    try_push(left.previous, bigram.left);
    try_push(bigram.left, left.next);
  }

  for (int index = 0; index >= 0; index = symbols[static_cast<std::size_t>(index)].next) {
    const Symbol& symbol = symbols[static_cast<std::size_t>(index)];
    if (symbol.length == 0) continue;
    const std::string_view view(symbol.text, symbol.length);
    const int token = lookup(view);
    if (token >= 0) {
      result.push_back(token);
    } else {
      append_byte_fallback(view, result);
    }
  }

  if (options.add_eos) result.push_back(config_.eos_id);
  return result;
}

std::string SentencePieceTokenizer::token_text(int token_id) const {
  if (token_id < 0 || static_cast<std::size_t>(token_id) >= pieces_.size()) return {};
  return pieces_[static_cast<std::size_t>(token_id)].text;
}

std::string SentencePieceTokenizer::decode(const TokenSequence& tokens) const {
  std::string result;
  result.reserve(tokens.size() * 4);

  for (const int token : tokens) {
    if (token < 0 || static_cast<std::size_t>(token) >= pieces_.size()) continue;
    const Piece& piece = pieces_[static_cast<std::size_t>(token)];
    if (piece.type == 3) continue;

    if (piece.type == 6 && piece.text.size() == 6 && piece.text.compare(0, 3, "<0x") == 0) {
      const int value = static_cast<int>(std::strtol(piece.text.substr(3, 2).c_str(), nullptr, 16));
      result.push_back(static_cast<char>(value));
      continue;
    }

    for (std::size_t offset = 0; offset < piece.text.size();) {
      if (piece.text.compare(offset, kSpaceMarkerLength, kSpaceMarker) == 0) {
        result.push_back(' ');
        offset += kSpaceMarkerLength;
      } else {
        result.push_back(piece.text[offset]);
        ++offset;
      }
    }
  }
  return result;
}

namespace {

void append_codepoint(unsigned int code_point, std::string& out) {
  if (code_point < 0x80) {
    out.push_back(static_cast<char>(code_point));
  } else if (code_point < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  }
}

std::size_t utf8_length(unsigned char lead) {
  if ((lead & 0x80) == 0x00) return 1;
  if ((lead & 0xE0) == 0xC0) return 2;
  if ((lead & 0xF0) == 0xE0) return 3;
  if ((lead & 0xF8) == 0xF0) return 4;
  return 1;
}

unsigned int decode_codepoint(std::string_view text, std::size_t offset, std::size_t& length) {
  const unsigned char lead = static_cast<unsigned char>(text[offset]);
  length = std::min(utf8_length(lead), text.size() - offset);
  if (length == 1) return lead;
  unsigned int value = lead & (0xFF >> (length + 1));
  for (std::size_t index = 1; index < length; ++index) {
    value = (value << 6) | (static_cast<unsigned char>(text[offset + index]) & 0x3F);
  }
  return value;
}

bool is_ascii_letter(unsigned int c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool is_digit(unsigned int c) { return c >= '0' && c <= '9'; }

bool is_space(unsigned int c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v' ||
         c == 0x00A0 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
         c == 0x202F || c == 0x205F || c == 0x3000;
}

bool is_punctuation_block(unsigned int c) {
  return (c >= 0x2000 && c <= 0x206F) || (c >= 0x3000 && c <= 0x303F) ||
         (c >= 0xFE30 && c <= 0xFE4F) || (c >= 0xFF00 && c <= 0xFF0F) ||
         (c >= 0xFF1A && c <= 0xFF20) || (c >= 0x2190 && c <= 0x2BFF);
}

bool is_letter(unsigned int c) {
  if (is_ascii_letter(c)) return true;
  if (c < 0x80) return false;
  if (is_space(c) || is_punctuation_block(c)) return false;
  return true;
}

bool is_newline(unsigned int c) { return c == '\n' || c == '\r'; }

}  // namespace

ByteBpeTokenizer::ByteBpeTokenizer(const ByteBpeTokenizerConfig& config) : config_(config) {
  byte_to_symbol_.resize(256);
  unsigned int spare = 0;
  for (unsigned int byte = 0; byte < 256; ++byte) {
    const bool printable = (byte >= '!' && byte <= '~') || (byte >= 0xA1 && byte <= 0xAC) ||
                           (byte >= 0xAE && byte <= 0xFF);
    std::string symbol;
    if (printable) {
      append_codepoint(byte, symbol);
    } else {
      append_codepoint(256 + spare, symbol);
      ++spare;
    }
    byte_to_symbol_[byte] = symbol;
    symbol_to_byte_[symbol] = static_cast<int>(byte);
  }

  if (!config_.tokenizer_json_path.empty()) {
    load_from_tokenizer_json(config_.tokenizer_json_path);
  } else {
    load_from_vocab_and_merges(config_.vocab_path, config_.merges_path);
  }
  finalise();

  Log::info("byte-level BPE tokenizer: %zu tokens, %zu merges, %zu special",
            id_to_token_.size(), merge_ranks_.size(), special_to_id_.size());
}

void ByteBpeTokenizer::load_from_tokenizer_json(const std::string& path) {
  const Json root = Json::parse_file(path);

  const Json* added = root.find("added_tokens");
  if (added != nullptr && added->is_array()) {
    for (std::size_t index = 0; index < added->size(); ++index) {
      const Json& entry = added->at(index);
      const Json* content = entry.find("content");
      const Json* id = entry.find("id");
      if (content == nullptr || id == nullptr) continue;
      const int token_id = static_cast<int>(id->as_int64(-1));
      special_to_id_[content->as_string()] = token_id;
      if (token_id >= 0) {
        if (static_cast<std::size_t>(token_id) >= id_to_token_.size()) {
          id_to_token_.resize(static_cast<std::size_t>(token_id) + 1);
        }
        id_to_token_[static_cast<std::size_t>(token_id)] = content->as_string();
      }
    }
  }

  const Json& model = root.at("model");
  const Json& vocab = model.at("vocab");
  const std::vector<std::string>& keys = vocab.keys();
  const std::vector<Json>& values = vocab.values();
  for (std::size_t index = 0; index < keys.size(); ++index) {
    const int token_id = static_cast<int>(values[index].as_int64(-1));
    if (token_id < 0) continue;
    token_to_id_[keys[index]] = token_id;
    if (static_cast<std::size_t>(token_id) >= id_to_token_.size()) {
      id_to_token_.resize(static_cast<std::size_t>(token_id) + 1);
    }
    id_to_token_[static_cast<std::size_t>(token_id)] = keys[index];
  }

  const Json& merges = model.at("merges");
  for (std::size_t index = 0; index < merges.size(); ++index) {
    const Json& entry = merges.at(index);
    std::string left;
    std::string right;
    if (entry.is_array() && entry.size() == 2) {
      left = entry.at(0).as_string();
      right = entry.at(1).as_string();
    } else if (entry.is_string()) {
      const std::string& line = entry.as_string();
      const std::size_t space = line.find(' ');
      if (space == std::string::npos) continue;
      left = line.substr(0, space);
      right = line.substr(space + 1);
    } else {
      continue;
    }
    merge_ranks_[left + " " + right] = static_cast<int>(index);
  }
}

void ByteBpeTokenizer::load_from_vocab_and_merges(const std::string& vocab_path,
                                                  const std::string& merges_path) {
  const Json vocab = Json::parse_file(vocab_path);
  const std::vector<std::string>& keys = vocab.keys();
  const std::vector<Json>& values = vocab.values();
  for (std::size_t index = 0; index < keys.size(); ++index) {
    const int token_id = static_cast<int>(values[index].as_int64(-1));
    if (token_id < 0) continue;
    token_to_id_[keys[index]] = token_id;
    if (static_cast<std::size_t>(token_id) >= id_to_token_.size()) {
      id_to_token_.resize(static_cast<std::size_t>(token_id) + 1);
    }
    id_to_token_[static_cast<std::size_t>(token_id)] = keys[index];
  }

  std::ifstream stream(merges_path);
  if (!stream) throw Error("cannot open merges file '" + merges_path + "'");
  std::string line;
  int rank = 0;
  while (std::getline(stream, line)) {
    if (line.empty() || line[0] == '#') continue;
    const std::size_t space = line.find(' ');
    if (space == std::string::npos) continue;
    merge_ranks_[line] = rank++;
  }
}

void ByteBpeTokenizer::finalise() {
  if (id_to_token_.empty()) throw Error("byte-level BPE tokenizer loaded no vocabulary");
  for (std::size_t index = 0; index < id_to_token_.size(); ++index) {
    if (!id_to_token_[index].empty()) {
      token_to_id_.emplace(id_to_token_[index], static_cast<int>(index));
    }
  }
}

std::string ByteBpeTokenizer::bytes_to_symbols(std::string_view raw) const {
  std::string mapped;
  mapped.reserve(raw.size() * 2);
  for (const char character : raw) {
    mapped += byte_to_symbol_[static_cast<unsigned char>(character)];
  }
  return mapped;
}

std::vector<std::string> ByteBpeTokenizer::split_words(std::string_view text) const {
  std::vector<std::string> words;
  std::size_t offset = 0;

  const auto peek = [&](std::size_t at, std::size_t& length) -> unsigned int {
    if (at >= text.size()) {
      length = 0;
      return 0;
    }
    return decode_codepoint(text, at, length);
  };

  while (offset < text.size()) {
    std::size_t length = 0;
    const unsigned int current = peek(offset, length);
    const std::size_t start = offset;

    if (current == '\'' && offset + 1 < text.size()) {
      static const char* kContractions[] = {"s", "t", "re", "ve", "m", "ll", "d"};
      bool matched = false;
      for (const char* suffix : kContractions) {
        const std::size_t suffix_length = std::char_traits<char>::length(suffix);
        if (offset + 1 + suffix_length > text.size()) continue;
        bool equal = true;
        for (std::size_t index = 0; index < suffix_length; ++index) {
          const char candidate = text[offset + 1 + index];
          const char lowered =
              (candidate >= 'A' && candidate <= 'Z') ? static_cast<char>(candidate + 32) : candidate;
          if (lowered != suffix[index]) {
            equal = false;
            break;
          }
        }
        if (equal) {
          offset += 1 + suffix_length;
          matched = true;
          break;
        }
      }
      if (matched) {
        words.emplace_back(text.substr(start, offset - start));
        continue;
      }
    }

    std::size_t probe = offset;
    std::size_t probe_length = 0;
    unsigned int probe_char = peek(probe, probe_length);
    if (!is_newline(probe_char) && !is_letter(probe_char) && !is_digit(probe_char) &&
        probe_length != 0) {
      const std::size_t after = probe + probe_length;
      std::size_t next_length = 0;
      const unsigned int next_char = peek(after, next_length);
      if (next_length != 0 && is_letter(next_char)) {
        probe = after;
        probe_char = next_char;
        probe_length = next_length;
      }
    }
    if (probe_length != 0 && is_letter(probe_char)) {
      std::size_t cursor = probe;
      while (cursor < text.size()) {
        std::size_t step = 0;
        const unsigned int character = peek(cursor, step);
        if (step == 0 || !is_letter(character)) break;
        cursor += step;
      }
      if (cursor > probe) {
        offset = cursor;
        words.emplace_back(text.substr(start, offset - start));
        continue;
      }
    }

    if (is_digit(current)) {
      std::size_t taken = 0;
      std::size_t cursor = offset;
      while (cursor < text.size() && taken < 3) {
        std::size_t step = 0;
        const unsigned int character = peek(cursor, step);
        if (step == 0 || !is_digit(character)) break;
        cursor += step;
        ++taken;
      }
      offset = cursor;
      words.emplace_back(text.substr(start, offset - start));
      continue;
    }

    {
      std::size_t cursor = offset;
      if (current == ' ') {
        std::size_t next_length = 0;
        const unsigned int next_char = peek(offset + 1, next_length);
        if (next_length != 0 && !is_space(next_char) && !is_letter(next_char) &&
            !is_digit(next_char)) {
          cursor = offset + 1;
        }
      }
      std::size_t run = cursor;
      while (run < text.size()) {
        std::size_t step = 0;
        const unsigned int character = peek(run, step);
        if (step == 0 || is_space(character) || is_letter(character) || is_digit(character)) break;
        run += step;
      }
      if (run > cursor) {
        while (run < text.size() && is_newline(static_cast<unsigned char>(text[run]))) ++run;
        offset = run;
        words.emplace_back(text.substr(start, offset - start));
        continue;
      }
    }

    {
      std::size_t cursor = offset;
      std::size_t last_newline = std::string::npos;
      while (cursor < text.size()) {
        std::size_t step = 0;
        const unsigned int character = peek(cursor, step);
        if (step == 0 || !is_space(character)) break;
        if (is_newline(character)) last_newline = cursor + step;
        cursor += step;
      }
      if (last_newline != std::string::npos) {
        offset = last_newline;
        words.emplace_back(text.substr(start, offset - start));
        continue;
      }
      if (cursor > offset) {
        std::size_t trailing = cursor;
        if (trailing < text.size()) {
          std::size_t back = trailing;
          std::size_t step = 0;
          while (back > offset) {
            std::size_t probe_back = back - 1;
            while (probe_back > offset && (static_cast<unsigned char>(text[probe_back]) & 0xC0) == 0x80) {
              --probe_back;
            }
            decode_codepoint(text, probe_back, step);
            back = probe_back;
            break;
          }
          if (back > offset) trailing = back;
        }
        offset = trailing > offset ? trailing : cursor;
        words.emplace_back(text.substr(start, offset - start));
        continue;
      }
    }

    offset += length == 0 ? 1 : length;
    words.emplace_back(text.substr(start, offset - start));
  }
  return words;
}

std::vector<std::string> ByteBpeTokenizer::apply_merges(const std::string& word) const {
  std::vector<std::string> symbols;
  for (std::size_t offset = 0; offset < word.size();) {
    const std::size_t length =
        std::min(utf8_length(static_cast<unsigned char>(word[offset])), word.size() - offset);
    symbols.push_back(word.substr(offset, length));
    offset += length;
  }
  if (symbols.size() < 2) return symbols;

  while (true) {
    int best_rank = std::numeric_limits<int>::max();
    std::size_t best_index = 0;
    bool found = false;

    for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
      const auto entry = merge_ranks_.find(symbols[index] + " " + symbols[index + 1]);
      if (entry == merge_ranks_.end()) continue;
      if (entry->second < best_rank) {
        best_rank = entry->second;
        best_index = index;
        found = true;
      }
    }
    if (!found) break;

    symbols[best_index] += symbols[best_index + 1];
    symbols.erase(symbols.begin() + static_cast<long>(best_index) + 1);
    if (symbols.size() < 2) break;
  }
  return symbols;
}

TokenSequence ByteBpeTokenizer::encode(std::string_view text, const EncodeOptions& options) const {
  TokenSequence result;
  if (options.add_bos && config_.bos_id >= 0) result.push_back(config_.bos_id);

  std::size_t cursor = 0;
  while (cursor < text.size()) {
    std::size_t special_at = std::string::npos;
    std::size_t special_length = 0;
    int special_id = -1;

    for (const auto& entry : special_to_id_) {
      const std::size_t found = text.find(entry.first, cursor);
      if (found == std::string::npos) continue;
      if (found < special_at) {
        special_at = found;
        special_length = entry.first.size();
        special_id = entry.second;
      }
    }

    const std::size_t chunk_end = special_at == std::string::npos ? text.size() : special_at;
    if (chunk_end > cursor) {
      const std::string_view chunk = text.substr(cursor, chunk_end - cursor);
      for (const std::string& word : split_words(chunk)) {
        for (const std::string& symbol : apply_merges(bytes_to_symbols(word))) {
          const auto entry = token_to_id_.find(symbol);
          if (entry != token_to_id_.end()) {
            result.push_back(entry->second);
          } else if (config_.unk_id >= 0) {
            result.push_back(config_.unk_id);
          }
        }
      }
    }

    if (special_at == std::string::npos) break;
    if (special_id >= 0) result.push_back(special_id);
    cursor = special_at + special_length;
  }

  if (options.add_eos && config_.eos_id >= 0) result.push_back(config_.eos_id);
  return result;
}

std::string ByteBpeTokenizer::token_text(int token_id) const {
  if (token_id < 0 || static_cast<std::size_t>(token_id) >= id_to_token_.size()) return {};
  return id_to_token_[static_cast<std::size_t>(token_id)];
}

std::string ByteBpeTokenizer::decode(const TokenSequence& tokens) const {
  std::string mapped;
  for (const int token : tokens) {
    if (token < 0 || static_cast<std::size_t>(token) >= id_to_token_.size()) continue;
    const std::string& piece = id_to_token_[static_cast<std::size_t>(token)];
    if (special_to_id_.count(piece) != 0) continue;
    mapped += piece;
  }

  std::string out;
  out.reserve(mapped.size());
  for (std::size_t offset = 0; offset < mapped.size();) {
    const std::size_t length =
        std::min(utf8_length(static_cast<unsigned char>(mapped[offset])), mapped.size() - offset);
    const auto entry = symbol_to_byte_.find(mapped.substr(offset, length));
    if (entry != symbol_to_byte_.end()) {
      out.push_back(static_cast<char>(entry->second));
    } else {
      out.append(mapped, offset, length);
    }
    offset += length;
  }
  return out;
}

std::shared_ptr<ITokenizer> TokenizerFactory::create(const std::string& model_directory,
                                                     int bos_token_id, int eos_token_id) {
  namespace fs = std::filesystem;
  const fs::path root(model_directory);

  if (fs::exists(root / "tokenizer_config.json")) {
    const Json settings = Json::parse_file((root / "tokenizer_config.json").string());
    const Json* add_bos = settings.find("add_bos_token");
    if (add_bos != nullptr && add_bos->is_bool() && !add_bos->as_bool(true)) bos_token_id = -1;
  }

  if (fs::exists(root / "tokenizer.model")) {
    SentencePieceTokenizerConfig config;
    config.model_path = (root / "tokenizer.model").string();
    config.bos_id = bos_token_id;
    config.eos_id = eos_token_id;
    config.unk_id = 0;
    return std::make_shared<SentencePieceTokenizer>(config);
  }

  ByteBpeTokenizerConfig config;
  config.bos_id = bos_token_id;
  config.eos_id = eos_token_id;
  config.unk_id = -1;
  if (fs::exists(root / "tokenizer.json")) {
    config.tokenizer_json_path = (root / "tokenizer.json").string();
  } else if (fs::exists(root / "vocab.json") && fs::exists(root / "merges.txt")) {
    config.vocab_path = (root / "vocab.json").string();
    config.merges_path = (root / "merges.txt").string();
  } else {
    throw Error("no tokenizer found under '" + model_directory + "'");
  }
  return std::make_shared<ByteBpeTokenizer>(config);
}

}  // namespace attnrank

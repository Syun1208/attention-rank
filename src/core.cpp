#include "attnrank/core.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace attnrank {

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::F32:
    case DType::I32:
      return 4;
    case DType::F16:
    case DType::BF16:
    case DType::I16:
      return 2;
    case DType::I64:
      return 8;
    case DType::I8:
    case DType::U8:
    case DType::Bool:
      return 1;
    case DType::Unknown:
    default:
      return 0;
  }
}

const char* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::I64: return "I64";
    case DType::I32: return "I32";
    case DType::I16: return "I16";
    case DType::I8: return "I8";
    case DType::U8: return "U8";
    case DType::Bool: return "BOOL";
    case DType::Unknown:
    default: return "UNKNOWN";
  }
}

DType dtype_from_safetensors(const std::string& name) {
  if (name == "F32" || name == "F64") return name == "F32" ? DType::F32 : DType::Unknown;
  if (name == "F16") return DType::F16;
  if (name == "BF16") return DType::BF16;
  if (name == "I64" || name == "U64") return DType::I64;
  if (name == "I32" || name == "U32") return DType::I32;
  if (name == "I16" || name == "U16") return DType::I16;
  if (name == "I8") return DType::I8;
  if (name == "U8") return DType::U8;
  if (name == "BOOL") return DType::Bool;
  return DType::Unknown;
}

DType dtype_from_torch_storage(const std::string& name) {
  if (name == "HalfStorage") return DType::F16;
  if (name == "FloatStorage") return DType::F32;
  if (name == "BFloat16Storage") return DType::BF16;
  if (name == "LongStorage") return DType::I64;
  if (name == "IntStorage") return DType::I32;
  if (name == "ShortStorage") return DType::I16;
  if (name == "CharStorage") return DType::I8;
  if (name == "ByteStorage") return DType::U8;
  if (name == "BoolStorage") return DType::Bool;
  return DType::Unknown;
}

std::int64_t TensorView::element_count() const {
  if (shape.empty()) return 0;
  return std::accumulate(shape.begin(), shape.end(), static_cast<std::int64_t>(1),
                         std::multiplies<std::int64_t>());
}

std::int64_t TensorView::dim(std::size_t index) const {
  if (index >= shape.size()) {
    throw Error("tensor '" + name + "' has no dimension " + std::to_string(index));
  }
  return shape[index];
}

namespace {

void encode_utf8(unsigned int code_point, std::string& out) {
  if (code_point < 0x80) {
    out.push_back(static_cast<char>(code_point));
  } else if (code_point < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else if (code_point < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  }
}

void escape_into(std::string_view value, std::string& out) {
  out.push_back('"');
  for (const char raw : value) {
    const unsigned char character = static_cast<unsigned char>(raw);
    switch (character) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (character < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", character);
          out += buffer;
        } else {
          out.push_back(raw);
        }
    }
  }
  out.push_back('"');
}

void number_into(double value, std::string& out) {
  if (!std::isfinite(value)) {
    out += "null";
    return;
  }
  char buffer[40];
  if (value == static_cast<double>(static_cast<long long>(value)) &&
      std::fabs(value) < 9.0e15) {
    std::snprintf(buffer, sizeof(buffer), "%lld", static_cast<long long>(value));
  } else {
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  }
  out += buffer;
}

}  // namespace

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : text_(text) {}

  Json parse_document() {
    skip_whitespace();
    Json value = parse_value(0);
    skip_whitespace();
    if (position_ != text_.size()) {
      fail("trailing characters after JSON document");
    }
    return value;
  }

 private:
  static constexpr int kMaxDepth = 256;

  [[noreturn]] void fail(const std::string& message) const {
    throw Error("JSON parse error at offset " + std::to_string(position_) + ": " + message);
  }

  void skip_whitespace() {
    while (position_ < text_.size()) {
      const char character = text_[position_];
      if (character == ' ' || character == '\t' || character == '\n' || character == '\r') {
        ++position_;
      } else {
        break;
      }
    }
  }

  char peek() const {
    if (position_ >= text_.size()) fail("unexpected end of input");
    return text_[position_];
  }

  void expect(char character) {
    if (position_ >= text_.size() || text_[position_] != character) {
      fail(std::string("expected '") + character + "'");
    }
    ++position_;
  }

  bool consume_literal(std::string_view literal) {
    if (text_.compare(position_, literal.size(), literal) == 0) {
      position_ += literal.size();
      return true;
    }
    return false;
  }

  Json parse_value(int depth) {
    if (depth > kMaxDepth) fail("maximum nesting depth exceeded");
    switch (peek()) {
      case '{': return parse_object(depth);
      case '[': return parse_array(depth);
      case '"': return Json::make_string(parse_string());
      case 't':
        if (consume_literal("true")) return Json::make_bool(true);
        fail("invalid literal");
      case 'f':
        if (consume_literal("false")) return Json::make_bool(false);
        fail("invalid literal");
      case 'n':
        if (consume_literal("null")) return Json::make_null();
        fail("invalid literal");
      case 'N':
        if (consume_literal("NaN")) return Json::make_null();
        fail("invalid literal");
      case 'I':
        if (consume_literal("Infinity")) return Json::make_null();
        fail("invalid literal");
      default: return parse_number();
    }
  }

  Json parse_object(int depth) {
    expect('{');
    std::vector<std::string> keys;
    std::vector<Json> values;
    skip_whitespace();
    if (peek() == '}') {
      ++position_;
      return Json::make_object(std::move(keys), std::move(values));
    }
    while (true) {
      skip_whitespace();
      keys.push_back(parse_string());
      skip_whitespace();
      expect(':');
      skip_whitespace();
      values.push_back(parse_value(depth + 1));
      skip_whitespace();
      const char character = peek();
      if (character == ',') {
        ++position_;
        continue;
      }
      if (character == '}') {
        ++position_;
        break;
      }
      fail("expected ',' or '}'");
    }
    return Json::make_object(std::move(keys), std::move(values));
  }

  Json parse_array(int depth) {
    expect('[');
    std::vector<Json> items;
    skip_whitespace();
    if (peek() == ']') {
      ++position_;
      return Json::make_array(std::move(items));
    }
    while (true) {
      skip_whitespace();
      items.push_back(parse_value(depth + 1));
      skip_whitespace();
      const char character = peek();
      if (character == ',') {
        ++position_;
        continue;
      }
      if (character == ']') {
        ++position_;
        break;
      }
      fail("expected ',' or ']'");
    }
    return Json::make_array(std::move(items));
  }

  std::string parse_string() {
    expect('"');
    std::string result;
    while (true) {
      if (position_ >= text_.size()) fail("unterminated string");
      const char character = text_[position_++];
      if (character == '"') break;
      if (character != '\\') {
        result.push_back(character);
        continue;
      }
      if (position_ >= text_.size()) fail("unterminated escape");
      const char escape = text_[position_++];
      switch (escape) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        case 'u': {
          unsigned int code_point = parse_hex4();
          if (code_point >= 0xD800 && code_point <= 0xDBFF && position_ + 1 < text_.size() &&
              text_[position_] == '\\' && text_[position_ + 1] == 'u') {
            position_ += 2;
            const unsigned int low = parse_hex4();
            if (low >= 0xDC00 && low <= 0xDFFF) {
              code_point = 0x10000 + ((code_point - 0xD800) << 10) + (low - 0xDC00);
            } else {
              encode_utf8(code_point, result);
              code_point = low;
            }
          }
          encode_utf8(code_point, result);
          break;
        }
        default: fail("invalid escape sequence");
      }
    }
    return result;
  }

  unsigned int parse_hex4() {
    if (position_ + 4 > text_.size()) fail("truncated unicode escape");
    unsigned int value = 0;
    for (int index = 0; index < 4; ++index) {
      const char digit = text_[position_++];
      value <<= 4;
      if (digit >= '0' && digit <= '9') {
        value |= static_cast<unsigned int>(digit - '0');
      } else if (digit >= 'a' && digit <= 'f') {
        value |= static_cast<unsigned int>(digit - 'a' + 10);
      } else if (digit >= 'A' && digit <= 'F') {
        value |= static_cast<unsigned int>(digit - 'A' + 10);
      } else {
        fail("invalid hex digit in unicode escape");
      }
    }
    return value;
  }

  Json parse_number() {
    const std::size_t start = position_;
    if (position_ < text_.size() && (text_[position_] == '-' || text_[position_] == '+')) ++position_;
    while (position_ < text_.size()) {
      const char character = text_[position_];
      const bool numeric = (character >= '0' && character <= '9') || character == '.' ||
                           character == 'e' || character == 'E' || character == '+' ||
                           character == '-';
      if (!numeric) break;
      ++position_;
    }
    if (position_ == start) fail("invalid number");
    const std::string literal(text_.substr(start, position_ - start));
    return Json::make_number(std::strtod(literal.c_str(), nullptr));
  }

  std::string_view text_;
  std::size_t position_ = 0;
};

Json Json::make_null() { return Json(); }

Json Json::make_bool(bool value) {
  Json result;
  result.kind_ = Kind::Bool;
  result.bool_value_ = value;
  return result;
}

Json Json::make_number(double value) {
  Json result;
  result.kind_ = Kind::Number;
  result.number_value_ = value;
  return result;
}

Json Json::make_string(std::string value) {
  Json result;
  result.kind_ = Kind::String;
  result.string_value_ = std::move(value);
  return result;
}

Json Json::make_array(std::vector<Json> items) {
  Json result;
  result.kind_ = Kind::Array;
  result.values_ = std::move(items);
  return result;
}

Json Json::make_object(std::vector<std::string> keys, std::vector<Json> values) {
  Json result;
  result.kind_ = Kind::Object;
  result.keys_ = std::move(keys);
  result.values_ = std::move(values);
  return result;
}

bool Json::as_bool(bool fallback) const {
  if (kind_ == Kind::Bool) return bool_value_;
  if (kind_ == Kind::Number) return number_value_ != 0.0;
  return fallback;
}

double Json::as_double(double fallback) const {
  if (kind_ == Kind::Number) return number_value_;
  if (kind_ == Kind::Bool) return bool_value_ ? 1.0 : 0.0;
  return fallback;
}

std::int64_t Json::as_int64(std::int64_t fallback) const {
  if (kind_ == Kind::Number) return static_cast<std::int64_t>(number_value_);
  if (kind_ == Kind::Bool) return bool_value_ ? 1 : 0;
  return fallback;
}

const std::string& Json::as_string() const {
  if (kind_ != Kind::String) throw Error("JSON value is not a string");
  return string_value_;
}

std::string Json::as_string_or(const std::string& fallback) const {
  return kind_ == Kind::String ? string_value_ : fallback;
}

std::size_t Json::size() const {
  if (kind_ == Kind::Array || kind_ == Kind::Object) return values_.size();
  return 0;
}

const Json& Json::at(std::size_t index) const {
  if (kind_ != Kind::Array || index >= values_.size()) {
    throw Error("JSON array index " + std::to_string(index) + " out of range");
  }
  return values_[index];
}

const Json* Json::find(std::string_view key) const {
  if (kind_ != Kind::Object) return nullptr;
  for (std::size_t index = 0; index < keys_.size(); ++index) {
    if (keys_[index] == key) return &values_[index];
  }
  return nullptr;
}

const Json& Json::at(std::string_view key) const {
  const Json* found = find(key);
  if (found == nullptr) throw Error("missing JSON key '" + std::string(key) + "'");
  return *found;
}

Json Json::parse(std::string_view text) { return JsonParser(text).parse_document(); }

Json Json::parse_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw Error("cannot open JSON file '" + path + "'");
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  const std::string content = buffer.str();
  return parse(content);
}

std::vector<Json> Json::parse_lines_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw Error("cannot open JSONL file '" + path + "'");
  std::vector<Json> documents;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    bool blank = true;
    for (const char character : line) {
      if (character != ' ' && character != '\t') {
        blank = false;
        break;
      }
    }
    if (blank) continue;
    try {
      documents.push_back(parse(line));
    } catch (const Error& error) {
      throw Error("in '" + path + "' line " + std::to_string(line_number) + ": " + error.what());
    }
  }
  return documents;
}

void Json::dump_into(std::string& out, int indent, int depth) const {
  const bool pretty = indent >= 0;
  const std::string pad = pretty ? std::string(static_cast<std::size_t>(indent * (depth + 1)), ' ') : std::string();
  const std::string close_pad = pretty ? std::string(static_cast<std::size_t>(indent * depth), ' ') : std::string();

  switch (kind_) {
    case Kind::Null: out += "null"; return;
    case Kind::Bool: out += bool_value_ ? "true" : "false"; return;
    case Kind::Number: number_into(number_value_, out); return;
    case Kind::String: escape_into(string_value_, out); return;
    case Kind::Array: {
      if (values_.empty()) {
        out += "[]";
        return;
      }
      out.push_back('[');
      for (std::size_t index = 0; index < values_.size(); ++index) {
        if (index != 0) out.push_back(',');
        if (pretty) {
          out.push_back('\n');
          out += pad;
        }
        values_[index].dump_into(out, indent, depth + 1);
      }
      if (pretty) {
        out.push_back('\n');
        out += close_pad;
      }
      out.push_back(']');
      return;
    }
    case Kind::Object: {
      if (values_.empty()) {
        out += "{}";
        return;
      }
      out.push_back('{');
      for (std::size_t index = 0; index < values_.size(); ++index) {
        if (index != 0) out.push_back(',');
        if (pretty) {
          out.push_back('\n');
          out += pad;
        }
        escape_into(keys_[index], out);
        out.push_back(':');
        if (pretty) out.push_back(' ');
        values_[index].dump_into(out, indent, depth + 1);
      }
      if (pretty) {
        out.push_back('\n');
        out += close_pad;
      }
      out.push_back('}');
      return;
    }
  }
}

std::string Json::dump(int indent) const {
  std::string out;
  dump_into(out, indent, 0);
  return out;
}

JsonWriter::JsonWriter(int indent) : indent_(indent) {}

void JsonWriter::newline_indent() {
  if (indent_ < 0) return;
  buffer_.push_back('\n');
  buffer_.append(static_cast<std::size_t>(indent_ * depth_), ' ');
}

void JsonWriter::prepare_value() {
  if (expect_value_) {
    expect_value_ = false;
    return;
  }
  if (!has_entry_.empty()) {
    if (has_entry_.back()) buffer_.push_back(',');
    has_entry_.back() = true;
    newline_indent();
  }
}

void JsonWriter::begin_object() {
  prepare_value();
  buffer_.push_back('{');
  ++depth_;
  has_entry_.push_back(false);
}

void JsonWriter::end_object() {
  const bool had_entry = has_entry_.back();
  has_entry_.pop_back();
  --depth_;
  if (had_entry) newline_indent();
  buffer_.push_back('}');
}

void JsonWriter::begin_array() {
  prepare_value();
  buffer_.push_back('[');
  ++depth_;
  has_entry_.push_back(false);
}

void JsonWriter::end_array() {
  const bool had_entry = has_entry_.back();
  has_entry_.pop_back();
  --depth_;
  if (had_entry) newline_indent();
  buffer_.push_back(']');
}

void JsonWriter::key(std::string_view name) {
  prepare_value();
  escape_into(name, buffer_);
  buffer_.push_back(':');
  if (indent_ >= 0) buffer_.push_back(' ');
  expect_value_ = true;
}

void JsonWriter::value_null() {
  prepare_value();
  buffer_ += "null";
}

void JsonWriter::value_bool(bool value) {
  prepare_value();
  buffer_ += value ? "true" : "false";
}

void JsonWriter::value_number(double value) {
  prepare_value();
  number_into(value, buffer_);
}

void JsonWriter::value_int(std::int64_t value) {
  prepare_value();
  buffer_ += std::to_string(value);
}

void JsonWriter::value_string(std::string_view value) {
  prepare_value();
  escape_into(value, buffer_);
}

MappedFile::MappedFile(const std::string& path) : path_(path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    throw Error("cannot open '" + path + "': " + std::strerror(errno));
  }

  struct stat status {};
  if (::fstat(descriptor, &status) != 0) {
    ::close(descriptor);
    throw Error("cannot stat '" + path + "': " + std::strerror(errno));
  }

  size_ = static_cast<std::size_t>(status.st_size);
  if (size_ == 0) {
    ::close(descriptor);
    throw Error("file '" + path + "' is empty");
  }

  void* mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
  ::close(descriptor);
  if (mapping == MAP_FAILED) {
    throw Error("cannot mmap '" + path + "': " + std::strerror(errno));
  }

  ::madvise(mapping, size_, MADV_SEQUENTIAL);
  data_ = static_cast<const std::uint8_t*>(mapping);
}

MappedFile::~MappedFile() { release(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : path_(std::move(other.path_)), data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    release();
    path_ = std::move(other.path_);
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void MappedFile::release() {
  if (data_ != nullptr) {
    ::munmap(const_cast<std::uint8_t*>(data_), size_);
    data_ = nullptr;
    size_ = 0;
  }
}

const std::uint8_t* MappedFile::at(std::size_t offset, std::size_t length) const {
  if (data_ == nullptr || offset > size_ || length > size_ - offset) {
    throw Error("out of bounds read in '" + path_ + "' at offset " + std::to_string(offset));
  }
  return data_ + offset;
}

}  // namespace attnrank

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace attnrank {

constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

inline double gibibytes(std::size_t bytes) { return static_cast<double>(bytes) / kBytesPerGiB; }

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message) : std::runtime_error(message) {}
};

enum class DType { F32, F16, BF16, I64, I32, I16, I8, U8, Bool, Unknown };

std::size_t dtype_size(DType dtype);
const char* dtype_name(DType dtype);
DType dtype_from_safetensors(const std::string& name);
DType dtype_from_torch_storage(const std::string& name);

struct TensorView {
  std::string name;
  DType dtype = DType::Unknown;
  std::vector<std::int64_t> shape;
  const std::uint8_t* data = nullptr;
  std::size_t byte_size = 0;

  std::int64_t element_count() const;
  std::int64_t dim(std::size_t index) const;
  std::size_t rank() const { return shape.size(); }
};

struct Document {
  std::string id;
  std::string title;
  std::string text;
};

struct ScoredDocument {
  std::size_t index = 0;
  double score = 0.0;
};

struct TokenSpan {
  int begin = 0;
  int end = 0;

  int length() const { return end - begin; }
  bool empty() const { return end <= begin; }
};

using TokenSequence = std::vector<int>;

class Json {
 public:
  enum class Kind { Null, Bool, Number, String, Array, Object };

  Json() = default;

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::Null; }
  bool is_bool() const { return kind_ == Kind::Bool; }
  bool is_number() const { return kind_ == Kind::Number; }
  bool is_string() const { return kind_ == Kind::String; }
  bool is_array() const { return kind_ == Kind::Array; }
  bool is_object() const { return kind_ == Kind::Object; }

  bool as_bool(bool fallback = false) const;
  double as_double(double fallback = 0.0) const;
  std::int64_t as_int64(std::int64_t fallback = 0) const;
  const std::string& as_string() const;
  std::string as_string_or(const std::string& fallback) const;

  std::size_t size() const;
  const Json& at(std::size_t index) const;
  const Json* find(std::string_view key) const;
  const Json& at(std::string_view key) const;
  bool contains(std::string_view key) const { return find(key) != nullptr; }

  const std::vector<std::string>& keys() const { return keys_; }
  const std::vector<Json>& values() const { return values_; }

  static Json parse(std::string_view text);
  static Json parse_file(const std::string& path);
  static std::vector<Json> parse_lines_file(const std::string& path);

  static Json make_null();
  static Json make_bool(bool value);
  static Json make_number(double value);
  static Json make_string(std::string value);
  static Json make_array(std::vector<Json> items);
  static Json make_object(std::vector<std::string> keys, std::vector<Json> values);

  std::string dump(int indent = -1) const;

 private:
  friend class JsonParser;

  void dump_into(std::string& out, int indent, int depth) const;

  Kind kind_ = Kind::Null;
  bool bool_value_ = false;
  double number_value_ = 0.0;
  std::string string_value_;
  std::vector<std::string> keys_;
  std::vector<Json> values_;
};

class JsonWriter {
 public:
  explicit JsonWriter(int indent);

  void begin_object();
  void end_object();
  void begin_array();
  void end_array();
  void key(std::string_view name);
  void value_null();
  void value_bool(bool value);
  void value_number(double value);
  void value_int(std::int64_t value);
  void value_string(std::string_view value);

  const std::string& str() const { return buffer_; }

 private:
  void prepare_value();
  void newline_indent();

  std::string buffer_;
  int indent_ = -1;
  int depth_ = 0;
  std::vector<bool> has_entry_;
  bool expect_value_ = false;
};

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

class Log {
 public:
  static void set_threshold(LogLevel level) { threshold() = level; }
  static LogLevel get_threshold() { return threshold(); }

  template <typename... Args>
  static void debug(const char* format, Args... args) { emit(LogLevel::Debug, "debug", format, args...); }
  template <typename... Args>
  static void info(const char* format, Args... args) { emit(LogLevel::Info, "info ", format, args...); }
  template <typename... Args>
  static void warn(const char* format, Args... args) { emit(LogLevel::Warn, "warn ", format, args...); }
  template <typename... Args>
  static void error(const char* format, Args... args) { emit(LogLevel::Error, "error", format, args...); }

 private:
  static LogLevel& threshold() {
    static LogLevel level = LogLevel::Info;
    return level;
  }

  static std::mutex& mutex() {
    static std::mutex instance;
    return instance;
  }

  static double elapsed_seconds() {
    using Clock = std::chrono::steady_clock;
    static const Clock::time_point start = Clock::now();
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  template <typename... Args>
  static void emit(LogLevel level, const char* tag, const char* format, Args... args) {
    if (static_cast<int>(level) < static_cast<int>(threshold())) return;
    std::lock_guard<std::mutex> guard(mutex());
    std::fprintf(stderr, "[%8.2fs %s] ", elapsed_seconds(), tag);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
    std::fprintf(stderr, format, args...);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    std::fputc('\n', stderr);
    std::fflush(stderr);
  }
};

class MappedFile {
 public:
  MappedFile() = default;
  explicit MappedFile(const std::string& path);
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  const std::uint8_t* data() const { return data_; }
  std::size_t size() const { return size_; }
  const std::string& path() const { return path_; }
  bool valid() const { return data_ != nullptr; }

  const std::uint8_t* at(std::size_t offset, std::size_t length) const;

 private:
  void release();

  std::string path_;
  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace attnrank

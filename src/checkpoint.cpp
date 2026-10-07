#include "attnrank/checkpoint.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace attnrank {

namespace {

constexpr std::size_t kSafetensorsHeaderPrefix = 8;

bool has_suffix(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::uint64_t read_u64_le(const std::uint8_t* source) {
  std::uint64_t value = 0;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

}  // namespace

void WeightRegistry::add_source(std::unique_ptr<IWeightSource> source) {
  if (source == nullptr) return;
  const IWeightSource* raw = source.get();
  sources_.push_back(std::move(source));
  for (const TensorView& tensor : raw->tensors()) {
    index_[tensor.name] = &tensor;
  }
}

const TensorView* WeightRegistry::find(const std::string& name) const {
  const auto found = index_.find(name);
  return found == index_.end() ? nullptr : found->second;
}

const TensorView& WeightRegistry::require(const std::string& name) const {
  const TensorView* found = find(name);
  if (found == nullptr) throw Error("checkpoint is missing tensor '" + name + "'");
  return *found;
}

std::vector<std::string> WeightRegistry::names() const {
  std::vector<std::string> result;
  result.reserve(index_.size());
  for (const auto& entry : index_) result.push_back(entry.first);
  std::sort(result.begin(), result.end());
  return result;
}

std::size_t WeightRegistry::total_bytes() const {
  std::size_t total = 0;
  for (const auto& entry : index_) total += entry.second->byte_size;
  return total;
}

SafetensorsWeightSource::SafetensorsWeightSource(const std::string& file_path)
    : origin_(file_path), file_(file_path) {
  if (file_.size() < kSafetensorsHeaderPrefix) throw Error("safetensors file too small: '" + file_path + "'");

  const std::uint64_t header_size = read_u64_le(file_.at(0, kSafetensorsHeaderPrefix));
  if (header_size == 0 || header_size > file_.size() - kSafetensorsHeaderPrefix) {
    throw Error("invalid safetensors header length in '" + file_path + "'");
  }

  const std::size_t data_origin = kSafetensorsHeaderPrefix + static_cast<std::size_t>(header_size);
  const std::string_view header_text(
      reinterpret_cast<const char*>(file_.at(kSafetensorsHeaderPrefix, header_size)),
      static_cast<std::size_t>(header_size));
  const Json header = Json::parse(header_text);
  if (!header.is_object()) throw Error("safetensors header is not an object");

  const std::vector<std::string>& keys = header.keys();
  const std::vector<Json>& values = header.values();
  tensors_.reserve(keys.size());

  for (std::size_t index = 0; index < keys.size(); ++index) {
    if (keys[index] == "__metadata__") continue;
    const Json& entry = values[index];

    TensorView tensor;
    tensor.name = keys[index];
    tensor.dtype = dtype_from_safetensors(entry.at("dtype").as_string());
    if (tensor.dtype == DType::Unknown) {
      throw Error("unsupported dtype '" + entry.at("dtype").as_string() + "' for tensor '" +
                          tensor.name + "'");
    }

    const Json& shape = entry.at("shape");
    tensor.shape.reserve(shape.size());
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      tensor.shape.push_back(shape.at(axis).as_int64());
    }

    const Json& offsets = entry.at("data_offsets");
    if (offsets.size() != 2) throw Error("bad data_offsets for tensor '" + tensor.name + "'");
    const std::size_t begin = static_cast<std::size_t>(offsets.at(0).as_int64());
    const std::size_t end = static_cast<std::size_t>(offsets.at(1).as_int64());
    if (end < begin) throw Error("inverted data_offsets for tensor '" + tensor.name + "'");

    tensor.byte_size = end - begin;
    tensor.data = file_.at(data_origin + begin, tensor.byte_size);

    const std::size_t expected = static_cast<std::size_t>(tensor.element_count()) * dtype_size(tensor.dtype);
    if (expected != tensor.byte_size) {
      throw Error("tensor '" + tensor.name + "' byte size mismatch");
    }
    tensors_.push_back(std::move(tensor));
  }

  Log::debug("safetensors '%s': %zu tensors", file_path.c_str(), tensors_.size());
}

bool SafetensorsWeightSourceFactory::accepts(const std::string& file_path) const {
  return has_suffix(file_path, ".safetensors");
}

std::unique_ptr<IWeightSource> SafetensorsWeightSourceFactory::create(const std::string& file_path) const {
  return std::make_unique<SafetensorsWeightSource>(file_path);
}

bool TorchArchiveWeightSourceFactory::accepts(const std::string& file_path) const {
  return has_suffix(file_path, ".bin") || has_suffix(file_path, ".pth") ||
         has_suffix(file_path, ".pt");
}

std::unique_ptr<IWeightSource> TorchArchiveWeightSourceFactory::create(const std::string& file_path) const {
  return std::make_unique<TorchArchiveWeightSource>(file_path);
}

CheckpointLoader::CheckpointLoader(std::vector<std::shared_ptr<IWeightSourceFactory>> factories)
    : factories_(std::move(factories)) {
  if (factories_.empty()) throw Error("CheckpointLoader requires at least one factory");
}

CheckpointLoader CheckpointLoader::with_default_factories() {
  return CheckpointLoader({std::make_shared<SafetensorsWeightSourceFactory>(),
                           std::make_shared<TorchArchiveWeightSourceFactory>()});
}

std::vector<std::string> CheckpointLoader::select_shards(const std::string& directory) const {
  namespace fs = std::filesystem;
  if (!fs::is_directory(directory)) {
    throw Error("model path is not a directory: '" + directory + "'");
  }

  std::vector<std::string> safetensors;
  std::vector<std::string> torch_archives;
  for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
    if (!entry.is_regular_file()) continue;
    const std::string path = entry.path().string();
    const std::string filename = entry.path().filename().string();
    if (has_suffix(filename, ".safetensors")) {
      safetensors.push_back(path);
    } else if (has_suffix(filename, ".bin") && filename.rfind("pytorch_model", 0) == 0) {
      torch_archives.push_back(path);
    }
  }

  std::vector<std::string>& selected = safetensors.empty() ? torch_archives : safetensors;
  if (selected.empty()) {
    throw Error("no weight shards (*.safetensors or pytorch_model*.bin) under '" + directory + "'");
  }
  std::sort(selected.begin(), selected.end());
  return selected;
}

WeightRegistry CheckpointLoader::load_directory(const std::string& directory) const {
  const std::vector<std::string> shards = select_shards(directory);
  WeightRegistry registry;

  for (const std::string& shard : shards) {
    const IWeightSourceFactory* selected = nullptr;
    for (const std::shared_ptr<IWeightSourceFactory>& factory : factories_) {
      if (factory->accepts(shard)) {
        selected = factory.get();
        break;
      }
    }
    if (selected == nullptr) {
      throw Error("no registered loader accepts shard '" + shard + "'");
    }
    Log::info("loading %s shard %s", selected->format_name(),
              std::filesystem::path(shard).filename().string().c_str());
    registry.add_source(selected->create(shard));
  }

  Log::info("checkpoint ready: %zu tensors, %.2f GiB mapped", registry.tensor_count(),
            gibibytes(registry.total_bytes()));
  return registry;
}

namespace {

constexpr std::uint32_t kEndOfCentralDirectory = 0x06054b50u;
constexpr std::uint32_t kZip64EndOfCentralDirectory = 0x06064b50u;
constexpr std::uint32_t kZip64Locator = 0x07064b50u;
constexpr std::uint32_t kCentralFileHeader = 0x02014b50u;
constexpr std::uint32_t kLocalFileHeader = 0x04034b50u;
constexpr std::uint16_t kZip64ExtraFieldId = 0x0001;
constexpr std::uint32_t kZip32Overflow = 0xFFFFFFFFu;
constexpr std::size_t kEndOfCentralDirectorySize = 22;
constexpr std::size_t kEndOfCentralDirectorySearchWindow = 66000;
constexpr std::size_t kZip64LocatorSize = 20;
constexpr std::size_t kCentralFileHeaderSize = 46;
constexpr std::size_t kLocalFileHeaderSize = 30;

std::uint16_t read_u16(const std::uint8_t* source) {
  std::uint16_t value = 0;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

std::uint32_t read_u32(const std::uint8_t* source) {
  std::uint32_t value = 0;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

std::uint64_t read_u64(const std::uint8_t* source) {
  std::uint64_t value = 0;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

struct ZipRecord {
  std::uint64_t data_offset = 0;
  std::uint64_t size = 0;
};

class ZipArchive {
 public:
  explicit ZipArchive(const MappedFile& file) : file_(file) { parse_central_directory(); }

  const ZipRecord* find(const std::string& name) const {
    const auto found = records_.find(name);
    return found == records_.end() ? nullptr : &found->second;
  }

  std::string detect_prefix() const {
    for (const auto& entry : records_) {
      const std::string suffix = "/data.pkl";
      if (entry.first.size() > suffix.size() &&
          entry.first.compare(entry.first.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return entry.first.substr(0, entry.first.size() - suffix.size());
      }
    }
    throw Error("torch archive '" + file_.path() + "' has no data.pkl record");
  }

 private:
  std::uint64_t locate_end_of_central_directory() const {
    const std::size_t size = file_.size();
    const std::size_t window = std::min<std::size_t>(size, kEndOfCentralDirectorySearchWindow);
    for (std::size_t back = kEndOfCentralDirectorySize; back <= window; ++back) {
      const std::size_t offset = size - back;
      if (read_u32(file_.at(offset, 4)) == kEndOfCentralDirectory) return offset;
    }
    throw Error("'" + file_.path() + "' is not a zip archive (no end of central directory)");
  }

  void parse_central_directory() {
    const std::uint64_t eocd = locate_end_of_central_directory();
    std::uint64_t entry_count = read_u16(file_.at(eocd + 10, 2));
    std::uint64_t directory_offset = read_u32(file_.at(eocd + 16, 4));

    if (eocd >= kZip64LocatorSize) {
      const std::uint64_t locator = eocd - kZip64LocatorSize;
      if (read_u32(file_.at(locator, 4)) == kZip64Locator) {
        const std::uint64_t zip64_eocd = read_u64(file_.at(locator + 8, 8));
        if (read_u32(file_.at(zip64_eocd, 4)) != kZip64EndOfCentralDirectory) {
          throw Error("corrupt zip64 end of central directory in '" + file_.path() + "'");
        }
        entry_count = read_u64(file_.at(zip64_eocd + 32, 8));
        directory_offset = read_u64(file_.at(zip64_eocd + 48, 8));
      }
    }

    std::uint64_t cursor = directory_offset;
    for (std::uint64_t index = 0; index < entry_count; ++index) {
      if (read_u32(file_.at(cursor, 4)) != kCentralFileHeader) {
        throw Error("corrupt central directory entry in '" + file_.path() + "'");
      }
      const std::uint16_t method = read_u16(file_.at(cursor + 10, 2));
      std::uint64_t uncompressed = read_u32(file_.at(cursor + 24, 4));
      const std::uint16_t name_length = read_u16(file_.at(cursor + 28, 2));
      const std::uint16_t extra_length = read_u16(file_.at(cursor + 30, 2));
      const std::uint16_t comment_length = read_u16(file_.at(cursor + 32, 2));
      std::uint64_t local_offset = read_u32(file_.at(cursor + 42, 4));
      const std::uint64_t compressed_raw = read_u32(file_.at(cursor + 20, 4));

      const std::string name(reinterpret_cast<const char*>(file_.at(cursor + kCentralFileHeaderSize, name_length)),
                             name_length);

      if (uncompressed == kZip32Overflow || local_offset == kZip32Overflow || compressed_raw == kZip32Overflow) {
        resolve_zip64_extra(cursor + kCentralFileHeaderSize + name_length, extra_length, uncompressed, local_offset,
                            compressed_raw);
      }

      if (method != 0) {
        throw Error("record '" + name + "' in '" + file_.path() +
                            "' is compressed; only stored records are supported");
      }

      const std::uint32_t local_signature = read_u32(file_.at(local_offset, 4));
      if (local_signature != kLocalFileHeader) {
        throw Error("corrupt local header for '" + name + "'");
      }
      const std::uint16_t local_name_length = read_u16(file_.at(local_offset + 26, 2));
      const std::uint16_t local_extra_length = read_u16(file_.at(local_offset + 28, 2));

      ZipRecord record;
      record.data_offset = local_offset + kLocalFileHeaderSize + local_name_length + local_extra_length;
      record.size = uncompressed;
      file_.at(record.data_offset, record.size);
      records_.emplace(name, record);

      cursor += kCentralFileHeaderSize + name_length + extra_length + comment_length;
    }
  }

  void resolve_zip64_extra(std::uint64_t extra_offset, std::uint16_t extra_length,
                           std::uint64_t& uncompressed, std::uint64_t& local_offset,
                           std::uint64_t compressed_raw) const {
    std::uint64_t cursor = extra_offset;
    const std::uint64_t end = extra_offset + extra_length;
    while (cursor + 4 <= end) {
      const std::uint16_t header_id = read_u16(file_.at(cursor, 2));
      const std::uint16_t field_size = read_u16(file_.at(cursor + 2, 2));
      if (header_id == kZip64ExtraFieldId) {
        std::uint64_t field_cursor = cursor + 4;
        const std::uint64_t field_end = field_cursor + field_size;
        if (uncompressed == kZip32Overflow && field_cursor + 8 <= field_end) {
          uncompressed = read_u64(file_.at(field_cursor, 8));
          field_cursor += 8;
        }
        if (compressed_raw == kZip32Overflow && field_cursor + 8 <= field_end) {
          field_cursor += 8;
        }
        if (local_offset == kZip32Overflow && field_cursor + 8 <= field_end) {
          local_offset = read_u64(file_.at(field_cursor, 8));
        }
        return;
      }
      cursor += 4 + field_size;
    }
  }

  const MappedFile& file_;
  std::unordered_map<std::string, ZipRecord> records_;
};

struct PickleValue;
using PickleRef = std::shared_ptr<PickleValue>;

struct PickleValue {
  enum class Kind {
    None, Bool, Int, Float, Text, Bytes, Tuple, List, Dict, Global, Storage, Tensor, Mark, Opaque
  };

  Kind kind = Kind::None;
  bool boolean = false;
  long long integer = 0;
  double floating = 0.0;
  std::string text;
  std::vector<PickleRef> items;
  std::vector<std::pair<PickleRef, PickleRef>> entries;
  std::string module_name;
  std::string global_name;
  std::string storage_key;
  DType storage_dtype = DType::Unknown;
  PickleRef storage;
  long long storage_offset = 0;
  std::vector<long long> sizes;
  std::vector<long long> strides;
};

PickleRef make_value(PickleValue::Kind kind) {
  PickleRef value = std::make_shared<PickleValue>();
  value->kind = kind;
  return value;
}

PickleRef make_integer(long long integer) {
  PickleRef value = make_value(PickleValue::Kind::Int);
  value->integer = integer;
  return value;
}

PickleRef make_text(std::string text) {
  PickleRef value = make_value(PickleValue::Kind::Text);
  value->text = std::move(text);
  return value;
}

class PickleMachine {
 public:
  PickleMachine(const std::uint8_t* data, std::size_t size, const std::string& origin)
      : data_(data), size_(size), origin_(origin) {}

  PickleRef run() {
    while (position_ < size_) {
      const std::uint8_t opcode = data_[position_++];
      if (!step(opcode)) break;
    }
    if (stack_.empty()) throw Error("pickle stream in '" + origin_ + "' produced no value");
    return stack_.back();
  }

 private:
  [[noreturn]] void fail(const std::string& message) const {
    throw Error("pickle error in '" + origin_ + "' at byte " + std::to_string(position_) +
                        ": " + message);
  }

  const std::uint8_t* take(std::size_t count) {
    if (position_ + count > size_) fail("unexpected end of stream");
    const std::uint8_t* pointer = data_ + position_;
    position_ += count;
    return pointer;
  }

  std::string take_line() {
    const std::size_t start = position_;
    while (position_ < size_ && data_[position_] != '\n') ++position_;
    if (position_ >= size_) fail("unterminated line");
    const std::string line(reinterpret_cast<const char*>(data_ + start), position_ - start);
    ++position_;
    return line;
  }

  std::string take_text(std::size_t count) {
    const std::uint8_t* pointer = take(count);
    return std::string(reinterpret_cast<const char*>(pointer), count);
  }

  void push(PickleRef value) { stack_.push_back(std::move(value)); }

  PickleRef pop() {
    if (stack_.empty()) fail("stack underflow");
    PickleRef value = stack_.back();
    stack_.pop_back();
    return value;
  }

  std::vector<PickleRef> pop_to_mark() {
    std::size_t index = stack_.size();
    while (index > 0 && stack_[index - 1]->kind != PickleValue::Kind::Mark) --index;
    if (index == 0) fail("no mark on stack");
    std::vector<PickleRef> items(stack_.begin() + static_cast<long>(index), stack_.end());
    stack_.resize(index - 1);
    return items;
  }

  void memo_put(std::size_t index) {
    if (stack_.empty()) fail("memo put with empty stack");
    if (memo_.size() <= index) memo_.resize(index + 1);
    memo_[index] = stack_.back();
  }

  void memo_get(std::size_t index) {
    if (index >= memo_.size() || memo_[index] == nullptr) fail("memo get for unset index");
    push(memo_[index]);
  }

  static long long decode_signed_long(const std::uint8_t* bytes, std::size_t count) {
    if (count == 0) return 0;
    unsigned long long magnitude = 0;
    for (std::size_t index = 0; index < count && index < 8; ++index) {
      magnitude |= static_cast<unsigned long long>(bytes[index]) << (8 * index);
    }
    if (count <= 8 && (bytes[count - 1] & 0x80) != 0) {
      for (std::size_t index = count; index < 8; ++index) {
        magnitude |= 0xFFull << (8 * index);
      }
    }
    return static_cast<long long>(magnitude);
  }

  static double decode_big_endian_double(const std::uint8_t* bytes) {
    std::uint64_t raw = 0;
    for (int index = 0; index < 8; ++index) {
      raw = (raw << 8) | bytes[index];
    }
    double value = 0.0;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
  }

  PickleRef make_global(std::string module_name, std::string global_name) {
    PickleRef value = make_value(PickleValue::Kind::Global);
    value->module_name = std::move(module_name);
    value->global_name = std::move(global_name);
    return value;
  }

  PickleRef build_storage(const PickleRef& descriptor) {
    if (descriptor->kind != PickleValue::Kind::Tuple || descriptor->items.size() < 5) {
      fail("unsupported persistent id layout");
    }
    const PickleRef& type_value = descriptor->items[1];
    const PickleRef& key_value = descriptor->items[2];

    PickleRef storage = make_value(PickleValue::Kind::Storage);
    storage->global_name =
        type_value->kind == PickleValue::Kind::Global ? type_value->global_name : type_value->text;
    storage->storage_dtype = dtype_from_torch_storage(storage->global_name);
    if (storage->storage_dtype == DType::Unknown) {
      fail("unsupported storage type '" + storage->global_name + "'");
    }
    storage->storage_key = key_value->text;
    return storage;
  }

  static std::vector<long long> to_integer_vector(const PickleRef& value) {
    std::vector<long long> result;
    if (value == nullptr) return result;
    for (const PickleRef& item : value->items) {
      result.push_back(item->integer);
    }
    return result;
  }

  PickleRef apply_reduce(const PickleRef& callable, const PickleRef& arguments) {
    if (callable->kind != PickleValue::Kind::Global) return make_value(PickleValue::Kind::Opaque);

    const std::string& name = callable->global_name;
    if (name == "_rebuild_tensor_v2" || name == "_rebuild_tensor_v3" || name == "_rebuild_tensor") {
      if (arguments->items.size() < 4) fail("malformed tensor rebuild arguments");
      PickleRef tensor = make_value(PickleValue::Kind::Tensor);
      tensor->storage = arguments->items[0];
      tensor->storage_offset = arguments->items[1]->integer;
      tensor->sizes = to_integer_vector(arguments->items[2]);
      tensor->strides = to_integer_vector(arguments->items[3]);
      return tensor;
    }
    if (name == "OrderedDict" || name == "dict") return make_value(PickleValue::Kind::Dict);
    return make_value(PickleValue::Kind::Opaque);
  }

  void set_items(const PickleRef& target, const std::vector<PickleRef>& flat) {
    if (target->kind != PickleValue::Kind::Dict) return;
    for (std::size_t index = 0; index + 1 < flat.size(); index += 2) {
      target->entries.emplace_back(flat[index], flat[index + 1]);
    }
  }

  bool step(std::uint8_t opcode) {
    switch (opcode) {
      case 0x80: take(1); return true;
      case 0x95: take(8); return true;
      case 0x94: memo_put(memo_next_++); return true;
      case '(': push(make_value(PickleValue::Kind::Mark)); return true;
      case '.': return false;
      case '0': pop(); return true;
      case '1': pop_to_mark(); return true;
      case 'N': push(make_value(PickleValue::Kind::None)); return true;
      case 0x88: {
        PickleRef value = make_value(PickleValue::Kind::Bool);
        value->boolean = true;
        push(value);
        return true;
      }
      case 0x89: {
        PickleRef value = make_value(PickleValue::Kind::Bool);
        value->boolean = false;
        push(value);
        return true;
      }
      case 'J': {
        std::int32_t value = 0;
        std::memcpy(&value, take(4), 4);
        push(make_integer(value));
        return true;
      }
      case 'K': push(make_integer(*take(1))); return true;
      case 'M': {
        std::uint16_t value = 0;
        std::memcpy(&value, take(2), 2);
        push(make_integer(value));
        return true;
      }
      case 'I': {
        const std::string line = take_line();
        if (line == "01") {
          PickleRef value = make_value(PickleValue::Kind::Bool);
          value->boolean = true;
          push(value);
        } else if (line == "00") {
          PickleRef value = make_value(PickleValue::Kind::Bool);
          value->boolean = false;
          push(value);
        } else {
          push(make_integer(std::strtoll(line.c_str(), nullptr, 10)));
        }
        return true;
      }
      case 'L': {
        std::string line = take_line();
        if (!line.empty() && line.back() == 'L') line.pop_back();
        push(make_integer(std::strtoll(line.c_str(), nullptr, 10)));
        return true;
      }
      case 0x8a: {
        const std::size_t count = *take(1);
        push(make_integer(decode_signed_long(take(count), count)));
        return true;
      }
      case 0x8b: {
        std::uint32_t count = 0;
        std::memcpy(&count, take(4), 4);
        push(make_integer(decode_signed_long(take(count), count)));
        return true;
      }
      case 'G': {
        PickleRef value = make_value(PickleValue::Kind::Float);
        value->floating = decode_big_endian_double(take(8));
        push(value);
        return true;
      }
      case 'F': {
        PickleRef value = make_value(PickleValue::Kind::Float);
        value->floating = std::strtod(take_line().c_str(), nullptr);
        push(value);
        return true;
      }
      case 'X': {
        std::uint32_t length = 0;
        std::memcpy(&length, take(4), 4);
        push(make_text(take_text(length)));
        return true;
      }
      case 0x8c: push(make_text(take_text(*take(1)))); return true;
      case 0x8d: {
        std::uint64_t length = 0;
        std::memcpy(&length, take(8), 8);
        push(make_text(take_text(static_cast<std::size_t>(length))));
        return true;
      }
      case 'V': push(make_text(take_line())); return true;
      case 'S': {
        std::string line = take_line();
        if (line.size() >= 2) line = line.substr(1, line.size() - 2);
        push(make_text(std::move(line)));
        return true;
      }
      case 'T': {
        std::uint32_t length = 0;
        std::memcpy(&length, take(4), 4);
        push(make_text(take_text(length)));
        return true;
      }
      case 'U': push(make_text(take_text(*take(1)))); return true;
      case 'B': {
        std::uint32_t length = 0;
        std::memcpy(&length, take(4), 4);
        PickleRef value = make_value(PickleValue::Kind::Bytes);
        value->text = take_text(length);
        push(value);
        return true;
      }
      case 'C': {
        PickleRef value = make_value(PickleValue::Kind::Bytes);
        value->text = take_text(*take(1));
        push(value);
        return true;
      }
      case ')': push(make_value(PickleValue::Kind::Tuple)); return true;
      case 0x85:
      case 0x86:
      case 0x87: {
        const std::size_t count = static_cast<std::size_t>(opcode - 0x84);
        PickleRef tuple = make_value(PickleValue::Kind::Tuple);
        tuple->items.resize(count);
        for (std::size_t index = count; index > 0; --index) tuple->items[index - 1] = pop();
        push(tuple);
        return true;
      }
      case 't': {
        PickleRef tuple = make_value(PickleValue::Kind::Tuple);
        tuple->items = pop_to_mark();
        push(tuple);
        return true;
      }
      case ']': push(make_value(PickleValue::Kind::List)); return true;
      case 'l': {
        PickleRef list = make_value(PickleValue::Kind::List);
        list->items = pop_to_mark();
        push(list);
        return true;
      }
      case 'a': {
        PickleRef item = pop();
        if (!stack_.empty()) stack_.back()->items.push_back(item);
        return true;
      }
      case 'e': {
        const std::vector<PickleRef> items = pop_to_mark();
        if (!stack_.empty()) {
          PickleRef& target = stack_.back();
          target->items.insert(target->items.end(), items.begin(), items.end());
        }
        return true;
      }
      case '}': push(make_value(PickleValue::Kind::Dict)); return true;
      case 'd': {
        PickleRef dictionary = make_value(PickleValue::Kind::Dict);
        set_items(dictionary, pop_to_mark());
        push(dictionary);
        return true;
      }
      case 's': {
        PickleRef value = pop();
        PickleRef key = pop();
        if (!stack_.empty() && stack_.back()->kind == PickleValue::Kind::Dict) {
          stack_.back()->entries.emplace_back(key, value);
        }
        return true;
      }
      case 'u': {
        const std::vector<PickleRef> flat = pop_to_mark();
        if (!stack_.empty()) set_items(stack_.back(), flat);
        return true;
      }
      case 'q': memo_put(*take(1)); return true;
      case 'r': {
        std::uint32_t index = 0;
        std::memcpy(&index, take(4), 4);
        memo_put(index);
        return true;
      }
      case 'p': memo_put(static_cast<std::size_t>(std::strtoull(take_line().c_str(), nullptr, 10)));
        return true;
      case 'h': memo_get(*take(1)); return true;
      case 'j': {
        std::uint32_t index = 0;
        std::memcpy(&index, take(4), 4);
        memo_get(index);
        return true;
      }
      case 'g': memo_get(static_cast<std::size_t>(std::strtoull(take_line().c_str(), nullptr, 10)));
        return true;
      case '2': {
        if (stack_.empty()) fail("dup on empty stack");
        push(stack_.back());
        return true;
      }
      case 'c': {
        const std::string module_name = take_line();
        const std::string global_name = take_line();
        push(make_global(module_name, global_name));
        return true;
      }
      case 0x93: {
        const PickleRef global_name = pop();
        const PickleRef module_name = pop();
        push(make_global(module_name->text, global_name->text));
        return true;
      }
      case 'Q': push(build_storage(pop())); return true;
      case 'P': {
        take_line();
        push(make_value(PickleValue::Kind::Opaque));
        return true;
      }
      case 'R': {
        const PickleRef arguments = pop();
        const PickleRef callable = pop();
        push(apply_reduce(callable, arguments));
        return true;
      }
      case 0x81: {
        const PickleRef arguments = pop();
        const PickleRef callable = pop();
        push(apply_reduce(callable, arguments));
        return true;
      }
      case 0x92: {
        pop();
        const PickleRef arguments = pop();
        const PickleRef callable = pop();
        push(apply_reduce(callable, arguments));
        return true;
      }
      case 'b': {
        const PickleRef state = pop();
        if (!stack_.empty() && state->kind == PickleValue::Kind::Dict) {
          PickleRef& target = stack_.back();
          if (target->kind == PickleValue::Kind::Dict) {
            target->entries.insert(target->entries.end(), state->entries.begin(), state->entries.end());
          }
        }
        return true;
      }
      case 'o':
      case 'i': {
        pop_to_mark();
        push(make_value(PickleValue::Kind::Opaque));
        return true;
      }
      default:
        fail("unsupported opcode 0x" + std::to_string(static_cast<int>(opcode)));
    }
  }

  const std::uint8_t* data_;
  std::size_t size_;
  std::string origin_;
  std::size_t position_ = 0;
  std::size_t memo_next_ = 0;
  std::vector<PickleRef> stack_;
  std::vector<PickleRef> memo_;
};

bool is_row_major_contiguous(const std::vector<long long>& sizes,
                             const std::vector<long long>& strides) {
  if (sizes.size() != strides.size()) return false;
  long long expected = 1;
  for (std::size_t index = sizes.size(); index > 0; --index) {
    if (sizes[index - 1] != 1 && strides[index - 1] != expected) return false;
    expected *= sizes[index - 1];
  }
  return true;
}

}  // namespace

TorchArchiveWeightSource::TorchArchiveWeightSource(const std::string& file_path)
    : origin_(file_path), file_(file_path) {
  const ZipArchive archive(file_);
  const std::string prefix = archive.detect_prefix();

  const ZipRecord* pickle_record = archive.find(prefix + "/data.pkl");
  if (pickle_record == nullptr) throw Error("missing data.pkl in '" + file_path + "'");

  PickleMachine machine(file_.at(pickle_record->data_offset, pickle_record->size),
                        static_cast<std::size_t>(pickle_record->size), file_path);
  const PickleRef root = machine.run();
  if (root->kind != PickleValue::Kind::Dict) {
    throw Error("'" + file_path + "' does not contain a state dict");
  }

  tensors_.reserve(root->entries.size());
  for (const auto& entry : root->entries) {
    const PickleRef& key = entry.first;
    const PickleRef& value = entry.second;
    if (key->kind != PickleValue::Kind::Text || value->kind != PickleValue::Kind::Tensor) continue;
    if (value->storage == nullptr || value->storage->kind != PickleValue::Kind::Storage) continue;

    if (!is_row_major_contiguous(value->sizes, value->strides)) {
      throw Error("tensor '" + key->text + "' in '" + file_path + "' is not contiguous");
    }

    const PickleValue& storage = *value->storage;
    const ZipRecord* data_record = archive.find(prefix + "/data/" + storage.storage_key);
    if (data_record == nullptr) {
      throw Error("missing storage record '" + storage.storage_key + "' in '" + file_path + "'");
    }

    TensorView tensor;
    tensor.name = key->text;
    tensor.dtype = storage.storage_dtype;
    tensor.shape.assign(value->sizes.begin(), value->sizes.end());

    const std::size_t item_size = dtype_size(tensor.dtype);
    tensor.byte_size = static_cast<std::size_t>(tensor.element_count()) * item_size;
    const std::size_t byte_offset =
        static_cast<std::size_t>(value->storage_offset) * item_size;
    tensor.data = file_.at(data_record->data_offset + byte_offset, tensor.byte_size);
    tensors_.push_back(std::move(tensor));
  }

  if (tensors_.empty()) throw Error("no tensors recovered from '" + file_path + "'");
  Log::debug("torch archive '%s': %zu tensors", file_path.c_str(), tensors_.size());
}

}  // namespace attnrank

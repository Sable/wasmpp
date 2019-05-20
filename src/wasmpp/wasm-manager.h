#ifndef WASM_WASM_MANAGER_H_
#define WASM_WASM_MANAGER_H_

#include <src/ir.h>
#include <third_party/wabt/src/stream.h>
#include <src/wasmpp/wasm-instructions.h>
#include <src/wasmpp/common.h>

namespace wasmpp {

class Memory {
public:
  Memory(uint64_t begin, uint64_t end);
  uint64_t Bytes() const { return end_ - begin_; }
  uint64_t Begin() const { return begin_; }
  uint64_t End() const { return end_; }
private:
  uint64_t begin_;
  uint64_t end_;
};

class MemoryManager {
private:
  std::vector<Memory*> memories_;
public:
  ~MemoryManager();
  // Allocate memory
  Memory* Allocate(uint64_t k);
  bool Free(Memory* m);
  uint64_t Pages();
};

class ContentManager {
private:
  wabt::ExprList* expr_list_ = nullptr;
  LabelManager* label_manager_ = nullptr;
public:
  // Extract the elements from input list
  // into structure list. Do not re-use the
  // input list as it will become empty after
  // calling this method
  void Insert(exprs_sptr e);
  ContentManager(LabelManager* label_manager, wabt::ExprList* expr_list);
  LabelManager* Label() const { return label_manager_; };
};
typedef ContentManager FuncBody;

struct DataEntry {
  union {
    uint8_t byte;
    uint32_t i32;
    uint64_t i64;
    float f32;
    double f64;
  } val;
  enum Kind {
    Byte,
    I32,
    I64,
    F32,
    F64
  } kind;
  uint32_t Size() const {
    if(kind == I32) return WASMPP_I32_SIZE;
    if(kind == I64) return WASMPP_I64_SIZE;
    if(kind == F32) return WASMPP_F32_SIZE;
    if(kind == F64) return WASMPP_F64_SIZE;
    assert(kind == Byte); return 1;
  }
  static DataEntry MakeI32(uint32_t val) { return {{.i32 = val}, I32}; }
  static DataEntry MakeI64(uint64_t val) { return {{.i64 = val}, I64}; }
  static DataEntry MakeF32(float val) { return {{.f32 = val}, F32}; }
  static DataEntry MakeF64(double val) { return {{.f64 = val}, F64}; }
  static DataEntry MakeByte(uint8_t val) { return {{.byte = val}, Byte}; }
};

class LabelManager {
private:
  int uid_ = 0;
public:
  std::string Next();
};

class ModuleManager {
private:
  wabt::Module module_;
  MemoryManager memory_manager_;
  LabelManager label_manager_;

  // Function copied from WastParser::CheckImportOrdering
  void CheckImportOrdering();
public:

  const wabt::Module& GetModule() const;
  MemoryManager& Memory() { return memory_manager_; }
  LabelManager& Label() { return label_manager_; }

  // Validation
  bool Validate();

  // Generate code
  std::string ToWat(bool folded, bool inline_import_export) const;
  wabt::OutputBuffer ToWasm() const;

  // Make sections entries
  wabt::Var MakeFunction(const char* name, wabt::FuncSignature sig, wabt::TypeVector locals,
                           std::function<void(FuncBody, std::vector<wabt::Var>, std::vector<wabt::Var>)> content);
  wabt::Var MakeFuncImport(std::string module, std::string function, wabt::FuncSignature sig);
  wabt::Var MakeMemory(uint64_t init_page, uint64_t max = 0, bool shared = false);
  void MakeData(wabt::Var var, uint32_t index, std::vector<DataEntry> entries);
};

} // namespace wasm

#endif

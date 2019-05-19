#ifndef WASM_WASM_MANAGER_H_
#define WASM_WASM_MANAGER_H_

#include <src/ir.h>
#include <third_party/wabt/src/stream.h>
#include <src/wasmpp/builtins/math-builtins.h>
#include <src/wasmpp/builtins/memory-builtins.h>
#include <src/wasmpp/builtins/system-builtins.h>
#include <src/wasmpp/wasm-instructions.h>

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

struct ModuleManagerOptions {
#define DEFINE_OPTIONS(var, name) \
    bool var = false;
#define ENABLE_OPTION(var, name) \
    var = true;
#define CREATE_MEMBERS(V) \
  V(DEFINE_OPTIONS)       \
  void EnableAll() {      \
     V(ENABLE_OPTION)     \
    }

  struct {
CREATE_MEMBERS(MATH_BUILTINS)
  } math;

  struct {
CREATE_MEMBERS(MEMORY_BUILTINS)
  } memory;

  struct {
CREATE_MEMBERS(SYSTEM_BUILTINS)
  } system;

#undef DEFINE_OPTIONS
#undef ENABLE_OPTION
#undef CREATE_MEMBERS
};

class ModuleManager;
struct BuiltinManager {
  BuiltinManager(ModuleManager* module_manager, ModuleManagerOptions* options);
  MathBuiltins math;
  MemoryBuiltins memory;
  SystemBuiltins system;
};

class ContentManager {
private:
  wabt::ExprList* expr_list_;
  bool parent_module_;
  union {
    ModuleManager* module_manager_;
    ContentManager* content_manager_;
  } parent;
  const BuiltinManager* builtins_;
  bool HasParent() const;
public:
  // Extract the elements from input list
  // into structure list. Do not re-use the
  // input list as it will become empty after
  // calling this method
  void Insert(exprs_sptr e);
  ContentManager(ModuleManager* mm, wabt::ExprList* expr_list);
  ContentManager(ContentManager* parent_ctn, wabt::ExprList* expr_list);
  std::string NextLabel() const;
  const BuiltinManager* Builtins() const { return builtins_; }
};

  typedef ContentManager FuncBody;

class ModuleManager {
private:
  wabt::Module module_;
  int uid_ = 0;
  MemoryManager memory_manager_;
  ModuleManagerOptions options_;
  BuiltinManager builtins_;

  // Function copied from WastParser::CheckImportOrdering
  void CheckImportOrdering();
public:

  ModuleManager(ModuleManagerOptions options = {}) : options_(options), builtins_(this, &options_) {}
  std::string NextLabel();
  const wabt::Module& GetModule() const;
  MemoryManager& Memory() { return memory_manager_; }
  const BuiltinManager& Builtins() const { return builtins_; }

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
};

} // namespace wasm

#endif

#ifndef WASM_WASM_MANAGER_H_
#define WASM_WASM_MANAGER_H_

#include <src/ir.h>
#include <third_party/wabt/src/stream.h>
#include <src/wasmpp/builtins/math-builtins.h>
#include <src/wasmpp/wasm-instructions.h>

namespace wasmpp {

struct Memory {
  uint64_t begin;
  uint64_t end;
  uint64_t Size() {
    assert(begin < end);
    return end - begin;
  }
};

class MemoryManager {
private:
  std::vector<Memory*> memories_;
public:
  ~MemoryManager();
  Memory* Allocate(uint64_t k);
  bool Free(Memory* m);
  uint64_t Pages();
};

struct ModuleManagerOptions {
  struct Math {
#define DEFINE_MATH_OPTIONS(var, name) \
  bool var = false;
  MATH_BUILTINS(DEFINE_MATH_OPTIONS)
#undef DEFINE_MATH_OPTIONS
  } math;
};

class ModuleManager;
struct BuiltinManager {
  BuiltinManager(ModuleManager* module_manager, ModuleManagerOptions* options) :
      math(module_manager, options) {}
  MathBuiltins math;
};

class ContentManager {
private:
  wabt::ExprList* expr_list_;
  bool parent_module_;
  union {
    ModuleManager* module_manager_;
    ContentManager* content_manager_;
  } parent;
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
};

  typedef ContentManager FuncBody;

class ModuleManager {
private:
  wabt::Module module_;
  int uid_ = 0;
  MemoryManager memory_manager_;
  ModuleManagerOptions options_;
public:
  BuiltinManager builtins;

  ModuleManager(ModuleManagerOptions options = {}) : options_(options), builtins(this, &options_) {}
  std::string NextLabel();
  const wabt::Module& GetModule() const;
  MemoryManager& Memory() { return memory_manager_; }

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

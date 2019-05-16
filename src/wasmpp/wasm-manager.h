#ifndef WASM_WASM_MANAGER_H_
#define WASM_WASM_MANAGER_H_

#include <src/ir.h>
#include <third_party/wabt/src/stream.h>

namespace wasmpp {

struct ContentManager {
  wabt::ExprList* exprList;
  // Extract the elements from input list
  // into structure list. Do not re-use the
  // input list as it will become empty after
  // calling this method
  void Insert(wabt::ExprList* e);
};

typedef ContentManager FuncBody;
typedef ContentManager BlockBody;

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

class ModuleManager {
private:
  wabt::Module module_;
  int uid_ = 0;

public:

  // Generate a unique id
  std::string GenerateUid();

  // Get module
  const wabt::Module& GetModule() const;

  // Generate wat code
  std::string ToWat(bool folded, bool inline_import_export) const;

  // Generate wasm code
  wabt::OutputBuffer ToWasm() const;

  // Make a new function in a module
  wabt::Var MakeFunction(const char* name, wabt::FuncSignature sig, wabt::TypeVector locals,
                           std::function<void(FuncBody, std::vector<wabt::Var>, std::vector<wabt::Var>)> content);

  // Make import
  wabt::Var MakeFuncImport(std::string module, std::string function, wabt::FuncSignature sig);

  // Make memory
  wabt::Var MakeMemory(uint64_t init_page, uint64_t max = 0, bool shared = false);
};

} // namespace wasm

#endif

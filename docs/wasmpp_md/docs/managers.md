# Managers
## Label Manager
Names generated for Wasm instructions are assigned by the label manager. The
latter simply creates a unique identifier on each request.

## Memory Manager
Memory manager in Wasm++ uses a first fit approach to allocate linear memory
space. The memory manager operates at Wasm-generation time. Thus, addresses can
directly be injected inside functions as `i32` values. Furthermore, the manager
can automatically compute the amount of linear memory pages, of size 64KB each,
required by a program.

### Example
#### C++
```
wasmpp::MemoryManager* mm = new wasmpp::FirstFit();
wasmpp::Memory* name = mm->Allocate(50);            // Allocate 50 bytes to store name string
wasmpp::Memory* age = mm->Allocate(1);              // Allocate 1 byte to store age
printf("Begin address: %d\tEnd address: %d\tSize: %d\n", 
        name->Begin(), name->End(), name->Bytes());
printf("Begin address: %d\tEnd address: %d\tSize: %d\n", 
        age->Begin(), age->End(), age->Bytes());
mm->Free(name);
mm->Free(age);
delete mm;
```

#### Output
```
Begin address: 0   End address: 50  Size: 50
Begin address: 50  End address: 51  Size: 1
```

## Module Manager
A module manager is the container of the [memory](#memory-manager) and
[label](#label-manager) managers, and hosts all the Wasm functions and other
sections living on the global scope of a module. This manager also uses
functionalities provided by the [WABT](https://github.com/WebAssembly/wabt)
library such as compiling the IR to WebAssembly bytecode or to Wat format, and
validating the correctness of the generated WebAssembly.

### Example
#### C++
```
using namespace std;
using namespace wabt;
using namespace wasmpp;

int main() {
  
  // Create a Wasm module
  ModuleManager module;
  
  // Create a function that loads the value of
  // an i32 address, that is passed as an argument,
  // from the linear memory
  TypeVector params = {Type::I32};
  TypeVector returns = {Type::I32};
  TypeVector locals = {};
  module.MakeFunction("load_example", {params, returns}, locals,
                      [&](FuncBody f, vector<Var> params, std::vector<Var> locals) {
    auto memory_load = MakeI32Load(MakeLocalGet(params[0]));
    f.Insert(memory_load);
  });
  
  // Create memory section
  uint32_t memory_pages = module.Memory().Pages();
  module.MakeMemory(memory_pages);

  // Validate and generate Wat
  assert(module.Validate());
  bool folded = false;
  bool inline_import_export = true;
  printf("%s\n", module.ToWat(folded, inline_import_export).c_str());
  return 0;
}
```
#### Output
```
(module
  (type (;0;) (func (param i32) (result i32)))
  (func $0 (export "load_example") (param $1 i32) (result i32)
    local.get 0
    i32.load)
  (memory $2 0))
```

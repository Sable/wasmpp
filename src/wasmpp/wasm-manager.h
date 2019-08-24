/*! @mainpage Wasm++
 *
 * This library allows creating <a href="https://github.com/WebAssembly/wabt">WABT</a>
 * IR objects with a simplified API.
 *
 * @file wasm-manager.h
 */

#ifndef WASM_WASM_MANAGER_H_
#define WASM_WASM_MANAGER_H_

#include <src/ir.h>
#include <third_party/wabt/src/stream.h>
#include <src/wasmpp/wasm-instructions.h>
#include <src/wasmpp/common.h>

namespace wasmpp {

/*!
 * @brief Block of bytes in the linear memory
 */
class Memory {
public:
  /*!
   * Define a memory block
   * @param begin
   * @param end
   */
  Memory(uint32_t begin, uint32_t end);
  /*!
   * Get bytes
   * @return Number of bytes
   */
  uint32_t Bytes() const { return end_ - begin_; }
  /*!
   * Get begin address
   * @return begin address
   */
  uint32_t Begin() const { return begin_; }
  /*!
   * Get end address
   * @return end address
   */
  uint32_t End() const { return end_; }
private:
  uint32_t begin_;
  uint32_t end_;
};

/*!
 * @brief Linear memory manager
 */
class MemoryManager {
protected:
  std::vector<Memory*> memories_;
public:
  ~MemoryManager();
  /*!
   * Allocate block in the linear memory
   * @param k Number of bytes
   * @return Pointer to the allocated block of memory
   */
  virtual Memory* Allocate(uint32_t k) = 0;
  /*!
   * Free block
   * @param m Pointer to the allocated block of memory
   * @return true if successful
   */
  virtual bool Free(const Memory* m);
  /*!
   * Get number of Wasm pages
   * @return Number of Wasm pages
   */
  virtual uint32_t Pages();
};

/*!
 * @brief First fit memory manager
 */
class FirstFit : public MemoryManager {
  Memory* Allocate(uint32_t k) override;
};

/*!
 * @brief Manage the content of a Wasm instruction block
 * e.g. <code>block</code>, <code>loop</code>, <code>if</code>, etc ...
 */
class ContentManager {
private:
  wabt::ExprList* expr_list_ = nullptr;
  LabelManager* label_manager_ = nullptr;
public:

  /*!
   * @brief Insert expression list
   * @warning Do not re-use the input list as it will
   * become empty after calling this method
   * @param e Expression list
   */
  void Insert(wabt::ExprList* e);

  /*!
   * @brief Create a new content manager
   * @param label_manager Label manager
   * @param expr_list Expression list
   */
  ContentManager(LabelManager* label_manager, wabt::ExprList* expr_list);

  /*!
   * @brief Get label manager
   * @return Label manager
   */
  LabelManager* Label() const { return label_manager_; };
};

/*!
 * @brief Function body block
 */
typedef ContentManager FuncBody;

/*!
 * @brief A data entry or unit in the linear memory
 */
struct DataEntry {
  union {
    uint8_t byte;
    uint32_t i32;
    uint64_t i64;
    float f32;
    double f64;
  } val;

  /*!
   * @brief Entry kind
   */
  enum Kind {
    Byte,
    I32,
    I64,
    F32,
    F64
  } kind;

  /*!
   * Entry size
   * @return Number of bytes
   */
  uint32_t Size() const {
    if(kind == I32) return WASMPP_I32_SIZE;
    if(kind == I64) return WASMPP_I64_SIZE;
    if(kind == F32) return WASMPP_F32_SIZE;
    if(kind == F64) return WASMPP_F64_SIZE;
    assert(kind == Byte); return 1;
  }

  /*!
   * Make an i32 data entry
   * @param val Value
   * @return Data entry
   */
  static DataEntry MakeI32(uint32_t val);
  /*!
   * Make an i64 data entry
   * @param val Value
   * @return Data entry
   */
  static DataEntry MakeI64(uint64_t val);
  /*!
   * Make an f32 data entry
   * @param val Value
   * @return Data entry
   */
  static DataEntry MakeF32(float val);
  /*!
   * Make an f64 data entry
   * @param val Value
   * @return Data entry
   */
  static DataEntry MakeF64(double val);
  /*!
   * Make a byte data entry
   * @param val Value
   * @return Data entry
   */
  static DataEntry MakeByte(uint8_t val);
};

/*!
 * @brief Manager for Wasm instruction labels e.g. <code>loop</code>,
 * locals, params, etc ...
 */
class LabelManager {
private:
  int uid_ = 0;
public:
  /*!
   * Next unique string
   * @return unique string
   */
  std::string Next();
};

/*!
 * @brief WebAssembly module manager
 */
class ModuleManager {
private:
  wabt::Module module_;
  FirstFit memory_manager_;
  LabelManager label_manager_;

  // Function copied from WastParser::CheckImportOrdering
  void CheckImportOrdering();
  void ResolveImplicitlyDefinedFunctionType(const wabt::FuncDeclaration& decl);

  // Helpers
  void MakeExport(std::string name, wabt::Var var, wabt::ExternalKind kind);
public:
  /*!
   * Get WABT module object
   * @return Module
   */
  const wabt::Module& GetModule() const;
  /*!
   * Get memory manager
   * @return Memory manager
   */
  MemoryManager& Memory() { return memory_manager_; }
  /*!
   * Get label manager
   * @return Label manager
   */
  LabelManager& Label() { return label_manager_; }

  /*!
   * Validate created Wasm module
   * @note Uses WABT validation
   * @return true if Wasm module is valid
   */
  bool Validate();

  /*!
   * Module to Wat format
   * @note Uses WABT Wat generator
   * @param folded Tree structure
   * @param inline_import_export Inline import/export
   * @return Wat code as string
   */
  std::string ToWat(bool folded, bool inline_import_export) const;

  /*!
   * Module to Wasm format
   * @note Uses WABT Wasm generator
   * @return Wasm binary in a buffer
   */
  wabt::OutputBuffer ToWasm() const;

  /*!
   * Make a Wasm function
   * @param name Export name
   * @param sig Function signature
   * @param locals Function locals
   * @param content Function body
   * @return Function variable used to call it e.g. <code>call $var</code>
   */
  wabt::Var MakeFunction(const char* name, wabt::FuncSignature sig, wabt::TypeVector locals,
                           std::function<void(FuncBody, std::vector<wabt::Var>, std::vector<wabt::Var>)> content);

  /*!
   * Make a Wasm import function <br/>
   * e.g. <code>("module" "function" (func $var (param i32) (result i32)))</code>
   * @param module Module name
   * @param function Function name
   * @param sig Function signature
   * @return Function variable used to call it e.g. <code>call $var</code>
   */
  wabt::Var MakeFuncImport(std::string module, std::string function, wabt::FuncSignature sig);

  /*!
   * Make a Wasm linear memory in the module
   * @param init_page Number of pages
   * @param max Maximum number of pages
   * @param shared Mark shared linear memory
   * @return Create memory reference variable
   */
  wabt::Var MakeMemory(uint32_t init_page, uint32_t max = 0, bool shared = false);

  /*!
   * Make a data section. <br/>
   * Insert data in little-endian format
   * @param var Linear memory reference variable
   * @param index Index where data is inserted
   * @param entries List of data entries
   */
  void MakeData(wabt::Var var, uint32_t index, std::vector<DataEntry> entries);

  /*!
   * Export linear memory
   * @param name Export name
   * @param var Linear memory reference variable
   */
  void MakeMemoryExport(std::string name, wabt::Var var);
#ifdef WABT_EXPERIMENTAL
  wabt::Var MakeNativeFunction(std::string function, wabt::FuncSignature sig);
#endif // WABT_EXPERIMENTAL
};

} // namespace wasm

#endif

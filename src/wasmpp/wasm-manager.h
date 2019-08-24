/*! \mainpage Wasm++
 *
 * This library allows creating <a href="https://github.com/WebAssembly/wabt">WABT</a>
 * IR objects with a simplified API.
 */

#ifndef WASM_WASM_MANAGER_H_
#define WASM_WASM_MANAGER_H_

#include <src/ir.h>
#include <third_party/wabt/src/stream.h>
#include <src/wasmpp/wasm-instructions.h>
#include <src/wasmpp/common.h>

namespace wasmpp {

class Memory {
public:
  Memory(uint32_t begin, uint32_t end);
  uint32_t Bytes() const { return end_ - begin_; }
  uint32_t Begin() const { return begin_; }
  uint32_t End() const { return end_; }
private:
  uint32_t begin_;
  uint32_t end_;
};

class MemoryManager {
protected:
  std::vector<Memory*> memories_;
public:
  ~MemoryManager();
  // Allocate memory
  virtual Memory* Allocate(uint32_t k) = 0;
  virtual bool Free(const Memory* m);
  virtual uint32_t Pages();
};

class FirstFit : public MemoryManager {
  Memory* Allocate(uint32_t k) override;
};

class ContentManager {
private:
  wabt::ExprList* expr_list_ = nullptr;
  LabelManager* label_manager_ = nullptr;
public:

  /*!
   * \brief Insert expression list
   * \warning Do not re-use the input list as it will
   * become empty after calling this method
   * \param e Expression list
   */
  void Insert(wabt::ExprList* e);

  /*!
   * \brief Manage the content of a Wasm instruction block
   * e.g. <code>block</code>, <code>loop</code>, <code>if</code>, etc ...
   * \param label_manager Label manager
   * \param expr_list Expression list
   */
  ContentManager(LabelManager* label_manager, wabt::ExprList* expr_list);

  /*!
   * \brief Get label manager
   * \return Label manager
   */
  LabelManager* Label() const { return label_manager_; };
};

/*!
 * \brief Function body block
 */
typedef ContentManager FuncBody;

/*!
 * \brief A data entry or unit in the linear memory
 */
struct DataEntry {
  /*!
   * Entry value
   */
  union {
    uint8_t byte;
    uint32_t i32;
    uint64_t i64;
    float f32;
    double f64;
  } val;

  /*!
   * Entry kind
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
   * \return Number of bytes
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

class LabelManager {
private:
  int uid_ = 0;
public:
  std::string Next();
};

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
  wabt::Var MakeMemory(uint32_t init_page, uint32_t max = 0, bool shared = false);
  void MakeData(wabt::Var var, uint32_t index, std::vector<DataEntry> entries);
  void MakeMemoryExport(std::string name, wabt::Var var);
#ifdef WABT_EXPERIMENTAL
  wabt::Var MakeNativeFunction(std::string function, wabt::FuncSignature sig);
#endif // WABT_EXPERIMENTAL
};

} // namespace wasm

#endif

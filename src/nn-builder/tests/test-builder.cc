#include <src/nn-builder/tests/matrix_test.h>
#include <iostream>
#include <getopt.h>
#include <fstream>

bool FLAG_to_wasm = false;
bool FLAG_to_wat = false;
std::string output_file;

void PrintUsage() {
  std::cout
      << "Test - Run test cases" << std::endl
      << "Usage: nn-test [OPTION]... [FILE]..." << std::endl
      << "    -w, --to-wasm    Print wasm" << std::endl
      << "    -W, --to-wat     Print wat" << std::endl
      << "    -o, --output     Output file" << std::endl
      << "    -h, --help       Display this help message" << std::endl;
}

void InitParams(int argc, char *argv[]) {

  struct option longOptions[] = {

      {"to-wasm", no_argument, 0, 'w'},
      {"to-wat", no_argument, 0, 'W'},
      {"output", required_argument, 0, 'o'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}
  };

  int optionIndex = 0;
  int c;
  while ((c = getopt_long(argc, argv, "hwWo:", longOptions, &optionIndex)) != -1) {
    switch (c) {
      case 'w':
        FLAG_to_wasm = true;
        break;
      case 'W':
        FLAG_to_wat = true;
        break;
      case 'o':
        output_file = optarg;
        break;
      case 'h':
      default:
        break;
    }
  }
}


int main(int argc, char *argv[]) {
  InitParams(argc, argv);

  if(!FLAG_to_wat && !FLAG_to_wasm && output_file.empty()) {
    PrintUsage();
    exit(0);
  }

  wasmpp::ModuleManager module_manager;

  // Import js functions
  nn::test::TestBuiltins test_builtins;
  test_builtins.assert_matrix_eq = module_manager.MakeFuncImport("Test", "assert_matrix_eq",
      {{wabt::Type ::I32, wabt::Type::I32, wabt::Type::I32, wabt::Type::I32},{}});

  // Allocate enough memory
  auto memory = module_manager.MakeMemory(500);
  module_manager.MakeMemoryExport("memory", memory);

  // Create matrix tests
  nn::test::MatrixSnippetTest matrix_snippet_test(&module_manager, &test_builtins);
  matrix_snippet_test.MatrixAddition_test_1();
  matrix_snippet_test.MatrixSubtraction_test_1();
  matrix_snippet_test.MatrixMultiplication_test_1();
  matrix_snippet_test.MatrixScalar_test_1();
  matrix_snippet_test.MatrixDot_test_1();
  matrix_snippet_test.MatrixDotLT_test_1();
  matrix_snippet_test.MatrixDotRT_test_1();
  matrix_snippet_test.MatrixVectorAddition_test_1();

  // Create matrix simd tests
  nn::test::MatrixSnippetSimdTest matrix_snippet_simd_test(&module_manager, &test_builtins);
  matrix_snippet_simd_test.MatrixAdditionSimd_test_1();
  matrix_snippet_simd_test.MatrixSubtractionSimd_test_1();
  matrix_snippet_simd_test.MatrixMultiplicationSimd_test_1();
  matrix_snippet_simd_test.MatrixScalarSimd_test_1();
  matrix_snippet_simd_test.MatrixVectorAdditionSimd_test_1();

  assert(module_manager.Validate());
  if(!output_file.empty()) {
    std::ofstream file;
    file.open(output_file);
    if(FLAG_to_wasm) {
      auto data = module_manager.ToWasm().data;
      file << std::string(data.begin(), data.end());
    } else if(FLAG_to_wat) {
      file << module_manager.ToWat(true, true);
    }
    file.close();
  }
  return 0;
}

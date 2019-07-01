const fs = require('fs');
const {CompiledModel} = require('../js/compiled_model');

process.on('unhandledRejection', error => {
  console.error(">> Make sure SIMD is enabled (e.g. nodejs --experimental-wasm-simd)");
  console.error(">> Error message:", error);
});

if(process.argv.length > 2) {
  const buf = fs.readFileSync(process.argv[2]);
  const lib = WebAssembly.instantiate(new Uint8Array(buf), CompiledModel.Imports());
  lib.then( wasm => {
    const compiled_model = new CompiledModel(wasm);
    compiled_model.UnitTest();
  })
} else {
    console.log("Missing argument: file.wasm");
}

const fs = require('fs');
const {CompiledModel} = require('./compiled_model');

process.on('unhandledRejection', error => {
  console.log("Make sure SIMD is enabled (e.g. nodejs --experimental-wasm-simd)");
});

if(process.argv.length > 2) {
  const buf = fs.readFileSync(process.argv[2]);
  const compiled_model = new CompiledModel();
  const lib = WebAssembly.instantiate(new Uint8Array(buf), compiled_model.Imports());
  lib.then( wasm => {
    compiled_model.SetWasm(wasm);
    compiled_model.UnitTest();
  })
} else {
    console.log("Missing argument: file.wasm");
}

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
    console.log("Training ...");
    compiled_model.Train();

    console.log("Testing ...");
    compiled_model.Test();

    // console.log("Predicting ...");
    // compiled_model.Predict(...);
    //
    // For example, in the logic example
    // if the prediction batch size is 2
    // then the function call will be:
    // compiled_model.Predict([[0,1],[1,0]]);
    // compiled_model.Predict([[1,1],[0,0]]);
    // compiled_model.Predict([[0,1],[1,1]]);
    // ...
  })
} else {
    console.log("Missing argument: file.wasm");
}

const fs = require('fs');
const {CompiledModel} = require('./compiled_model');

// This is a workaround to omit trap-handlers
// in the generated machine code
// Comment: https://github.com/nodejs/node/issues/14927#issuecomment-482919665
require("../../../third_party/gyp/trap-handler/build/Release/th");

// Warning to enable SIMD in Node
process.on('unhandledRejection', error => {
  console.error(">> Make sure SIMD is enabled (e.g. nodejs --experimental-wasm-simd)");
  console.error(">> Error message:", error);
});

if(process.argv.length > 2) {
  const buf = fs.readFileSync(process.argv[2]);
  const compiled_model = new CompiledModel();
  const lib = WebAssembly.instantiate(new Uint8Array(buf), compiled_model.Imports());
  lib.then( wasm => {
    compiled_model.SetWasm(wasm);
    console.log("Training ...");
    compiled_model.Train({
      log_accuracy: true,
      epochs: 1
    });
    //compiled_model.LogTrainForward();
    //compiled_model.LogTrainBackward();
    //compiled_model.PrintTrainingConfusionMatrix();

    // console.log("Testing ...");
    // compiled_model.Test();

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

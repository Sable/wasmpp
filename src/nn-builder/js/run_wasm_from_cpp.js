const fs = require('fs');
const {CompiledModel} = require('./compiled_model');
const mnist = require('mnist');

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

    // Load mnist data
    let mnist_data = {training:[]};
    for(let i=0; i < 224; i++) {
      for(let j=0; j < 10; j++) {
        let l = [0,0,0,0,0,0,0,0,0,0];
        l[j] = 1;
        mnist_data.training.push(
          {input: mnist[j].get(i), output: l}
        );
      }
    }
    let train_obj = mnist_data.training;
    let train_data = [];
    let train_labels = [];
    for(let i=0; i < train_obj.length; i++) {
      train_data.push(train_obj[i].input);
      train_labels.push(train_obj[i].output);
    }
    
    console.log("Training ...");
    compiled_model.Train(train_data, train_labels, {
      log_accuracy: true,
      log_error: true,
      log_time: true,
      epochs: 10
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

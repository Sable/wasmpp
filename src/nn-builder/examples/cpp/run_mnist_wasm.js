const fs = require('fs');
const {CompiledModel} = require('../../js/compiled_model');
const mnist = require('mnist');

// This is a workaround to omit trap-handlers
// in the generated machine code
// Comment: https://github.com/nodejs/node/issues/14927#issuecomment-482919665
require("../../../../third_party/gyp/trap-handler/build/Release/th");

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
    let mnist_data = mnist.set(2240*2,2240);
    let train_data    = [];
    let train_labels  = [];
    mnist_data.training.forEach((x) => {train_data.push(x.input); train_labels.push(x.output)});
    let test_data    = [];
    let test_labels  = [];
    mnist_data.test.forEach((x) => {test_data.push(x.input); test_labels.push(x.output)});
    
    // Encode data
    let training = compiled_model.EncodeTrainingData(train_data, train_labels);
    let testing = compiled_model.EncodeTestingData(test_data, test_labels);
    let prediction = compiled_model.EncodePredictionData(train_data);

    console.log("Training ...");
    compiled_model.Train(training, {
      log_accuracy: true,
      log_error: true,
      log_time: true,
      log_forward: true,
      log_backward: true,
      log_conf_mat: true,
      epochs: 5,
      learning_rate: 0.02
    });

    console.log("Testing ...");
    compiled_model.Test(testing, {
      log_time: true,
      log_accuracy: true,
      log_conf_mat: true,
      log_error: true
    });

    console.log("Predicting ...");
    compiled_model.Predict(prediction, {
      log_time: true,
      log_result: true,
      result_mode: "hardmax"
    });
  })
} else {
    console.log("Missing argument: mnist.wasm");
}

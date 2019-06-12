if(process.argv.length <= 2) {
  console.log("Missing argument: /PATH/TO/nnb_js.js");
  process.exit(1);
}

process.on('unhandledRejection', error => {
  console.error(">> Make sure SIMD is enabled (e.g. nodejs --experimental-wasm-simd)");
  console.error(">> Error message:", error);
});

const nnb  = require(process.argv[2]);
const mnist = require('mnist');
const {CompiledModel} = require("../../../js/compiled_model");

function LoadMnist() {
  let mnist_data = mnist.set(6000, 1000);
  // Load training
  let train_data = [];
  let train_labels = [];
  mnist_data.training.forEach((entry) => {
    train_data.push(entry.input);
    train_labels.push(entry.output);
  });
  // Load testing
  let test_data = [];
  let test_labels = [];
  mnist_data.test.forEach((entry) => {
    test_data.push(entry.input);
    test_labels.push(entry.output);
  });
  return {
    train_data: train_data,
    train_labels: train_labels,
    test_data: test_data,
    test_labels: test_labels
  }
}

nnb.onRuntimeInitialized = function() {
  // Create model options
  let options = new nnb.ModelOptions();
  options.log_training_error = true;
  options.log_training_accuracy = true;
  options.log_training_time = true;
  options.log_testing_error = true;
  options.log_testing_accuracy = true;
  options.log_testing_confusion_matrix = true;
  options.log_testing_time = true;
  options.log_prediction_results = true;
  options.log_prediction_time = true;
  options.use_simd = true;

  // Create model
  let model = new nnb.Model(options);

  // Create layers
  let l0 = new nnb.DenseInputLayerDescriptor(784);
  l0.SetKeepProb(1.0);
  model.AddDenseInputLayer(l0);

  let l1 = new nnb.DenseHiddenLayerDescriptor(100, "sigmoid");
  l1.SetKeepProb(1.0);
  l1.SetWeightType("xavier_uniform");
  model.AddDenseHiddenLayer(l1);

  let l2 = new nnb.DenseOutputLayerDescriptor(10, "sigmoid");
  l2.SetWeightType("lecun_uniform");
  model.AddDenseOutputLayer(l2);

  // Start compiling
  let data = LoadMnist();
  let epoch = 10;
  let training_batch = 4;
  let testing_batch = 1;
  let prediction_batch = 1;
  let loss = "mean-squared-error";
  let learning_rate = 0.02;
  model.SetLayers();
  model.CompileLayers(training_batch, testing_batch, prediction_batch, loss);
  model.CompileTrainingFunction(epoch, learning_rate, nnb.ToMatrix(data.train_data), nnb.ToMatrix(data.train_labels));
  model.CompileTestingFunction(nnb.ToMatrix(data.test_data), nnb.ToMatrix(data.test_labels));
  model.CompilePredictionFunctions();
  model.CompileWeightsFunctions();
  model.CompileInitialization();
  if(model.Validate()) {
    let buffer = nnb.ToUint8Array(model.ToWasm());
    let compiled_model = new CompiledModel();
    const lib = WebAssembly.instantiate(buffer, compiled_model.Imports());
    lib.then(wasm => {
      compiled_model.SetWasm(wasm);
      console.log("Training ...");
      compiled_model.Train();
      console.log("Testing ...");
      compiled_model.Test();
    });
  }
}

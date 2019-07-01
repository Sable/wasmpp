// This class stores information about model weights
// this is useful for extracting trained model weights
// or importing pre-trained model weights
class LayerWeights {
  _layer;
  _weights;
  _bias;
  constructor(layer) {
    this._layer = layer;
  }
  _Float32ArrayFromBuffer(buffer, offset, byte_size) {
    return new Float32Array(buffer, offset, byte_size / Float32Array.BYTES_PER_ELEMENT);
  }
  _Float32ArrayFromArray(array) {
    return new Float32Array(array);
  }
  ImportWeightsFromBuffer(buffer, offset, byte_size) {
    this._weights = this._Float32ArrayFromBuffer(buffer, offset, byte_size);
  }
  ImportWeightsFromArray(array) {
    this._weights = this._Float32ArrayFromArray(array);
  }
  ImportBiasFromBuffer(buffer, offset, byte_size) {
    this._bias = this._Float32ArrayFromBuffer(buffer, offset, byte_size);
  }
  ImportBiasFromArray(array) {
    this._bias = this._Float32ArrayFromArray(array);
  }
  CopyWeights(layer_weights) {
    if(this._layer === layer_weights._layer) {
      if(this._weights.length === layer_weights._weights.length && this._bias.length === layer_weights._bias.length) {
        this._weights.set(layer_weights._weights);
        this._bias.set(layer_weights._bias);
      } else {
        console.error("Failed to copy weights/bias: arrays size are different (Weights: %d and %d), (Bias: %d and %d)",
          this._weights.length, layer_weights._weights.length, this._bias.length, layer_weights._bias.length);
        return false;
      }
    } else {
      console.error("Failed to copy weights: layer id are different (%d != %d)", this._layer, layer_weights._layer);
      return false;
    }
    return true;
  }
  ToJson() {
    return {
      layer: this._layer,
      weights: Array.from(this._weights),
      bias: Array.from(this._bias)
    }
  }
}

// Class for logging model details
class ModelLogger {
  constructor(){}
  // Forward timing
  log_forward_Time() {
    console.log("\n>> Forward algorithm steps time:");
  }
  log_forward_A_1(time) {
    console.log("A) Z[l] = W[l] . A[l-1] + b[l]");
    console.log("    1) Z[l] = W[l] . A[l-1]:", time);
  }
  log_forward_A_2(time) {
    console.log("    2) Z[l] = Z[l] + b[l]:", time);
  }
  log_forward_B(time) {
    console.log("B) A[l] = g[l](Z[l]):", time);
  }

  // Backward timing
  log_backward_Time() {
    console.log("\n>> Backward algorithm steps time:");
  }
  log_backward_A(time) {
    console.log("A) dA[L] = dJ(T, A[L]):", time);
  }
  log_backward_B(time) {
    console.log("B) [Softmax] dZ[L] = dJ(T, A[L]):", time);
  }
  log_backward_C_1(time) {
    console.log("C) dZ[l] = dA[l] * g'[l](Z[l])");
    console.log("    1) dZ[l] = g'[l](Z[l]):", time);
  }
  log_backward_C_2(time) {
    console.log("    2) dZ[l] = dA[l] * dZ[l]:", time);
  }
  log_backward_D_1(time) {
    console.log("D) dW[l] = (1/m) (dZ[l] . A[l-1]^T + l2_decay W[l]) + l1_decay sign(W[l]))");
    console.log("    1) dW[l] = dZ[l] . A[l-1]^T:", time);
  }
  log_backward_D_2_1(time) {
    console.log("    2) 1) dW[l] = dW[l] + l1_decay sign(W[l]) + l2_decay W[l]:", time);
  }
  log_backward_D_2_2_1(time) {
    console.log("       2) dW[l] = dW[l] + l1_decay sign(W[l]) + l2_decay W[l]");
    console.log("          1) dW[l] = dW[l] + l1_decay W[l]:", time);
  }
  log_backward_D_2_2_2(time) {
    console.log("          2) dW[l] = dW[l] + l2_decay W[l]:", time);
  }
  log_backward_D_3(time) {
    console.log("    3) dW[l] = (1/m) dW[l]:", time);
  }
  log_backward_E_1(time) {
    console.log("E) db[l] = (1/m) dZ[l]");
    console.log("    1) db[l] = SUM(dZ[l], row wise):", time);
  }
  log_backward_E_2(time) {
    console.log("    2) db[l] = (1/m) db[l]:", time);
  }
  log_backward_F(time) {
    console.log("F) dA[l-1] = W[l]^T . dZ[l]:", time);
  }
  log_backward_G(time) {
    console.log("G) W[l] = W[l] - alpha * dW[l]:", time);
  }
  log_backward_H(time) {
    console.log("H) b[l] = b[l] - alpha * db[l]:", time);
  }
}

// This class is the type of the training, 
// testing and predicting parameter
class EncodedData { 
  X; 
  x_size;
  x_count;
  Y;
  y_size;
  y_count;
  good = false;
}

// This class is a wrapper for the generated Wasm model
// It also contains the import functions from JS to Wasm
// Most functions simply calls the Wasm exported functions
// but some might process the arguments in order to pass
// them correctly to Wasm functions
class CompiledModel {
  _wasm = null;
  _imports = {};
  _logger = new ModelLogger();

  constructor(wasm) {
    this._wasm = wasm;
  }

  // Get exports from Wasm to JS
  Exports() {
    if (this._wasm == null) {
      console.error("Wasm instance was not set");
      return null;
    }
    return this._wasm.instance.exports;
  }

  _CallExport(key, ...args) {
    if(key in this.Exports()) {
      return this.Exports()[key].apply(null, args);
    }
    this._WarnNotFound("Function '"+key+"' not found")
    return null;
  }

  Memory() {
    return this.Exports().memory;
  }

  _EncodeArray(data, entry_size, batch_size) {
    // We assume that data and batch size
    // have already been checked
    //
    // This encoder partially trainsposes, flatten 
    // and stores the data in a Float32Array
    //  Batch = 2
    //  [
    //    [0,  1, 2],
    //    [3,  4, 5],
    //    [6,  7, 8],
    //    [9, 10, 11]
    //  ];
    // Becomes:
    // Float32Array [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11]
    let index = 0;
    let encoded = new Float32Array(data.length * entry_size);
    let total_batches = data.length / batch_size;
    for(let batch=0; batch < total_batches; batch++) {
      let data_begin = batch * batch_size;
      let data_end = data_begin + batch_size;
      // Check if the input provided is correct
      for(let r = data_begin; r < data_end; r++) {
        if(data[r].length != entry_size) {
          console.log("Input shape is incorrect. Expecting", entry_size, "but received", data[r].length);
          return false;
        }
      }
      for (let c = 0; c < entry_size; c++) {
        for (let r = data_begin; r < data_end; r++) {
          encoded[index++] = data[r][c];
        }
      }
    }
    return encoded;
  }

  _EncodeInput(data, labels, batch_size) {
    // Load model info
    let input_size = this._LayerSize(0);
    let output_size = this._LayerSize(this._TotalLayers() - 1);
    // Check if input is valid
    if(data.length === 0) {
      console.log("Data cannot be empty");
      return false;
    }
    if(labels !== null && data.length !== labels.length) {
      console.log("Data and labels should be equal");
      return false;
    }
    if(data.length % batch_size != 0) {
      console.error("Data size",data.length,"should be divisible by the number of batch",batch_size);
      return false;
    }
    // Encode data
    let result = new EncodedData();
    result.X = this._EncodeArray(data, input_size, batch_size);
    result.x_size = input_size;
    result.x_count = data.length;
    // Encode labels
    if(labels !== null) {
      result.Y = this._EncodeArray(labels, output_size, batch_size);
      result.y_size = output_size;
      result.y_count = labels.length;
    }
    // Check if encoding was successful
    if(result.X !== false && (labels === null || result.Y !== false)) {
      result.good = true;
    }
    return result;
  }

  EncodeTrainingData(data, labels) {
    return this._EncodeInput(data, labels, this._TrainingBatchSize());
  }

  EncodeTestingData(data, labels) {
    return this._EncodeInput(data, labels, this._TestingBatchSize());
  }

  EncodePredictionData(data) {
    return this._EncodeInput(data, null, this._PredictionBatchSize());
  }

  _LayerSize(id) {
    return this.Exports()["layer_" + id + "_size"]();
  }

  // Run train Wasm function
  Train(input, config) {
    if(!input.good) {
      console.log("Trainting input seems bad, skipping ...");
      return false;
    }
    // Load model batch information
    let batch_size = this._TrainingBatchSize();
    let batches_in_memory = this._TrainingBatchesInMemory();
    let number_of_batches = input.x_count / batch_size;
    
    // Configuration        Value                     Default
    config                  = config                  || {};
    config.log_epoch_num    = config.log_epoch_num    || true;
    config.log_accuracy     = config.log_accuracy     || false;
    config.log_error        = config.log_error        || false;
    config.log_time         = config.log_time         || false;
    config.log_forward      = config.log_forward      || false;
    config.log_backward     = config.log_backward     || false;
    config.log_conf_mat     = config.log_conf_mat     || false;
    config.epochs           = config.epochs           || 0;
    config.learning_rate    = config.learning_rate    || 0.01;

    // Update learning rate
    this._SetLearningRate(config.learning_rate);

    // Training variables
    let total_time = 0.0;
    let data_offset = this._TrainingDataOffset();
    let labels_offset = this._TrainingLabelsOffset();

    // Train for each epoch
    for(let e=0; e < config.epochs; e++) {
      // Epoch variables
      let total_hits = 0;
      let average_cost = 0.0;
      let epoch_time = new Date().getTime();
      for(let i=0; i < number_of_batches; i += batches_in_memory) {
        // Load new batches in memory and train
        let batches_inserted = this._CopyBatchesToMemory(input, data_offset, labels_offset, i,
                                                         batch_size, batches_in_memory);
        // Start training
        this.Exports().train_batches_in_memory(batches_inserted);

        // Update training details
        if(config.log_accuracy) {
          total_hits += this._TrainingBatchesAccuracy();
        }
        if(config.log_error) {
          average_cost += this._TrainingBatchesError();
        }
      }
      // Update time
      epoch_time = new Date().getTime() - epoch_time;
      total_time += epoch_time;
      // Log after end of epoch
      if(config.log_epoch_num) {
        console.log("Epoch",e+1);
      }
      if(config.log_accuracy) {
        console.log(">> Accuracy:  ", total_hits / input.x_count);
      }
      if(config.log_error) {
        console.log(">> Error:     ", average_cost / number_of_batches);
      }
      if(config.log_time) {
        console.log(">> Epoch time:", epoch_time, "ms");
        console.log(">> Total time:", total_time, "ms");
      }
    }
    if(config.log_forward) {
      this._LogTrainForward();
    }
    if(config.log_backward) {
      this._LogTrainBackward();
    }
    if(config.log_conf_mat) {
      this._LogTrainingConfusionMatrix();
    }
  }
  
  // Run test Wasm function
  Test(input, config) {
    if(!input.good) {
      console.log("Testing input seems bad, skipping ...");
      return false;
    }
    // Load model batch information
    let batch_size = this._TestingBatchSize();
    let batches_in_memory = this._TestingBatchesInMemory();
    let number_of_batches = input.x_count / batch_size;
    
    // Configuration        Value                     Default
    config                  = config                  || {};
    config.log_accuracy     = config.log_accuracy     || false;
    config.log_error        = config.log_error        || false;
    config.log_time         = config.log_time         || false;
    config.log_conf_mat     = config.log_conf_mat     || false;

    // Testing results
    let total_hits = 0;
    let average_cost = 0.0;
    let data_offset = this._TestingDataOffset();
    let labels_offset = this._TestingLabelsOffset();
    let test_time = new Date().getTime();
    for(let i=0; i < number_of_batches; i += batches_in_memory) {
      // Load new batches in memory and test
      let batches_inserted = this._CopyBatchesToMemory(input, data_offset, labels_offset, i,
                                                       batch_size, batches_in_memory);
      // Start testing
      this.Exports().test_batches_in_memory(batches_inserted);

      // Update testing details
      if(config.log_accuracy) {
        total_hits += this._TestingBatchesAccuracy();
      }
      if(config.log_error) {
        average_cost += this._TestingBatchesError();
      }
    }
    // Update time
    test_time = new Date().getTime() - test_time;
    // Log testing results
    if(config.log_accuracy) {
      console.log(">> Test Accuracy:  ", total_hits / input.x_count);
    }
    if(config.log_error) {
      console.log(">> Test Error:     ", average_cost / number_of_batches);
    }
    if(config.log_time) {
      console.log(">> Test Time: ", test_time, "ms");
    }
    if(config.log_conf_mat) {
      this._LogTestingConfusionMatrix();
    }
  }

  _GetLearningRate() {
    return this.Exports().get_learning_rate();
  }

  _SetLearningRate(val) {
    this.Exports().set_learning_rate(val);
  }

  _CopyBatchesToMemory(input, x_offset, y_offset, batch_index, batch_size, batch_in_memory) {
    // Compute data indicies
    let x_batch_float_size  = batch_size * input.x_size;
    let x_begin             = batch_index * x_batch_float_size;
    let x_length            = batch_in_memory * x_batch_float_size;
    // Detech if batch in memory value is more than there is actually
    if(x_begin + x_length > input.X.length) {
      let remaining_entries = (input.X.length - x_begin) / input.x_size;
      // Update number of batch in memory
      batch_in_memory = remaining_entries / batch_size;
      // Recompute length
      x_length = batch_in_memory * x_batch_float_size;
    }

    // Copy data
    let data_array = new Float32Array(input.X.buffer, x_begin* Float32Array.BYTES_PER_ELEMENT, x_length);
    let data_memory = new Float32Array(this.Memory().buffer, x_offset, x_length);
    data_memory.set(data_array);

    if(y_offset !== null) {
      // Compute labels indices
      let y_batch_float_size  = batch_size * input.y_size;
      let y_begin             = batch_index * y_batch_float_size;
      let y_length            = batch_in_memory * y_batch_float_size;

      // Copy labels
      let labels_array = new Float32Array(input.Y.buffer, y_begin* Float32Array.BYTES_PER_ELEMENT, y_length);
      let labels_memory = new Float32Array(this.Memory().buffer, y_offset, y_length);
      labels_memory.set(labels_array);
    }

    // Return number of batches inserted in the memory
    return batch_in_memory;
  }

  _TrainingDataOffset() {
    return this.Exports().training_data_offset();
  }

  _TestingDataOffset() {
    return this.Exports().testing_data_offset();
  }

  _TrainingLabelsOffset() {
    return this.Exports().training_labels_offset();
  }

  _TestingLabelsOffset() {
    return this.Exports().testing_labels_offset();
  }

  _TrainingBatchesAccuracy() {
    let key = "training_batches_hits";
    if(key in this.Exports()) {
      return this.Exports()[key]();
    } else {
      this._WarnNotFound("Training batches accuracy function not found");
    }
    return 0;
  }

  _TestingBatchesAccuracy() {
    let key = "testing_batches_hits";
    if(key in this.Exports()) {
      return this.Exports()[key]();
    } else {
      this._WarnNotFound("Testing batches accuracy function not found");
    }
    return 0;
  }

  _TrainingBatchesError() {
    let key = "training_batches_error";
    if(key in this.Exports()) {
      return this.Exports()[key]();
    } else {
      this._WarnNotFound("Training batches error function not found");
    }
    return 0;
  }

  _TestingBatchesError() {
    let key = "testing_batches_error";
    if(key in this.Exports()) {
      return this.Exports()[key]();
    } else {
      this._WarnNotFound("Testing batches error function not found");
    }
    return 0;
  }

  _TrainingBatchSize() {
    return this.Exports().training_batch_size();
  }

  _TestingBatchSize() {
    return this.Exports().testing_batch_size();
  }

  _TrainingBatchesInMemory() {
    return this.Exports().training_batches_in_memory();
  }

  _TestingBatchesInMemory() {
    return this.Exports().testing_batches_in_memory();
  }

  // Log training forward details
  _LogTrainForward() {
    let found = false;
    Object.keys(this.Exports()).forEach((func) => {
      if(func.startsWith("log_forward_")) {
        this._logger[func](this._CallExport(func));
        found = true;
      }
    });
    if(!found) {
      this._WarnNotFound("Forward functions were not found");
    }
  }

  // Log training backward details
  _LogTrainBackward() {
    let found = false;
    Object.keys(this.Exports()).forEach((func) => {
      if(func.startsWith("log_backward_")) {
        this._logger[func](this.Exports()[func]());
        found = true;
      }
    });
    if(!found) {
      this._WarnNotFound("Backward functions were not found");
    }
  }

  _LogTrainingConfusionMatrix() {
    let matrix_offset_key = "training_confusion_matrix_offset";
    let matrix_side = this._LayerSize(this._TotalLayers() - 1);
    if(matrix_offset_key in this.Exports()) {
      console.table(this._MakeF32Matrix(this.Exports()[matrix_offset_key](), matrix_side, matrix_side));
    } else {
      this._WarnNotFound("Training confusion matrix function not found");
    }
  }

  _LogTestingConfusionMatrix() {
    let matrix_offset_key = "testing_confusion_matrix_offset";
    let matrix_side = this._LayerSize(this._TotalLayers() - 1);
    if(matrix_offset_key in this.Exports()) {
      console.table(this._MakeF32Matrix(this.Exports()[matrix_offset_key](), matrix_side, matrix_side));
    } else {
      this._WarnNotFound("Testing confusion matrix function not found");
    }
  }

  _LogPredictionResult() {
    let offset = this._CallExport("prediction_result_offset");
    if(offset != null) {
      console.table(this._MakeF32Matrix(offset, this._LayerSize(this._TotalLayers() - 1), this._PredictionBatchSize()));
    }
  }

  // Run unit test Wasm function
  UnitTest() {
    Object.keys(this.Exports()).forEach((func) => {
      if (func.startsWith("test_")) {
        console.log(">>  Testing function:", func);
        console.time("    exectuion time");
        this.Exports()[func]();
        console.timeEnd("    exectuion time");
      }
    });
  }

  // Run predict Wasm function
  Predict(input, config) {
    if(!input.good) {
      console.log("Prediction input seems bad, skipping ...");
      return false;
    }

    // Load model batch information
    let batch_size = this._PredictionBatchSize();
    let number_of_batches = input.x_count / batch_size;
    
    // Configuration        Value                     Default
    config                  = config                  || {};
    config.log_time         = config.log_time         || false;
    config.log_result       = config.log_result       || false;

    // Prediction results
    let data_offset = this._PredictionDataOffset();
    let pred_time = new Date().getTime();
    for(let i=0; i < number_of_batches; i++) {
      // Load new batches in memory and predict
      this._CopyBatchesToMemory(input, data_offset, null, i, batch_size, 1);

      // Start predicting
      this.Exports().predict_batch();

      // Log result
      if(config.log_result) {
        this._LogPredictionResult();
      }
    }
    // Update time
    pred_time = new Date().getTime() - pred_time;

    // Log prediction details
    if(config.log_time) {
      console.log(">> Prediction time:", pred_time, "ms");
    }
  }

  ExtractWeights() {
    let weights = [];
    for (let l = 0; l < this._TotalLayers(); l++) {
      let weight_info = this._WeightInfo(l);
      let bias_info = this._BiasInfo(l);
      if (weight_info != null && bias_info != null) {
        let layer_weight = new LayerWeights(l);
        layer_weight.ImportWeightsFromBuffer(this.Memory().buffer, weight_info.offset, weight_info.byte_size);
        layer_weight.ImportBiasFromBuffer(this.Memory().buffer, bias_info.offset, bias_info.byte_size);
        weights.push(layer_weight.ToJson());
      }
    }
    return weights;
  }

  ImportWeights(weights_array) {
    for(var i=0; i < weights_array.length; i++) {
      // Wrap JSON in a LayerWeight object
      let imported_layer_weights = new LayerWeights(weights_array[i].layer);
      imported_layer_weights.ImportWeightsFromArray(weights_array[i].weights);
      imported_layer_weights.ImportBiasFromArray(weights_array[i].bias);
      // Load model weights info
      let weights_info = this._WeightInfo(weights_array[i].layer);
      let bias_info = this._BiasInfo(weights_array[i].layer);
      if (weights_info != null && bias_info != null) {
        // Wrap Wasm model weight in a LayerWeight object
        let model_layer_weights = new LayerWeights(weights_array[i].layer);
        model_layer_weights.ImportWeightsFromBuffer(this.Memory().buffer,
          weights_info.offset, weights_info.byte_size);
        model_layer_weights.ImportBiasFromBuffer(this.Memory().buffer,
          bias_info.offset, bias_info.byte_size);
        // Set weights
        if (!model_layer_weights.CopyWeights(imported_layer_weights)) {
          console.log("Import failed!");
          return false;
        }
      } else {
        console.error("Import failed: Layer %d does not exists!", weights_array[i].layer);
        return false;
      }
    };
    return true;
  }

  _PredictionDataOffset() {
    return this.Exports().prediction_data_offset();
  }

  _PredictionBatchSize() {
    return this.Exports().prediction_batch_size();
  }

  _TotalLayers() {
    return this.Exports().total_layers();
  }

  _WeightInfo(layer_index) {
    let offset_func = 'layer_' + layer_index + '_weight_offset';
    let length_func = 'layer_' + layer_index +'_weight_byte_size';
    if(this.Exports()[offset_func] !== undefined
      && this.Exports()[length_func] !== undefined) {
      return {
        offset: this.Exports()[offset_func](),
        byte_size: this.Exports()[length_func]()
      }
    }
    return null;
  }

  _BiasInfo(layer_index) {
    let offset_func = 'layer_' + layer_index + '_bias_offset';
    let length_func = 'layer_' + layer_index + '_bias_byte_size';
    if(this.Exports()[offset_func] !== undefined
      && this.Exports()[length_func] !== undefined) {
      return {
        offset: this.Exports()[offset_func](),
        byte_size: this.Exports()[length_func]()
      }
    }
    return null;
  }

  _WarnNotFound(pre_msg) {
    console.log(pre_msg +". Make sure you compiled the model with the correct options.");
  }

  _MakeF32Matrix(index, rows, cols) {
    let view = new Float32Array(this.Memory().buffer, index);
    let table = [];
    for (let r = 0; r < rows; ++r) {
      table.push([]);
      for (let c = 0; c < cols; ++c) {
        table[r].push(view[r * cols + c]);
      }
    }
    return table;
  }

  // Initialize imports
  static Imports() {
    let math_imports = {
      exp: Math.exp,
      log: Math.log,
      random: Math.random
    };

    let system_imports = {
      print: console.log,
      time: () => {
        return new Date().getTime();
      },
      print_table_f32: (index, rows, cols) => {
        console.table(this._MakeF32Matrix(index, rows, cols));
      }
    };

    let test_imports = {
      assert_f32_eq: (val1, val2) => {
        if(val1 !== val2) {
          console.error("F32 equality failed!", val1, "!=", val2);
        }
      },
      assert_matrix_eq: (mat1_index, mat2_index, rows, cols) => {
        let mat1 = new Float32Array(this.Memory().buffer, mat1_index, rows * cols);
        let mat2 = new Float32Array(this.Memory().buffer, mat2_index, rows * cols);
        for (let i = 0; i < rows * cols; i++) {
          if (mat1[i] !== mat2[i]) {
            console.error("Matrix equality failed!");
            system_imports.print_table_f32(mat1_index, rows, cols);
            system_imports.print_table_f32(mat2_index, rows, cols);
            return;
          }
        }
      }
    };
    return {
      "Math": math_imports,
      "System": system_imports,
      "Test": test_imports,
    };
  }
}

var module = module || { exports: {} };
module.exports.CompiledModel = CompiledModel;

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
    console.log("A) dA[L] = L(T, A[L]):", time);
  }
  log_backward_B_1(time) {
    console.log("B) dZ[l] = dA[l] * g'[l](Z[l])");
    console.log("    1) dZ[l] = g'[l](Z[l]):", time);
  }
  log_backward_B_2(time) {
    console.log("    2) dZ[l] = dA[l] * dZ[l]:", time);
  }
  log_backward_C_1(time) {
    console.log("C) dW[l] = (1/m) dZ[l] . A[l-1]^T");
    console.log("    1) dW[l] = dZ[l] . A[l-1]^T:", time);
  }
  log_backward_C_2(time) {
    console.log("    2) dW[l] = (1/m) dW[l]:", time);
  }
  log_backward_D_1(time) {
    console.log("D) db[l] = (1/m) dZ[l]");
    console.log("    1) db[l] = SUM(dZ[l], row wise):", time);
  }
  log_backward_D_2(time) {
    console.log("    2) db[l] = (1/m) db[l]:", time);
  }
  log_backward_E(time) {
    console.log("E) dA[l-1] = W[l]^T . dZ[l]:", time);
  }
  log_backward_F_1(time) {
    console.log("F) W[l] = W[l] - alpha * dW[l]");
    console.log("    1) dW[l] = alpha * dW[l]:", time);
  }
  log_backward_F_2(time) {
    console.log("    2) W[l] = W[l] - dW[l]:", time);
  }
  log_backward_G_1(time) {
    console.log("G) b[l] = b[l] - alpha * db[l]");
    console.log("    1) db[l] = alpha * db[l]:", time);
  }
  log_backward_G_2(time) {
    console.log("    2) b[l] = b[l] - db[l]:", time);
  }
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

  constructor() {
    this._imports = this._InitImports();
  }

  // Set the wasm instance
  SetWasm(wasm) {
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

  // Get imports from JS to Wasm
  Imports() {
    return this._imports;
  }

  Memory() {
    return this.Exports().memory;
  }

  // Run train Wasm function
  Train(data, labels, config) {
    // Register the total training time
    let total_time = 0.0;
    // Model details
    let batch_size = this._TrainingBatchSize();
    let batches_in_memory = this._TrainingBatchesInMemory();
    let batches_in_memory_count = Math.ceil(data.length / (batches_in_memory * batch_size));;
    let number_of_batches = data.length / batch_size;
    if(data.length != labels.length) {
      console.error("Data size should be equal to the labels size");
      return false;
    }
    if(data.length % batch_size != 0) {
      console.error("Data size",data.length,"should be divisible by the number of batch",batch_size);
      return false;
    }
    // Set configuration
    config = config || {};
    config.log_accuracy = config.log_accuracy || false;
    config.log_error = config.log_error || false;
    config.log_time = config.log_time || false;
    config.epochs = config.epochs || 0;
    for(let e=0; e < config.epochs; e++) {
      // Epoch results
      let total_hits = 0;
      let average_cost = 0.0;
      let train_time = 0.0;
      let copy_time = 0.0;
      for(let i=0; i < batches_in_memory_count; i++) {
        // Load new batches in memory and train
        let time = new Date().getTime();
        let batches_inserted = this._InsertBatchesInMemory(this._TrainingDataOffset(), 
          data, this._TrainingLabelsOffset(), labels, i * batches_in_memory * batch_size, 
          batch_size, batches_in_memory);
        copy_time += new Date().getTime() - time;

        // Start training
        time = new Date().getTime();
        this.Exports().train_batches_in_memory(batches_inserted);
        train_time += new Date().getTime() - time;
        
        // Update training details
        if(config.log_accuracy) {
          total_hits += this._TrainingBatchesAccuracy();
        }
        if(config.log_error) {
          average_cost += this._TrainingBatchesError();
        }
      }
      // Update total time
      total_time += train_time + copy_time;
      // Log after end of epoch
      console.log("Epoch",e+1);
      if(config.log_accuracy) {
        console.log(">> Accuracy:  ", total_hits / data.length);
      }
      if(config.log_error) {
        console.log(">> Error:     ", average_cost / number_of_batches);
      }
      if(config.log_time) {
        console.log(">> Copy time: ", copy_time, "ms");
        console.log(">> Train time:", train_time, "ms");
        console.log(">> Epoch time:", train_time + copy_time, "ms")
        console.log(">> Total time:", total_time, "ms");
      }
    }
    if(config.log_time) {
      console.log(">> Time/Batch:", total_time / number_of_batches);
    }
  }

  _InsertBatchesInMemory(data_offset, data, labels_offset, labels, from, batch_size, num_batches) {
    let count = 0;
    let to = from + batch_size * num_batches;
    // Compute batch sizes
    let data_batch_bytes = data[from].length * batch_size * Float32Array.BYTES_PER_ELEMENT;
    let labels_batch_bytes = labels[from].length * batch_size * Float32Array.BYTES_PER_ELEMENT;
    // Copy batches into linear memory
    for(let i=from; i < to && i < data.length; i+=batch_size) {
      // Copy data and labels
      this._InsertBatchInMemory(data_offset, data, i, batch_size);
      this._InsertBatchInMemory(labels_offset, labels, i, batch_size);
      // Move to the next batch
      data_offset += data_batch_bytes;
      labels_offset += labels[i].length * batch_size * Float32Array.BYTES_PER_ELEMENT;
      // Count how many batches were copied
      count++;
    }
    return count;
  }

  _InsertBatchInMemory(memory_offset, data, from, batch_size) {
    let index = 0;
    let entry_size = data[from].length;
    let memory = new Float32Array(this.Memory().buffer, memory_offset, entry_size * batch_size);
    for (let c = 0; c < entry_size; c++) {
      for (let r = 0; r < batch_size; r++) {
        memory[index++] = data[from + r][c];
      }
    }
  }

  _TrainingDataOffset() {
    return this.Exports().training_data_offset();
  }

  _TrainingLabelsOffset() {
    return this.Exports().training_labels_offset();
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

  _TrainingBatchesError() {
    let key = "training_batches_error";
    if(key in this.Exports()) {
      return this.Exports()[key]();
    } else {
      this._WarnNotFound("Training batches error function not found");
    }
    return 0;
  }

  _TrainingBatchSize() {
    return this.Exports().training_batch_size();
  }

  _TrainingBatchesInMemory() {
    return this.Exports().training_batches_in_memory();
  }

  // Run test Wasm function
  Test() {
    this.Exports().test();
  }

  // Log training forward details
  LogTrainForward() {
    let found = false;
    Object.keys(this.Exports()).forEach((func) => {
      if(func.startsWith("log_forward_")) {
        this._logger[func](this.Exports()[func]());
        found = true;
      }
    });
    if(!found) {
      this._WarnNotFound("Forward functions were not found");
    }
  }

  // Log training backward details
  LogTrainBackward() {
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

  PrintTrainingConfusionMatrix() {
    let matrix_offset_key = "training_confusion_matrix_offset";
    let output_size_key = "layer_" + (this.Exports().total_layers()-1) + "_size";
    let matrix_side = this.Exports()[output_size_key]();
    if(!(matrix_offset_key in Object.keys(this.Exports()))) {
      console.table(this._MakeF32Matrix(this.Exports()[matrix_offset_key](), matrix_side, matrix_side));
    } else {
      this._WarnNotFound("Training confusion matrix function not found");
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
  Predict(data) {
    let offset = this._PredictionInputOffset();
    let batch_size = this._PredictionBatchSize();

    if (data === undefined || data.length === 0 || batch_size !== data.length) {
      console.error("Data size should match the batch size %d != %d", data.length, batch_size);
      return false;
    }

    let index = 0;
    let memory = new Float32Array(this.Memory().buffer, offset, data[0].length * batch_size);
    for (let c = 0; c < data[0].length; c++) {
      for (let r = 0; r < data.length; r++) {
        memory[index++] = data[r][c];
      }
    }
    this.Exports().predict();
    return true;
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

  _PredictionInputOffset() {
    return this.Exports().prediction_input_offset();
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
  _InitImports() {
    let math_imports = {
      exp: Math.exp,
      log: Math.log,
      random: Math.random
    };

    let message_imports = {
      log_training_time: (epoch, time_epoch, time_total) => {
        console.log("Training time at epoch", epoch + 1, "is", time_epoch, "ms", 
          "and total time so far is", time_total, "ms");
      },
      log_training_error: (epoch, error) => {
        console.log("Training Error in epoch", epoch + 1, ":", error);
      },
      log_training_accuracy: (epoch, acc) => {
        console.log("Training Accuracy in epoch", epoch + 1, ":",
          Math.round(acc * 10000) / 10000);
      },
      log_testing_time: (time) => {
        console.log("Testing time:", time, "ms");
      },
      log_testing_error: (error) => {
        console.log("Testing Error:", error);
      },
      log_testing_accuracy: (acc) => {
        console.log("Testing Accuracy:", Math.round(acc * 10000) / 10000);
      },
      log_prediction_time: (time) => {
        console.log("Prediction time:", time, "ms");
      },
    };

    let system_imports = {
      print: console.log,
      time: () => {
        return new Date().getTime();
      },
      print_table_f32: (index, rows, cols) => {
        let view = new Float32Array(this.Memory().buffer, index);
        let table = [];
        for (let r = 0; r < rows; ++r) {
          table.push([]);
          for (let c = 0; c < cols; ++c) {
            table[r].push(view[r * cols + c]);
          }
        }
        console.table(table);
      }
    };

    let test_imports = {
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
      "Message": message_imports,
      "System": system_imports,
      "Test": test_imports,
    };
  }
}

var module = module || { exports: {} };
module.exports.CompiledModel = CompiledModel;

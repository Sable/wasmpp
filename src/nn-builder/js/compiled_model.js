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

// This class is a wrapper for the generated Wasm model
// It also contains the import functions from JS to Wasm
// Most functions simply calls the Wasm exported functions
// but some might process the arguments in order to pass
// them correctly to Wasm functions
class CompiledModel {
  _wasm = null;
  _imports = {};

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

  // Run train Wasm function
  Train() {
    if (this.Exports() != null) {
      this.Exports().train();
    }
  }

  // Run test Wasm function
  Test() {
    if (this.Exports() != null) {
      this.Exports().test();
    }
  }

  // Run unit test Wasm function
  UnitTest() {
    if (this.Exports() != null) {
      Object.keys(this.Exports()).forEach((func) => {
        if (func.startsWith("test_")) {
          console.log(">>  Testing function:", func);
          console.time("    exectuion time");
          this.Exports()[func]();
          console.timeEnd("    exectuion time");
        }
      });
    }
  }

  // Run predict Wasm function
  Predict(data) {
    if (this.Exports() != null) {
      let offset = this._PredictionInputOffset();
      let batch_size = this._PredictionBatchSize();

      if (data === undefined || data.length === 0 || batch_size !== data.length) {
        console.error("Data size should match the batch size %d != %d", data.length, batch_size);
        return false;
      }

      let index = 0;
      let memory = new Float32Array(this.Exports().memory.buffer, offset, data[0].length * batch_size);
      for (let c = 0; c < data[0].length; c++) {
        for (let r = 0; r < data.length; r++) {
          memory[index++] = data[r][c];
        }
      }
      this.Exports().predict();
      return true;
    }
    return false;
  }

  ExtractWeights() {
    let weights = [];
    if (this.Exports() != null) {
      for (let l = 0; l < this._TotalLayers(); l++) {
        let weight_info = this._WeightInfo(l);
        let bias_info = this._BiasInfo(l);
        if (weight_info != null && bias_info != null) {
          let layer_weight = new LayerWeights(l);
          layer_weight.ImportWeightsFromBuffer(this.Exports().memory.buffer, weight_info.offset, weight_info.byte_size);
          layer_weight.ImportBiasFromBuffer(this.Exports().memory.buffer, bias_info.offset, bias_info.byte_size);
          weights.push(layer_weight.ToJson());
        }
      }
      return weights;
    }
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
        model_layer_weights.ImportWeightsFromBuffer(this.Exports().memory.buffer,
          weights_info.offset, weights_info.byte_size);
        model_layer_weights.ImportBiasFromBuffer(this.Exports().memory.buffer,
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
    if(this.Exports() != null) {
      return this.Exports().prediction_input_offset();
    }
    return false;
  }

  _PredictionBatchSize() {
    if(this.Exports() != null) {
      return this.Exports().prediction_batch_size();
    }
    return false;
  }

  _TotalLayers() {
    if(this.Exports() != null) {
      return this.Exports().total_layers();
    }
    return 0;
  }

  _WeightInfo(layer_index) {
    let offset_func = 'weight_offset_' + layer_index;
    let length_func = 'weight_byte_size_' + layer_index;
    if(this.Exports() != null
      && this.Exports()[offset_func] !== undefined
      && this.Exports()[length_func] !== undefined) {
      return {
        offset: this.Exports()[offset_func](),
        byte_size: this.Exports()[length_func]()
      }
    }
    return null;
  }

  _BiasInfo(layer_index) {
    let offset_func = 'bias_offset_' + layer_index;
    let length_func = 'bias_byte_size_' + layer_index;
    if(this.Exports() != null
      && this.Exports()[offset_func] !== undefined
      && this.Exports()[length_func] !== undefined) {
      return {
        offset: this.Exports()[offset_func](),
        byte_size: this.Exports()[length_func]()
      }
    }
    return null;
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
        console.log("Training time at epoch", epoch, "is", time_epoch, "ms", 
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
        if (this.Exports() != null) {
          let view = new Float32Array(this.Exports().memory.buffer, index);
          let table = [];
          for (let r = 0; r < rows; ++r) {
            table.push([]);
            for (let c = 0; c < cols; ++c) {
              table[r].push(view[r * cols + c]);
            }
          }
          console.table(table);
        }
      }
    };

    let test_imports = {
      assert_matrix_eq: (mat1_index, mat2_index, rows, cols) => {
        if (this.Exports() != null) {
          let mat1 = new Float32Array(this.Exports().memory.buffer, mat1_index, rows * cols);
          let mat2 = new Float32Array(this.Exports().memory.buffer, mat2_index, rows * cols);
          for (let i = 0; i < rows * cols; i++) {
            if (mat1[i] !== mat2[i]) {
              console.error("Matrix equality failed!");
              imports.System.print_table_f32(mat1_index, rows, cols);
              imports.System.print_table_f32(mat2_index, rows, cols);
              return;
            }
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

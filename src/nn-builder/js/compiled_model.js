class CompiledModel {
  #wasm = null;
  #imports = {};

  constructor() {
    this.#imports = this._InitImports();
  }

  // Set the wasm instance
  SetWasm(wasm) {
    this.#wasm = wasm;
  }

  // Get exports from Wasm to JS
  Exports() {
    if (this.#wasm == null) {
      console.error("Wasm instance was not set");
      return null;
    }
    return this.#wasm.instance.exports;
  }

  // Get imports from JS to Wasm
  Imports() {
    return this.#imports;
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
    if (data === undefined) {
      console.error("Data cannot be undefined");
      return;
    }
    if (this.Exports() != null) {
      let offset = this.Exports().prediction_input_offset();
      let batch_size = this.Exports().prediction_batch_size();
      if (this._InsertData(this.Exports().memory.buffer, data, batch_size, offset)) {
        this.Exports().predict();
      }
    }
  }

  // Insert data into buffer
  _InsertData(buffer, data, batch_size, offset) {
    if (data.length === 0 || batch_size !== data.length) {
      console.error("Data size should match the batch size %d != %d", data.length, batch_size);
      return false;
    }

    let index = 0;
    let memory = new Float32Array(buffer, offset, data[0].length * batch_size);
    for (let c = 0; c < data[0].length; c++) {
      for (let r = 0; r < data.length; r++) {
        memory[index++] = data[r][c];
      }
    }
    return true;
  }

  // Initialize imports
  _InitImports() {
    let math_imports = {
      exp: Math.exp,
      log: Math.log,
      random: Math.random
    };

    let message_imports = {
      log_training_time: (time) => {
        console.log("Training time:", time, "ms");
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

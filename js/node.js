const fs = require('fs');
var g_wasm = null;
const imports = {
  "Math": {
    exp: Math.exp,
    log: Math.log,
    random: Math.random
  },
  "Message": {
    log_training_time: (time) => {console.log("Training time:", time, "ms");},
    log_training_error: (epoch, error) => {
      console.log("Training Error in epoch", epoch+1, ":", error);
    },
    log_training_accuracy: (epoch, acc) => {
      console.log("Training Accuracy in epoch", epoch+1, ":", 
        Math.round(acc * 10000) / 10000);
    },
    log_testing_time: (time) => {console.log("Testing time:", time, "ms");},
    log_testing_error: (error) => {console.log("Testing Error:", error);},
    log_testing_accuracy: (acc) => {
      console.log("Testing Accuracy:", Math.round(acc * 10000) / 10000);
    },
  },
  "System": {
    print: console.log,
    time: () => { return new Date().getTime(); },
    print_table_f32: (index, rows, cols) => {
      let view = new Float32Array(g_wasm.instance.exports.memory.buffer, index);
      let table = [];
      for(let r=0; r < rows; ++r) {
        table.push([]);
        for(let c=0; c < cols; ++c) {
          table[r].push(view[r * cols + c]);
        }
      }
      console.table(table);
    }
  },
  "Test": {
    assert_matrix_eq: (mat1_index, mat2_index, rows, cols) => {
      let mat1 = new Float32Array(g_wasm.instance.exports.memory.buffer, mat1_index, rows*cols);
      let mat2 = new Float32Array(g_wasm.instance.exports.memory.buffer, mat2_index, rows*cols);
      for(let i=0; i < rows*cols; i++) {
        if(mat1[i] != mat2[i]) {
          console.error("Matrix equality failed!");
          imports.System.print_table_f32(mat1_index, rows, cols);
          imports.System.print_table_f32(mat2_index, rows, cols);
          return;
        }
      }
    }
  }
}

function InsertData(buffer, data, batch_size, offset) {
  if(data.length == 0 || batch_size != data.length) {
    console.error("Data size should match the batch size");
    return false;
  }

  var index = 0;
  let memory = new Float32Array(buffer, offset, data[0].length * batch_size);
  for(var c=0; c < data[0].length; c++) {
    for(var r=0; r < data.length; r++) {
      memory[index++] = data[r][c];
    }
  }
  return true;
}

if(process.argv.length > 2) {
    const buf = fs.readFileSync(process.argv[2]);
    const lib = WebAssembly.instantiate(new Uint8Array(buf), imports);
    lib.then( wasm => {
      g_wasm = wasm;
      console.log('WASM Ready.');
      if(wasm.instance.exports.train != undefined) {
        console.log("Training ...");
        console.log(wasm.instance.exports.train())
      }
      if(wasm.instance.exports.test != undefined) {
        console.log("Testing ...");
        console.log(wasm.instance.exports.test())
      }
      if(wasm.instance.exports.predict) {
        console.log("Predicting ...");
        let offset = wasm.instance.exports.prediction_input_offset();
        let batch_size = wasm.instance.exports.prediction_batch_size();
        let data = [[0, 0], [0, 1],[1,0],[1,1]];
        if(InsertData(wasm.instance.exports.memory.buffer, data, batch_size, offset)) {
          wasm.instance.exports.predict();
        }
      }
      Object.keys(wasm.instance.exports).forEach((func) => {
        if(func.startsWith("test_")) {
          console.log(">>  Testing function:", func);
          console.time("    exectuion time")
          wasm.instance.exports[func]();
          console.timeEnd("    exectuion time");
        }
      });
    })
} else {
    console.log("Missing argument: file.wasm");
}

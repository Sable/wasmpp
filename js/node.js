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
      console.log("Training error in epoch", epoch+1, ":", error);
    },
    log_testing_time: (time) => {console.log("Testing time:", time, "ms");},
    log_testing_error: (error) => {console.log("Testing error:", error);},
  },
  "System": {
    print: console.log,
    time: () => { return new Date().getTime(); },
    print_table_f32: function(index, rows, cols) {
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
  }
}

if(process.argv.length > 2) {
    const buf = fs.readFileSync(process.argv[2]);
    const lib = WebAssembly.instantiate(new Uint8Array(buf), imports);
    lib.then( wasm => {
        g_wasm = wasm;
        console.log('WASM Ready.');
        console.log(wasm.instance.exports.train())
        console.log(wasm.instance.exports.test())
    })
} else {
    console.log("Missing argument: file.wasm");
}

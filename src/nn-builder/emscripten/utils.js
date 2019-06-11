// UTILS FUNCTIONS
// Functions in this file are accessible
// inside the onRuntimeInitialized callback
//
// Example:
// var my_module = require('...')
// my_module.onRuntimeInitialized = function() {
//  my_module.my_util_function();
// }

var Module = Module || {};
Module["ToMatrix"] = function(array) {
  var f32_matrix = new Module["F32Matrix"]();      
  for(var r=0; r < array.length; r++) {
    var f32_array = new Module["F32Array"]();
    for(var c=0; c < array[0].length; c++) {
      f32_array.push_back(array[r][c]);
    }
    f32_matrix.push_back(f32_array);
  }                                          
  return f32_matrix;
}

Module["ToUint8Array"] = function(byte_array) {
  var array = new Uint8Array(byte_array.size());
  for (var i = 0; i < byte_array.size(); i++) {
      array[i] = byte_array.get(i);
  }
  return array;
}


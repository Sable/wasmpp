// UTILS FUNCTIONS
// Functions in this file are accessible
// inside the onRuntimeInitialized callback
//
// Example:
// var my_module = require('...')
// my_module.onRuntimeInitialized = function() {
//  my_module.my_util_function();
// }

Module["ToUint8Array"] = function(byte_array) {
  var array = new Uint8Array(byte_array.size());
  for (var i = 0; i < byte_array.size(); i++) {
      array[i] = byte_array.get(i);
  }
  return array;
}


#include <node.h>
#include <v8.h>
#include <napi.h>

using v8::Local;
using v8::Object;

void Initialize(Local<Object> exports) {
  v8::V8::EnableWebAssemblyTrapHandler(true);
}

NODE_MODULE(NODE_GYP_MODULE_NAME, Initialize)

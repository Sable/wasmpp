#!/bin/bash

# VARS
DOCS_DST="$PWD/../docs"
WASMPP_DOXYGEN="$DOCS_DST/wasmpp_api"
WASMDDNN_DOXYGEN="$DOCS_DST/wasmdnn_api"
DOXYGEN_DST="$PWD/build/doxygen"

# Build doxygen
echo ">> Building doxygen files ..."
mkdir -p $DOXYGEN_DST
cd $DOXYGEN_DST
cmake ../../..
make doxygen_wasmpp && make doxygen_wasmdnn

if [ $? -eq 0 ]; then
  # Copy doxygen
  echo ">> Copying generated doxygen files ..."
  cp -ar "$DOXYGEN_DST/wasmpp_api/html" $WASMPP_DOXYGEN
  cp -ar "$DOXYGEN_DST/wasmdnn_api/html" $WASMDDNN_DOXYGEN
  echo ">> Done"
else
  echo ">> Error: One or more targets failed!"
fi


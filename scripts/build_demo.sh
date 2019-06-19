#!/bin/bash

DEMO_DIR="../docs/demo"
BUILD_DIR="./build"

echo ">> Removing old builds (if any) ..."
rm -rf $BUILD_DIR

echo ">> Removing old demo ..."
rm -rf $DEMO_DIR

echo ">> Creating a build directoy ..."
mkdir $BUILD_DIR

echo ">> Checking if emcmake and emmake are available ..."
command -v emcmake > /dev/null && echo "++ emcmake found" \
  || (echo "!! emcmake not found" && exit 1)
command -v emmake > /dev/null && echo "++ emmake found" \
  || (echo "!! emmake not found" && exit 1)

echo ">> Starting the build ..."
cd $BUILD_DIR
emcmake cmake ../../
emmake make nnb_js
cd ..

echo ">> Copying files into ../docs/demo ..."
mkdir $DEMO_DIR
cp -v $BUILD_DIR/bin/nnb_js.* $DEMO_DIR
cp -v ../src/nn-builder/js/compiled_model.js $DEMO_DIR
cp -v ../src/nn-builder/examples/js/nnb.html $DEMO_DIR

echo "Done!"

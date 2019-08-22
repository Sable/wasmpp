#!/bin/bash

# VARS
DOCS_DST="$PWD/../docs"
MKDOCS="$PWD/../third_party/mkdocs/mkdocs"

# Build WasmDNN docs
DOCS_MD_DST="$DOCS_DST/wasmdnn_md"
DOCS_HTML_DST="$DOCS_DST/wasmdnn_html"

cd $DOCS_MD_DST
python $MKDOCS build -d "$DOCS_HTML_DST"

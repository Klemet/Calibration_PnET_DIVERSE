#!/bin/bash

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

# Build the Jupyter Book
jupyter-book build calibrationJupyterBook/

# Remove current html files
rm -r ./docs/*

# Copy the built HTML files to the docs directory
cp -r ./calibrationJupyterBook/_build/html/* ./docs

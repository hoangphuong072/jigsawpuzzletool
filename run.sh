#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$PWD"
cd src
python3 crop.py
cd ..
rm -r output.zip
zip -r output.zip output
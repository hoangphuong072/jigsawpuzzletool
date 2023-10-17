#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$PWD"
rm -r output.zip
unzip input.zip
cd src
python3 crop.py
cd ..

zip -r output.zip output
ls

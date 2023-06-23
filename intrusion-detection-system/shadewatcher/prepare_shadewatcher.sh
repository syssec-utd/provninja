#!/bin/bash
git checkout data-processing-scripts 
cd ./syssec-data-processing

# disable glob (wildcard expansion),
# since the intermediate scripts will perform expansion on their own.
set -f

# filesystem accessible dataset path
data_dir="/data"
# evaluation output directory
test_dir="tests"
mkdir -p $test_dir

# shadewatcher storage dir
export STORE_DIR=$data_dir/shadewatcher_store

# model prefix
model="model"
# training epochs
epochs=300
# threshold for classification
threshold=2.0

pip install pandas tabulate
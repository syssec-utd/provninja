#!/bin/bash

### train on benign dataset for non-gadget
python3.6 shadewatcher_train.py "$STORE_DIR/non-gadget-benign-*" $model-non-gadget --gnn_args="--epoch $epochs --threshold=$threshold --val_size 0.5"

# verify on benign dataset to obtain fn, tp
python3.6 shadewatcher_eval.py "$STORE_DIR/non-gadget-benign-*" $STORE_DIR/$model-non-gadget $test_dir/$model-non-gadget.csv --benign --threshold=$threshold
# evaluate on anomaly dataset to obtain tn, fp
python3.6 shadewatcher_eval.py "$STORE_DIR/non-gadget-anomaly-*" $STORE_DIR/$model-non-gadget $test_dir/$model-non-gadget.csv --threshold=$threshold

### train on benign dataset for gadget
python3.6 shadewatcher_train.py "$STORE_DIR/gadget-benign-*" $model-gadget --gnn_args="--epoch $epochs --threshold=$threshold --val_size 0.5"

# verify on benign dataset to obtain fn, tp
python3.6 shadewatcher_eval.py "$STORE_DIR/gadget-benign-*" $STORE_DIR/$model-gadget $test_dir/$model-gadget.csv --benign --threshold=$threshold
# evaluate on anomaly dataset to obtain tn, fp
python3.6 shadewatcher_eval.py "$STORE_DIR/gadget-anomaly-*" $STORE_DIR/$model-gadget $test_dir/$model-gadget.csv --threshold=$threshold
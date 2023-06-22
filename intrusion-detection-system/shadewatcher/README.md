# Shadewatcher Evaluation

## Setup

A Dockerized version of the Shadewatcher repository is prepared in [./Dockerfile](./Dockerfile)

Build the image:

```shell
docker build . -t shadewatcher
```

Once built, run the container interactively:

```shell
docker run -it --mount type=bind,source="$(pwd)",target=/data -e DATASET_DIR=/data --name shadewatcher shadewatcher
```

## Execution

Inside the Container, navigate to the `ShadeWatcher` directory, then switch to the Syssec Lab branch containing additional processing scripts

```shell
cd /ShadeWatcher
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

# metric dependencies
pip install pandas tabulate
# display metrics in tabular form
python3.6 stat_eval.py $test_dir
```

Example output (subject to training & parameter changes):

```shell
| filename                   |   tn |   tp |
|:---------------------------|-----:|-----:|
| tests/model-gadget.csv     |  634 |   31 |
| tests/model-non-gadget.csv | 1233 |   48 |
```




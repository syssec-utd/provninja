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

rm -rf shadewatcher_store/ tests/

# disable glob (wildcard expansion),
# since the intermediate scripts will perform expansion on their own.
set -f

# filesystem accessible dataset path
data_dir="/data"
# hyphen-joined replacement of data_dir path
path_dir="${data_dir//\//-}"
path_dir="${path_dir#-}"
# evaluation output directory
test_dir="tests"
mkdir -p $test_dir

# data caching directory
store_dir="shadewatcher_store"
# model prefix
model="model"
# threshold for classification
threshold=2.0


# parse and cache shadewatcher representations of ASI JSON graph data
python3.6 shadewatcher_parse.py "$data_dir/gadget/*/*/graph.json $data_dir/non-gadget/*/*/graph.json"


### train on benign dataset for non-gadget
python3.6 shadewatcher_train.py "$store_dir/$path_dir-non-gadget-benign-*" $model-non-gadget --gnn_args="--epoch 200 --threshold=$threshold"

# verify on benign dataset to obtain fn, tp
python3.6 shadewatcher_eval.py "$store_dir/$path_dir-non-gadget-benign-*" $store_dir/$model-non-gadget $test_dir/$model-non-gadget.csv --benign --threshold=$threshold
# evaluate on anomaly dataset to obtain tn, fp
python3.6 shadewatcher_eval.py "$store_dir/$path_dir-non-gadget-anomaly-*" $store_dir/$model-non-gadget $test_dir/$model-non-gadget.csv --threshold=$threshold


### train on benign dataset for gadget
python3.6 shadewatcher_train.py "$store_dir/$path_dir-gadget-benign-*" $model-gadget --gnn_args="--epoch 200 --threshold=$threshold"

# verify on benign dataset to obtain fn, tp
python3.6 shadewatcher_eval.py "$store_dir/$path_dir-gadget-benign-*" $store_dir/$model-gadget $test_dir/$model-gadget.csv --benign --threshold=$threshold
# evaluate on anomaly dataset to obtain tn, fp
python3.6 shadewatcher_eval.py "$store_dir/$path_dir-gadget-anomaly-*" $store_dir/$model-gadget $test_dir/$model-gadget.csv --threshold=$threshold

# metric dependencies
pip install pandas tabulate
# display metrics in tabular form
python3.6 compare_eval.py $test_dir

```





# Shadewatcher Evaluation

## Setup

A Dockerized version of the Shadewatcher repository is prepared in [./Dockerfile](./Dockerfile)

Build the image:

```shell
docker build . -t shadewatcher
```

Once built, run the container interactively:

```shell
docker run -it --mount type=bind,source="$(pwd)",target=/data --name shadewatcher shadewatcher
```

## Execution

Inside the Container, navigate to the `ShadeWatcher` directory, then source the scripts:
1. [prepare_shadewatcher.sh](./prepare_shadewatcher.sh)
2. [run_shadewatcher_experiments.sh](./run_shadewatcher_experiments.sh)

```shell
cd /ShadeWatcher
. /data/prepare_shadewatcher.sh
. /data/run_shadewatcher_experiments.sh
exit
```

Example output (subject to training & parameter changes):

```shell
| filename                   |   tn |   tp |
|:---------------------------|-----:|-----:|
| tests/model-non-gadget.csv | 1233 |   48 |
| tests/model-gadget.csv     |  634 |   31 |
```





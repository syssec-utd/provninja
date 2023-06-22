# Evading Provenance-Based ML Detectors with Adversarial System Actions

Reproducibility artifacts for the paper _Evading Provenance-Based ML Detectors with Adversarial System Actions_.

## Overview


## Folder structure

| Folder | Description|
| -------|-----------|
| `gadget-finder`| Folder containing the code and data to execute the gadget-finder algorithms. |
| `intrusion-detection-system`| Folder containing the code and data files for IDS execution. |

### Environment Setup

We will use `conda` as the python environment manager. Install the project dependencies from the [provng.yml](provng.yml) using this command:

```bash
conda env update --name provng --file provng.yml
```

Activate the conda environment before running the experiments by running this command

```bash
conda activate provng
```

### Gadget Finder

* [Gadget Finder](gadget-finder/gadget-finder.py)
  * Finds the possible gadget chains between two programs as identified in [input.csv](gadget-finder/input.csv)
  * You can check a sample output in [output](gadget-finder/output) directory.
  
Running the gadget finder script:

```bash
python gadget-finder.py -i input.csv -p FrequencyDB/SAMPLE_WINDOWS_FREQUENCY_DB.csv -o output/gadgets.txt
```

### Path-based IDS

#### SIGL[[1]](#references)

* [sigl](intrusion-detection-system/path-based/sigl.py)
  * Driver script for SIGL, which is an Autoencoder based IDS that detects anomalous paths.
  * Sample causal paragraphs and feature vectors for Enterprise APT available in [sample-enterprise-data](intrusion-detection-system/path-based/sample-enterprise-data) directory.
  
Running the SIGL script:

```bash
python sigl.py
```

#### ProvDetector[[2]](#references)

* [provdetector](intrusion-detection-system/path-based/provdetector.py)
  * Driver script for ProvDetector, which is an LOF based IDS that detects anomalous paths.
  * Sample causal paragraphs and feature vectors for Enterprise APT available in [sample-enterprise-data](intrusion-detection-system/path-based/sample-enterprise-data) directory.
  
Running the ProvDetector script:

```bash
python provdetector.py
```

### Graph-based IDS

#### S-GAT

* [S-GAT](intrusion-detection-system/graph-based/gnnDriver.py)
  * Driver script for S-GAT, which is an GNN based IDS that detects anomalous graph using graph structure and attributes, e.g., node/edge types.
  * Run [download_sample_supply_chain_data.sh](intrusion-detection-system/graph-based/download_sample_supply_chain_data.sh) to download and unzip the sample Supply-Chain APT data from [Google Drive](https://drive.google.com/file/d/1Jz0ZuiZlUEZdAgqlnfmpN2_X0Cms6Sl8/view?usp=sharing)
  * The weighted average F1 score on the provided data with the provided model should be 0.88.
  
Running the S-GAT script:

```bash
python gnnDriver.py gat -if 5 -hf 10 -lr 0.001 -e 20 -n 5 -bs 128 -bi -s
```

#### Prov-GAT

* [Prov-GAT](intrusion-detection-system/graph-based/gnnDriver.py)
  * Driver script for Prov-GAT, which is an GNN based IDS that detects anomalous graph using node and edge attributes on top of features used by S-GAT feature.
  * Run [download_sample_supply_chain_data.sh](intrusion-detection-system/graph-based/download_sample_supply_chain_data.sh) to download and unzip the sample Supply-Chain APT data from [Google Drive](https://drive.google.com/file/d/1Jz0ZuiZlUEZdAgqlnfmpN2_X0Cms6Sl8/view?usp=sharing)
  * The weighted average F1 score on the provided data with the provided model should be 0.95.

Running the Prov-GAT script:

```bash
python gnnDriver.py gat -if 768 -hf 10 -lr 0.001 -e 20 -n 5 -bs 128 -bi
```

#### ProvNinja Graph

* [ProvNinja-Graph](intrusion-detection-system/graph-based/provninjaGraph.py)
  * Driver script for ProvNinja-Graph which is an adversarial example generator.
  * Run [download_sample_supply_chain_data.sh](intrusion-detection-system/graph-based/download_sample_supply_chain_data.sh) to download and unzip the sample Supply-Chain APT data from [Google Drive](https://drive.google.com/file/d/1Jz0ZuiZlUEZdAgqlnfmpN2_X0Cms6Sl8/view?usp=sharing)
  * Output will be in directory [adversarial_examples](intrusion-detection-system/graph-based/adversarial_examples).
  * The evasion rate should be approximately 168 / 198 true positives for the provided data with the provided models.

Running the ProvNinja-Graph script:

```bash
python provninjaGraph.py
```

#### ShadeWatcher

* [ShadeWatcher](intrusion-detection-system/shadewatcher/README.md)
  * Demonstrate usage of gadgets on edge-classification model [ShadeWatcher](https://jun-zeng.github.io/file/shadewatcher_paper.pdf) 
  * Follow [README](intrusion-detection-system/shadewatcher/README.md) instructions on bootstrapping a testing env. and running dataset
  * Gadget graphs evade edge-level anomaly detections much more often than raw attack



## Citing us

```
@inproceedings{mukherjee2023sec,
	title        = {Evading Provenance-Based ML Detectors with Adversarial System Actions},
	author       = {Kunal Mukherjee and Josh Wiedemeier and Tianhao Wang and James Wei and Feng Chen and Muhyun Kim and Murat Kantarcioglu and Kangkook Jee},
	year         = 2023,
	booktitle    = {Proceedings of USENIX Security Symposium (SEC)},
	series       = {USENIX '23}
}
```

## References 

[1] X. Han, X. Yu, T. Pasquier, et al., “_Sigl: Securing software installations through deep graph learning_,” in
USENIX Security Symposium (SEC), 2021. <br>
[2] Q. Wang, W. U. Hassan, D. Li, et al., “_You Are What
You Do: Hunting Stealthy Malware via Data Provenance Analysis_,” in Network and Distributed System
Security Symposium (NDSS), Feb. 2020. <br>

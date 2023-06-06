# Graph-based IDS

## S-GAT

* [S-GAT](intrusion-detection-system/graph-based/gnnDriver.py)
  * Driver script for S-GAT, which is an GNN based IDS that detects anomalous graph using graph structure and attributes, e.g., node/edge types.
  * Download the [sample-supply-chain-data.zip](https://drive.google.com/file/d/1Jz0ZuiZlUEZdAgqlnfmpN2_X0Cms6Sl8/view?usp=sharing) and put it in `intrusion-detection-system/graph-based/` directory. Then unzip it.
  * Sample processed graphs for Supply-Chain APT available in [sample-supply-chain-data](intrusion-detection-system/graph-based/sample-supply-chain-data) directory.
  
Running the S-GAT script:

```bash
python gnnDriver.py gat -if 5 -hf 10 -lr 0.001 -e 20 -n 5 -bs 128 -bi -s
```

Output

```bash
2023-06-21 01:04:21,305 | INFO  | Malware Detection with GAT model using /home/USER/provninja/provninja/intrusion-detection-system/graph-based/sample-supply-chain-data as input data directory
2023-06-21 01:04:21,305 | INFO  | Structural graph training is turned on
2023-06-21 01:04:21,305 | INFO  | Bidirectional dataset is turned on
2023-06-21 01:04:21,305 | INFO  | 5 Layer GNN. Input Feature Size: 5. Hidden Layer Size(s): 10. Loss Rate: 0.001
2023-06-21 01:04:21,305 | INFO  | LR Scheduler is off
2023-06-21 01:04:21,305 | INFO  | Batch size: 128. Number of workers: 1
2023-06-21 01:04:21,305 | INFO  | Input Device: cpu
2023-06-21 01:04:21,305 | INFO  | Stratified sampler is enabled
2023-06-21 01:04:21,305 | INFO  | Training on 20 epochs...
Done loading data from cached files.
Done loading data from cached files.
Done loading data from cached files.
2023-06-21 01:04:24,705 | INFO  | Length of dataset: 4276
2023-06-21 01:04:24,729 | INFO  | Evaluating on Device: cpu
2023-06-21 01:04:24,730 | INFO  | # Parameters in model: 8426
2023-06-21 01:04:24,730 | INFO  | # Trainable parameters in model: 8426
2023-06-21 01:04:24,731 | INFO  | Stratified sampler enabled
2023-06-21 01:04:27,987 | INFO  | === test stats ===
2023-06-21 01:04:27,987 | INFO  | Number Correct: 742
2023-06-21 01:04:27,988 | INFO  | Number Graphs in test Data: 851
2023-06-21 01:04:27,988 | INFO  | test accuracy: 0.87192
2023-06-21 01:04:27,989 | INFO  | [[530 106]
 [  3 212]]
2023-06-21 01:04:27,994 | INFO  |               precision    recall  f1-score   support

      Benign       0.99      0.83      0.91       636
     Anomaly       0.67      0.99      0.80       215

    accuracy                           0.87       851
   macro avg       0.83      0.91      0.85       851
weighted avg       0.91      0.87      0.88       851
```

## Prov-GAT

* [Prov-GAT](intrusion-detection-system/graph-based/gnnDriver.py)
  * Driver script for Prov-GAT, which is an GNN based IDS that detects anomalous graph using node and edge attributes on top of features used by S-GAT feature. 
  * Download the [sample-supply-chain-data.zip](https://drive.google.com/file/d/1Jz0ZuiZlUEZdAgqlnfmpN2_X0Cms6Sl8/view?usp=sharing) and put it in `intrusion-detection-system/graph-based/` directory. Then unzip it.
  * Sample processed graphs for Supply-Chain APT available in [sample-supply-chain-data](intrusion-detection-system/graph-based/sample-supply-chain-data) directory.
  
Running the Prov-GAT script:

```bash
python gnnDriver.py gat -if 768 -hf 10 -lr 0.001 -e 20 -n 5 -bs 128 -bi
```

Output

```bash
2023-06-21 01:03:12,985 | INFO  | Malware Detection with GAT model using /home/USER/provninja/provninja/intrusion-detection-system/graph-based/sample-supply-chain-data as input data directory
2023-06-21 01:03:12,985 | INFO  | Structural graph training is turned off
2023-06-21 01:03:12,985 | INFO  | Bidirectional dataset is turned on
2023-06-21 01:03:12,985 | INFO  | 5 Layer GNN. Input Feature Size: 768. Hidden Layer Size(s): 10. Loss Rate: 0.001
2023-06-21 01:03:12,985 | INFO  | LR Scheduler is off
2023-06-21 01:03:12,985 | INFO  | Batch size: 128. Number of workers: 1
2023-06-21 01:03:12,985 | INFO  | Input Device: cpu
2023-06-21 01:03:12,985 | INFO  | Stratified sampler is enabled
2023-06-21 01:03:12,985 | INFO  | Training on 20 epochs...
Done loading data from cached files.
Done loading data from cached files.
Done loading data from cached files.
2023-06-21 01:03:16,470 | INFO  | Length of dataset: 4276
2023-06-21 01:03:16,491 | INFO  | Evaluating on Device: cpu
2023-06-21 01:03:16,491 | INFO  | # Parameters in model: 115231
2023-06-21 01:03:16,492 | INFO  | # Trainable parameters in model: 115231
2023-06-21 01:03:16,493 | INFO  | Stratified sampler enabled
2023-06-21 01:03:20,440 | INFO  | === test stats ===
2023-06-21 01:03:20,440 | INFO  | Number Correct: 811
2023-06-21 01:03:20,440 | INFO  | Number Graphs in test Data: 851
2023-06-21 01:03:20,440 | INFO  | test accuracy: 0.95300
2023-06-21 01:03:20,441 | INFO  | [[613  23]
 [ 17 198]]
2023-06-21 01:03:20,447 | INFO  |               precision    recall  f1-score   support

      Benign       0.97      0.96      0.97       636
     Anomaly       0.90      0.92      0.91       215

    accuracy                           0.95       851
   macro avg       0.93      0.94      0.94       851
weighted avg       0.95      0.95      0.95       851
```

## ProvNinja Graph

* [ProvNinja-Graph](intrusion-detection-system/graph-based/provninjaGraph.py)
  * Driver script for ProvNinja-Graph which is an adversarial example generator.
  * Download the [sample-supply-chain-data.zip](https://drive.google.com/file/d/1Jz0ZuiZlUEZdAgqlnfmpN2_X0Cms6Sl8/view?usp=sharing) and put it in `intrusion-detection-system/graph-based/` directory. Then unzip it.
  * Sample processed graphs for Supply-Chain APT available in [sample-supply-chain-data](intrusion-detection-system/graph-based/sample-supply-chain-data) directory.
  * Output will be in directory [adversarial_examples](intrusion-detection-system/graph-based/adversarial_examples).

Running the ProvNinja-Graph script:

```bash
python provninjaGraph.py
```

Output:

```bash
Done loading data from cached files.
Done loading data from cached files.
Done loading data from cached files.
Attacking nd_220323_0105_24
Adversal examples found.
file=nd_220323_0105_24 label=1.0 original pred=0.9951713681221008 new pred=0.006628171540796757
!!!Attack SUCCESSFUL for graph nd_220323_0105_24 :) !!!
saving to adversarial_examples\nd_220323_0105_24



Attacking nd_220323_0111_35
Adversal examples found.
file=nd_220323_0111_35 label=1.0 original pred=0.9999735355377197 new pred=0.19754895567893982
!!!Attack SUCCESSFUL for graph nd_220323_0111_35 :) !!!
saving to adversarial_examples\nd_220323_0111_35



Attacking nd_220323_0109_07
Attack failed for graph nd_220323_0109_07 :(
saving to adversarial_examples\nd_220323_0109_07



...
Attacking nd_220323_0842_53
Adversal examples found.
file=nd_220323_0842_53 label=1.0 original pred=0.9993903636932373 new pred=0.19370944797992706
!!!Attack SUCCESSFUL for graph nd_220323_0842_53 :) !!!
saving to adversarial_examples\nd_220323_0842_53



Detection evaded for 168 / 198 true positive samples
Precision:0.6714285714285714
Recall:0.2186046511627907
F1:0.3298245614035088
```

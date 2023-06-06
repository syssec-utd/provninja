# Path-based IDS

## SIGL

* [sigl](intrusion-detection-system/path-based/sigl.py) [[1]](#references)
  * Driver script for SIGL, which is an Autoencoder based IDS that detects anomalous paths.
  * Sample causal paragraphs and feature vectors for Enterprise APT available in [sample-enterprise-data](intrusion-detection-system/path-based/sample-enterprise-data) directory.
  
Running the SIGL script:

```bash
python sigl.py
```

Output

```bash
2023-06-21 00:46:40,381 | INFO  | **************************Enterprise APT**************************
98/98 [==============================] - 0s 1ms/step
17/17 [==============================] - 0s 3ms/step
2023-06-21 00:46:40,998 | INFO  | Accuracy:0.943609022556391
2023-06-21 00:46:40,998 | INFO  | Precision:0.9125874125874126
2023-06-21 00:46:40,998 | INFO  | Recall:0.981203007518797
2023-06-21 00:46:40,998 | INFO  | F1:0.9456521739130436
2023-06-21 00:46:40,999 | INFO  |
2023-06-21 00:46:41,175 | INFO  | **************************Gadget Enterprise APT**************************
98/98 [==============================] - 0s 1ms/step
17/17 [==============================] - 0s 768us/step
2023-06-21 00:46:41,625 | INFO  | Accuracy:0.5876865671641791
2023-06-21 00:46:41,626 | INFO  | Precision:0.8051948051948052
2023-06-21 00:46:41,626 | INFO  | Recall:0.23134328358208955
2023-06-21 00:46:41,626 | INFO  | F1:0.35942028985507246
```

## ProvDetector

* [provdetector](intrusion-detection-system/path-based/provdetector.py) [[2]](#references)
  * Driver script for ProvDetector, which is an LOF based IDS that detects anomalous paths.
  * Sample causal paragraphs and feature vectors for Enterprise APT available in [sample-enterprise-data](intrusion-detection-system/path-based/sample-enterprise-data) directory.
  
Running the ProvDetector script:

```bash
python provdetector.py
```

Output

```bash
2023-06-21 00:44:41,104 | INFO  | **************************Enterprise APT**************************
2023-06-21 00:44:41,105 | INFO  | Accuracy:0.793233082706767
2023-06-21 00:44:41,105 | INFO  | Precision:1.0
2023-06-21 00:44:41,105 | INFO  | Recall:0.793233082706767
2023-06-21 00:44:41,105 | INFO  | F1:0.8846960167714885
2023-06-21 00:44:41,105 | INFO  |
2023-06-21 00:44:41,105 | INFO  | **************************Gadget Enterprise APT**************************
2023-06-21 00:44:41,106 | INFO  | Accuracy:0.11940298507462686
2023-06-21 00:44:41,106 | INFO  | Precision:1.0
2023-06-21 00:44:41,106 | INFO  | Recall:0.11940298507462686
2023-06-21 00:44:41,106 | INFO  | F1:0.21333333333333335
```

## References 

[1] X. Han, X. Yu, T. Pasquier, et al., “_Sigl: Securing software installations through deep graph learning_,” in
USENIX Security Symposium (SEC), 2021. <br>
[2] Q. Wang, W. U. Hassan, D. Li, et al., “_You Are What
You Do: Hunting Stealthy Malware via Data Provenance Analysis_,” in Network and Distributed System
Security Symposium (NDSS), Feb. 2020.

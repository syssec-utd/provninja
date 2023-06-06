import logging
import os

# data preparation
import pandas as pd
import numpy as np
import pickle

# outlier/anomaly detection
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s\t| %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)

# base file location
base = os.getcwd()
benign = os.path.join(base, "sample-enterprise-data", "benign-fv.csv")
anomaly = os.path.join(base, "sample-enterprise-data", "anomaly-fv.csv")
gadget = os.path.join(base, "sample-enterprise-data", "gadget-fv.csv")


def printStat(CONFUSION_MATRIX):

    result = CONFUSION_MATRIX.ravel()
    if len(result) == 4:
        tn, fp, fn, tp = result
    else:
        tp = result

    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    f1 = (2 * recall * precision) / (recall + precision)
    accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)

    log.info("Accuracy:" + str(accuracy))
    log.info("Precision:" + str(precision))
    log.info("Recall:" + str(recall))
    log.info("F1:" + str(f1))


def filter_outlier(DF_BENIGN_DATA):
    outlier_frac = 0.001
    model = LocalOutlierFactor(contamination=outlier_frac)
    pred = model.fit_predict(DF_BENIGN_DATA)

    inlier_ind = np.where(pred == 1)
    inlier_val = DF_BENIGN_DATA.iloc[inlier_ind]

    return pd.DataFrame(inlier_val)


def save_model(clf, filename):
    with open(filename, "wb") as f:
        pickle.dump(clf, f)


def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def main():
    log.info("Running the LOF Model...")
    log.debug(f"**************************Data Locations**************************")
    log.debug(f"benign path: {benign}")
    log.debug(f"anomaly path: {anomaly}")
    log.debug(f"gadget path: {gadget}")

    df_ben_processed = pd.read_csv(benign, header=None)
    df_ana = pd.read_csv(anomaly, header=None)
    df_gat = pd.read_csv(gadget, header=None)

    # benign data
    # df_ben_processed = filter_outlier(df_ben_processed)

    # model specification
    # model = LocalOutlierFactor(n_neighbors=50, contamination=0.04, novelty=True)
    # model.fit(df_ben_processed)
    #
    # # save the model
    # save_model(model, os.path.join(base, "models", "enterprise_apt_lof.pkl"))

    # load the model
    model = load_model(os.path.join(base, "models", "enterprise_apt_lof.pkl"))

    pred2 = model.predict(df_ana)
    pred3 = model.predict(df_gat)

    log.info("")
    log.info("**************************Enterprise APT**************************")
    # for anomaly
    y_pred_ana = pred2 == 1
    y_true_ana = [1 for _ in range(len(y_pred_ana))]
    printStat(confusion_matrix(y_true_ana, y_pred_ana))
    log.info("")

    log.info("**************************Gadget Enterprise APT**************************")
    # for gadget
    y_pred_gad = pred3 == 1
    y_true_gad = [1 for _ in range(len(y_pred_gad))]
    printStat(confusion_matrix(y_true_gad, y_pred_gad))


if __name__ == "__main__":
    main()

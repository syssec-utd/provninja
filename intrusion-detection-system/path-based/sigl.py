import os
import logging

import numpy as np
import pandas as pd

from keras.layers import Input, Dense
from keras import regularizers
from keras.models import Model, load_model

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# base directory for models
base = os.getcwd()
benign = os.path.join(base, "sample-enterprise-data", "benign-fv.csv")
anomaly = os.path.join(base, "sample-enterprise-data", "anomaly-fv.csv")
gadget = os.path.join(base, "sample-enterprise-data", "gadget-fv.csv")

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s\t| %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
Description: Contains utility functions for the Autoencoder class
""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


class AutoencoderUtils:

    # constructor
    def __init__(self):
        log.debug("AutoencoderUtils init")

    """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
    Input:
    fileNameTrain: file path
    fileNameAbnormal: file path

    Output:
    Returns pandas dataframe x_train, x_test, x_abnormal

    Description:
    helper function to get the data
    """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

    def getData(self, fileNameTrain, fileNameAbnormal):

        x_train_df = pd.read_csv(fileNameTrain)
        x_abnormal_df = pd.read_csv(fileNameAbnormal, header=None)

        x_train = x_train_df.values
        x_abnormal = x_abnormal_df.values

        log.debug("x_train:" + str(len(x_train)))
        log.debug("x_abnormal:" + str(len(x_abnormal)))

        x_train, x_test = train_test_split(x_train, test_size=0.2)

        return x_test, x_train, x_abnormal

    """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
    Input: 
    autoencoder: the autoencoder model
    X_test: test feature vector
    y_test: label
    threshold: given threshold

    Description:
    Helper function to plot the mse of X and threshold for the data
    """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

    def evalaute(self, autoencoder, x, y, threshold):

        tn, fp, fn, tp = 0, 0, 0, 0

        predictions = autoencoder.predict(x)
        mse = np.mean(np.power(x - predictions, 2), axis=1)

        error_df = pd.DataFrame(
            list(
                zip(
                    list(mse.values.reshape(1, len(mse))[0]),
                    list(y.values.reshape(1, len(y))[0]),
                )
            ),
            columns=["reconstruction_error", "true_class"],
        )

        y_pred = [
            1 if e > threshold else 0 for e in error_df.reconstruction_error.values
        ]
        conf_matrix = confusion_matrix(error_df.true_class, y_pred)

        result = conf_matrix.ravel()
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

    """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
    Input: 
    autoencoder: autoencoder model
    X: dataset

    Output:
    the mse threshold

    Description:
    get the mse threshold corresponding to the 80th percentile
    also print out info regarding the threshold
    plot the threshold w.r.t to train data mse
    """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

    def getThreashold(self, autoencoder, x):

        predictions = autoencoder.predict(x)
        mse = np.mean(np.power(x - predictions, 2), axis=1)

        PERCENTILE = 0.93
        threshold = np.quantile(mse, PERCENTILE)

        return threshold

    """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
    Input:
    model: model that will used to get the threshold and do evaluation
    modelName: a string to display
    train: benign training data
    anomaly: anomalous data

    Description:
    Main driver function
    """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

    def driver(self, model, modelName, benign, anomaly):

        log.info(f"**************************{modelName}**************************")
        x_train, x_test, x_abnormal = self.getData(benign, anomaly)

        x_test = x_test[: len(x_abnormal)]
        x_abnormal = x_abnormal[: len(x_test)]

        threshold = self.getThreashold(model, pd.DataFrame(x_train))

        X_test = pd.DataFrame(np.concatenate([x_test, x_abnormal], axis=0))
        Y_test = pd.DataFrame(
            [0 for _ in range(len(x_test))] + [1 for _ in range(len(x_abnormal))]
        )

        self.evalaute(model, X_test, Y_test, threshold)


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
Input: path to train, test, and anomaly dataset
Output: runs the autoencoder and produces a log file along with a model if generated
Description: trains a model or uses a user supplied model to run autoencoder
""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


class Autoencoder:
    """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
    Input: path to train, test, and anomaly dataset
    Description: initializes Autoencoder class
    """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

    def __init__(self, trainFilename, anomolyFilename):
        log.debug("Autoencoder Class Initialization")
        self.trainFilename = trainFilename
        self.anomolyFilename = anomolyFilename
        self.autoencoderUtils = AutoencoderUtils()
        self.name = "enterprise_apt_autoencoder"
        self.modelFilename = os.path.join(base, "models", self.name + ".h5")

    """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
    Input: filename of specified model
    Output: None
    Description: Runs the autoencoder with the specified model
    """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

    def runWithModel(self, modelName):
        log.debug("Running Autoencoder And Generating Model")
        log.debug("RUN NAME: " + str(self.name))
        log.debug("ANOMALY FILEPATH: " + self.anomolyFilename)
        log.debug("MODEL FILEPATH: " + self.modelFilename)
        log.debug("Running Autoencoder With Model: " + str(self.modelFilename))
        model = load_model(self.modelFilename)
        self.autoencoderUtils.driver(
            model, modelName, self.trainFilename, self.anomolyFilename
        )

    """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
    Input: none
    Output: None
    Description: Runs the autoencoder with a model generated from the training data
    """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

    def run(self, modelName):
        print("DEBUG: Autoencoder Running")
        print("DEBUG: Running Autoencoder And Generating Model")
        print("DEBUG: UNIQUE RUN NAME: " + str(self.name))
        print("DEBUG: TRAIN FILEPATH: " + self.trainFilename)
        print("DEBUG: ANOMALY FILEPATH: " + self.anomolyFilename)
        print("DEBUG: MODEL FILEPATH: " + self.modelFilename)

        x_train, x_test, _ = self.autoencoderUtils.getData(
            self.trainFilename, self.anomolyFilename
        )
        model = self.getAutoencoderModel(x_train, x_test)

        self.autoencoderUtils.driver(
            model, modelName, self.trainFilename, self.anomolyFilename
        )

    """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
    Input: training and test data, number of epochs to train for and batch size
    :param X_train: feature vector for training the model
    :param x_test: feature vector for testing the model
    :param numEpochs: number of epochs to run for, default 600
    :param batchSize: batch size for model, default 128
    Output: returns the autoencoder model
    Description: This function is used the training and test data to create an autoencoder model.
    """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

    def getAutoencoderModel(self, x_train, x_test, numberEpochs=30, batchSize=128):
        print("DEBUG: Getting Autoencoder Model")

        # number of features
        input_dim = x_train.shape[1]

        # bottle necking the feature
        encoding_dim = int(input_dim / 2)
        hidden_dim = int(encoding_dim / 4)
        input_layer = Input(shape=(input_dim,))

        # NN shape
        encoder = Dense(
            encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5)
        )(input_layer)
        encoder = Dense(hidden_dim, activation="relu")(encoder)
        decoder = Dense(encoding_dim, activation="relu")(encoder)
        decoder = Dense(input_dim, activation="tanh")(decoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(
            optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]
        )

        autoencoder.fit(
            x_train,
            x_train,
            epochs=numberEpochs,
            batch_size=batchSize,
            shuffle=True,
            validation_data=(x_test, x_test),
            verbose=1,
        )

        # save the auto encoder and then run from it
        autoencoder.save(self.modelFilename)
        return autoencoder


def main():
    log.info("Running the Autoencoder Model...")

    autoencoder = Autoencoder(benign, anomaly)
    # autoencoder.run('Enterprise APT')
    autoencoder.runWithModel("Enterprise APT")

    log.info("")
    autoencoder = Autoencoder(benign, gadget)
    autoencoder.runWithModel("Gadget Enterprise APT")


if __name__ == "__main__":
    main()

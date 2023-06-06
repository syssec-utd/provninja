import logging
import os
import dgl
import warnings

import numpy as np
import tkinter as tk
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plot

from datetime import datetime
from collections.abc import Mapping, Sequence

from dgl.dataloading import GraphDataLoader

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

from samplers import StratifiedBatchSampler

DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 1

# suppress PyTorch cuda warnings
warnings.filterwarnings(
    "ignore",
    message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
)

from dataloaders.AnomalyBenignDataset import AnomalyBenignDataset


def get_binary_train_val_test_datasets(
    dataset_dir_path,
    benign_folder_name,
    anomaly_folder_name,
    node_attributes_map,
    relation_attributes_map,
    bidirection,
    force_reload=False,
    verbose=True,
):
    train_dataset = AnomalyBenignDataset(
        os.path.join(dataset_dir_path, "train"),
        benign_folder_name,
        anomaly_folder_name,
        node_attributes_map,
        relation_attributes_map,
        bidirection=bidirection,
        force_reload=force_reload,
        verbose=verbose,
    )
    val_dataset = AnomalyBenignDataset(
        os.path.join(dataset_dir_path, "validation"),
        benign_folder_name,
        anomaly_folder_name,
        node_attributes_map,
        relation_attributes_map,
        bidirection=bidirection,
        force_reload=force_reload,
        verbose=verbose,
    )
    test_dataset = AnomalyBenignDataset(
        os.path.join(dataset_dir_path, "test"),
        benign_folder_name,
        anomaly_folder_name,
        node_attributes_map,
        relation_attributes_map,
        bidirection=bidirection,
        force_reload=force_reload,
        verbose=verbose,
    )

    return train_dataset, val_dataset, test_dataset


def cal_acc_and_loss(num_correct_predictions, len_of_data, loss_history):
    """
    calculates accuracy and average loss

    :param num_correct_predictions:
    :type num_correct_predictions:
    :param len_of_data:
    :type len_of_data:
    :param loss_history:
    :type loss_history:
    :return:
    :rtype:
    """
    acc = float(num_correct_predictions) / len_of_data
    loss = float(sum(loss_history)) / len(loss_history)
    return acc, loss


def outputInputArguments(
    logger,
    nn_type,
    dataset_dir_path,
    number_epochs,
    number_of_layers,
    input_feature_size,
    hidden_feature_size,
    loss_rate,
    dynamic_lr,
    batch_size,
    num_workers,
    device,
    structural,
    remove_stratified_sampler,
    train_validation_confusion_matrix,
    benign_downsampling_training,
    anomaly_threshold,
    bidirection=False,
):
    logger.info(
        f"Malware Detection with {nn_type.upper()} model "
        f"using {dataset_dir_path} as input data directory"
    )
    logger.info(f'Structural graph training is turned {"on" if structural else "off"}')
    logger.info(f'Bidirectional dataset is turned {"on" if bidirection else "off"}')
    logger.info(
        f"{number_of_layers} Layer GNN. Input Feature Size: {input_feature_size}. Hidden Layer Size(s):"
        f" {hidden_feature_size}. Loss "
        f"Rate:"
        f" {loss_rate}"
    )
    logger.info(f'LR Scheduler is {"on" if dynamic_lr else "off"}')
    logger.info(f"Batch size: {batch_size}. Number of workers: {num_workers}")
    logger.info(f"Input Device: {device}")

    logger.info(
        f'Stratified sampler is {"disabled" if remove_stratified_sampler else "enabled"}'
    )

    if train_validation_confusion_matrix:
        logger.info(f"Outputting Training and Validation Confusion Matrices")
    if benign_downsampling_training:
        logger.info(f"Benign Down Sampling: {benign_downsampling_training}")
    if anomaly_threshold:
        logger.info(
            f"Variable Prediction Threshold for Anomalous graphs have been enabled & set to {anomaly_threshold}"
        )

    logger.info(f"Training on {number_epochs} epochs...")


def outputEpochStats(epoch, train_acc, train_loss, val_acc, val_loss, logger):
    logger.info(
        f"Epoch {epoch}: "
        f"Training Accuracy: {train_acc:.5f}, "
        f"Average Training Loss: {train_loss:.5f}, "
        f"Validation Accuracy: {val_acc:.5f}, "
        f"Average Validation Loss: {val_loss:.5f}"
    )


def outputStats(
    data_type,
    num_correct,
    len_data,
    pred_y_vals,
    true_y_vals,
    dataset_labels,
    logger,
    incorrect_graphs=None,
):
    """
    :param data_type: type of data (train/val/test)
    :type type:
    :param num_correct: correct predictions in test data
    :type num_correct:
    :param len_data: length of test data (how many samples)
    :type len_data:
    :param pred_y_vals: class prediction from the model for each sample in test_data
    :type pred_y_vals: # true_y_vals -> actual label for each sample in test_data
    :param true_y_vals:
    :type true_y_vals:
    :param dataset_labels:
    :type dataset_labels:
    :return:
    :rtype:
    """
    logger.info(f"=== {data_type} stats ===")
    logger.info(f"Number Correct: {num_correct}")

    if incorrect_graphs:
        logger.info(f"Incorrect graph names: {incorrect_graphs}")

    logger.info(f"Number Graphs in {data_type} Data: {len_data}")
    logger.info(f"{data_type} accuracy: {(float(num_correct) / len_data):.5f}")

    logger.info(confusion_matrix(true_y_vals, pred_y_vals))

    if data_type == "test":
        logger.info(
            classification_report(true_y_vals, pred_y_vals, target_names=dataset_labels)
        )


def outputMatplotModelResults(
    training_accuracy_hist,
    validation_accuracy_hist,
    training_loss_hist,
    validation_loss_hist,
):
    fig, axs = plot.subplots(2)
    fig.suptitle("Model Results")
    axs[0].set_title("Accuracy")
    axs[0].plot(training_accuracy_hist, "ro", label="Training Accuracy")
    axs[0].plot(validation_accuracy_hist, "bo", label="Validation Accuracy")
    axs[0].legend()

    axs[1].set_title("Loss")
    axs[1].plot(training_loss_hist, "ro", label="Training Loss")
    axs[1].plot(validation_loss_hist, "bo", label="Validation Loss")
    axs[1].legend()

    plot.show()


def outputPerEpochStatsToSummaryWriter(
    summaryWriter, epoch, train_acc, train_loss, val_acc, val_loss
):
    """
    output training acc, training loss, val accuracy, and val loss to TensorBoard

    :param summaryWriter:
    :type summaryWriter:
    :param epoch:
    :type epoch:
    :param train_acc:
    :type train_acc:
    :param train_loss:
    :type train_loss:
    :param val_acc:
    :type val_acc:
    :param val_loss:
    :type val_loss:
    :return:
    :rtype:
    """
    summaryWriter.add_scalar("Training Accuracy", train_acc, epoch)
    summaryWriter.add_scalar("Average Training Loss", train_loss, epoch)
    summaryWriter.add_scalar("Validation Accuracy", val_acc, epoch)
    summaryWriter.add_scalar("Average Validation Loss", val_loss, epoch)


def outputTestStatsToSummaryWriter(
    summaryWriter, test_num_correct, len_test_data, len_dataset
):
    summaryWriter.add_scalar("Test Accuracy", float(test_num_correct) / len_test_data)
    summaryWriter.add_scalar("Test Number Correct", test_num_correct)
    summaryWriter.add_scalar("Length of Test Dataset", len_test_data)
    summaryWriter.add_scalar("Length of Entire Dataset", len_dataset)


# Gets number of parameters in PyTorch model
def getTotalParams(model):
    return sum(p.numel() for p in model.parameters())


# Gets number of TRAINABLE parameters in PyTorch model
def getTotalTrainableParams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def saveProbabilityDistibution(data):
    bins = np.arange(0, 1, 0.01)  # fixed bin size

    plot.xlim([min(data) - 0.01, max(data) + 0.01])

    plot.hist(data, bins=bins, alpha=0.5)
    plot.title("Probability distribution of anomaly (fixed bin size)")
    plot.xlabel("Probability of anomaly (bin size = 0.01)")
    plot.ylabel("count")

    plot.savefig(
        os.path.join(
            "images", f"prob_dist_{datetime.now().strftime('%b%d_%H-%M-%S')}.png"
        )
    )


# structural_flag = true when training on structural graph data, else otherwise
# returns correct feature aggregation function
def getAndcheckStructrualTrainingState(structural_flag, feature_aggregation_func):
    if structural_flag:
        if feature_aggregation_func:
            raise AssertionError(
                "Training on only structural graph data. Feature aggregation function should be None"
            )
        feature_aggregation_func = lambda _: None

    if not structural_flag and feature_aggregation_func is None:
        raise AssertionError(
            "Training w/ predefined node feature set for each graph. Feature aggregation function"
            "sbould not be None"
        )

    return feature_aggregation_func


# model: is the model to train with
# summaryWriter: SummaryWriter object to write to (for tensorboard support)
# train_dataset: DGLDataset dataset object containing graph data for training
# validation_dataset: DGLDataset dataset object containing graph data for validation
# test_dataset: DGLDataset dataset object containing graph data for testing
# loss_rate: loss rate used in optimizer
# dynamic_lr: if we wish to adjust learning rate based on number of epochs
# epochs: number of epochs to train on
# logger: logger object to log to
# num_workers: Number of python workers (ie separate threads) to run on
# batch_size: How many graphs we want each minibatch to have
def train_binary_graph_classification(
    model,
    summaryWriter,
    train_dataset,
    validation_dataset,
    test_dataset,
    loss_rate,
    epochs,
    logger,
    structural,
    feature_aggregation_func=None,
    dynamic_lr=False,
    num_workers=DEFAULT_NUM_WORKERS,
    batch_size=DEFAULT_BATCH_SIZE,
    device=None,
    remove_stratified_sampler=False,
    train_validation_confusion_matrix=False,
    benign_downsampling_training=None,
    anomaly_threshold=None,
):

    # make sure that if structural is on, then a feature aggregation func must be provided to the func
    # if structural is off, make sure no feature aggregation func is present
    feature_aggregation_func = getAndcheckStructrualTrainingState(
        structural, feature_aggregation_func
    )

    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

    logger.info(f"Training on Device: {device}")
    outputBinaryCounts(train_dataset, validation_dataset, test_dataset, logger)

    logger.info(f"# Parameters in model: {getTotalParams(model)}")
    logger.info(f"# Trainable parameters in model: {getTotalTrainableParams(model)}")

    model.to(device)

    if remove_stratified_sampler:
        logger.info(f"Stratified sampler not enabled")

        # this is just for consistency sake so the user doesnt think we can have batch size > 1 when rss=True
        assert (
            batch_size == 1
        ), "Batch size must be 1 for stratified sampler to be disabled"

        train_dataloader = GraphDataLoader(
            train_dataset,
            collate_fn=collate_func,
            num_workers=num_workers,
            shuffle=True,
        )
        val_dataloader = GraphDataLoader(
            validation_dataset,
            collate_fn=collate_func,
            num_workers=num_workers,
            shuffle=False,
        )
        test_dataloader = GraphDataLoader(
            test_dataset,
            collate_fn=collate_func,
            num_workers=num_workers,
            shuffle=False,
        )
    else:
        logger.info(f"Stratified sampler enabled")

        train_batch_sampler = StratifiedBatchSampler(
            train_dataset.labels, batch_size=batch_size, shuffle=True
        )
        val_batch_sampler = StratifiedBatchSampler(
            validation_dataset.labels, batch_size=batch_size, shuffle=False
        )
        test_batch_sampler = StratifiedBatchSampler(
            test_dataset.labels, batch_size=batch_size, shuffle=False
        )

        train_dataloader = GraphDataLoader(
            train_dataset,
            collate_fn=collate_func,
            num_workers=num_workers,
            batch_sampler=train_batch_sampler,
        )
        val_dataloader = GraphDataLoader(
            validation_dataset,
            collate_fn=collate_func,
            num_workers=num_workers,
            batch_sampler=val_batch_sampler,
        )
        test_dataloader = GraphDataLoader(
            test_dataset,
            collate_fn=collate_func,
            num_workers=num_workers,
            batch_sampler=test_batch_sampler,
        )

    # compute class weights for sample weighting (
    # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4)
    dataset_labels_numpy = th.cat(
        (train_dataset.labels, validation_dataset.labels, test_dataset.labels), 0
    ).numpy()

    assert 0 in dataset_labels_numpy, "Dataset must contain at least one 0 label"
    assert 1 in dataset_labels_numpy, "Dataset must contain at least one 1 label"

    class_weights = compute_class_weight(
        class_weight="balanced", classes=[0, 1], y=dataset_labels_numpy
    )

    class_weights = th.tensor(class_weights, dtype=th.float, device=device)
    logger.info(f"Computed weights for loss function: {class_weights}")

    opt = th.optim.Adam(model.parameters(), lr=loss_rate)

    if dynamic_lr:
        scheduler = ReduceLROnPlateau(opt, "min", patience=5, verbose=True)

    training_accuracy_hist = []
    validation_accuracy_hist = []

    avg_training_loss_hist = []
    avg_validation_loss_hist = []

    for epoch in range(epochs):
        train_loss_history = []
        training_num_correct = 0
        train_y_pred = []
        train_y_true = []
        train_incorrect_graphs = (
            [] if remove_stratified_sampler else None
        )  # if remove_stratified_sampler is turned on
        # lets print incorrect graph names since
        # batch size would = 1 anyways
        for batched_graphs, batched_labels in train_dataloader:
            if benign_downsampling_training and not remove_stratified_sampler:

                unbatched_graphs = dgl.unbatch(batched_graphs)
                benign_unbatched_graphs = np.array(
                    [
                        graph
                        for graph, label in zip(unbatched_graphs, batched_labels)
                        if label == 0
                    ]
                )
                anomaly_unbatched_graphs = np.array(
                    [
                        graph
                        for graph, label in zip(unbatched_graphs, batched_labels)
                        if label == 1
                    ]
                )

                chosen_benign_unbatch_graphs_indx = th.randperm(
                    len(benign_unbatched_graphs)
                )[: int(len(benign_unbatched_graphs) * benign_downsampling_training)]
                chosen_benign_unbatch_graphs = benign_unbatched_graphs[
                    chosen_benign_unbatch_graphs_indx.numpy().astype(int)
                ]

                batched_graphs = dgl.batch(
                    list(
                        np.concatenate(
                            (chosen_benign_unbatch_graphs, anomaly_unbatched_graphs),
                            axis=None,
                        )
                    )
                )
                batched_labels = th.tensor(
                    [
                        0.0
                        for _ in range(
                            int(
                                len(benign_unbatched_graphs)
                                * benign_downsampling_training
                            )
                        )
                    ]
                    + [1.0 for _ in range(len(anomaly_unbatched_graphs))]
                )

                logger.info(
                    f"Benign Batched Graphs: {len([label for label in batched_labels if label == 0])} Anomaly Batched Graphs: {len([label for label in batched_labels if label == 1])}"
                )

            batched_graphs = batched_graphs.to(device)
            batched_labels = th.reshape(
                batched_labels, (batched_labels.shape[0], 1)
            ).to(device)

            # prediction
            pred = model(batched_graphs, feature_aggregation_func(batched_graphs))

            # calculate loss
            # we need to calculate the weight for each batch element before we pass it into F.binary_cross_entropy
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html#torch.nn.functional.binary_cross_entropy
            with batched_graphs.local_scope():
                loss_weights = th.tensor(
                    [
                        class_weights[1] if rounded_prediction else class_weights[0]
                        for rounded_prediction in th.round(pred)
                    ],
                    dtype=th.float,
                    device=device,
                ).reshape((-1, 1))

            loss = F.binary_cross_entropy(pred, batched_labels, weight=loss_weights)
            train_loss_history.append(loss.item())

            # get prediction
            if anomaly_threshold:
                pred = (pred > anomaly_threshold).float()
            else:
                pred = th.round(pred)

            num_correct = (pred == batched_labels).sum().item()
            training_num_correct += num_correct

            if (
                remove_stratified_sampler and num_correct == 0
            ):  # batch size = 1, so if training_num_correct
                # is 0, then it misclassified
                train_incorrect_graphs.append(
                    (
                        batched_graphs.folder_name,
                        f"misclassified as {'benign' if batched_labels[0].item() else 'anomaly'}",
                    )
                )

            train_y_pred = train_y_pred + th.reshape(pred, (pred.shape[0],)).tolist()
            train_y_true = train_y_true + batched_labels.tolist()

            # optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

        if train_validation_confusion_matrix:
            outputStats(
                "training",
                training_num_correct,
                len(train_dataset),
                train_y_pred,
                train_y_true,
                ["Benign", "Anomaly"],
                logger,
                incorrect_graphs=train_incorrect_graphs,
            )

        validation_loss_history = []
        validation_num_correct = 0
        validation_y_pred = []
        validation_y_true = []
        validation_incorrect_graphs = [] if remove_stratified_sampler else None
        for batched_graphs, batched_labels in val_dataloader:
            batched_graphs = batched_graphs.to(device)
            batched_labels = th.reshape(
                batched_labels, (batched_labels.shape[0], 1)
            ).to(device)

            with batched_graphs.local_scope():
                pred = model(batched_graphs, feature_aggregation_func(batched_graphs))

                # we need to calculate the weight for each batch element before we pass it into F.binary_cross_entropy
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html#torch.nn.functional.binary_cross_entropy
                loss_weights = th.tensor(
                    [
                        class_weights[1] if rounded_prediction else class_weights[0]
                        for rounded_prediction in th.round(pred)
                    ],
                    dtype=th.float,
                    device=device,
                ).reshape(-1, 1)

                loss = F.binary_cross_entropy(pred, batched_labels, weight=loss_weights)
                validation_loss_history.append(loss.item())

                # for binary classification, use rounding
                if anomaly_threshold:
                    pred = (pred > anomaly_threshold).float()
                else:
                    pred = th.round(pred)

                num_correct = (pred == batched_labels).sum().item()
                validation_num_correct += num_correct

                if remove_stratified_sampler and num_correct == 0:
                    validation_incorrect_graphs.append(
                        (
                            batched_graphs.folder_name,
                            f"misclassified as {'benign' if batched_labels[0].item() else 'anomaly'}",
                        )
                    )

                validation_y_pred = (
                    validation_y_pred + th.reshape(pred, (pred.shape[0],)).tolist()
                )
                validation_y_true = validation_y_true + batched_labels.tolist()

        if train_validation_confusion_matrix:
            outputStats(
                "validation",
                validation_num_correct,
                len(validation_dataset),
                validation_y_pred,
                validation_y_true,
                ["Benign", "Anomaly"],
                logger,
                incorrect_graphs=validation_incorrect_graphs,
            )

        training_accuracy, avg_training_loss = cal_acc_and_loss(
            training_num_correct, len(train_dataset), train_loss_history
        )
        validation_accuracy, avg_validation_loss = cal_acc_and_loss(
            validation_num_correct, len(validation_dataset), validation_loss_history
        )

        if dynamic_lr:
            scheduler.step(avg_validation_loss)

        outputEpochStats(
            epoch,
            training_accuracy,
            avg_training_loss,
            validation_accuracy,
            avg_validation_loss,
            logger,
        )

        training_accuracy_hist.append(training_accuracy)
        avg_training_loss_hist.append(avg_training_loss)
        validation_accuracy_hist.append(validation_accuracy)
        avg_validation_loss_hist.append(avg_validation_loss)

        outputPerEpochStatsToSummaryWriter(
            summaryWriter,
            epoch,
            training_accuracy,
            avg_training_loss,
            validation_accuracy,
            avg_validation_loss,
        )

    test_num_correct = 0
    test_y_pred = []
    test_y_true = []
    anomaly_prob = []
    test_incorrect_graphs = [] if remove_stratified_sampler else None
    for batched_graphs, batched_labels in test_dataloader:
        batched_graphs = batched_graphs.to(device)
        batched_labels = batched_labels.to(device)

        pred = model(batched_graphs, feature_aggregation_func(batched_graphs))
        if anomaly_threshold:
            pred = (pred > anomaly_threshold).float()
        else:
            anomaly_prob.append(pred[batched_labels.long() == 1].view(-1).tolist())

            pred = th.round(pred)

        pred = th.reshape(pred, (pred.shape[0],))

        num_correct = (pred == batched_labels).sum().item()
        test_num_correct += num_correct

        if remove_stratified_sampler and num_correct == 0:
            test_incorrect_graphs.append(
                (
                    batched_graphs.folder_name,
                    f"misclassified as {'benign' if batched_labels[0].item() else 'anomaly'}",
                )
            )

        test_y_pred = test_y_pred + pred.tolist()
        test_y_true = test_y_true + batched_labels.tolist()

    if not anomaly_threshold:
        saveProbabilityDistibution(
            [round(val, 2) for vals in anomaly_prob for val in vals]
        )

    outputStats(
        "test",
        test_num_correct,
        len(test_dataset),
        test_y_pred,
        test_y_true,
        ["Benign", "Anomaly"],
        logger,
        incorrect_graphs=test_incorrect_graphs,
    )
    outputTestStatsToSummaryWriter(
        summaryWriter,
        test_num_correct,
        len(test_dataset),
        len(train_dataset) + len(validation_dataset) + len(test_dataset),
    )

    try:
        outputMatplotModelResults(
            training_accuracy_hist,
            validation_accuracy_hist,
            avg_training_loss_hist,
            avg_validation_loss_hist,
        )
    except tk.TclError:
        logger.info(f"could not connect to display (maybe running in server)")


def evaluate_binary_graph_classification(
    model,
    summaryWriter,
    test_dataset,
    logger,
    structural,
    num_workers=DEFAULT_NUM_WORKERS,
    batch_size=DEFAULT_BATCH_SIZE,
    feature_aggregation_func=None,
    device=None,
    remove_stratified_sampler=False,
    anomaly_threshold=None,
):

    # make sure that if structural is on, then a feature aggregation func must be provided to the func
    # if structural is off, make sure no feature aggregation func is present
    feature_aggregation_func = getAndcheckStructrualTrainingState(
        structural, feature_aggregation_func
    )

    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

    logger.info(f"Evaluating on Device: {device}")

    logger.info(f"# Parameters in model: {getTotalParams(model)}")
    logger.info(f"# Trainable parameters in model: {getTotalTrainableParams(model)}")

    model.to(device)

    if remove_stratified_sampler:
        logger.info(f"Stratified sampler not enabled")

        test_dataloader = GraphDataLoader(
            test_dataset,
            collate_fn=collate_func,
            num_workers=num_workers,
            shuffle=False,
        )
    else:
        logger.info(f"Stratified sampler enabled")

        test_batch_sampler = StratifiedBatchSampler(
            test_dataset.labels, batch_size=batch_size, shuffle=False
        )

        test_dataloader = GraphDataLoader(
            test_dataset,
            collate_fn=collate_func,
            num_workers=num_workers,
            batch_sampler=test_batch_sampler,
        )

    test_num_correct = 0
    test_y_pred = []
    test_y_true = []
    anomaly_prob = []
    test_incorrect_graphs = [] if remove_stratified_sampler else None
    for batched_graphs, batched_labels in test_dataloader:
        batched_graphs = batched_graphs.to(device)
        batched_labels = batched_labels.to(device)

        pred = model(batched_graphs, feature_aggregation_func(batched_graphs))
        if anomaly_threshold:
            pred = (pred > anomaly_threshold).float()
        else:
            anomaly_prob.append(pred[batched_labels.long() == 1].view(-1).tolist())

            pred = th.round(pred)

        pred = th.reshape(pred, (pred.shape[0],))

        num_correct = (pred == batched_labels).sum().item()
        test_num_correct += num_correct

        if remove_stratified_sampler and num_correct == 0:
            test_incorrect_graphs.append(
                (
                    batched_graphs.folder_name,
                    f"misclassified as {'benign' if batched_labels[0].item() else 'anomaly'}",
                )
            )

        test_y_pred = test_y_pred + pred.tolist()
        test_y_true = test_y_true + batched_labels.tolist()

    if not anomaly_threshold:
        saveProbabilityDistibution(
            [round(val, 2) for vals in anomaly_prob for val in vals]
        )

    outputStats(
        "test",
        test_num_correct,
        len(test_dataset),
        test_y_pred,
        test_y_true,
        ["Benign", "Anomaly"],
        logger,
        incorrect_graphs=test_incorrect_graphs,
    )
    outputTestStatsToSummaryWriter(
        summaryWriter, test_num_correct, len(test_dataset), len(test_dataset)
    )


def outputBinaryCounts(train_dataset, validation_dataset, test_dataset, logger):
    logger.info(
        f"Number benign in training dataset: {sum(label.item() == 0 for _, label in train_dataset)}"
    )
    logger.info(
        f"Number anomaly in training dataset: {sum(label.item() == 1 for _, label in train_dataset)}"
    )

    logger.info(
        f"Number benign in validation dataset: {sum(label.item() == 0 for _, label in validation_dataset)}"
    )
    logger.info(
        f"Number anomaly in validation dataset: {sum(label.item() == 1 for _, label in validation_dataset)}"
    )

    logger.info(
        f"Number benign in test dataset: {sum(label.item() == 0 for _, label in test_dataset)}"
    )
    logger.info(
        f"Number anomaly in test dataset: {sum(label.item() == 1 for _, label in test_dataset)}"
    )


def outputClassCounts(train_dataset, validation_dataset, test_dataset, logger):
    assert (
        train_dataset.label_types
        == validation_dataset.label_types
        == test_dataset.label_types
    ), ("Dataset label " "types should be " "the same.")

    dataset_label_types = train_dataset.label_types  # use one of label types

    unique, counts = np.unique(train_dataset.labels, return_counts=True)
    train_count_dict = dict(zip(unique, counts))

    unique, counts = np.unique(validation_dataset.labels, return_counts=True)
    validation_count_dict = dict(zip(unique, counts))

    unique, counts = np.unique(test_dataset.labels, return_counts=True)
    test_count_dict = dict(zip(unique, counts))

    for i in range(len(dataset_label_types)):
        dataset_label = dataset_label_types[i]
        if i in train_count_dict:
            logger.info(
                f"Train Dataset contains {train_count_dict[i]} labels of {dataset_label}"
            )
        else:
            logger.info(f"Train Dataset contains 0 labels of {dataset_label}")

    for i in range(len(dataset_label_types)):
        dataset_label = dataset_label_types[i]
        if i in validation_count_dict:
            logger.info(
                f"Validation Dataset contains {validation_count_dict[i]} labels of {dataset_label}"
            )
        else:
            logger.info(f"Validation Dataset contains 0 labels of {dataset_label}")

    for i in range(len(dataset_label_types)):
        dataset_label = dataset_label_types[i]
        if i in test_count_dict:
            logger.info(
                f"Test Dataset contains {test_count_dict[i]} labels of {dataset_label}"
            )
        else:
            logger.info(f"Test Dataset contains 0 labels of {dataset_label}")


# Based upon dgl.dataloading.GraphCollator.collate
# used for dataloader collation & to inject folder name into a batched graph
def collate_func(items):
    elem = items[0]
    elem_type = type(elem)
    if isinstance(elem, dgl.DGLHeteroGraph):
        batched_graphs = dgl.batch(items)
        batched_graphs.folder_name = (
            elem.folder_name
        )  # assign folder name for the batch of graphs to be the name
        # of the first graph
        return batched_graphs
    elif th.is_tensor(elem):
        return th.stack(items, 0)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            return collate_func([F.tensor(b) for b in items])
        elif elem.shape == ():  # scalars
            return th.tensor(items)
    elif isinstance(elem, float):
        return th.tensor(items, dtype=th.float64)
    elif isinstance(elem, int):
        return th.tensor(items)
    elif isinstance(elem, (str, bytes)):
        return items
    elif isinstance(elem, Mapping):
        return {key: collate_func([d[key] for d in items]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate_func(samples) for samples in zip(*items)))
    elif isinstance(elem, Sequence):
        # check to make sure that the elements in batch have consistent size
        item_iter = iter(items)
        elem_size = len(next(item_iter))
        if not all(len(elem) == elem_size for elem in item_iter):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*items)
        return [collate_func(samples) for samples in transposed]

    raise TypeError("collate_func encountered an unexpected input type")

import argparse
import logging
import os

from datetime import datetime

import torch

from nn_types.gat import KLayerHeteroRGAT
from gnnUtils import *
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)
date_time = datetime.now().strftime("%b%d_%H-%M-%S")

# dir to store outputs
os.makedirs("logs", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Graph Classifier for Prov Graph Dataset (ie benign or malicious)
class BinaryHeteroClassifier(nn.Module):
    def __init__(
        self,
        gnn_type,
        num_layers,
        input_feature_size,
        hidden_dimension_size,
        graph_relation_list,
        graph_node_types,
        structural=True,
    ):
        super(BinaryHeteroClassifier, self).__init__()
        assert gnn_type in ["gat"], "Supported GNN types are [gat]"

        self.gnn = KLayerHeteroRGAT(
            num_layers,
            input_feature_size,
            hidden_dimension_size,
            hidden_dimension_size,
            graph_relation_list,
            graph_node_types,
            structural=structural,
        )

        self.classifier = nn.Linear(hidden_dimension_size, 1)

    def forward(self, graph, feature_set=None, eweight=None):
        x = self.gnn(graph, feature_set, eweight=eweight)

        # classify graph as benign/malicious
        with graph.local_scope():
            graph.ndata["x"] = x
            hg = 0
            for ntype in graph.ntypes:
                if graph.num_nodes(ntype):
                    hg = hg + dgl.mean_nodes(graph, "x", ntype=ntype)

            return th.sigmoid(self.classifier(hg))  # output probability

    def get_node_features(self, graph):
        return self.gnn.get_node_features(graph)


def main():
    parser = argparse.ArgumentParser(
        description="Runner script for Heterogeneous GNNs."
    )
    parser.add_argument(
        "nn_type",
        type=str,
        choices=["gcn", "gat", "mlp"],
        help="Type of neural network used",
    )
    parser.add_argument(
        "-if", "--input_feature", type=int, help="Input feature size.", required=True
    )
    parser.add_argument(
        "-hf", "--hidden_feature", type=int, help="Hidden feature size.", required=True
    )
    parser.add_argument(
        "-lr", "--loss_rate", type=float, default=0.01, help="Loss rate.", required=True
    )
    parser.add_argument(
        "-e", "--epochs", type=int, help="Number of epochs.", required=True
    )
    parser.add_argument(
        "-n",
        "--layers",
        type=int,
        default=2,
        help="Number of layers in the GNN.",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Number of python workers (ie separate threads) to run on",
        required=False,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        help="How many graphs we want each minibatch to have",
        required=True,
    )
    parser.add_argument(
        "-dlr",
        "--dynamic_lr",
        help="Add this flag if you wish to adjust LR based upon num epochs",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-bi",
        "--bidirection",
        help="Add this flag if you wish to train with bidirectional graphs",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--force_reload",
        help="Reload the dataset without using cached data",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run. Options are cpu, cuda, cuda:N. N is the index of the cuda device which can be fetched using nvidia-smi command.",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--structural",
        help="Train with ONLY structural graph information (i.e. do not use node features)",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-rss",
        "--remove_stratified_sampler",
        help="Add this flag if you wish to remove stratified sampling",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-tvcm",
        "--train_validation_confusion_matrix",
        help="Add this flag if you want to print confusion matrix for train and validation dataset",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-bdst",
        "--benign_downsampling_training",
        type=float,
        default=0.0,
        help="A percentage for benign downsampling for training [0.0-1.0]",
        required=False,
    )
    parser.add_argument(
        "-at",
        "--anomaly_threshold",
        type=float,
        default=0.0,
        help="Threshold for classification of anomalous graphs [0.0-1.0]",
        required=False,
    )
    parser.add_argument(
        "-vpta",
        "--variable_pred_threshold_anomaly",
        type=float,
        default=0.0,
        help="Variable Prediction Threshold for Anomaly",
        required=False,
    )

    parsed_arguments = parser.parse_args()

    neural_network_type = parsed_arguments.nn_type.lower()
    dataset_dir_path = os.path.join(os.getcwd(), "sample-supply-chain-data")
    input_feature_size = parsed_arguments.input_feature
    hidden_feature_size = parsed_arguments.hidden_feature
    loss_rate = parsed_arguments.loss_rate
    epochs = parsed_arguments.epochs
    num_layers = parsed_arguments.layers
    num_workers = parsed_arguments.workers
    batch_size = parsed_arguments.batch_size
    dynamic_lr = parsed_arguments.dynamic_lr
    bidirection = parsed_arguments.bidirection
    structural = parsed_arguments.structural
    remove_stratified_sampler = parsed_arguments.remove_stratified_sampler
    force_reload = parsed_arguments.force_reload
    train_validation_confusion_matrix = (
        parsed_arguments.train_validation_confusion_matrix
    )
    benign_downsampling_training = parsed_arguments.benign_downsampling_training
    anomaly_threshold = parsed_arguments.anomaly_threshold

    log_name = (
        f"{neural_network_type}_{input_feature_size}_{hidden_feature_size}_{loss_rate}"
        f"_{epochs}_{num_layers}_{batch_size}{'_bidirection' if bidirection else ''}"
    )

    global device

    if parsed_arguments.device is not None:
        device = th.device(parsed_arguments.device)
    else:
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s\t| %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(os.getcwd(), "logs", log_name + ".log"),
                mode="w",
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )

    outputInputArguments(
        log,
        neural_network_type,
        dataset_dir_path,
        epochs,
        num_layers,
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
        bidirection=bidirection,
    )

    # load dataset
    if structural:
        node_attributes = {"ProcessNode": [], "SocketChannelNode": [], "FileNode": []}
    else:
        # string embedding
        node_attributes = {
            "ProcessNode": ["EXE_NAME"],
            "SocketChannelNode": ["LOCAL_INET_ADDR"],
            "FileNode": ["FILENAME_SET"],
        }

    relation_attributes = {
        ("ProcessNode", "PROC_CREATE", "ProcessNode"): [],
        ("ProcessNode", "READ", "FileNode"): [],
        ("ProcessNode", "WRITE", "FileNode"): [],
        ("ProcessNode", "FILE_EXEC", "FileNode"): [],
        ("ProcessNode", "WRITE", "FileNode"): [],
        ("ProcessNode", "WRITE", "SocketChannelNode"): [],
        ("ProcessNode", "READ", "SocketChannelNode"): [],
        ("ProcessNode", "IP_CONNECTION_EDGE", "ProcessNode"): [],
        ("ProcessNode", "IP_CONNECTION_EDGE", "FileNode"): [],
    }

    # This is for training with predefined node features (ie we are not ONLY training on structural data)
    # Given a graph (or a batched graph), return the node features for each node type in the graph
    # that we wish to supply to the model to train with
    def feature_aggregation_function(graph):
        return {
            "FileNode": graph.nodes["FileNode"].data["FILENAME_SET"]
            if graph.num_nodes("FileNode")
            else torch.empty(0),
            "ProcessNode": graph.nodes["ProcessNode"].data["EXE_NAME"]
            if graph.num_nodes("ProcessNode")
            else torch.empty(0),
            "SocketChannelNode": graph.nodes["SocketChannelNode"].data[
                "LOCAL_INET_ADDR"
            ]
            if graph.num_nodes("SocketChannelNode")
            else torch.empty(0),
        }

    agg_func = None if structural else feature_aggregation_function

    # add inverse relationships
    if bidirection:
        for relation_attribute in list(relation_attributes.keys()):
            flipped_relation = relation_attribute[::-1]
            if flipped_relation not in relation_attributes:
                relation_attributes[flipped_relation] = relation_attributes[
                    relation_attribute
                ]

    train_dataset, val_dataset, test_dataset = get_binary_train_val_test_datasets(
        dataset_dir_path,
        "benign",
        "anomaly",
        node_attributes,
        relation_attributes,
        bidirection=bidirection,
        force_reload=force_reload,
        verbose=True,
    )

    dataset_length = len(train_dataset) + len(val_dataset) + len(test_dataset)
    log.info(f"Length of dataset: {dataset_length}")

    writer = SummaryWriter(os.path.join(os.getcwd(), "runs", log_name))

    model = BinaryHeteroClassifier(
        neural_network_type,
        num_layers,
        input_feature_size,
        hidden_feature_size,
        list(relation_attributes.keys()),
        list(node_attributes.keys()),
        structural=structural,
    )

    # train_binary_graph_classification(
    #     model,
    #     writer,
    #     train_dataset,
    #     val_dataset,
    #     test_dataset,
    #     loss_rate,
    #     epochs,
    #     log,
    #     structural,
    #     feature_aggregation_func=agg_func,
    #     dynamic_lr=dynamic_lr,
    #     num_workers=num_workers,
    #     batch_size=batch_size,
    #     device=device,
    #     remove_stratified_sampler=remove_stratified_sampler,
    #     train_validation_confusion_matrix=train_validation_confusion_matrix,
    #     benign_downsampling_training=benign_downsampling_training,
    #     anomaly_threshold=anomaly_threshold)
    #
    # th.save(model.state_dict(),
    #         os.path.join(os.getcwd(), 'models', log_name + '.bin'))

    model.load_state_dict(
        th.load(os.path.join(os.getcwd(), "models", log_name + ".bin"), map_location=device)
    )
    model.eval()

    evaluate_binary_graph_classification(
        model,
        writer,
        test_dataset,
        log,
        structural,
        num_workers,
        batch_size,
        feature_aggregation_func=agg_func,
        device=device,
        remove_stratified_sampler=remove_stratified_sampler,
        anomaly_threshold=anomaly_threshold,
    )

    writer.close()


if __name__ == "__main__":
    main()

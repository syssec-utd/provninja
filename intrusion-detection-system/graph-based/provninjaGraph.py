import argparse
import logging
import os
import copy
import numpy as np
import pickle
import json
import sys

import torch

from gnnDriver import BinaryHeteroClassifier
from gnnUtils import get_binary_train_val_test_datasets

from transformers import AutoTokenizer, AutoModel
from pathlib import PureWindowsPath, Path

import plotly.io as pio

pio.templates.default = "plotly_white"

bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
bert_model = AutoModel.from_pretrained("microsoft/codebert-base")

if len(sys.argv) > 1 and sys.argv[1] == 'structural':
    train_command_line = "gat -if 5 -hf 10 -lr 0.001 -e 20 -n 5 -bs 128 -bi -s".split(" ")
    print("structural configuration")
else:
    train_command_line = "gat -if 768 -hf 10 -lr 0.001 -e 20 -n 5 -bs 128 -bi".split(" ")


DEFAULT_NUM_WORKERS = 10

parser = argparse.ArgumentParser(description="Runner script for ProvNinja-Graph.")

parser.add_argument(
    "nn_type", type=str, choices=["gat"], help="Type of neural network used"
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
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs.", required=True)
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
    help="Train with ONLY structural graph information (ie do not use node features)",
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
    help="A percentage for benign downsampling for training",
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

parsed_arguments = parser.parse_args(train_command_line)

neural_network_type = parsed_arguments.nn_type.lower()
dataset_dir_path = os.path.join(os.getcwd(), "sample-supply-chain-data")
input_feature_size = parsed_arguments.input_feature
hidden_feature_size = parsed_arguments.hidden_feature
loss_rate = parsed_arguments.loss_rate
epochs = parsed_arguments.epochs
num_layers = parsed_arguments.layers
num_workers = parsed_arguments.workers
batch_size = parsed_arguments.batch_size
bidirection = parsed_arguments.bidirection
force_reload = parsed_arguments.force_reload
structural = parsed_arguments.structural
remove_stratified_sampler = parsed_arguments.remove_stratified_sampler
train_validation_confusion_matrix = parsed_arguments.train_validation_confusion_matrix
benign_downsampling_training = parsed_arguments.benign_downsampling_training
variable_pred_threshold_anomaly = parsed_arguments.variable_pred_threshold_anomaly

log = logging.getLogger(__name__)

log_name = (
    f"{neural_network_type}_{input_feature_size}_{hidden_feature_size}_{loss_rate}"
    f"_{epochs}_{num_layers}_{batch_size}{'_bidirection' if bidirection else ''}"
)

with Path('gadget_files/gadget-chain.json').open('r') as gadget_chains:
    gadget_dict = json.load(gadget_chains)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if parsed_arguments.device is not None:
    device = torch.device(parsed_arguments.device)

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


def feature_aggregation_function(graph):
    return {
        "FileNode": graph.nodes["FileNode"].data["FILENAME_SET"]
        if graph.num_nodes("FileNode")
        else torch.empty(0),
        "ProcessNode": graph.nodes["ProcessNode"].data["EXE_NAME"]
        if graph.num_nodes("ProcessNode")
        else torch.empty(0),
        "SocketChannelNode": graph.nodes["SocketChannelNode"].data["LOCAL_INET_ADDR"]
        if graph.num_nodes("SocketChannelNode")
        else torch.empty(0),
    }


agg_func = None if structural else feature_aggregation_function

if bidirection:
    for relation_attribute in list(relation_attributes.keys()):
        flipped_relation = relation_attribute[::-1]
        if flipped_relation not in relation_attributes:
            relation_attributes[flipped_relation] = relation_attributes[
                relation_attribute
            ]

if structural:
    THRESHOLD = 0.85
else:
    THRESHOLD = 0.5

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

model = BinaryHeteroClassifier(
    neural_network_type,
    num_layers,
    input_feature_size,
    hidden_feature_size,
    list(relation_attributes.keys()),
    list(node_attributes.keys()),
    structural=structural,
)

model.load_state_dict(
    torch.load(os.path.join(os.getcwd(), "models", log_name + ".bin"), map_location=device)
)
model = model.to(device)


attrib_feature_map = {}
for ntype, attribs in node_attributes.items():
    attrib_feature_map[ntype] = {}
    for attrib in attribs:
        attrib_feature_map[ntype][attrib] = {}

for graph, _ in train_dataset:
    if not hasattr(graph, "additional_node_data"):
        continue
    for attrib, data_dict in graph.ndata.items():
        for ntype, data in data_dict.items():
            for i in range(data.shape[0]):
                orig_str = graph.additional_node_data[ntype][i][attrib]
                attrib_feature_map[ntype][attrib][orig_str] = data[i]


def find_gadgets(source_process_name, process_to_replace, target_process_name):
    # print(f'find gadget: {source_process_name}->({process_to_replace})->{target_process_name}')

    return gadget_dict[str((source_process_name, target_process_name))]


def find_camoflauge(process_name):
    # print(f'find camouflage: {process_name}')

    with Path(f'gadget_files/{Path(process_name).name}.json').open('r') as gadget_file:
        camouflage_dict = json.load(gadget_file)

    parse_file_camo = lambda relation, target, count: (relation.split('_')[0].upper(), int(count), Path(target).name)
    file_camo = [parse_file_camo(relation, target, count) for [relation, target], count in camouflage_dict['files']]

    parse_ip_camo = lambda relation, target, count: (relation.split('_')[0].upper(), int(count), target)
    ip_camo = [parse_ip_camo(relation, target, count) for [relation, _, target], count in camouflage_dict['ips']]

    return {"FileNode": file_camo, "SocketChannelNode": ip_camo}


def provninja_attack(orig_graph, label):

    def add_node(graph, node_type):
        if (
            hasattr(graph, "additional_node_data")
            and node_type in graph.additional_node_data
        ):
            graph.additional_node_data[node_type].append({})

        data = {}
        for attrib in node_attributes[node_type]:
            attrib_str = np.random.choice(
                list(attrib_feature_map[node_type][attrib].keys())
            )
            attrib_tensor = attrib_feature_map[node_type][attrib][attrib_str]
            data[attrib] = attrib_tensor.unsqueeze(0).to(graph.device)
            if (
                hasattr(graph, "additional_node_data")
                and node_type in graph.additional_node_data
            ):
                graph.additional_node_data[node_type][-1][attrib] = attrib_str

        graph.add_nodes(1, data=data, ntype=node_type)
        return graph.num_nodes(ntype=node_type) - 1

    def add_process_node(graph, process_name):
        node_type = "ProcessNode"
        if (
            hasattr(graph, "additional_node_data")
            and node_type in graph.additional_node_data
        ):
            graph.additional_node_data[node_type].append({"EXE_NAME": process_name})

        data = {}
        for attrib in node_attributes[node_type]:
            if attrib == "EXE_NAME":
                input_str = process_name
                if len(input_str) == 1 or (
                    len(input_str) <= 3 and input_str[-1] in ["/" or "'"]
                ):
                    input_str = "root"
                else:
                    input_str = PureWindowsPath(
                        input_str
                    ).name

                tokens = bert_tokenizer.tokenize(input_str)
                ids = bert_tokenizer.convert_tokens_to_ids(tokens)

                embedding = bert_model(torch.tensor(ids)[None, :])[0][0]
                embedding = embedding.sum(0).detach().cpu().numpy()

                attrib_tensor = torch.tensor(embedding)
                data[attrib] = attrib_tensor.unsqueeze(0).to(graph.device)
            else:
                raise NotImplementedError()

        graph.add_nodes(1, data=data, ntype=node_type)
        return graph.num_nodes(ntype=node_type) - 1

    def remove_node(graph, node_type, node_id):
        graph.remove_nodes([node_id], ntype=node_type)
        if hasattr(graph, "additional_node_data"):
            graph.additional_node_data[node_type].pop(node_id)

    def remove_isolated_nodes(graph):
        in_degrees = {
            ntype: np.zeros(graph.num_nodes(ntype=ntype)) for ntype in graph.ntypes
        }
        out_degrees = {
            ntype: np.zeros(graph.num_nodes(ntype=ntype)) for ntype in graph.ntypes
        }
        for etype in graph.canonical_etypes:
            in_degrees[etype[2]] = np.add(
                in_degrees[etype[2]], graph.in_degrees(etype=etype).numpy()
            )
            out_degrees[etype[0]] = np.add(
                out_degrees[etype[0]], graph.out_degrees(etype=etype).numpy()
            )
        for ntype in graph.ntypes:
            isolated_nodes = (
                (in_degrees[ntype] == 0) & (out_degrees[ntype] == 0)
            ).nonzero()[0]
            if len(isolated_nodes) > 0:
                graph.remove_nodes(isolated_nodes, ntype=ntype)

    def add_edge(graph, relation_type, source_node, target_node, bidirection=bidirection):
        source_node_type, edge_type, target_node_type = relation_type
        graph.add_edges(
            [source_node],
            [target_node],
            etype=(source_node_type, edge_type, target_node_type),
        )
        if bidirection:
            graph.add_edges(
                [target_node],
                [source_node],
                etype=(target_node_type, edge_type, source_node_type),
            )

    def apply_camoflauge(graph, source_node, camoflauge, num_actions=10):
        allowed_relations = []
        for target_node_type, actions in camoflauge.items():
            for action in actions:
                allowed_relations.append(
                    (action[1], ("ProcessNode", action[0], target_node_type), action[2])
                )

        for i in range(num_actions):
            actions_prob = np.array([x[0] for x in allowed_relations])
            actions_prob = actions_prob / np.sum(actions_prob)
            relation = allowed_relations[
                np.random.choice(len(allowed_relations), p=actions_prob)
            ][1]
            new_node = add_node(graph, relation[2])
            add_edge(graph, relation, source_node, new_node)

    def find_process_creator(graph, target_node, exclude=[]):
        etype = ("ProcessNode", "PROC_CREATE", "ProcessNode")
        for i in range(len(graph.edges(etype=etype)[0])):
            if (
                graph.edges(etype=etype)[1][i] == target_node
                and graph.edges(etype=etype)[0][i] not in exclude
            ):
                return graph.edges(etype=etype)[0][i]
        return None

    print(f"Attacking {orig_graph.folder_name}")

    orig_graph = copy.deepcopy(orig_graph)

    graph = copy.deepcopy(orig_graph).to(device)

    if not structural:
        orig_pred = model(graph, agg_func(graph))
    else:
        orig_pred = model(graph)

    adversal_graph = None

    etype = "PROC_CREATE"
    gadget_node_type = "ProcessNode"
    gadget_edge_type = "PROC_CREATE"

    for idx in range(len(orig_graph.edges(etype=etype))):

        node_to_replace = orig_graph.edges(etype=etype)[0][idx]
        dst_node = orig_graph.edges(etype=etype)[1][idx]
        src_node = find_process_creator(orig_graph, node_to_replace, exclude=[dst_node])

        if src_node is None:
            continue

        src_node_name = orig_graph.additional_node_data[gadget_node_type][src_node]['EXE_NAME']
        to_replace_node_name = orig_graph.additional_node_data[gadget_node_type][node_to_replace]['EXE_NAME']
        dst_node_name = orig_graph.additional_node_data[gadget_node_type][dst_node]['EXE_NAME']

        gadget_chains = find_gadgets(src_node_name, to_replace_node_name, dst_node_name)

        # Try with all the gadgets
        attack_succeed = False
        for gadget_chain in gadget_chains:
            attack_graph = copy.deepcopy(orig_graph)
            remove_node(attack_graph, gadget_node_type, node_to_replace)
            if src_node > node_to_replace:
                src_node -= 1
            if dst_node > node_to_replace:
                dst_node -= 1

            prev_gadget_node = src_node
            for gadget in gadget_chain:
                gadget_node = add_process_node(attack_graph, gadget)
                add_edge(
                    attack_graph,
                    (gadget_node_type, gadget_edge_type, gadget_node_type),
                    prev_gadget_node,
                    gadget_node,
                )

                camoflauge = find_camoflauge(gadget)
                apply_camoflauge(attack_graph, gadget_node, camoflauge)
                prev_gadget_node = gadget_node

            add_edge(
                attack_graph,
                (gadget_node_type, gadget_edge_type, gadget_node_type),
                prev_gadget_node,
                dst_node,
            )

            remove_isolated_nodes(attack_graph)

            graph = copy.deepcopy(attack_graph).to(device)

            if not structural:
                attack_pred = model(graph, agg_func(graph))
            else:
                attack_pred = model(graph)

            if label == 1 and attack_pred < THRESHOLD:
                adversal_graph = copy.deepcopy(attack_graph)

                attack_succeed = True

                print("Adversal examples found.")
                print(f"file={orig_graph.folder_name} label={float(label)} original pred={float(orig_pred)} new pred={float(attack_pred)}")
                break
        if attack_succeed:
            break

    if adversal_graph is None:
        print(f"Attack failed for graph {orig_graph.folder_name} :(")
    else:
        print(f"!!!Attack SUCCESSFUL for graph {orig_graph.folder_name} :) !!!")

        if not structural:
            result_dir = os.path.join("adversarial_examples", orig_graph.folder_name)
        else:
            result_dir = os.path.join("adversarial_examples_structural", orig_graph.folder_name)
        print("saving to", result_dir)

        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, "original_graph.pkl"), "wb") as f:
            pickle.dump(orig_graph, f)
        if adversal_graph is not None:
            with open(os.path.join(result_dir, "adversarial_graph.pkl"), "wb") as f:
                pickle.dump(adversal_graph, f)

    return adversal_graph


correctly_identified_anomoly_graphs = []
fp = 0
tn = 0
total_attack = 0
for i in range(len(test_dataset)):
    orig_graph, label = test_dataset[i]

    graph = copy.deepcopy(orig_graph).to(device)
    if not structural:
        pred = model(graph, agg_func(graph))
    else:
        pred = model(graph)

    if label == 0 and float(pred) < THRESHOLD:
        tn += 1
    if label == 0 and float(pred) > THRESHOLD:
        fp += 1
    if label == 1:
        total_attack += 1

    if float(pred) > THRESHOLD and label == 1:
        correctly_identified_anomoly_graphs.append(i)


successful_attacks = 0
for graph_idx in correctly_identified_anomoly_graphs:
    graph, label = test_dataset[graph_idx]
    attack_graph = provninja_attack(graph, label)
    print("\n\n")
    if attack_graph is not None:
        successful_attacks += 1

fn = successful_attacks
tp = total_attack - successful_attacks
print(f"Detection evaded for {successful_attacks} / {len(correctly_identified_anomoly_graphs)} true positive samples")

precision = 1.0 * tp / (tp + fp)
recall = 1.0 * tp / (tp + fn)
f1 = (2 * recall * precision) / (recall + precision)
print("Precision:" + str(precision))
print("Recall:" + str(recall))
print("F1:" + str(f1))


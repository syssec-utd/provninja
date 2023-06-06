# import logging
import logging
import os
import socket
from datetime import datetime
import argparse
from pathlib import Path
import pandas as pd

# set up logging variables
log = logging.getLogger(__name__)
unique_str = datetime.now().strftime("%b%d_%H-%M-%S")

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s\t| %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(
                os.getcwd(), "logs", f"{unique_str}_{socket.gethostname()}.log"
            ),
            mode="w",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

THRESHOLD = 0.003


def rare_edge(transition_matrix, source, destination):
    try:
        if transition_matrix.loc[source, destination] > THRESHOLD:
            return True
        else:
            return False
    except KeyError:
        return False


def find_all_paths(
    transition_matrix, source, destination, visited, path, visited_nodes, all_paths
):
    visited.append(source)
    path.append(source)
    visited_nodes.add(source)
    if source == destination:
        all_paths.append(path[:])
    else:
        for neighbor in transition_matrix.columns:
            if neighbor not in visited_nodes and rare_edge(
                transition_matrix, source, neighbor
            ):
                find_all_paths(
                    transition_matrix,
                    neighbor,
                    destination,
                    visited,
                    path,
                    visited_nodes,
                    all_paths,
                )
    visited.remove(source)
    path.pop()
    visited_nodes.remove(source)


def get_transition_matrix(df):
    src_counts = df.groupby("src")["count"].sum().reset_index()

    transition_matrix = df.merge(src_counts, on="src", suffixes=["", "_src"])
    transition_matrix["probability"] = 100 * (
        transition_matrix["count"] / transition_matrix["count_src"]
    )

    transition_matrix = transition_matrix.pivot(
        index="src", columns="dst", values="probability"
    )
    transition_matrix = transition_matrix.fillna(0.0)

    return transition_matrix


def get_regularity_score(transition_matrix, path):
    score = 1
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        probability = transition_matrix.loc[src, dst]
        score *= probability
    return score


def postprocessing_result(path_to_score_dict):

    path_length = lambda path: len(path.split(" -> "))
    decay_factor = 4
    norm_score = lambda path, score: score * pow(decay_factor, path_length(path))
    normalized_path_to_score_dict = {
        path: norm_score(path, score) for path, score in path_to_score_dict.items()
    }

    # Normalize the scores to be between 0 and 10
    max_score = max(normalized_path_to_score_dict.values())
    min_score = min(normalized_path_to_score_dict.values())
    for path, score in normalized_path_to_score_dict.items():
        normalized_path_to_score_dict[path] = (
            10 * (score - min_score) / (max_score - min_score)
        )

    # Sort the paths based on the highest score
    sorted_paths = sorted(
        normalized_path_to_score_dict.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_paths


def find_gadget_chains(input_file, processed_file, output_file):
    log.info(f"input:{input_file}, proc:{processed_file}, output:{output_file}")

    input_df = pd.read_csv(input_file)
    proc_df = pd.read_csv(processed_file)

    transition_matrix = get_transition_matrix(proc_df)

    source = input_df["entry"].values[0]
    destination = input_df["target"].values[0]

    log.info(f"src: {source} dst:{destination}")

    visited = []
    path = []
    visited_nodes = set()
    all_paths = []
    find_all_paths(
        transition_matrix, source, destination, visited, path, visited_nodes, all_paths
    )

    if all_paths:
        log.info(f"{len(all_paths)} # of path exists from {source} to {destination}")

        sorted_paths = postprocessing_result(
            {
                " -> ".join(path): get_regularity_score(transition_matrix, path)
                for path in all_paths
            }
        )

        with open(output_file, "w") as of:
            for path, score in sorted_paths:
                log.info(f"{path}: {score:.2e}")
                of.write(f"{path}: {score:.2e}\n")
    else:
        log.info(f"No path exists from {source} to {destination}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Gadget Finder")

    parser.add_argument(
        "-i",
        "--input",
        action="store",
        type=str,
        default="input.csv",
        help="Configuration File Name",
    )
    parser.add_argument(
        "-p",
        "--processed",
        action="store",
        type=str,
        default="proccreate_windows_processed.csv",
        help="Configuration File Name",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        default="output.txt",
        help="Output File Name",
    )

    args = parser.parse_args()

    find_gadget_chains(args.input, args.processed, args.output)

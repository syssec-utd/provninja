import io
import pickle
import base64

import dgl
import os

import numpy as np
import pandas as pd
import torch as th

from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from base64 import b64decode


class ProvDataset(DGLDataset):

    @property
    def __cache_file_name(self):
        raise NotImplementedError

    def __init__(self,
                 name,
                 input_dir,
                 node_attributes_map,
                 relation_attributes_map,
                 bidirection=False,
                 force_reload=False,
                 verbose=False):
        self.graphs = []
        self.labels = []

        self.node_attributes_map = node_attributes_map
        self.relation_attributes_map = relation_attributes_map

        self.bidirection = bidirection

        super(ProvDataset, self).__init__(name=name,
                                          raw_dir=input_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    # transforms set of .csv nodes and .csv edges from input_dir into a DGL heterograph using node_attributes
    # and relation_attributes map
    def __processGraph(self, input_dir, label) -> None:
        graph_name = os.path.basename(input_dir)
        node_csv_files = []
        edge_csv_files = []

        csv_files = [
            f for f in os.scandir(input_dir)
            if f.is_file() and f.path.endswith('.csv') and f.name not in [
                'scoring_stats.csv', 'graph_properties.csv', 'file.csv',
                'process.csv', 'path_score.csv'
            ]
        ]

        while csv_files:
            file = csv_files.pop()
            file_name = file.name[:-4]  # remove .csv extension
            if '~' in file_name:  # relations have ~
                relation = tuple(file_name.split('~'))

                # TODO: Investigate why we have these strange relations.
                if relation[0] != 'ProcessNode' or relation in [
                    ('ProcessNode', 'FILE_EXEC', 'ProcessNode'),
                    ('ProcessNode', 'READ', 'ProcessNode'),
                    ('ProcessNode', 'WRITE', 'ProcessNode'),
                    ('ProcessNode', 'PROC_CREATE', 'FileNode'),
                    ('ProcessNode', 'IP_CONNECTION_EDGE', 'SocketChannelNode'),
                    ('ProcessNode', 'PROC_CREATE', 'SocketChannelNode'),
                ]:
                    print(
                        f'{input_dir} was skipped in the dataset because invalid relation {relation} exists'
                    )
                    return

                if relation not in self.relation_attributes_map:
                    raise ValueError(
                        f'Error processing "{file.path}". Relation type not found in relation_attributes_map.'
                    )
                edge_csv_files.append(file)
            else:
                if file_name not in self.node_attributes_map:
                    raise ValueError(
                        f'Error processing "{file.path}". Node type not found in node_attributes_map.'
                    )
                node_csv_files.append(file)

        # mapping of node_type => pandas dataframe of the node_type's .csv
        node_dict = {}

        node_addit_data_dict = {}

        # mapping of relation => pandas dataframe of the relation's .csv
        relation_dict = {}

        for node_file in node_csv_files:
            node_file_name = node_file.name[:-4]

            df = pd.read_csv(node_file.path)
            node_dict[node_file_name] = df

            with open(os.path.join(input_dir, node_file_name + '.pickle'),
                      'rb') as f:
                node_addit_data_dict[node_file_name] = pickle.load(f)

        num_nodes_dict = {
            nodeType: nodeList.shape[0]
            for nodeType, nodeList in node_dict.items()
        }

        # add missing nodes with 0 nodes to ensure consistent graph schema
        for node_name in self.node_attributes_map:
            if node_name not in num_nodes_dict:
                num_nodes_dict[node_name] = 0

        # maps relation => edges in the graph
        graph_data = {}
        for relation_file in edge_csv_files:
            df = pd.read_csv(relation_file.path)

            relation = tuple(relation_file.name[:-4].split(
                '~'))  # get relation from file name

            relation_dict[relation] = df

            u = df['u'].to_numpy()
            v = df['v'].to_numpy()

            graph_data[relation] = (u, v)

            # add bidirectional edges (of flipped relation) if specified
            if self.bidirection:
                flipped_relation = relation[::-1]  # reverses relation tuple

                if flipped_relation == relation:  # handle relations like (ProcessNode, CREATE, ProcessNode)
                    u_with_flipped = np.concatenate([u, v])
                    v_with_flipped = np.concatenate([v, u])
                    graph_data[relation] = (u_with_flipped, v_with_flipped)
                else:
                    graph_data[flipped_relation] = (v, u)

        # add missing relations with 0 edges to ensure consistent graph schema
        for relation in self.relation_attributes_map:
            if relation not in graph_data:
                graph_data[relation] = (th.tensor([], dtype=th.int64),
                                        th.tensor([], dtype=th.int64))

            if self.bidirection:
                flipped_relation = relation[::-1]  # reverses relation tuple
                if flipped_relation not in graph_data:
                    graph_data[flipped_relation] = (th.tensor([],
                                                              dtype=th.int64),
                                                    th.tensor([],
                                                              dtype=th.int64))

        graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        graph.folder_name = graph_name

        # assign node additional data
        graph.additional_node_data = node_addit_data_dict

        # assign node attributes
        graph.nodes_additional_data = {}
        for node_type in graph.ntypes:
            if graph.num_nodes(node_type):  # if node count > 0
                for attribute in self.node_attributes_map[node_type]:
                    data_numpy = node_dict[node_type][attribute].to_numpy()
                    if data_numpy.dtype == object:
                        np_list = [
                            np.load(io.BytesIO(b64decode(b64_string)))
                            for b64_string in data_numpy
                        ]
                        graph.nodes[node_type].data[attribute] = th.from_numpy(
                            np.array(np_list))
                    else:
                        graph.nodes[node_type].data[attribute] = th.from_numpy(
                            data_numpy)

        # assign edge attributes
        for relation, attributes in self.relation_attributes_map.items():
            for attribute in attributes:
                if relation not in relation_dict:
                    numpy_relation_attribute = np.array([])
                else:
                    numpy_relation_attribute = relation_dict[relation][
                        attribute].to_numpy()

                if self.bidirection:
                    flipped_relation = relation[::
                                                -1]  # reverses relation tuple

                    if flipped_relation == relation:  # handle relations like (ProcessNode, CREATE, ProcessNode)
                        doubled_relation_attribute = np.concatenate([
                            numpy_relation_attribute, numpy_relation_attribute
                        ])
                        graph.edges[relation].data[attribute] = th.from_numpy(
                            doubled_relation_attribute)
                    else:
                        graph.edges[relation].data[attribute] = th.from_numpy(
                            numpy_relation_attribute)

                        graph.edges[flipped_relation].data[
                            attribute] = th.from_numpy(
                                numpy_relation_attribute)
                else:
                    graph.edges[relation].data[attribute] = th.from_numpy(
                        numpy_relation_attribute)
        total_num_nodes = 0
        for node_type in graph.ntypes:
            total_num_nodes += graph.num_nodes(node_type)

        if total_num_nodes != 0:
            self.graphs.append(graph)
            self.labels.append(th.tensor([int(label)]))

    def save(self):
        graph_path = os.path.join(
            self.save_path,
            self.__cache_file_name) if not self.bidirection else os.path.join(
                self.save_path,
                self.__cache_file_name.split('.')[0] + '_bidirectional.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})

        # output graph folder names
        graph_folder_names_pickle_path = os.path.join(
            self.save_path,
            self.__cache_file_name.split('.')[0] + '_folder_names.pickle')
        graph_folder_names = [graph.folder_name for graph in self.graphs]

        with open(graph_folder_names_pickle_path, 'wb') as output_pickle_file:
            pickle.dump(graph_folder_names, output_pickle_file)

        # output node additional data
        graph_node_addit_data_pickle_path = os.path.join(
            self.save_path,
            self.__cache_file_name.split('.')[0] + '_node_addit_data.pickle')
        graph_node_addit_data = [
            graph.additional_node_data for graph in self.graphs
        ]

        with open(graph_node_addit_data_pickle_path,
                  'wb') as output_pickle_file:
            pickle.dump(graph_node_addit_data, output_pickle_file)

    def load(self):
        graph_path = os.path.join(
            self.save_path,
            self.__cache_file_name) if not self.bidirection else os.path.join(
                self.save_path,
                self.__cache_file_name.split('.')[0] + '_bidirectional.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']

        # load in graph folder names
        graph_folder_names_pickle_path = os.path.join(
            self.save_path,
            self.__cache_file_name.split('.')[0] + '_folder_names.pickle')
        with open(graph_folder_names_pickle_path, 'rb') as input_pickle_file:
            graph_folder_names = pickle.load(input_pickle_file)

        assert len(graph_folder_names) == len(self.graphs), 'Folder names pickle file is not consistent with graph ' \
                                                            'binary'

        # load in graph node additional data
        graph_node_addit_data_pickle_path = os.path.join(
            self.save_path,
            self.__cache_file_name.split('.')[0] + '_node_addit_data.pickle')
        with open(graph_node_addit_data_pickle_path,
                  'rb') as input_pickle_file:
            graph_node_addit_data = pickle.load(input_pickle_file)

        assert len(graph_node_addit_data) == len(self.graphs), 'Graph node additional data pickle file is not ' \
                                                               'consistent with graph binary'

        for i in range(len(self.graphs)):
            self.graphs[i].folder_name = graph_folder_names[i]
            self.graphs[i].additional_node_data = graph_node_addit_data[i]

    def has_cache(self):
        return os.path.exists(
            os.path.join(self.save_path, self.__cache_file_name
                         ) if not self.bidirection else os.path.
            join(self.save_path,
                 self.__cache_file_name.split('.')[0] + '_bidirectional.bin'))

    @property
    def num_labels(self):
        _, num = th.unique(self.labels, return_counts=True)
        return num.shape[0]

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

from .BaseDataloader import ProvDataset
import os
import torch as th


class AnomalyBenignDataset(ProvDataset):
    _ProvDataset__cache_file_name = '_anomaly_benign_prov_dataset.bin'

    # input_dir - input directory path containing two folders for anomaly and benign .csv files
    # file struture should be as follows:
    # input_dir/
    #   |-- benign_folder_name/
    #       |-- 1/                    # name here does not matter
    #           |-- FileNode.csv
    #           |-- ProcessNode.csv
    #           |-- ProcessNode_READ_FileNode.csv
    #           |-- ...
    #   |-- anomaly_folder_name/
    #       |-- 1/                    # name here does not matter
    #           |-- FileNode.csv
    #           |-- ProcessNode.csv
    #           |-- ProcessNode_READ_FileNode.csv
    #           |-- ...

    # node_attributes_map - (Node => attribute list) dict containing attributes we wish to import from .csv for each node
    # relation_attributes_map - (relation => attribute list) dict containg attributes we wish to import from .csv for each relation

    #     node_attributes = {
    #     "ProcessNode": ["AGENT_ID", "TYPE", "REF_ID", "PROC_ORDINAL", "PID", "PROC_STARTTIME"],
    #     "FileNode": ["AGENT_ID", "TYPE", "REF_ID"]
    # }

    # relation_attributes = {
    #     ('ProcessNode', 'PROC_CREATE', 'ProcessNode'): ['EVENT_START','EVENT_END','TIME_START','TIME_END','EVENT_START_STR','EVENT_END_STR','_label','IS_ALERT','OPTYPE'],
    #     ('ProcessNode', 'READ', "FileNode"): ['EVENT_START','EVENT_END','TIME_START','TIME_END','EVENT_START_STR','EVENT_END_STR','_label','IS_ALERT','OPTYPE'],
    #     ('ProcessNode', 'WRITE', 'FileNode'): ['EVENT_START','EVENT_END','TIME_START','TIME_END','EVENT_START_STR','EVENT_END_STR','_label','IS_ALERT','OPTYPE'],
    # }
    #
    # Note: Node and Relation names in the dict will determine the .csv file we import from
    # ie node_attributes = {'FileNode' => [a]} will import only the 'a' attribute from FileNode.csv
    #
    # force_reload is if we want to bypass the cached processed data (if there is one) and recreate the dataset from scratch again
    # verbose will print out status of data loader as it loads the data
    def __init__(self,
                 input_dir,
                 benign_folder_name,
                 anomaly_folder_name,
                 node_attributes_map,
                 relation_attributes_map,
                 bidirection=False,
                 force_reload=False,
                 verbose=False):
        self.benign_folder = os.path.join(input_dir, benign_folder_name)
        self.anomaly_folder = os.path.join(input_dir, anomaly_folder_name)

        super(AnomalyBenignDataset,
              self).__init__(name='Anomaly Benign Provenance Graph',
                             input_dir=input_dir,
                             node_attributes_map=node_attributes_map,
                             relation_attributes_map=relation_attributes_map,
                             bidirection=bidirection,
                             force_reload=force_reload,
                             verbose=verbose)

    def process(self):
        benign_subfolders = [
            f.path for f in os.scandir(self.benign_folder) if f.is_dir()
        ]
        anomaly_subfolders = [
            f.path for f in os.scandir(self.anomaly_folder) if f.is_dir()
        ]

        for benign_file in benign_subfolders:
            self._ProvDataset__processGraph(benign_file, 0)

        for anomaly_file in anomaly_subfolders:
            self._ProvDataset__processGraph(anomaly_file, 1)

        self.labels = th.tensor(
            self.labels,
            dtype=th.float)  # convert label list to tensor for saving

        num_benign_processed = sum(label == 0 for label in self.labels)
        num_anomaly_processed = sum(label == 1 for label in self.labels)

        print(
            f'Processed {num_benign_processed}/{len(benign_subfolders)} benign graphs ({float(num_benign_processed) / len(benign_subfolders) * 100:.2f}%)'
        )
        print(
            f'Processed {num_anomaly_processed}/{len(anomaly_subfolders)} anomaly graphs ({float(num_anomaly_processed) / len(anomaly_subfolders) * 100:.2f}%)'
        )


if __name__ == '__main__':
    node_attributes = {
        'ProcessNode': ['EXE_NAME'],
        'SocketChannelNode': ['LOCAL_INET_ADDR'],
        'FileNode': ['AGENT_ID', 'FILENAME_SET']
    }

    relation_attributes = {
        ('ProcessNode', 'PROC_CREATE', 'ProcessNode'): [],
        ('ProcessNode', 'READ', 'FileNode'): [],
        ('ProcessNode', 'WRITE', 'FileNode'): [],
        ('ProcessNode', 'FILE_EXEC', 'FileNode'): [],
        ('ProcessNode', 'WRITE', 'FileNode'): [],
        ('ProcessNode', 'WRITE', 'SocketChannelNode'): [],
        ('ProcessNode', 'READ', 'SocketChannelNode'): [],
        ('ProcessNode', 'IP_CONNECTION_EDGE', 'ProcessNode'): [],
        ('ProcessNode', 'IP_CONNECTION_EDGE', 'FileNode'): [],
    }

    dataset = AnomalyBenignDataset(
        '/home/gxz170001/Documents/repos/prov-ng-gnn/data/excel.exe_string',
        'benign',
        'anomaly',
        node_attributes,
        relation_attributes,
        bidirection=True,
        force_reload=True,
        verbose=True)
    # input_dir = "/home/vincent/gnn/input_dir"
    # dataset = AnomalyBenignDataset(input_dir,
    #                                'benign',
    #                                'anomaly',
    #                                node_attributes,
    #                                relation_attributes,
    #                                verbose=True)

    graph, label = dataset[0]

    print('Node types:', graph.ntypes)
    print('Edge types:', graph.etypes)
    print('Canonical edge types:',
          graph.canonical_etypes)  # prints out relations

    print(dataset.graphs[0])

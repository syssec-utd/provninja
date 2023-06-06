import torch
from torch import nn
import numpy as np

string_embedding_features_map = {
    'ProcessNode': 'EXE_NAME',
    'FileNode': 'FILENAME_SET',
    'SocketChannelNode': 'REMOTE_INET_ADDR',
}


class StringEmbeddingLayer(nn.Module):
    """
    Character level embedding for filenames.
    Only handles ASCII characters.
    Adopted from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    """

    def __init__(self, output_size):
        super(StringEmbeddingLayer, self).__init__()

        self.n_letters = 256
        self.output_size = output_size
        self.max_length = 100
        self.num_rnn_layers = 2

        self.rnn_layers = nn.RNN(self.n_letters, self.output_size,
                                 self.num_rnn_layers)

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def string_to_tensor(self, line):
        tensor = torch.zeros(self.max_length, 1, self.n_letters)
        line = line.encode('utf-8')
        for li, letter in enumerate(line):
            if li >= self.max_length:
                break
            tensor[li][0][letter] = 1
        return tensor.to(self.device)

    def forward(self, string_batch):
        if len(string_batch) == 0:
            return torch.tensor([], device=self.device)
        inputs = []
        for string in string_batch:
            inputs.append(self.string_to_tensor(string))
        inputs = torch.cat(inputs, dim=1)
        hidden = torch.zeros(self.num_rnn_layers,
                             inputs.shape[1],
                             self.output_size,
                             device=self.device)

        outputs, hidden = self.rnn_layers(inputs, hidden)

        return outputs[-1]

    @property
    def device(self):
        return next(self.parameters()).device


def path_data_to_filename(path):
    # Convert the path in the dataset into filename. Support batch operation.
    if isinstance(path, list) or isinstance(path, np.ndarray):
        return [path_data_to_filename(p) for p in path]
    else:
        if path is not str:
            path = ''
        return path.split('/')[-1].split('\\')[-1]

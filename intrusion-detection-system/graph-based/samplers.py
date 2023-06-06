import torch as th

from sklearn.model_selection import StratifiedKFold


# Adapted from Reuben Feinman
# (https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/6)
class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, labels, batch_size, shuffle=True):
        assert len(labels.shape) == 1, 'label array must be 1D'

        if th.is_tensor(labels):
            self.labels = labels.detach().cpu().numpy().astype(int)
        else:
            self.labels = labels.numpy().astype(int)

        n_batches = len(self.labels) // batch_size
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = th.randn(len(self.labels), 1).numpy()

        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = th.randint(0, int(1e8), size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.labels):
            yield test_idx

    def __len__(self):
        return len(self.labels) // self.batch_size

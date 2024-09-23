"""Definition of Dataloader"""

import numpy as np


class DataLoader:

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        def combine_batch_dicts(batch):
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            return batch_dict

        def batch_to_numpy(batch):
            """Transform all values of the given batch dict to numpy arrays"""
            numpy_batch = {}
            for key, value in batch.items():
                numpy_batch[key] = np.array(value)
            return numpy_batch

        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))
        else:
            index_iterator = iter(range(len(self.dataset)))

        batch = []
        for index in index_iterator:
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch_to_numpy(combine_batch_dicts(batch))
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch_to_numpy(combine_batch_dicts(batch))

    def __len__(self):
        length = None

        if self.drop_last:
            length = len(self.dataset) // self.batch_size
        else:
            length = int(np.ceil(len(self.dataset) / self.batch_size))

        return length

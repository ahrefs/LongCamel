import torch
import numpy as np
from datetime import datetime
import torch.distributed as dist
from functools import lru_cache
import bisect
import json
from transformers import AutoTokenizer
from dataclasses import dataclass
IGNORE_INDEX = -100

def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass

class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        print(f"{datetime.now().strftime('%H:%M:%S')}, on rank: {dist.get_rank()}, Loading data from {path}...")
        super().__init__()
        with open(path, "r") as f:
            data = f.readlines()
        self.data = [json.loads(line) for line in data]
        self.data = [{'input_ids': torch.tensor(sample['input_ids']),'labels':torch.tensor(sample['labels'])} for sample in self.data]
        print(f"{datetime.now().strftime('%H:%M:%S')}, on rank: {dist.get_rank()} loaded total {self.__len__()} samples...")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind]


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, weights = None, offset=0, shuffle=False):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        self.real_sizes = [len(d) for d in self.datasets]

        if weights is not None: # each dataset is already random shuffled
            total_samples = min([size / weight for size, weight in zip(self.real_sizes, weights)])
            new_real_sizes = [int(total_samples * w) for w in weights]
            assert new_real_sizes <= self.real_sizes
            self.real_sizes = new_real_sizes
        self.cumulative_sizes = np.cumsum(self.real_sizes)
        self.offset = offset
        self.id_map = None

        if shuffle:
            import random
            id_map = list(range(self.cumulative_sizes[-1]))
            random.shuffle(id_map)
            self.id_map = {new_id: old_id for new_id, old_id in enumerate(id_map)}
        print(f"{datetime.now().strftime('%H:%M:%S')}, on rank: {dist.get_rank()} loaded total {len(self.cumulative_sizes)} datasets, {self.cumulative_sizes[-1]} samples...")

    def __len__(self):
        return self.cumulative_sizes[-1]-self.offset

    def __getitem__(self, idx):
        idx = idx + self.offset
        if self.id_map:
            dataset_idx, sample_idx = self._get_dataset_and_sample_index(self.id_map[idx])
        else:
            dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx][sample_idx]

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]  # no need this line??
        return dataset_idx, sample_idx

@dataclass
class DefaultDataCollatorForFinetune():
    tokenizer: AutoTokenizer
    max_seq_len: int
    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)[:,:self.max_seq_len]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)[:,:self.max_seq_len]
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),)




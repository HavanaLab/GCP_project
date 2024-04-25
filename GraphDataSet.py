import glob
import random

import torch
from torch.utils.data import DataLoader

from parse_graphs import ConvertToTensor
from torch.utils.data import Dataset


class GraphDataSet(Dataset):
    def __init__(self, graph_dir, batch_size):
        self.bs = batch_size
        self.gd = graph_dir

        self.files_list = []
        indexes = torch.randperm(len(self.files_list))
        self.files_list = [self.files_list[i] for i in indexes]
        self.current_file = 0
        self.current_instance = 0

        self.jsons = glob.glob('{}/*.json'.format(self.gd))
        self.idx_mapping = self.get_idx_mapping()

    def __len__(self):
        return len(self.idx_mapping)

    def get_idx_mapping(self):
        idxs = [i for i in range(len(self.jsons))]
        idx_mapping = []
        while idxs:
            tmp_idx = random.sample(idxs, k=min(self.bs, len(idxs)))
            for r in tmp_idx:
                idxs.remove(r)
            idx_mapping.append(tmp_idx)
        return idx_mapping

    def transform(self, idx):
        jsons = [self.jsons[i] for i in idx]
        return ConvertToTensor.get_batch(jsons)

    def __getitem__(self, idx):
        gc, labels, cn, split, mvc = self.transform(self.idx_mapping[idx])
        return gc, labels, cn, split, mvc


if __name__ == '__main__':
    ds = GraphDataSet('/home/dov/openu/Project/pytorch_GNN/GCP_project/adversarial-training', 3)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    print(next(iter(dl)))

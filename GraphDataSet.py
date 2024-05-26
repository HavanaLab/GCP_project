import glob
import random

from torch.utils.data import DataLoader

from parse_graphs import ConvertToTensor
from torch.utils.data import Dataset


class GraphDataSet(Dataset):
    def __init__(self, graph_dir, batch_size, get_true=False, limit=None):
        self.bs = batch_size
        self.gd = graph_dir
        self.get_true = get_true
        self.jsons = glob.glob('{}/*.json'.format(self.gd))
        limit = limit or len(self.jsons)
        self.jsons = self.jsons[:limit]
        self.idx_mapping = self.get_idx_mapping()

    def shuffle(self):
        self.idx_mapping = self.get_idx_mapping()

    def __len__(self):
        return len(self.idx_mapping)

    def get_idx_mapping(self):
        idxs = [i for i in range(len(self.jsons))]
        idx_mapping = []
        while idxs:
            tmp_idx = random.sample(idxs, k=min(self.bs, len(idxs)))
            # tmp_idx = idxs[:min(self.bs, len(idxs))]
            for r in tmp_idx:
                idxs.remove(r)
            idx_mapping.append(tmp_idx)
        return idx_mapping

    def transform(self, idx):
        jsons = [self.jsons[i] for i in idx]
        return ConvertToTensor.get_batch(jsons)#, get_true=self.get_true)

    def __getitem__(self, idx):
        gc, labels, cn, split, mvc = self.transform(self.idx_mapping[idx])
        return gc, labels, cn, split, mvc


if __name__ == '__main__':
    ds = GraphDataSet('/home/dov/openu/Project/pytorch_GNN/GCP_project/adversarial-training', 3)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    print(next(iter(dl)))

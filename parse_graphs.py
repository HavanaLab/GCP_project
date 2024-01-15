
__auther__ = 'dovkess@gmail.com'

import glob
import json
import math
import random
import torch


class ConvertToTensor(object):
    '''
    This class converts JSON graphs into torch tensors to deal with.
    '''
    def __init__(self, graph_dir_path):
        self._gp = glob.glob('{}/*.json'.format(graph_dir_path))

    @staticmethod
    def get_one(g):
        # load graph
        jg = json.load(open(g))
        mat = torch.Tensor(jg['m']).reshape([jg['v'], jg['v']])
        mat_adv = mat.clone()
        cc = jg['change_coord']
        mat_adv[cc[0], cc[1]] = 1
        mat_adv[cc[1], cc[0]] = 1
        return mat, mat_adv, jg['c_number']

    @staticmethod
    def get_batch(jsons):
        # load batch of graphs
        loaded = [json.load(open(j)) for j in jsons]
        size = sum([l['v'] for l in loaded])*2
        ret_mat = torch.zeros((int(size), int(size)))
        shift = 0
        ret_labels = []
        color_nm = []
        split = []
        for l in loaded:
            mat = torch.Tensor(l['m']).reshape([l['v'], l['v']])
            mat_adv = mat.clone()
            cc = l['change_coord']
            mat_adv[cc[0], cc[1]] = 1
            mat_adv[cc[1], cc[0]] = 1
            m = mat.shape[0]
            ret_mat[shift:shift+m, shift:shift+m] = mat
            ret_mat[shift+m:shift+(2*m), shift+m:shift+(2*m)] = mat_adv
            ret_labels += [1, 0]
            color_nm += [l['c_number'], l['c_number']]
            shift += 2*m
            split += [m, m]
        return ret_mat, ret_labels, color_nm, split

    def random_graph(self):
        return self.get_one(random.choice(self._gp))


if __name__ == '__main__':
    ctt = ConvertToTensor('/home/dov/openu/Project/pytorch_GNN/GCP_project/adversarial-training-final_data')
    ctt.random_graph()

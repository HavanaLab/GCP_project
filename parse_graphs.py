
__auther__ = 'dovkess@gmail.com'

import glob
import json
import math
import random
import torch
import numpy as np


class ConvertToTensor(object):
    '''
    This class converts JSON graphs into torch tensors to deal with.
    '''
    def __init__(self, graph_dir_path, device='cuda'):
        self._gp = glob.glob('{}/*.json'.format(graph_dir_path))
        self.device = device

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
    def get_batch(jsons, device='cpu'):
        # load batch of graphs
        loaded = []
        for j in jsons:
            f = open(j, 'r')
            loaded.append(json.load(f))
            f.close()
        # loaded = [json.load(open(j)) for j in jsons]
        size = sum([l['v'] for l in loaded])*2
        ret_mat = torch.zeros((int(size), int(size)))
        mvc = torch.zeros((int(size), 2*int(sum([l['c_number'] for l in loaded]))))
        shift = 0
        color_shift = 0
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
            mvc[shift:shift+m, color_shift:color_shift+l['c_number']] = 1
            mvc[shift+m:shift+(2*m), color_shift+l['c_number']:color_shift+(2*l['c_number'])] = 1
            color_shift += 2*l['c_number']
            ret_labels += [1, 0]
            color_nm += [l['c_number'], l['c_number']]
            shift += 2*m
            split += [m, m]
        return torch.Tensor(ret_mat).to(device), torch.Tensor(ret_labels).to(device), torch.Tensor(color_nm).to(device), torch.Tensor(split).to(device), torch.Tensor(mvc).to(device)

    def random_graph(self):
        return self.get_one(random.choice(self._gp))


if __name__ == '__main__':
    ctt = ConvertToTensor('/home/dov/openu/Project/pytorch_GNN/GCP_project/adversarial-training-final_data')
    ctt.random_graph()

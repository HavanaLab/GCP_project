__auther__ = 'dovkess@gmail.com'

import glob
import json
import math
import random
import torch
import numpy as np

import networkx as nx
from networkx.algorithms import bipartite

class ConvertToTensor(object):
    '''
    This class converts JSON graphs into torch tensors to deal with.
    '''
    def __init__(self, graph_dir_path, device='cuda'):
        self._gp = glob.glob('{}/*.json'.format(graph_dir_path))
        self.device = device

    BATCH_CACHE = {}
    @staticmethod
    def get_batch(jsons, device='cpu'):
        # load batch of graphs
        V_matricies = []
        C_matricies = []
        breaking_edges = []
        splits = []
        colors = []

        # jsons= [jsons[0]]*len(jsons)
        # jsons= ["/home/elad/Documents/kcol/tmp/json/train/m13991.graph.json"]*len(jsons)

        jsons2 = []
        for j in jsons:
            jsons2.append(j)
            jsons2.append(j)
        jsons = jsons2

        for j in jsons:
            if j not in ConvertToTensor.BATCH_CACHE:
                with open(j, 'r') as f:
                    l = json.load(f)
                n = l['v']
                v_mat = torch.Tensor(l['m']).reshape([n,n])
                c = l['c_number']
                be = l['change_coord']
                split = v_mat.shape[0]
                c_mat = torch.ones(n,c)

                # G = bipartite.random_graph(n//2, n-n//2, 0.5)
                # c=2
                # v_mat2 = torch.Tensor((nx.to_numpy_array(G)*1.0).tolist())
                # # be = torch.randint(n//2, n, (2,)) if torch.randint(0, 2, (1,)) > 0.5 else torch.randint(0, n//2, (2,))
                # c_mat = torch.ones(n, c)

                ConvertToTensor.BATCH_CACHE[j] = (v_mat, c_mat, be, c, split)

            v_mat, c_mat, b_edges, c, s = ConvertToTensor.BATCH_CACHE[j]
            V_matricies.append(v_mat)
            C_matricies.append(c_mat)
            breaking_edges.append(b_edges)
            colors.append(c)
            splits.append(s)

        labels = torch.randint(0, 2, size=(len(V_matricies),)).float()
        labels = torch.tensor([1.0,0.0]*(len(jsons)//2))
        # labels = torch.zeros(len(V_matricies))
        # labels[:len(V_matricies)//2] = 1

        for i in range(len(V_matricies)):
            if labels[i] == 0:
                v, u = breaking_edges[i]
                V_matricies[i][v][u] = V_matricies[i][u][v] = 1

        # v_mat_temp =[]
        # c_mat_temp = []
        # breaking_edges_temp = []
        # colors_temp = []
        # splits_temp = []
        # labels_temp = []
        # for i in range(len(V_matricies)):
        #     for j in range(2):
        #         v_mat_temp.append(V_matricies[i].clone())
        #         c_mat_temp.append(C_matricies[i].clone())
        #         u,v = breaking_edges[i]
        #         v_mat_temp[-1][u,v] = v_mat_temp[-1][v, u] = 1
        #         breaking_edges_temp.append(breaking_edges[i])
        #         colors_temp.append(colors[i])
        #         splits_temp.append(splits[i])
        #         labels_temp.append(1-j)
        # V_matricies = v_mat_temp
        # C_matricies = c_mat_temp
        # breaking_edges = breaking_edges_temp
        # colors = colors_temp
        # splits = splits_temp
        # labels = torch.tensor(labels_temp).float()

        # V_matricies_rotated = []
        # for i in range(0, len(V_matricies), 2):
        #     perm = torch.randperm(V_matricies[i].shape[0])
        #     for j in range(2):
        #         V_matricies_rotated.append(V_matricies[i+j][perm][:, perm])
        # V_mat = torch.block_diag(*V_matricies_rotated)
        V_mat = torch.block_diag(*V_matricies)
        C_mat = torch.block_diag(*C_matricies)

        return V_mat.to(device), labels.to(device), torch.tensor(colors).to(device), torch.Tensor(splits).to(device), C_mat.to(device)

    def random_graph(self):
        return self.get_one(random.choice(self._gp))
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


if __name__ == '__main__':
    ctt = ConvertToTensor('/home/dov/openu/Project/pytorch_GNN/GCP_project/adversarial-training-final_data')
    ctt.random_graph()

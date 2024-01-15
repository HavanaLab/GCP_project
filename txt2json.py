
__auther__ = 'dovkess@gmail.com'

import glob
import json
import os


def from_text_to_JSON(path):
    for g in glob.glob('{}/*.graph'.format(path)):
        text_graph = open(g).readlines()
        start = False
        diff_flag = False
        c_number_flag = False
        v = 0
        m = []
        change_coord = (0, 0)
        c_number = 0
        for l in text_graph:
            if 'DIMENSION' in l:
                v = int(l.split(' ')[1].split('\n')[0])
            if start and 'DIFF_EDGE' not in l:
                for f in l.split():
                    m.append(float(f.split('\n')[0]))
            if 'EDGE_WEIGHT_SECTION' in l:
                start = True
            if diff_flag:
                change_coord = (int(l.split()[0]), int(l.split()[1].split('\n')[0]))
                diff_flag = False
            if 'DIFF_EDGE' in l:
                start = False
                diff_flag = True
            if c_number_flag:
                c_number = int(l.split('\n')[0])
                c_number_flag = False
            if 'CHROM_NUMBER' in l:
                c_number_flag = True

        json.dump({'c_number': c_number, 'change_coord': change_coord, 'm': m, 'v': v}, open('{}.json'.format(g), 'w'))


from_text_to_JSON('/home/dov/openu/Project/pytorch_GNN/GCP_project/adversarial-training-final_data')
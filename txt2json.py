
__auther__ = 'dovkess@gmail.com'

import glob
import json
import os
from contextlib import ExitStack

import numpy as np
from tqdm import tqdm

from gurobipy import Model, GRB

import gurobipy as gpy

gurobi_licnese = False

count = 0
options = {
    "WLSACCESSID" : "d2cbfec6-a740-4a94-86f2-0f033ad4633d",
    "WLSSECRET" : "48dc3657-016b-48bf-b831-8c1303efe0f4",
    "LICENSEID" : 2521086,
}
with ExitStack() as stack:
    env = stack.enter_context(gpy.Env(params=options)) if gurobi_licnese else None
    def solve_csp(M, n_colors):
        # global count
        # count += 1
        N = len(M)
        # model = Model()
        if N<=n_colors:
            return {i:i for i in range (n_colors)}
        with gpy.Model(env=env) as model:
            model.setParam('OutputFlag', 0)

            # Create variables
            x = {}
            for i in range(N):
                for j in range(n_colors):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

            # Each node is assigned exactly one color
            for i in range(N):
                model.addConstr(sum(x[i, j] for j in range(n_colors)) == 1)

            # Adjacent nodes do not share the same color
            for i in range(N):
                for j in range(i + 1, N):
                    if M[i][j] == 1:
                        for k in range(n_colors):
                            model.addConstr(x[i, k] + x[j, k] <= 1)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                solution = {}
                for i in range(N):
                    for j in range(n_colors):
                        if x[i, j].x > 0.5:
                            solution[i] = j
                return solution
            elif model.status == GRB.INFEASIBLE:
                return None
            else:
                raise Exception("Gurobi is unsure about the problem")

    def from_text_to_JSON(path):
        files = glob.glob('{}/*.graph'.format(path))
        for i, g in enumerate(tqdm(files)):
            # print('/rworking On {} from {}'.format(i, len(files)))
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
            m = np.array(m).reshape(v, v)
            m2 = m.copy()
            m2[change_coord[0], change_coord[1]]= m2[change_coord[1], change_coord[0]] = 1
            if not (solve_csp(m, c_number) is not None and solve_csp(m, c_number-1) is None and solve_csp(m2,c_number) is None):
                print('Problem with {}'.format(g))
            # json.dump({'c_number': c_number, 'change_coord': change_coord, 'm': m, 'v': v}, open('{}.json'.format(g), 'w'))


    from_text_to_JSON('./data4')

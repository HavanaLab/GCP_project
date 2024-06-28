import random
import time
from contextlib import ExitStack

import numpy as np
import sys, os, json, argparse, itertools
from ortools.sat.python import cp_model
import cProfile

from gurobipy import Model, GRB

import gurobipy as gpy

gurobi_licnese = True
gurobi_licnese_range_help = lambda ra: ra[1] >6 or  (ra[0]>46 and ra[1]>5) or (ra[0]>56 and ra[1]>4)
gurobi_licnese_range = (lambda ta: gurobi_licnese_range_help(ta)) if gurobi_licnese else (lambda ta: not gurobi_licnese_range_help(ta))

count = 0
options = {
    "WLSACCESSID": "d2cbfec6-a740-4a94-86f2-0f033ad4633d",
    "WLSSECRET": "48dc3657-016b-48bf-b831-8c1303efe0f4",
    "LICENSEID": 2521086,
}

options2 = {
    "WLSACCESSID": "e1989b66-f136-4ffc-8232-42a1d0131c5d",
    "WLSSECRET": "072668a4-0722-4b0c-a4d7-f02bc40aa91b",
    "LICENSEID": 2523648,
}

# import cplex
# from cplex.exceptions import CplexError
# def solve_csp_cplex(adj_matrix, c):
#     try:
#         num_nodes = len(adj_matrix)
#         my_prob = cplex.Cplex()
#         my_prob.parameters.output.writelevel = 0
#         my_prob.set_log_stream(None)
#         my_prob.set_error_stream(None)
#         my_prob.set_warning_stream(None)
#         my_prob.set_results_stream(None)
#
#         # Create binary variables for each node-color combination
#         var_names = []
#         for i in range(num_nodes):
#             for color in range(c):
#                 var_names.append(f"x_{i}_{color}")
#         my_prob.variables.add(names=var_names, types=[my_prob.variables.type.binary] * (num_nodes * c))
#
#         # Constraint: Each node must be assigned exactly one color
#         for i in range(num_nodes):
#             my_prob.linear_constraints.add(
#                 lin_expr=[cplex.SparsePair(ind=[var_names[i * c + color] for color in range(c)], val=[1.0] * c)],
#                 senses=["E"],
#                 rhs=[1.0],
#                 names=[f"node_{i}_color"]
#             )
#
#         # Constraint: Adjacent nodes must have different colors
#         for i in range(num_nodes):
#             for j in range(i + 1, num_nodes):
#                 if adj_matrix[i][j] == 1:
#                     for color in range(c):
#                         my_prob.linear_constraints.add(
#                             lin_expr=[cplex.SparsePair(ind=[var_names[i * c + color], var_names[j * c + color]],
#                                                        val=[1.0, -1.0])],
#                             senses=["L"],
#                             rhs=[0.0],
#                             names=[f"adj_{i}_{j}_color_{color}"]
#                         )
#
#         # Set objective: minimize 0 (dummy objective)
#         my_prob.objective.set_sense(my_prob.objective.sense.minimize)
#
#         # Solve the ILP
#         my_prob.solve()
#
#         # Check if feasible solution exists
#         if my_prob.solution.get_status() == my_prob.solution.status.optimal:
#             return True
#         else:
#             return None
#
#     except CplexError as exc:
#         print(f"Error: {exc}")


def solve_csp2(M, n_colors, nmin=40):
    model = cp_model.CpModel()
    N = len(M)

    variables = [model.NewIntVar(0, n_colors - 1, '{i}'.format(i=i)) for i in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            if M[i][j] == 1:
                model.Add(variables[i] != variables[j])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = int(((10.0 / nmin) * N))
    status = solver.Solve(model)

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        solution = dict()
        for k in range(N):
            solution[k] = solver.Value(variables[k])
        return solution
    elif status == cp_model.INFEASIBLE:
        return None
    else:
        raise Exception("CSP is unsure about the problem")



def self_conrained_solve_csp(M, n_colors, gurobi_licnese=True):
    options = {
        "WLSACCESSID": "d2cbfec6-a740-4a94-86f2-0f033ad4633d",
        "WLSSECRET": "48dc3657-016b-48bf-b831-8c1303efe0f4",
        "LICENSEID": 2521086,
    }

    N = len(M)

    if N <= n_colors:
        return {i: i for i in range(n_colors)}
    with ExitStack() as stack:
        env = stack.enter_context(gpy.Env(params=options)) if gurobi_licnese else None
        model = stack.enter_context(gpy.Model(env=env)) if gurobi_licnese else gpy.Model("GCP")
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


def self_contained_min_coloring(M):
    N = len(M)  # Number of nodes
    max_colors = N  # Maximum number of colors

    with ExitStack() as stack:
        env = stack.enter_context(gpy.Env(params=options)) if gurobi_licnese else None
        model = stack.enter_context(gpy.Model(env=env)) if gurobi_licnese else gpy.Model("GCP")
        model.setParam('OutputFlag', 0)

        # Create variables
        x = model.addVars(N, max_colors, vtype=gpy.GRB.BINARY)
        y = model.addVar(vtype=gpy.GRB.INTEGER)

        # Each node is assigned exactly one color
        model.addConstrs((gpy.quicksum(x[i, j] for j in range(max_colors)) == 1 for i in range(N)))

        # Adjacent nodes do not share the same color
        for i in range(N):
            for j in range(i + 1, N):
                if M[i][j] == 1:  # If nodes i and j are adjacent
                    for k in range(max_colors):
                        model.addConstr(x[i, k] + x[j, k] <= 1)

        # If a node is colored with color j, then y must be at least j
        for i in range(N):
            for j in range(max_colors):
                model.addConstr(y >= j * x[i, j])

        # Objective: minimize the highest color index used
        model.setObjective(y, gpy.GRB.MINIMIZE)

        model.optimize()

        if model.status == gpy.GRB.OPTIMAL:
            # Extract solution
            solution = {}
            for i in range(N):
                for j in range(max_colors):
                    if x[i, j].x > 0.5:
                        solution[i] = j
            return solution, y.x + 1
        elif model.status == gpy.GRB.INFEASIBLE:
            return None, None
        else:
            raise Exception("Gurobi is unsure about the problem")


with ExitStack() as stack:
    env = stack.enter_context(gpy.Env(params=options)) if gurobi_licnese else None


    def solve_csp(M, n_colors):
        # global count
        # count += 1
        N = len(M)
        # model = Model()
        if N <= n_colors:
            return {i: i for i in range(n_colors)}
        with ExitStack() as stack2:
            model = stack2.enter_context(gpy.Model(env=env)) if gurobi_licnese else gpy.Model("GCP")
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


    def min_coloring(M):
        N = len(M)  # Number of nodes
        max_colors = N  # Maximum number of colors

        with gpy.Model(env=env) as model:
            model.setParam('OutputFlag', 0)

            # Create variables
            x = model.addVars(N, max_colors, vtype=gpy.GRB.BINARY)
            y = model.addVar(vtype=gpy.GRB.INTEGER)

            # Each node is assigned exactly one color
            model.addConstrs((gpy.quicksum(x[i, j] for j in range(max_colors)) == 1 for i in range(N)))

            # Adjacent nodes do not share the same color
            for i in range(N):
                for j in range(i + 1, N):
                    if M[i][j] == 1:  # If nodes i and j are adjacent
                        for k in range(max_colors):
                            model.addConstr(x[i, k] + x[j, k] <= 1)

            # If a node is colored with color j, then y must be at least j
            for i in range(N):
                for j in range(max_colors):
                    model.addConstr(y >= j * x[i, j])

            # Objective: minimize the highest color index used
            model.setObjective(y, gpy.GRB.MINIMIZE)

            model.optimize()

            if model.status == gpy.GRB.OPTIMAL:
                # Extract solution
                solution = {}
                for i in range(N):
                    for j in range(max_colors):
                        if x[i, j].x > 0.5:
                            solution[i] = j
                return solution, y.x + 1
            elif model.status == gpy.GRB.INFEASIBLE:
                return None, None
            else:
                raise Exception("Gurobi is unsure about the problem")



    def is_cn(Ma, cn_i):
        if solve_csp(Ma, cn_i - 1) == None:
            return True
        else:
            return False


    def find_diff_edge2(Ma, CN, not_edges):
        for k, (i, j) in enumerate(not_edges):
            Ma[i, j] = Ma[j, i] = 1

            sol = solve_csp(Ma, CN)
            if sol is None:  #diff_edge found
                diff_edge = (i, j)
                Ma[i, j] = Ma[j, i] = 0  #backtrack
                return diff_edge
        #end for
        return None


    def find_diff_edge(Ma, CN, not_edges):
        left = 0
        right = len(not_edges) - 1
        not_edges1 = not_edges[:, 0]
        not_edges2 = not_edges[:, 1]

        ls = []
        rs = []
        ms = []
        cs = []

        while left <= right:
            ls.append(left)
            rs.append(right)
            mid = (left + right) // 2
            ms.append(mid)
            # Add edges in the middle of the list
            edges1 = not_edges1[:mid + 1]
            edges2 = not_edges2[:mid + 1]
            Ma[edges1, edges2] = Ma[edges2, edges1] = 1
            sol = solve_csp(Ma, CN)
            cs.append(sol is None)
            if sol is None:  # diff_edge found
                right = mid - 1
            else:
                left = mid + 1
            Ma[edges1, edges2] = Ma[edges2, edges1] = 0  # Reset the edges

        success_map = {i:s for i,s in zip(ms, cs)}
        mid = ms[-1]
        def success(mid):
            return success_map[mid - 1] == False if success_map[mid] == True else success_map[mid + 1] == True
        # If we found the diff_edge, return it
        if left <= len(not_edges) - 1 and right >= 0 and success(mid):
            edges1 = not_edges1[:mid + 1]
            edges2 = not_edges2[:mid + 1]
            if success_map[mid] == True:
                u, v = edges1[-1], edges2[-1]
                edges1 = edges1[:-1]
                edges2 = edges2[:-1]
                Ma[edges1, edges2] = Ma[edges2, edges1] = 1
                Ma[u, v] = Ma[v, u] = 0
            else:
                Ma[edges1, edges2] = Ma[edges2, edges1] = 1
                u, v = not_edges1[mid + 1], not_edges2[mid + 1]

            # Mb = Ma.copy()
            # Mb[u, v] = Mb[v, u] = 1
            # if solve_csp(Ma, CN) is None or solve_csp(Mb, CN) is not None:
            #     print("!!!!!!!!!! not c color" if success_map[mid] else "yes c color", ls, rs, ms, cs)
            # Ma[edges1, edges2] = Ma[edges2, edges1] = 0
            return [u,v]

        return None


    def create_dataset(nmin, nmax, path, samples):
        if samples > 1 and not os.path.exists(path):
            os.makedirs(path)
        #end if
        z = len(os.listdir(path))
        er = 0
        #probability intervals for each CN, given a nmax size, we calculated it outside
        prob_constraints = {3: (0.01, 0.1), 4: (0.1, 0.2), 5: (0.2, 0.3), 6: (0.2, 0.3), 7: (0.3, 0.4), 8: (0.4, 0.5)}

        ranges = []
        instances = []
        for i in range(40, 61):
            for p in range(min(prob_constraints.keys()), 1+max(prob_constraints.keys())):
                ranges.append((i, p))
                instances+= [(i, p)]*1191
        while z in range(min(samples+len(ranges), len(instances))):
            # ra = ranges[z % len(ranges)]
            # if gurobi_licnese_range(ra):
            #     z += 1
            #     continue
            # print(ra)
            N = np.random.randint(nmin, nmax + 1)

            # N = ra[0]
            # Ma = np.zeros((N, N))
            Cn = np.random.randint(min(list(prob_constraints.keys())), max(list(prob_constraints.keys())) + 1)
            # Cn = ra[1]
            N,Cn = instances[z]

            lim_inf, lim_sup = prob_constraints[Cn][0], prob_constraints[Cn][1]
            p_connected = random.uniform(lim_inf, lim_sup)
            Ma = gen_matrix(N, p_connected)

            try:
                init_sol = solve_csp(Ma, Cn) is not None
                is_exact_cn = is_cn(Ma, Cn)
                # _, solved_c = min_coloring(Ma)
                if init_sol and is_exact_cn:  #if solved_c == Cn:
                    deg_rank = degree_ranking(
                        Ma)  #we sort edges by their current degrees to increase the chances of finding the diff edge
                    for w in deg_rank:
                        not_edges = np.array([[w, j] for j in range(N) if w != j and Ma[w, j] == 0])
                        random.shuffle(not_edges)
                        diff_edge = find_diff_edge(Ma, Cn, not_edges)
                        if diff_edge is not None:
                            if samples == 1: return Ma, Cn, diff_edge
                            # Write graph to file
                            write_graph(Ma, Ma, diff_edge, "{}/m{}.graph".format(path, z), False, cn=Cn, from_f="adding")
                            z += 1
                            # if (z-1) % (samples//10) == 0:
                            print('{}% Complete'.format(np.round(100 * z / samples)), Cn, Ma.shape, flush=True)
                            #end if
                            break
                        #end if
                        else:
                            #print("Cant find diff_edge")
                            er += 1
                        #end else
                    #end for
                #end if
                elif not init_sol:  # elif solved_c>Cn:
                    #remove edges to find a derived instance which satisfies the current cn
                    edges = np.array([[i, j] for i in range(N) for j in range(i + 1, N) if Ma[i, j] == 1])
                    random.shuffle(edges)
                    diff_edge = find_diff_edge_below(Ma, Cn, edges)
                    if diff_edge is not None:
                        if samples == 1:
                            return Ma, Cn, diff_edge
                        # Write graph to file
                        write_graph(Ma, Ma, diff_edge, "{}/m{}.graph".format(path, z), False, cn=Cn, from_f="pruning")
                        z += 1
                        # if (z-1) % (samples//10) == 0:
                        print('{}% Complete'.format(np.round(100 * z / samples)), Cn, Ma.shape, flush=True)
                        #end if
                    #end if
                    else:
                        #print("Cant find diff_edge")
                        er += 1
                    #end else
            except Exception as error:
                print(repr(error), Cn, Ma.shape)
                er += 1
        #end while
        print('Could not solve n-color for {} random generated graphs'.format(er))


    #end

    def find_diff_edge_below2(Ma, Cn, edges):
        # global count
        # count += 1
        for k, (i, j) in enumerate(edges):
            Ma[i, j] = Ma[j, i] = 0
            sol = solve_csp(Ma, Cn)
            if sol is not None and is_cn(Ma, Cn): return [i, j]
            # if sol is None: break
        return None


    def find_diff_edge_below(Ma, Cn, edges):
        left = 0
        right = len(edges) - 1
        edges1 = edges[:, 0]
        edges2 = edges[:, 1]

        while left <= right:
            mid = (left + right) // 2
            # Remove edges in the middle of the list
            edges1_mid = edges1[left:mid + 1]
            edges2_mid = edges2[left:mid + 1]
            Ma[edges1_mid, edges2_mid] = Ma[edges2_mid, edges1_mid] = 0

            sol = solve_csp(Ma, Cn)
            if sol is not None and is_cn(Ma, Cn):  # diff_edge found
                # If the graph is colorable, the diff_edge is in the left half
                right = mid - 1
                Ma[edges1_mid, edges2_mid] = Ma[edges2_mid, edges1_mid] = 1  # Reset the edges
            else:
                # If the graph is not colorable, the diff_edge is in the right half
                left = mid + 1

        # If we found the diff_edge, return it
        if left <= len(edges) - 1 and right >= 0:
            u, v = edges[left]
            Ma[u, v] = Ma[v, u] = 0
            return edges[left]

        return None

    def gen_matrix(N, prob):
        Ma = np.zeros((N, N))
        Ma = np.random.choice([0, 1], size=(N, N), p=[1 - prob, prob])
        i_lower = np.tril_indices(N, -1)
        Ma[i_lower] = Ma.T[i_lower]  # make the matrix symmetric
        np.fill_diagonal(Ma, 0)
        return Ma

    def degree_ranking(Ma):
        degrees = np.sum(Ma, axis=1)
        degrees = np.max(degrees) + 1 - degrees
        degree_rank = np.argsort(degrees)
        return degree_rank

    def write_graph(Ma, Mw, diff_edge, filepath, int_weights=False, cn=0, from_f=""):
        # diff_edge_vallue = Ma[*diff_edge]
        # _,c = min_coloring(Ma)
        # diff_edge_vallue2 = Ma[*diff_edge]=Ma[*(diff_edge[::-1])]=1
        # _,c2= min_coloring(Ma)
        # print(from_f, cn, Ma.shape, diff_edge_vallue, diff_edge_vallue2, c, c2, cn>c, c2!=c, c2==cn)
        #
        # Ma[*diff_edge]=Ma[*(diff_edge[::-1])] = diff_edge_vallue

        # Mb = Ma.copy()
        # Mb[diff_edge[0], diff_edge[1]] = Mb[diff_edge[1], diff_edge[0]] = 1
        # if not (solve_csp(Ma, cn) is not None and solve_csp(Ma, cn - 1) is None and solve_csp(Mb, cn) is None):
        #     raise Exception("Problem with the graph and your algo", from_f)

        with open(filepath, "w") as out:

            n, m = Ma.shape[0], len(np.nonzero(Ma)[0])

            out.write('TYPE : Graph Coloring\n')

            out.write('DIMENSION: {n}\n'.format(n=n))

            out.write('EDGE_DATA_FORMAT: EDGE_LIST\n')
            out.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
            out.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX \n')

            # List edges in the (generally not complete) graph
            out.write('EDGE_DATA_SECTION\n')
            for (i, j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
                out.write("{} {}\n".format(i, j))
            #end
            out.write('-1\n')

            # Write edge weights as a complete matrix
            out.write('EDGE_WEIGHT_SECTION\n')
            for i in range(n):
                if int_weights:
                    out.write('\t'.join([str(int(Mw[i, j])) for j in range(n)]))
                else:
                    out.write('\t'.join([str(float(Mw[i, j])) for j in range(n)]))
                #end
                out.write('\n')
            #end

            # Write diff edge
            out.write('DIFF_EDGE\n')
            out.write('{}\n'.format(' '.join(map(str, diff_edge))))
            if cn > 0:
                # Write chromatic number
                out.write('CHROM_NUMBER\n')
                out.write('{}\n'.format(cn))

            out.write('EOF\n')
        #end


    #end

    def main():
        # Define argument parser
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('-samples', default=150_000, type=int, help='How many samples?')
        parser.add_argument('-path', default='data_final4', type=str, help='Save path')
        parser.add_argument('-nmin', default=40, type=int, help='Min. number of vertices')
        parser.add_argument('-nmax', default=60, type=int, help='Max. number of vertices')
        parser.add_argument('--train', action='store_true', help='To define the seed')

        # Parse arguments from command line
        args = parser.parse_args()
        # random_seed = 1327 if vars(args)['train'] else 3712
        # random.seed(random_seed)
        # np.random.seed(random_seed)
        t1 = time.perf_counter()

        print('Creating {} instances'.format(vars(args)['samples']), flush=True)
        create_dataset(
            vars(args)['nmin'], vars(args)['nmax'],
            samples=vars(args)['samples'],
            path=vars(args)['path']
        )
        print(time.perf_counter() - t1)


    # end

    if __name__ == '__main__':
        # cProfile.run('main()', sort='cumtime')
        # pr = cProfile.Profile()
        # pr.enable()
        main()
        # pr.disable()
        # pr.dump_stats('{}/profile.pstat'.format("./"))
        # import pstats
        # stats = pstats.Stats('./profile.pstat')
        # stats.sort_stats('tottime')
        # stats.print_stats()

# import os
# import json
# import numpy as np
# mat = np.zeros((60-40+1, 8-3+1))
# for j in os.listdir("../tmp/json/train"):
#     with open(f"../tmp/json/train/{j}", "r") as f:
#         data = json.load(f)
#         n = data["v"]-40
#         c = data["c_number"]-3
#         mat[n][c] += 1
# print(mat)

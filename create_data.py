import pickle
import random
import numpy as np
import sys, os, json, argparse, itertools
from ortools.sat.python import cp_model

from file_system_storage import FS

# probability intervals for each CN, given a nmax size, we calculated it outside
prob_constraints = {3: (0.01, 0.1), 4: (0.1, 0.2), 5: (0.2, 0.3), 6: (0.2, 0.3), 7: (0.3, 0.4), 8: (0.4, 0.5)}


def solve_csp(M, n_colors, nmin=40):
    model = cp_model.CpModel()
    N = len(M)
    variables = []

    variables = [model.NewIntVar(0, n_colors - 1, '{i}'.format(i=i)) for i in range(N)]

    for i in range(N):
        # maybe change to numpy matrix operation
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


def is_cn(Ma, cn_i):
    if solve_csp(Ma, cn_i - 1) == None:
        return True
    else:
        return False


def find_diff_edge(Ma, CN, not_edges):
    for k, (i, j) in enumerate(not_edges):
        Ma[i, j] = Ma[j, i] = 1

        sol = solve_csp(Ma, CN)
        if sol is None:  # diff_edge found
            diff_edge = (i, j)
            Ma[i, j] = Ma[j, i] = 0  # backtrack
            return diff_edge
    # end for
    return None


def create_dataset(nmin, nmax, path, samples):
    if samples > 1 and not os.path.exists(path): os.makedirs(path)
    z = 0
    er = 0
    instances_map = {}

    while z in range(samples):
        N = np.random.randint(nmin, nmax + 1)
        Cn = np.random.randint(3, 8 + 1)
        lim_inf, lim_sup = prob_constraints[Cn][0], prob_constraints[Cn][1]

        Ma, Cn, diff_edge, success = create_instance(Cn, N, lim_inf, lim_sup)
        if success:
            if N not in instances_map: instances_map[N] = {}
            if Cn not in instances_map[N]: instances_map[N][Cn] = []
            instances_map[N][Cn].append((Ma.tolist(), Cn, list(diff_edge)))
            # data = pickle.dumps([instances_map[N][Cn][-1]])
            # FS().upload_data(
            #     data, os.path.join(path, f'{N}/{Cn}/N{N}_C{Cn}_{len(instances_map[N][Cn]) - 1}.pkl')
            # )
            print(f"created instance #{z}")

        z += success
        er += (1 - success)
        if samples > 10 and (z - 1) % (samples // 10) == 0:
            print('{}% Complete'.format(np.round(100 * z / samples)), flush=True)

    # end while
    print('Could not solve n-color for {} random generated graphs. Was able to create {} instances.'.format(er, z))
    print('Saving instances to disk...')
    for N in instances_map:
        for Cn, instances in instances_map[N].items():
            data = pickle.dumps(instances)
            FS().upload_data(
                data, os.path.join(path, f'{N}/{Cn}/N{N}_C{Cn}_all.pkl')
            )
    print('Done.')


def create_instance(Cn, N, lim_inf, lim_sup):
    """
    if diff_edge is None, then the instance is not valid and should be considered as an error
    """
    p_connected = random.uniform(lim_inf, lim_sup)
    Ma = gen_matrix(N, p_connected)
    try:
        init_sol = solve_csp(Ma, Cn)
        if init_sol is not None and is_cn(Ma, Cn):
            deg_rank = degree_ranking(
                Ma)  # we sort edges by their current degrees to increase the chances of finding the diff edge
            for w in deg_rank:
                j_indices = np.where(Ma[w, :] == 0)[0]  # Find the indices where Ma[w, j] == 0
                not_edges = [(w, j) for j in j_indices if j != w]
                random.shuffle(not_edges)
                diff_edge = find_diff_edge(Ma, Cn, not_edges)
                if diff_edge is not None: break
            return Ma, Cn, diff_edge, diff_edge is not None
        elif init_sol is None:
            # remove edges to find a derived instance which satisfies the current cn
            edges = np.transpose(np.where(Ma != 0))
            random.shuffle(edges)
            diff_edge = None
            for k, (i, j) in enumerate(edges):
                Ma[i, j] = Ma[j, i] = 0
                if solve_csp(Ma, Cn) is not None and is_cn(Ma, Cn):
                    diff_edge = (i, j)
                    break
            return Ma, Cn, diff_edge, diff_edge is not None
        else:
            return Ma, Cn, None, False
    except Exception as error:
        print(repr(error))
        return Ma, Cn, None, False


def gen_matrix(N, prob):
    # Ma = np.zeros((N, N))
    Ma = np.random.choice([0, 1], size=(N, N), p=[1 - prob, prob])
    i_lower = np.tril_indices(N, -1)
    Ma[i_lower] = Ma.T[i_lower]  # make the matrix symmetric
    np.fill_diagonal(Ma, 0)
    return Ma


def degree_ranking(Ma):
    return np.argsort(Ma.sum(axis=1))[::-1]
    # G = nx.from_numpy_matrix(Ma)
    # deg = np.asarray(gp.degree_sequence(G))
    # deg = (np.amax(deg) + 1) - deg  # higher degree comes first
    # deg_rank = np.argsort(deg)
    # return deg_rank


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-samples', default=2 ** 15, type=int, help='How many samples?')
    parser.add_argument('-path', default='adversarial-training', type=str, help='Save path')
    parser.add_argument('-nmin', default=40, type=int, help='Min. number of vertices')
    parser.add_argument('-nmax', default=60, type=int, help='Max. number of vertices')
    parser.add_argument('--train', action='store_true', help='To define the seed')

    # Parse arguments from command line
    args = parser.parse_args()
    # args.samples = 2
    random_seed = 1327 if vars(args)['train'] else 3712
    random.seed(random_seed)
    np.random.seed(random_seed)

    print('Creating {} instances'.format(vars(args)['samples']), flush=True)
    create_dataset(
        vars(args)['nmin'], vars(args)['nmax'],
        samples=vars(args)['samples'],
        path=vars(args)['path']
    )
    # end
import time
from collections import defaultdict
import numpy as np
import argparse

debug = True
K = 15
folder = 'case/Heuristic/map2/'


def read_social_network(network_name):
    LINE = 1e-7
    with open(network_name, 'r') as network_buffer:
        n, m = map(int, network_buffer.readline().split())
        cut_graph1 = defaultdict(list)
        cut_graph2 = defaultdict(list)
        nei = defaultdict(list)
        for _ in range(m):
            u, v, p1, p2 = map(float, network_buffer.readline().split())
            u = int(u)
            v = int(v)
            if p1 > LINE:
                cut_graph1[u].append((v, p1))
            if p2 > LINE:
                cut_graph2[u].append((v, p2))
            nei[u].append(v)

    return cut_graph1, cut_graph2, nei


def read_init_seed_set(init_name):
    with open(init_name, 'r') as file1:
        k1, k2 = map(int, file1.readline().split())
        i1 = [int(file1.readline()) for _ in range(k1)]
        i2 = [int(file1.readline()) for _ in range(k2)]

    return set(i1), set(i2)


def write_sets_to_file(set1, set2, filename):
    with open(filename, 'w') as file:
        file.write(f"{len(set1)} {len(set2)}\n")
        for num in set1:
            file.write(f"{num}\n")
        for num in set2:
            file.write(f"{num}\n")


def sim_one(cut1, cut2, nei12, sim_num):
    single1 = defaultdict(set)
    single2 = defaultdict(set)
    for loop_num in range(sim_num):
        for key in cut1.keys():
            exp_cur = set()
            exp_cur.add(key)
            activated = exp_cur.copy()
            act = exp_cur.copy()
            while act:
                new_act = set()
                for node in act:
                    exp_cur = exp_cur.union(nei12[node])
                    if node in cut1.keys():
                        for nei, p1 in cut1[node]:
                            if nei not in activated:
                                if np.random.random() < p1:
                                    new_act.add(nei)
                activated.update(new_act)
                act = new_act
            if loop_num == 0:
                single1[key] = exp_cur
            else:
                single1[key] = single1[key] & exp_cur

        for key in cut2.keys():
            exp_cur = set()
            exp_cur.add(key)
            activated = exp_cur.copy()
            act = exp_cur.copy()
            while act:
                new_act = set()
                for node in act:
                    exp_cur = exp_cur.union(nei12[node])
                    if node in cut2.keys():
                        for nei, p2 in cut2[node]:
                            if nei not in activated:
                                if np.random.random() < p2:
                                    new_act.add(nei)
                activated.update(new_act)
                act = new_act
            if loop_num == 0:
                single2[key] = exp_cur
            else:
                single2[key] = single2[key] & exp_cur

    return single1, single2


def greedy(cut1, cut2, i1, i2, sig1, sig2, n, k):
    s1 = set()
    s2 = set()
    r1 = set()
    r2 = set()
    for i in i1:
        r1 = r1 | sig1[i]
    for i in i2:
        r2 = r2 | sig2[i]
    init = n - len(r1 ^ r2)
    exp_max1 = exp_max2 = init
    while len(s1) + len(s2) < k:
        u1 = i1 | s1
        u2 = i2 | s2
        v1_id = v2_id = -1
        for key in cut1.keys():
            if key not in u1:
                sim_res = n - len((r1 | sig1[key]) ^ r2)
                if sim_res > exp_max1:
                    exp_max1 = sim_res
                    v1_id = key

        for key in cut2.keys():
            if key not in u2:
                sim_res = n - len(r1 ^ (r2 | sig2[key]))
                if sim_res > exp_max2:
                    exp_max2 = sim_res
                    v2_id = key

        if v1_id == -1 and v2_id == -1:
            break
        elif v1_id != -1 and v2_id != -1:
            if exp_max1 >= exp_max2:
                s1.add(v1_id)
                r1 = r1 | sig1[v1_id]
            else:
                s2.add(v2_id)
                r2 = r2 | sig2[v2_id]
        else:
            if v1_id == -1:
                s2.add(v2_id)
                r2 = r2 | sig2[v2_id]
            else:
                s1.add(v1_id)
                r1 = r1 | sig1[v1_id]

        exp_max1 = exp_max2 = (max(exp_max1, exp_max2))

    print(max(exp_max1, exp_max2))
    return s1, s2


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network", help="social network input absolute path")
    parser.add_argument("-i", "--initial", help="initial seed set absolute path")
    parser.add_argument("-b", "--balanced", help="balanced seed set absolute path")
    parser.add_argument("-k", "--budget", help="budget integer value")
    args = parser.parse_args()

    if debug:
        network_file = folder + 'dataset'
        init_file = folder + 'seed'
        balanced_file = folder + 'out'
        k_in = K
    else:
        network_file = args.network
        init_file = args.initial
        balanced_file = args.balanced
        k_in = int(args.budget)

    cut_graph1, cut_graph2, nei = read_social_network(network_file)
    seti1, seti2 = read_init_seed_set(init_file)
    single1, single2 = sim_one(cut_graph1, cut_graph2, nei, 3)

    balanced_set1, balanced_set2 = greedy(cut_graph1, cut_graph2, seti1, seti2, single1, single2, len(nei.keys()), k_in)
    write_sets_to_file(balanced_set1, balanced_set2, balanced_file)

    end = time.time()
    print(end - start)

import copy
from collections import defaultdict

import numpy as np
import argparse
import time

debug = False
folder = 'case/Evolutionary/map3/'
K = 0

global sig1, sig2, ires1, ires2, n, nodes
global s_sig1, s_sig2
N = 100
global numbers, prob
global sl1, sl2, skey1, skey2
global dic_num1, dic_num2, dic_num_w, dic_prob1, dic_prob2, dic_prob_w


def read_social_network(network_name):
    LINE = 1e-5
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


def init_group(i1, i2):
    res1 = set()
    res2 = set()
    for i in i1:
        res1 = res1 | sig1[i]
    for i in i2:
        res2 = res2 | sig2[i]

    return res1, res2


def generate_structure(nums: int):
    pop = []

    for _ in range(nums):
        gen = []
        k_num = np.random.randint(K - 3, K + 1)
        indices = np.random.choice(dic_num_w, size=k_num, p=dic_prob_w, replace=False).tolist()
        for index in indices:
            id = index // 2
            if index % 2 == 0:
                if id > sl1:
                    gen.append(s_sig2[id] + nodes)
                else:
                    gen.append(s_sig1[id])
            else:
                if id > sl2:
                    gen.append(s_sig1[id])
                else:
                    gen.append(s_sig2[id] + nodes)
        pop.append((-0.1, gen))
    return pop


def fitness_calc(sample):
    r1 = copy.deepcopy(ires1)
    r2 = copy.deepcopy(ires2)
    for i in sample:
        if i < nodes:
            r1 = r1 | sig1[i]
        else:
            r2 = r2 | sig2[i - nodes]

    sigma_x = nodes - len(r1 ^ r2)
    if len(sample) > K:
        return -sigma_x, sample
    return sigma_x, sample


def new_population(pop):
    for i, gen in enumerate(pop):
        if gen[0] == -0.1:
            pop[i] = fitness_calc(gen[1])
    sorted_list = sorted(pop, key=lambda x: x[0], reverse=True)
    sorted_list = sorted_list[:N]
    return sorted_list


def cross_over(pa1, pa2):
    child1 = []
    child2 = []
    length = min(len(pa1), len(pa2))
    type = 0
    if type == 0 or length <= 2:  # single point
        c_point = np.random.randint(nodes * 2)
        for i in pa1:
            if i < c_point:
                child1.append(i)
            else:
                child2.append(i)
        for i in pa2:
            if i < c_point:
                child2.append(i)
            else:
                child1.append(i)
    elif type == 1:  # two point
        c_points = sorted(np.random.choice(nodes * 2, size=2, replace=False))
        for i in pa1:
            if c_points[0] <= i < c_points[1]:
                child2.append(i)
            else:
                child1.append(i)
        for i in pa2:
            if c_points[0] <= i < c_points[1]:
                child1.append(i)
            else:
                child2.append(i)
    elif type == 3:  # uniform
        rand_list = np.random.rand(length)
        for i in range(length):
            if rand_list[i] < 0.5:
                child1.append(pa1[i])
                child2.append(pa2[i])
            else:
                child1.append(pa2[i])
                child2.append(pa1[i])
    return sorted(child1), sorted(child2)


def mutation(sample: list):
    type = np.random.randint(2)
    length = len(sample)
    if type == 0 or length < 3:
        pass
    elif type == 1:
        is_add = np.random.rand()
        if is_add < 0.5:
            if length < K:
                val = np.random.randint(n)
                sample.append(val)
        else:
            id = np.random.randint(length)
            sample.pop(id)
    elif type == 2:
        id = np.random.randint(length)
        sample.pop(id)
        val = np.random.randint(n)
        sample.append(val)
    return sample


def new_son(pop):
    draw = np.random.choice(numbers, size=2, p=prob, replace=False)
    son = cross_over(pop[draw[0] - 1][1], pop[draw[1] - 1][1])
    son1 = mutation(son[0])
    son2 = mutation(son[1])
    return (-0.1, son1), (-0.1, son2)


def Evolutionary():
    pop = generate_structure(N * 2)
    pop = new_population(pop)
    for i in range(20):
        new_pop = []
        for _ in range(N):
            sons = new_son(pop)
            new_pop.append(sons[0])
            new_pop.append(sons[1])
        pop = new_population(new_pop)
    return pop[0][1]


def write_sets_to_file(sample, filename):
    set1 = []
    set2 = []

    for i in sample:
        if i < nodes:
            set1.append(i)
        else:
            set2.append(i - nodes)

    with open(filename, 'w') as file:
        file.write(f"{len(set1)} {len(set2)}\n")
        for num in set1:
            file.write(f"{num}\n")
        for num in set2:
            file.write(f"{num}\n")


def strategy_before_evol():
    global sig1, sig2, sl1, sl2, s_sig1, s_sig2, n, skey1, skey2
    global numbers, prob, dic_num1, dic_num2, dic_num_w, dic_prob1, dic_prob2, dic_prob_w

    numbers = list(range(1, N + 1))
    prob = [1 / (i + 1) for i in range(N)]
    prob = [p / sum(prob) for p in prob]
    skey1 = list(sig1.keys())
    skey2 = list(sig2.keys())
    sl1 = len(skey1)
    sl2 = len(skey2)
    n = sl1 + sl2
    dic_num1 = list(range(1, sl1 + 1))
    dic_num2 = list(range(1, sl2 + 1))
    dic_prob1 = [1 / (i + 1) for i in range(sl1)]
    dic_prob1 = [p / sum(dic_prob1) for p in dic_prob1]
    dic_prob2 = [1 / (i + 1) for i in range(sl2)]
    dic_prob2 = [p / sum(dic_prob2) for p in dic_prob2]
    dic_num_w = list(range(1, n + 1))
    dic_prob_w = [1 / (i + 1) for i in range(n)]
    dic_prob_w = [p / sum(dic_prob_w) for p in dic_prob_w]
    s_sig1 = list(dict(sorted(sig1.items(), key=lambda item: len(item[1]), reverse=True)).keys())
    s_sig2 = list(dict(sorted(sig2.items(), key=lambda item: len(item[1]), reverse=True)).keys())


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
        K = 14
    else:
        network_file = args.network
        init_file = args.initial
        balanced_file = args.balanced
        K = int(args.budget)

    cut_graph1, cut_graph2, nei = read_social_network(network_file)
    nodes = len(nei.keys())
    seti1, seti2 = read_init_seed_set(init_file)
    sig1, sig2 = sim_one(cut_graph1, cut_graph2, nei, 3)
    ires1, ires2 = init_group(seti1, seti2)

    strategy_before_evol()
    sam = Evolutionary()
    # write_sets_to_file(balanced_set1, balanced_set2, balanced_file)
    write_sets_to_file(sam, balanced_file)
    end = time.time()
    print(end - start)

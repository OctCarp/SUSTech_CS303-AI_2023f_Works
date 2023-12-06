import numpy as np
import networkx as nx
import argparse
import time

debug = True


def read_social_network(network_name):
    with open(network_name, 'r') as network_buffer:
        n, m = map(int, network_buffer.readline().split())
        graph = nx.DiGraph()
        for _ in range(m):
            u, v, p1, p2 = map(float, network_buffer.readline().split())
            graph.add_edge(int(u), int(v), p1=p1, p2=p2)

    return graph


def read_seed_set(init_name, balanced_name):
    with open(init_name, 'r') as file1:
        k1, k2 = map(int, file1.readline().split())
        i1 = [int(file1.readline()) for _ in range(k1)]
        i2 = [int(file1.readline()) for _ in range(k2)]

    with open(balanced_name, 'r') as file2:
        k1, k2 = map(int, file2.readline().split())
        s1 = [int(file2.readline()) for _ in range(k1)]
        s2 = [int(file2.readline()) for _ in range(k2)]

    return set(i1 + s1), set(i2 + s2)


def simulation(graph, set_u1, set_u2, loop_num) -> float:
    exp_nums = 0
    rand_list1 = np.random.rand(loop_num, graph.number_of_edges())
    rand_list2 = np.random.rand(loop_num, graph.number_of_edges())
    node_num = graph.number_of_nodes()

    node_es = {}
    for node in graph.nodes:
        node_es[node] = graph.edges(node, data=True)

    for loop_id in range(loop_num):
        exp1 = set_u1.copy()
        exp2 = set_u2.copy()

        activated = set_u1.copy()
        active = set_u1.copy()
        rand_id = 0
        while active:
            new_active = set()
            for node in active:
                edges = node_es[node]
                for ori, nei, p in edges:
                    if nei not in activated:
                        exp1.add(nei)
                        if rand_list1[loop_id][rand_id] < p['p1']:
                            new_active.add(nei)
                        rand_id = rand_id + 1
            activated.update(new_active)
            active = new_active

        activated = set_u2.copy()
        active = set_u2.copy()
        rand_id = 0
        while active:
            new_active = set()
            for node in active:
                edges = node_es[node]
                for ori, nei, p in edges:
                    if nei not in activated:
                        exp2.add(nei)
                        if rand_list2[loop_id][rand_id] < p['p2']:
                            new_active.add(nei)
                        rand_id = rand_id + 1
            activated.update(new_active)
            active = new_active

        exp_nums = exp_nums + node_num - len(set.symmetric_difference(exp1, exp2))

    return exp_nums / loop_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network", help="social network input absolute path")
    parser.add_argument("-i", "--initial", help="initial seed set absolute path")
    parser.add_argument("-b", "--balanced", help="balanced seed set absolute path")
    parser.add_argument("-k", "--budget", help="budget integer value")
    parser.add_argument("-o", "--object", help="object value output absolute path")
    args = parser.parse_args()

    if debug:
        folder = '../case/Heuristic/map3/'
        network_file = folder + 'dataset'
        init_file = folder + 'seed'
        balanced_file = folder + 'out'
        output_file = '../testcase/out.txt'
        k_in = 15
    else:
        network_file = args.network
        init_file = args.initial
        balanced_file = args.balanced
        output_file = args.object
        k_in = args.budget

    social = read_social_network(network_file)
    setu1, setu2 = read_seed_set(init_file, balanced_file)
    sims = 500
    start = time.time()
    ans = simulation(social, setu1, setu2, sims)
    print(time.time() - start)

    with open(output_file, "w") as file:
        file.write(str(ans))

    print(ans)

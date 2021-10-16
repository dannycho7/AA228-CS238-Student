import sys

import networkx as nx
import numpy as np
from math import prod
from pandas import read_csv
import random
from scipy.special import loggamma as lgamma


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


class GraphExplorer:
    def __init__(self, D, G):
        self.D = D
        self.r = D.max(axis=0)
        self.n_vars = D.shape[1]
        # mutable
        self.G = G
        self.q = np.array([prod(r.take(list(G.predecessors(i))))
                           if G.in_degree(i) > 0 else 1 for i in range(self.n_vars)])
        self.M = self.get_M()
        self.var_scores = [self.get_score_for_var(i) for i in range(
            self.n_vars)]  # maintain scores coming from each var

    def get_score_for_var(self, i):
        var_counts = self.M[i]
        score = lgamma(self.r[i]) * self.q[i]  # a_ij0
        score -= lgamma(self.r[i] + var_counts.sum(axis=1)
                        ).sum()  # a_ij0 + m_ij0
        score += lgamma(1 + var_counts).sum()
        # note: lgamma(a_ijk) term is not here as lgamma(1) == 0
        return score

    def get_j(self, i, entry):
        if self.G.in_degree(i) == 0:
            return 0
        preds = list(self.G.predecessors(i))
        pred_shapes = [self.r[pred] for pred in preds]
        pred_idxs = entry.take(preds) - 1
        return np.ravel_multi_index(pred_idxs, pred_shapes)

    def get_M(self):
        # M is a dict i => (q_i, r_i)
        M = {i: np.zeros((self.q[i], self.r[i])) for i in range(self.n_vars)}
        for entry in self.D:
            for i, val in enumerate(entry):
                k = val - 1
                M[i][self.get_j(i, entry), k] += 1
        return M

    def bayesian_score_unif_prior(self):
        # we can drop logP(G), since we are using uniform priors.
        return sum(self.var_scores)

    def add_edge(self, x, y):
        self.G.add_edge(x, y)
        self.q[y] *= self.r[x]
        self.M = self.get_M()
        self.var_scores[y] = self.get_score_for_var(y)

    def remove_edge(self, x, y):
        self.G.remove_edge(x, y)
        self.q[y] //= self.r[x]
        self.M = self.get_M()
        self.var_scores[y] = self.get_score_for_var(y)

    # returns None if edge isn't a candidate for addition else score
    def score_with_new_edge(self, x, y):
        if x == y or self.G.has_edge(x, y):
            return None
        self.add_edge(x, y)
        score = self.bayesian_score_unif_prior()
        acyclic_edge = nx.is_directed_acyclic_graph(self.G)
        self.remove_edge(x, y)
        return score if acyclic_edge else None


def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    in_df = read_csv(infile)
    G = nx.DiGraph()
    idx2names = []
    for i, col in enumerate(in_df.columns):
        idx2names.append(col)
        G.add_node(i)

    D = in_df.to_numpy()
    gexpl = GraphExplorer(D, G)

    # K2 Search
    max_parents = 2

    # optimize G w/ K2 Search
    ordered_vars = list(range(gexpl.n_vars))
    random.shuffle(ordered_vars)
    for i in ordered_vars:
        curr_score = gexpl.bayesian_score_unif_prior()
        for _ in range(max_parents):
            best_score, best_pred = -float("inf"), None
            for pred in range(gexpl.n_vars):
                score = gexpl.score_with_new_edge(pred, i)
                if score is not None and score > best_score:
                    best_score = score
                    best_pred = pred
            if best_score > curr_score:
                curr_score = best_score
                gexpl.add_edge(best_pred, i)
                print(f"Adding edge {best_pred} -> {i}")
            else:
                break
    print(f"final bayesian score = {gexpl.bayesian_score_unif_prior()}")
    write_gph(G, idx2names, outfile)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()

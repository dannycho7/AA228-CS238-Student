import sys

import networkx as nx
import numpy as np
from math import prod
from pandas import read_csv
from scipy.special import loggamma as lgamma

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


class GraphExplorer:
    def __init__(self, D, G):
        self.D = D
        self.G = G
        self.n_vars = D.shape[1]
        self.r = self.get_r()
        self.q = self.get_q()

    def get_r(self):
        return self.D.max(axis=0)

    def get_q(self):
        return np.array([prod(self.r.take(list(self.G.predecessors(i))))
                         if self.G.in_degree(i) > 0 else 1 for i in range(self.n_vars)])

    def get_j(self, i, entry):
        if self.G.in_degree(i) == 0:
            return 0
        preds = list(self.G.predecessors(i))
        pred_shapes = [self.r[pred] for pred in preds]
        pred_idxs = entry.take(preds) - 1
        return np.ravel_multi_index(pred_idxs, pred_shapes)

    def M(self):
        # M is a dict i => (q_i, r_i)
        M = {i: np.zeros((self.q[i], self.r[i])) for i in range(self.n_vars)}
        for entry in self.D:
            for i, val in enumerate(entry):
                k = val - 1
                M[i][self.get_j(i, entry), k] += 1
        return M
    
    def bayesian_score_unif_prior(self):
        # we can drop logP(G), since we are using uniform priors.
        var_scores = [0] * self.n_vars # maintain scores coming from each var
        M = self.M()
        for i, var_counts in M.items():
            var_scores[i] += lgamma(self.r[i]) * self.q[i] # a_ij0
            var_scores[i] -= lgamma(self.r[i] + var_counts.sum(axis=1)).sum() # a_ij0 + m_ij0
            var_scores[i] += lgamma(1 + var_counts).sum()
            # note: lgamma(a_ijk) term is not here as lgamma(1) == 0
        return sum(var_scores)


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

    G.add_edge(0, 1)
    G.add_edge(2, 3)
    G.add_edge(4, 5)

    gexpl = GraphExplorer(D, G)
    print(gexpl.bayesian_score_unif_prior())
    # optimize G

    write_gph(G, idx2names, outfile)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()

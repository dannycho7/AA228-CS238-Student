import sys

from collections import defaultdict
import numpy as np
from pandas import read_csv
import random

def export_policy(policy, filename):
    with open(filename, 'w') as f:
        for i, (_, a) in enumerate(sorted(policy.items())):
            maybe_newline = "\n" if (i != len(policy) - 1) else ""
            f.write(f"{a}{maybe_newline}")

def random_policy(samples_df):
    states = set(samples_df['s'])
    actions = set(samples_df['a'])
    return {s: random.sample(actions, 1)[0] for s in states}

def q_learn_policy(samples_df, discount=0.9, lr=0.1, num_iter=5):
    Q = defaultdict(lambda: defaultdict(int))
    for iter in range(num_iter):
        print(f"Processing iter {iter + 1}")
        for i in samples_df.index:
            sample = samples_df.iloc[i]
            max_Q_sp_a = max(v for v in Q[sample['sp']].values()) if len(Q[sample['sp']]) > 0 else 0
            Q[sample['s']][sample['a']] += lr * (sample['r'] + discount * (max_Q_sp_a - Q[sample['s']][sample['a']]))
    return {s: max((v, a) for a, v in av.items())[1] for s, av in Q.items()}

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project2.py <infile>.csv <outfile>.policy")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    samples_df = read_csv(inputfilename)
    # export_policy(random_policy(samples_df), outputfilename)
    export_policy(q_learn_policy(samples_df), outputfilename)

if __name__ == '__main__':
    main()

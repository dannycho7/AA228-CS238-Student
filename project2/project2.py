import sys

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

def q_learn_policy(samples_df):
    pass

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project2.py <infile>.csv <outfile>.policy")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    samples_df = read_csv(inputfilename)
    export_policy(random_policy(samples_df), outputfilename)
    # q_learn_policy(samples_df)

if __name__ == '__main__':
    main()

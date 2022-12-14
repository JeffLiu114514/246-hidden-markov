# File: classify.py
# Purpose:  Starter code for the main experiment for CSC 246 P3 F22.

import argparse
from hmm import *


def constructor(filename, sample):
    with open(filename, 'rb') as inp:
        pi = np.load(inp)
        A = np.load(inp)
        B = np.load(inp)
        N = np.load(inp)
        observable_token = np.load(inp)
    return HMM_one(N, observable_token, sample, pi, A, B)


def main():
    parser = argparse.ArgumentParser(description='Program to test a neural network.')
    parser.add_argument('--test_path', default=None, help='Path to the test data.')
    parser.add_argument('--num_test_files', type=int, default=1000, help='Number of test files used.')

    args = parser.parse_args()

    num_files = args.num_test_files

    correct = 0
    total = 0

    # test samples from positive datapath    
    samples = load_subdir(os.path.join(args.test_path, 'pos'), num_files)
    for sample in samples:
        pos_hmm = constructor("pos_model", sample)
        neg_hmm = constructor("neg_model", sample)
        pos_prob = pos_hmm.test()
        neg_prob = neg_hmm.test()
        if pos_prob > neg_prob:
            correct += 1
        total += 1

    # test samples from negative datapath
    samples = load_subdir(os.path.join(args.test_path, 'neg'), num_files)
    for sample in samples:
        pos_hmm = constructor("pos_model", sample)
        neg_hmm = constructor("neg_model", sample)
        pos_prob = pos_hmm.test()
        neg_prob = neg_hmm.test()
        if pos_prob < neg_prob:
            correct += 1
        total += 1

    # report accuracy  (no need for F1 on balanced data)
    print(f"{correct / total}")


if __name__ == '__main__':
    main()

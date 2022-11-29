# File: classify.py
# Purpose:  Starter code for the main experiment for CSC 246 P3 F22.

import argparse  
from hmm import *

def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--pos_hmm', default=None, help='Path to the positive class hmm.')
    parser.add_argument('--neg_hmm', default=None, help='Path to the negative class hmm.')
    parser.add_argument('--datapath', default=None, help='Path to the test data.')

    args = parser.parse_args()

    # Load HMMs 
    pos_hmm = load_hmm(args.pos_hmm)
    neg_hmm = load_hmm(args.neg_hmm)

    correct = 0
    total = 0

    # test samples from positive datapath    
    samples = load_subdir(os.path.join(args.datapath, 'pos'))
    for sample in samples:
        if pos_hmm.LL(sample) > neg_hmm.LL(sample):
            correct += 1
        total += 1
            
    # test samples from negative datapath
    samples = load_subdir(os.path.join(args.datapath, 'neg'))
    for sample in samples:
        if pos_hmm.LL(sample) < neg_hmm.LL(sample):
            correct += 1
        total += 1
        
    # report accuracy  (no need for F1 on balanced data)
    print("%d/%d correct; accuracy %f"%(correct, total, correct/total))
    
if __name__ == '__main__':
    main()

# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.

import argparse
import copy
import math
import os
import numpy as np
from tqdm import tqdm


# A utility class for bundling together relevant parameters - you may modify if you like.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# num_states -- this should be an integer recording the number of hidden states
#
# pi -- this should be the distribution over the first hidden state of a sequence
#
# transitions -- this should be a num_states x num_states matrix of transition probabilities
#
# emissions -- this should be a num_states x vocab_size matrix of emission probabilities
#              (i.e., the probability of generating token X when operating in state K)
#
# vocab_size -- this should be an integer recording the vocabulary size; 255 is a safe upper bound
#
# Note: You may want to add fields for expectations.
class HMM_one:
    def __init__(self, num_states, observable_token, sample, pi, A, B):
        if isinstance(observable_token, np.ndarray):
            self.observable_token = observable_token.tolist()
        else:
            self.observable_token = observable_token
        self.M = len(observable_token)
        self.N = num_states
        self.T = len(sample)
        self.sample = sample
        self.pi = pi
        self.A = A
        self.B = B

        self.logProb = - math.inf

        self.alpha = np.zeros(shape=(self.T, self.N))
        self.beta = np.zeros(shape=(self.T, self.N))
        self.gamma = np.zeros(shape=(self.T, self.N))
        self.digamma = np.zeros(shape=(self.T - 1, self.N, self.N))
        self.c = np.zeros(self.T)

    def test(self):
        self.alpha_pass()
        self.compute_log()
        return self.logProb

    def alpha_pass(self):
        # compute alpha0(i)
        self.alpha = np.zeros(shape=(self.T, self.N))
        self.alpha[0, :] = self.pi * self.B[:, self.observable_token.index(self.sample[0])].T

        # scale the alpha0(i)
        self.c[0] = 1 / (self.alpha[0, :]).sum()
        self.alpha[0, :] *= self.c[0]

        for t in range(1, self.T):
            self.alpha[t, :] = self.alpha[t - 1, :].dot(self.A) * self.B[:,
                                                                  self.observable_token.index(self.sample[t])].T
            self.c[t] = 1 / (self.alpha[t, :]).sum()
            self.alpha[t, :] *= self.c[t]

    def beta_pass(self):
        # Let βT −1(i) = 1, scaled by cT −1
        self.beta = np.zeros(shape=(self.T, self.N))
        self.beta[-1, :] = 1 * self.c[-1]

        # β -pass
        for t in range(self.T - 2, -1, -1):
            # for i in range(0, self.N):
            #     self.beta[t][i] = 0
            #     for j in range(0, self.N):
            #         self.beta[t][i] += self.A[i][j] * self.B[j][self.sample[t + 1]] * self.beta[t + 1][j]
            #     # scale βt(i) with same scale factor as αt(i)
            #     self.beta[t][i] = self.c[t] * self.beta[t][i]
            self.beta[t, :] = (self.A @ (
                    self.B[:, self.observable_token.index(self.sample[t + 1])] * self.beta[t + 1]))
            self.beta[t, :] *= self.c[t]

    def gammas(self):
        # gamma
        self.gamma = (self.alpha * self.beta) / self.c.reshape(-1, 1)

        # digamma
        for t in range(0, self.T - 1):
            # self.digamma[t] = self.alpha[t] * self.A * self.B[:, self.sample[t + 1]] * self.beta[t + 1]
            alpha_part = self.alpha[t, :].reshape(-1, 1) * self.A
            beta_part = self.B[:, self.observable_token.index(self.sample[t + 1])] * self.beta[t + 1].reshape(1, -1)
            self.digamma[t, :, :] = alpha_part * beta_part

    def compute_log(self):
        self.logProb = -(np.log(self.c)).sum()

    def run(self):
        self.alpha_pass()
        self.beta_pass()
        self.gammas()
        self.compute_log()
        # print("alpha")
        # print(self.alpha)
        # print("beta")
        # print(self.beta)
        # print("digamma")
        # print(self.digamma)


class HMM_all:
    # The constructor should initalize all the model parameters.
    # you may want to write a helper method to initialize the emission probabilities.
    def __init__(self, num_states, observable_token, max_iters, samples, tolerance, pos_or_neg):
        self.M = len(observable_token)
        self.observable = observable_token
        self.N = num_states
        self.samples = samples
        self.num_files = len(samples)
        self.tolerance = tolerance
        self.pos_or_neg = pos_or_neg

        # initialize πi ≈ 1/N, aij ≈ 1/N, bj (k) ≈ 1/M
        self.pi = np.random.uniform(0.5 / self.N, 1.5 / self.N, size=self.N)
        self.pi /= self.pi.sum()
        self.A = np.random.uniform(0.5 / self.N, 1.5 / self.N, size=self.N * self.N).reshape(self.N,
                                                                                             self.N)
        for row in self.A:
            row /= row.sum()
        self.B = np.random.uniform(0.5 / self.M, 1.5 / self.M, size=self.N * self.M).reshape(self.N,
                                                                                             self.M)
        for row in self.B:
            row /= row.sum()

        self.iters = 0
        self.max_iters = max_iters
        self.oldLL = - math.inf
        self.LL = - math.inf

    def reestimate(self):
        digammas, gammas, likelihoods = [], [], []
        for sample in tqdm(self.samples):
            hmm = HMM_one(self.N, self.observable, sample, self.pi, self.A, self.B)
            hmm.run()
            digamma, gamma, likelihood = hmm.digamma, hmm.gamma, hmm.logProb
            digammas.append(digamma)
            gammas.append(gamma)
            likelihoods.append(likelihood)

        # reestimate pi
        new_pi = np.zeros(self.N)
        for gamma in gammas:
            new_pi += gamma[0]
        new_pi /= len(self.samples)
        self.pi = new_pi

        # reestimate A
        numer = np.zeros(shape=(self.N, self.N))
        denom = np.zeros(shape=self.N)
        for digamma, gamma in zip(digammas, gammas):
            numer += digamma.sum(axis=0)
            denom += gamma[:-1].sum(axis=0)
        self.A = numer / denom.reshape(-1, 1)

        # reestimate B
        observable_index = [[self.observable.index(x) for x in sample] for sample in self.samples]
        numer = np.zeros(shape=(self.M, self.N))
        denom = np.zeros(shape=self.N)
        for i, gamma in enumerate(gammas):
            for j in range(len(observable_index[i])):
                numer[observable_index[i][j]] += gamma[j]
            denom += gamma.sum(axis=0)
        numer += 0.00000001
        self.B = numer.T / denom.reshape(-1, 1)

        n = copy.deepcopy(self.LL)
        self.LL = sum(likelihoods) / len(self.samples)
        self.oldLL = n

        # print("pi A B")
        # print(self.pi)
        # print(self.A)
        # print(self.B)

    def iteration(self):
        if self.iters == 0:
            self.iters += 1
            return True

        if self.iters < self.max_iters and self.LL > (self.oldLL + self.tolerance):
            print("iter = {} out of {}, log likelihood = {}.".format(self.iters, self.max_iters,
                                                                     self.LL))
            self.iters += 1
            return True
        else:
            if self.iters == self.max_iters:
                print("iter = {} out of {}, log likelihood = {}.".format(self.iters, self.max_iters,
                                                                         self.LL))
                print("Max iterations reached.")
                if self.pos_or_neg == "positive":
                    self.save_model("pos_model")
                else:
                    self.save_model("neg_model")
            else:
                if self.oldLL < self.LL < (self.oldLL + self.tolerance):
                    print("iter = {} out of {}, log likelihood = {}.".format(self.iters, self.max_iters,
                                                                             self.LL))
                    print("Increase in likelihood is smaller than tolerance. Converged.")
                    if self.pos_or_neg == "positive":
                        self.save_model("pos_model")
                    else:
                        self.save_model("neg_model")
                else:
                    print("iter = {} out of {}, log likelihood = {}.".format(self.iters, self.max_iters,
                                                                             self.LL))
                    print("Likelihood decreasing. Early stopping.")
            print("Converged in {} iterations.".format(self.iters))
            return False

    # Save the complete model to a file (most likely using np.save and pickles)
    def save_model(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'wb') as out:
            np.save(out, self.pi)
            np.save(out, self.A)
            np.save(out, self.B)
            np.save(out, self.N)
            np.save(out, self.observable)


# Load all the files in a subdirectory and return a giant list.
def load_subdir(path, num_files):
    data = []
    for filename in os.listdir(path)[:num_files]:
        with open(os.path.join(path, filename)) as fh:
            data.append(fh.read())
    return data


def train(hmm):
    while hmm.iteration():
        hmm.reestimate()


def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    # parser.add_argument('--dev_path', default=None, help='Path to development (i.e., testing) data.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--max_iters', type=int, default=30, help='The maximum number of EM iterations (default 30)')
    # parser.add_argument('--model_out', default=None, help='Filename to save the final model.')
    parser.add_argument('--hidden_states', type=int, default=10,
                        help='The number of hidden states to use. (default 10)')
    parser.add_argument('--num_files', type=int, default=1000, help='The number of files used to train. (default 1000)')
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='The minimum improvement per iter. It is regarded to be converged after the improvement '
                             'is less than tolerance. (default 0.01)')
    parser.add_argument('--both', type=int, default=0,
                        help='Train both pos and neg or not. 0 for both, 1 for pos, 2 for neg')
    args = parser.parse_args()
    # pos_hmm = "C:\\Users\\23566\\PycharmProjects\\CSC246_Project3\\dataset_aclImdbNorm\\aclImdbNorm\\train\\pos"
    # neg_hmm = "C:\\Users\\23566\\PycharmProjects\\CSC246_Project3\\dataset_aclImdbNorm\\aclImdbNorm\\train\\neg"
    # N = 4
    # maxIters = 100
    # num_files = 300
    # pos_samples = load_subdir(pos_hmm, num_files)
    # tolerance = 0.01
    all_observable_token = list(''.join(chr(i) for i in range(255)))
    num_files = args.num_files

    N = args.hidden_states
    maxIters = args.max_iters
    tolerance = args.tolerance

    if args.both == 0:
        prepare_train(os.path.join(args.train_path, "pos"), num_files, N, all_observable_token, maxIters, tolerance,
                      "positive")
        prepare_train(os.path.join(args.train_path, "neg"), num_files, N, all_observable_token, maxIters, tolerance,
                      "negative")
    elif args.both == 1:
        prepare_train(os.path.join(args.train_path, "pos"), num_files, N, all_observable_token, maxIters, tolerance,
                      "positive")
    elif args.both == 2:
        prepare_train(os.path.join(args.train_path, "neg"), num_files, N, all_observable_token, maxIters, tolerance,
                      "negative")

    # test
    # np.random.seed(42)
    #
    # observations = ['3L', '2M', '1S', '3L', '3L', '3L']
    #
    # states = ['1H', '2C']
    # observables = ['1S', '2M', '3L']
    #
    # hmm = HMM_all(2, [0, 1, 2], 100, [[2, 1, 0, 2, 2, 2]])
    #
    # train(hmm)


def prepare_train(train_path, num_files, N, all_observable_token, maxIters, tolerance, pos_or_neg):
    samples = load_subdir(train_path, num_files)
    observed_file_list = []
    for file in samples:
        observed_line = []
        for char in list(file):
            observed_line.append(char)
            # if char in observable:
            #     pass
            # else:
            #     observable.append(char)
        observed_file_list.append(observed_line)

    # observable_token = observable
    sample = observed_file_list

    print(
        "Using {} files to train {} model with {} hidden states and {} max iterations.".format(len(samples), pos_or_neg,
                                                                                               N, maxIters))
    hmm = HMM_all(N, all_observable_token, maxIters, sample, tolerance, pos_or_neg)
    train(hmm)


if __name__ == '__main__':
    main()

import argparse
import subprocess
import pandas as pd
import re


def main():
    parser = argparse.ArgumentParser(description='Program to train a hidden markov model and test the classifier.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--max_iters', type=int, default=30, help='The maximum number of EM iterations (default 30)')
    # parser.add_argument('--hidden_states', type=int, default=10,
    #                     help='The number of hidden states to use. (default 10)')
    parser.add_argument('--num_files', type=int, default=1000, help='The number of files used to train. (default 1000)')
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='The minimum improvement per iter. It is regarded to be converged after the improvement '
                             'is less than tolerance. (default 0.01)')

    args = parser.parse_args()

    data = dict()
    num_hidden = [40]
    re_max_iter = re.compile(r"Converged in (\d+) iterations")
    cur_array = []
    for i in num_hidden:
        train = subprocess.run(
            ["python", "hmm.py", "--train_path", f"{args.train_path}", "--max_iters", f"{args.max_iters}",
             "--hidden_states", f"{i}", "--num_files", f"{args.num_files}", "--tolerance", f"{args.tolerance}",
             "--both", f"{1}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        matched = re_max_iter.search(f"{train.stdout.decode('utf-8')}")
        print(matched.group(1))
        cur_array.append(matched.group(1))
        print(f"{i} hidden variables finished, result: {matched.group(1)}\n")

    data[1] = cur_array

    final = pd.DataFrame(data, index=num_hidden).T
    final.to_csv(f"./result_convergence.csv")


if __name__ == '__main__':
    main()

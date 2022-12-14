CSC246 Project3
Junfei Liu (jliu137)

run instruction:
For classifier:
you can train the model and test the classifier simply by running
>python test_classifier.py --train_path --max_iters --num_files --tolerance --test_path --num_test_files
with correct parameters after each argument correspondingly. Type
>python test_classifier.py -h
for more detailed information.
For example, I run by the following on my device:
>python test_classifier.py --train_path C:\\Users\\23566\\PycharmProjects\\CSC246_Project3\\dataset_aclImdbNorm\\aclImdbNorm\\train --max_iters 10 --num_files 100 --tolerance 0.01 --test_path C:\\Users\\23566\\PycharmProjects\\CSC246_Project3\\dataset_aclImdbNorm\\aclImdbNorm\\test --num_test_files 100

For convergence:
you can train the model and test the convergence simply by running
>python test_convergence.py --train_path --max_iters --num_files --tolerance
with correct parameters after each argument correspondingly. Type
>python test_convergence.py -h
for more detailed information.
For example, I run by the following on my device:
>python test_convergence.py --train_path C:\\Users\\23566\\PycharmProjects\\CSC246_Project3\\dataset_aclImdbNorm\\aclImdbNorm\\train --max_iters 10 --num_files 100 --tolerance 0.01

or if you want to train model on your own, you can type
>python hmm.py --train_path --max_iters --hidden_states --num_files --tolerance
with correct parameters after each argument correspondingly. Type
>python hmm.py -h
for more detailed information.
For example, I run by the following on my device:
>python hmm.py --train_path C:\\Users\\23566\\PycharmProjects\\CSC246_Project3\\dataset_aclImdbNorm\\aclImdbNorm\\train --max_iters 10 --hidden_states 5 --num_files 1000 --tolerance 0.01


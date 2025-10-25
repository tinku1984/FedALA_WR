import h5py
import numpy as np

file_path = '/media/sparsh/NCLAB/pFL/PFLlib/results/MNIST_FedAvg_test_0.h5'

with h5py.File(file_path, 'r') as hf:
    rs_test_acc = np.array(hf.get('rs_test_acc'))
    print(rs_test_acc)
import sys
import scipy.io
import numpy as np
from skimage.io import imread
from pdb import set_trace
from datetime import datetime

def eprint(*args, **kwargs):
    print(str(datetime.now().strftime('%H:%M:%S')),":", *args, file=sys.stderr, **kwargs)
    sys.stderr.flush()
    sys.stdout.flush()

def parse_svhn(path):
    mat = scipy.io.loadmat(path)
    X = mat['X']
    y = mat['y'][:, 0]
    return X, y

class InputLoader(object):


    def __init__(self, dataset):
        self.dataset = dataset
        self.load_dataset_names()

    def load_dataset_names(self):
        if self.dataset == 'svhn':
            self.train_X, self.train_y = parse_svhn('svhn/train_32x32.mat')
            self.test_X, self.test_y = parse_svhn('svhn/test_32x32.mat')

    def get_train_data(self):
        return self.train_X, self.train_y

    def get_test_data(self):
        return self.test_X, self.test_y

    def get_num_classes(self):
        if self.dataset == 'svhn':
            return 10

def main():
    input_loader = InputLoader('svhn')

if __name__=='__main__':
    main()

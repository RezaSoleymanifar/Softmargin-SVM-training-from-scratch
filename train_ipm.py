import numpy as np
from SVMIPM import SVM
import sys
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


lmbda = sys.argv[1]
max_it  = sys.argv[2]
seed = sys.argv[3] if sys.argv[3] != 'None' else None

obj = SVM(data = data, lmbda = lmbda, max_it = max_it, seed = seed)
obj.int_pnt_method()

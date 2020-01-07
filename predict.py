import numpy as np
import sys
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
data = load_breast_cancer()

seed = sys.argv[1]

def asgn_lbl(vec):
    """
    Assigns -1 or +1 depending on sign
    :param vec: input vector
    :return: vector with same length as vec with -1, and +1 entries
    """
    temp_vec = vec[:]
    for i in range(len(vec)):
        if vec[i] >= 0:
            temp_vec[i] = +1
        elif vec[i] < 0:
            temp_vec[i] = -1
    return temp_vec

cur_theta, cur_sol = np.load(seed, allow_pickle=True)
X, y = data['data'], data['target']
m, n = np.shape(X)
y[y == 0] = -1
w = cur_sol[0:n]
b = cur_sol[-2]
y_pred = X @ w + b
y_pred = asgn_lbl(y_pred)
result = accuracy_score(y, y_pred)
print('accuracy: ', result)
np.savetxt("predictions.csv", y_pred, delimiter=",")

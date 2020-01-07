import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
#Please make sure to open HW5.py to run the model

class SVM:

    def __init__(self, data, lmbda = 1, max_it = 100, seed = None):
        """
        Initializes an SVM instance
        :param lmbda: \Lambda as in documentation
        :param max_it: maximum number of iterations
        """
        self.max_it = int(max_it)
        self.lmbda = float(lmbda)
        self.X, self.y = data['data'], data['target']
        self.y[self.y == 0] = -1
        self.y = self.y.reshape(-1, 1)
        self.m, self.n = np.shape(self.X)
        self.cur_sol, self.cur_Q = self.gen_seed(seed)
        self.sys_update()
        self.obj_val_hist = [self.cur_obj_val]
    
    def asgn_lbl(self, vec):
        """
        Assigns +1 or -1 depending on value of vec
        :param vec: input
        :return: Array of +1 and -1's
        """
        temp_vec = vec[:]
        for i in range(len(vec)):
            if vec[i] > 0:
                temp_vec[i] = +1
            elif vec[i] <= 0:
                temp_vec[i] = -1
        return temp_vec

    def gen_seed(self, seed):
        """
        Initializes a starting point for Ellipsoid method
        :return: ellipsoid center and matrix
        """
        if seed == None:
            w0 = np.zeros((self.n, 1))
            b0 = 0
            cur_sol = np.block([[w0], [b0]])
            R = 1000
            Q = R * np.eye(self.n + 1)
            
        else:
            result = np.load(seed, allow_pickle=True)[1]
            w0 = result[0: self.n]
            b0 = result[-2][0]
            R = 2
            Q = R * np.eye(self.n + 1)
            cur_sol = np.block([[w0], [b0]])
        return cur_sol, Q

    def calc_pmtrs(self):
        """
        Calculates parameters w and bias b out of current solution
        :return: w, b
        """
        w = self.cur_sol[0: self.n]
        b = self.cur_sol[-1][0]
        return w, b

    def calc_obj_val(self):
        #Calculates the objective value corresponding to documentation
        result = (1 / self.m) * np.sum(self.g) + self.lmbda * ((np.linalg.norm(self.cur_w)) ** 2)
        return result

    def calc_g(self):
        """Calculates g_i corresponding to functions in documentation"""
        g = np.maximum(0, 1 - np.multiply(self.y, self.X @ self.cur_w + self.cur_b))
        return g

    def calc_grad_g(self):
        """
        Calculates gradient of functions g_i
        :return: gradient vector (n + 1, 1)
        """
        grad_g = [None] * self.m
        for i in range(self.m):
            if self.g[i] > 0:
                grad_g[i] = (-self.y[i]) * np.block([[(self.X[i]).reshape(-1,1)],[1]])
            else:
                grad_g[i] = np.zeros((self.n + 1, 1))
        return grad_g

    def calc_grad(self):
        """
        Calls the first order oracle to return gradient at current solution
        :return: gradient (n + 1, 1)
        """
        result = (1 / self.m) * sum(self.grad_g) + np.vstack((2 * self.lmbda * self.cur_w, 0))
        return result

    def calc_pred_err(self):
        """
        Calculates the classification prediction error
        :return:
        """
        y_pred = self.X @ self.cur_w + self.cur_b
        y_pred = self.asgn_lbl(y_pred)
        result = accuracy_score(self.y, y_pred)
        return result

    def save_sol(self):
        """
        Saves current solution to a pickle file
        :return: None
        """
        eps0 = np.maximum(0, 1 - np.multiply(self.y, self.X @ self.cur_w + self.cur_b)) + 0.5
        t0 = (1 / self.m) * np.sum(eps0) + self.lmbda * ((np.linalg.norm(self.cur_w)) ** 2) + 0.5
        save_sol = np.block([[self.cur_w.reshape(-1, 1)],[eps0.reshape(-1, 1)],[self.cur_b],[t0]])
        np.save('best_seed_elps.npy', [1, save_sol])
        return None

    def sys_update(self):
        """
        Updates instance variables with most recent solution
        :return:
        """
        self.cur_w, self.cur_b = self.calc_pmtrs()
        self.g = self.calc_g()
        self.grad_g = self.calc_grad_g()
        self.grad = self.calc_grad()
        self.cur_obj_val = self.calc_obj_val()


    def elpsd_method(self):
        """
        Runs the ellipsoid method
        :return: Objective value history
        """
        N = self.n + 1
        for it_ctr in range(self.max_it):
            print('---------------')
            print('Ellipsoid // Iter #', it_ctr + 1)
            print('---------------')
            print('obj: ', self.cur_obj_val)
            print('log-norm: = ', np.log(np.linalg.norm(self.grad)))
            print('accuracy: ', self.calc_pred_err())

            self.cur_sol = self.cur_sol - (1 / (N + 1)) * ((self.cur_Q @ self.grad)
                                                              / (np.sqrt(self.grad.T @ self.cur_Q @ self.grad)))

            self.cur_Q = (N**2 / (N**2 -1)) * (self.cur_Q - (2 / (N + 1)) *
                                               ((self.cur_Q @ (self.grad @ self.grad.T) @ self.cur_Q)
                                                                            / (self.grad.T @ self.cur_Q @ self.grad)))
            self.sys_update()

            self.obj_val_hist.append(self.cur_obj_val)
            plt.plot(self.obj_val_hist, 'b')
            plt.title('Learning curve')
            plt.xlabel('Iterations', fontsize=14)
            plt.ylabel('Objective function', fontsize=14)
            plt.grid(True)
            plt.pause(0.001)
            plt.clf()
            if it_ctr % 10 == 0:
                self.save_sol()
        return None

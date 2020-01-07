# %load SVMIPM.py
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#The eipgraph form of SVM is transformed into a QCLP. variables are stacked as [w, \epsilon, b, t].
#Please make sure to open HW5.py to run the model

class SVM:
    def __init__(self, data, lmbda = 1, max_it = 100, seed=None):
        """
        Initializes IPM method
        :param lmbda: equivalent to \lambda in documentation
        :param max_it: maximum number of iterations (phase 2)
        """
        self.max_it = int(max_it)
        self.lmbda = float(lmbda)
        self.obj_val_hist = []
        self.X, self.y = data['data'], data['target']
        self.y[self.y == 0] = -1
        self.m, self.n = np.shape(self.X)
        self.gen_seed(seed)
        self.c, self.Gamma, self.gamma = self.init_qdrc()
        self.sys_update()
    
    def gen_seed(self, seed):
        """
        Generates a starting point for IPM
        :return: a feasible solution with it's corresponding theta
        """
        if seed == None:
            w0 = np.zeros(self.X.shape[1])
            b0 = 0
            eps0 = np.maximum(0, 1 - np.multiply(self.y, self.X @ w0 + b0)) + 0.5
            t0 = (1 / self.m) * np.sum(eps0) + self.lmbda * ((np.linalg.norm(w0)) ** 2) + 0.5
            self.cur_sol = np.block([[w0.reshape(-1, 1)],[eps0.reshape(-1, 1)],[b0],[t0]])
            self.cur_theta = 1
        else:
            self.cur_theta, self.cur_sol = np.load(seed, allow_pickle=True)
        return None

    def base_vec(self, size, idx):
        """
        Creates a basis vector
        :param size: size of basis vector
        :param idx: index idx will be set to one
        :return: base vector, shape(size, 1)
        """
        vec = np.zeros((size, 1))
        vec[idx] = 1
        return vec

    def asgn_lbl(self, vec):
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

    def calc_pmtrs(self):
        """
        Retrieves w, b, \epsilon, t out of current solution
        :return: w, \epsilon, b
        """
        w = self.cur_sol[0: self.n]
        eps = self.cur_sol[self.n: -2]
        b = self.cur_sol[-2][0]
        return w, eps, b

    def init_qdrc(self):
        """
        Initializes the QCLP formulation of SVM
        :return: Coefficients of QCLP
        """
        Gamma = np.block([[np.eye(self.n), np.zeros((self.n,self.m+2))],
                                  [np.zeros((self.m+2, self.n)), np.zeros((self.m+2, self.m+2))]])
        gamma = [None]*(self.m + 1)
        c = np.block([[np.zeros((self.n + self.m + 1,1))], [1]])
        for i in range(self.m+1):
            if i == 0:
                gamma[i] = np.block([[np.zeros((self.n, 1))], [(-1 / self.m) * np.ones((self.m, 1))], [0], [1]])
            else:
                gamma[i] = np.block([[(self.y[i-1]*self.X[i-1]).reshape(-1,1)],
                                     [self.base_vec(self.m, i-1)], [self.y[i-1]], [0]])
        return c, Gamma, gamma

    def init_qdrc2(self):
        """
        Initializes the QCLP formulation of SVM
        :return: QCLP constraint values
        """
        q = [None]*(self.m + 1)
        u = [None]*(self.m)
        for i in range(self.m+1):
            if i == 0:
                q[i] = (-self.lmbda * ((self.cur_sol.T @ self.Gamma @ self.cur_sol))
                        + (self.gamma[i].T @ self.cur_sol))[0][0]
            else:
                q[i] = (self.gamma[i].T @ self.cur_sol -1)[0][0]

        for i in range(self.m):
            u[i] = ((self.base_vec(self.m + self.n + 2, i+self.n)).T @ self.cur_sol)[0][0]
        return q, u

    def calc_Ft_val(self):
        """
        Calculates IPM objective value
        :return: F_{\theta}(x) = \theta * c @ x + F(x)
        """
        temp_q = [np.log(self.q[i]) for i in range(self.m + 1)]
        temp_u = [np.log(self.u[i]) for i in range(self.m)]
        result = (self.cur_theta * (self.c.T) @ self.cur_sol)[0][0] - sum(temp_q) - sum(temp_u)
        return result

    def calc_obj_val(self):
        """
        Calculates SVM objective value
        :return: SVM objective value
        """
        g = np.maximum(0, 1 - np.multiply(self.y, self.X @ self.cur_w + self.cur_b))
        result = (1 / self.m) * np.sum(g) + self.lmbda * ((np.linalg.norm(self.cur_w)) ** 2)
        return result

    def calc_grad_hess(self):
        """
        Calculates the gradient and hessian of QCLP constraints
        """
        grad_q = [None] * (self.m + 1)
        grad_u = [None] * self.m
        for i in range(self.m+ 1):
            if i == 0 :
                grad_q[i] = -2 * self.lmbda * (self.Gamma @ self.cur_sol) + self.gamma[i]
            else:
                grad_q[i] = self.gamma[i]
        for i in range(self.m):
            grad_u[i] = self.base_vec(self.m + self.n + 2, i + self.n)
        hess_q0 = -2 * self.lmbda * self.Gamma
        return grad_q, grad_u, hess_q0

    def calc_grad_hess_F(self):
        """
        Calculates the gradient and hessian of F_{\theta}
        :return:
        """
        temp_q = [-(self.grad_q[i]/self.q[i]) for i in range(self.m + 1)]
        temp_u = [-(self.grad_u[i]/self.u[i]) for i in range(self.m)]
        grad_F = sum(temp_u)+ sum(temp_q)
        temp_q = [((self.grad_q[i]) @ (self.grad_q[i]).T) / (self.q[i] ** 2) for i in range(self.m + 1)]
        temp_u = [((self.grad_u[i]) @ (self.grad_u[i]).T) / (self.u[i] ** 2) for i in range(self.m)]
        hess_F = sum(temp_u) + sum(temp_q) - self.hess_q0 / (self.q[0])
        return grad_F, hess_F

    def calc_Ntn_dcr(self):
        """
        Calculates the Newton Decrement
        :return: Newton Decrement
        """
        result = np.sqrt((self.cur_theta * self.c + self.grad_F).T @
                         self.inv_hess_F @ (self.cur_theta * self.c + self.grad_F))[0][0]
        return result

    def save_sol(self):
        """
        Saves current solution to an npy file
        :return: None
        """
        np.save('best_seed.npy', [self.cur_theta, self.cur_sol])

    def calc_pred_acc(self):
        """
        Calculates the SVM classification prediction accuracy
        :return: accuracy \in [0,1]
        """
        w = self.cur_sol[0:self.n]
        b = self.cur_sol[-2]
        y_pred = self.X @ w + b
        y_pred = self.asgn_lbl(y_pred)
        result = accuracy_score(self.y, y_pred)
        return result

    def sys_update(self):
        """
        Updates all dependent variables with most the recent solution
        :return: None
        """
        self.cur_w, self.cur_eps, self.cur_b = self.calc_pmtrs()
        self.q, self.u = self.init_qdrc2()
        self.grad_q, self.grad_u, self.hess_q0 = self.calc_grad_hess()
        self.grad_F, self.hess_F = self.calc_grad_hess_F()
        self.inv_hess_F = np.linalg.inv(self.hess_F)
        self.cur_obj_val = self.calc_obj_val()
        self.cur_Ft_val = self.calc_Ft_val()
        self.Ntn_dcr = self.calc_Ntn_dcr()
        return None

    def initialize_ipm(self):
        """
        Runs until Newton Decrement < 0.25
        :return: None
        """
        it_ctr = 0
        while(self.Ntn_dcr > 0.25):
            it_ctr += 1
            print('---------------')
            print('Phase 1 // Iter #', it_ctr)
            print('---------------')
            print('Nd: ', self.Ntn_dcr)
            print('Ft ', self.cur_Ft_val)
            print('obj ', self.cur_obj_val)
            print('log-norm = ', np.log(np.linalg.norm(self.cur_theta * self.c + self.grad_F)))
            print('accuracy: ', self.calc_pred_acc())
            self.obj_val_hist.append(self.cur_obj_val)
            dmp_fac = 1 / (1 + self.Ntn_dcr)
            self.cur_sol = self.cur_sol - dmp_fac * self.inv_hess_F @  ((self.cur_theta * self.c) + self.grad_F)
            self.sys_update()
            
            if it_ctr % 10 == 0:
                self.save_sol()
        return None

    def int_pnt_method(self):
        """
        Interior point method implementation of SVM
        Uses initialization to reach quadratic convergence zone first
        :return: Objective value history
        """
        if self.Ntn_dcr > 0.25:
            self.initialize_ipm()
        plt.ion()
        for it_ctr in range(self.max_it):
            print('---------------')
            print('Phase 2 // Iter #', it_ctr + 1)
            print('---------------')
            print('Nd: ', self.Ntn_dcr)
            print('obj: ', self.cur_obj_val)
            print('Ft: ', self.cur_Ft_val)
            print('theta: ', self.cur_theta)
            print('log-norm: = ', np.log(np.linalg.norm(self.cur_theta * self.c + self.grad_F)))
            print('accuracy: ', self.calc_pred_acc())
            g_rate = 1.5
            v = 2 * self.m + 1
            self.obj_val_hist.append(self.cur_obj_val)
            plt.plot(self.obj_val_hist)
            plt.title('Learning curve')
            plt.xlabel('Iterations', fontsize=14)
            plt.ylabel('Objective function', fontsize=14)
            plt.grid(True)
            plt.pause(0.001)
            plt.clf()
            self.cur_theta = self.cur_theta * (1 + g_rate / np.sqrt(v))
            self.cur_sol = self.cur_sol - self.inv_hess_F @ (self.cur_theta * self.c + self.grad_F)
            self.sys_update()

            if it_ctr % 10 == 0:
                self.save_sol()
        return None

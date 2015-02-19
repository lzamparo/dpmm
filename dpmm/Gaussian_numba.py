# -*- coding: utf-8 -*-
import itertools, random, sys

import numpy as np
from scipy import linalg
import math


class Gaussian:
    def __init__(self, X=np.zeros((0,1)), kappa_0=0, nu_0=1.0001, mu_0=None, 
            Psi_0=None): # Psi is also called Lambda or T
        # See http://en.wikipedia.org/wiki/Conjugate_prior 
        # Normal-inverse-Wishart conjugate of the Multivariate Normal
        # or see p.18 of Kevin P. Murphy's 2007 paper:"Conjugate Bayesian 
        # analysis of the Gaussian distribution.", in which Psi = T
        self.n_points = X.shape[0]
        self.n_var = X.shape[1]

        self._hash_covar = None
        self._inv_covar = None

        if mu_0 == None: # initial mean for the cluster
            self._mu_0 = np.zeros((1, self.n_var))
        else:
            self._mu_0 = mu_0
        assert(self._mu_0.shape == (1, self.n_var))

        self._kappa_0 = kappa_0 # mean fraction

        self._nu_0 = nu_0 # degrees of freedom
        if self._nu_0 < self.n_var:
            self._nu_0 = self.n_var
        self.nu = nu_0 - self.n_var + 1

        if Psi_0 == None:
            self._Psi_0 = 10*np.eye(self.n_var) # TODO this 10 factor should be a prior, ~ dependent on the mean distance between points of the dataset
        else:
            self._Psi_0 = Psi_0
        assert(self._Psi_0.shape == (self.n_var, self.n_var))

        if X.shape[0] > 0:
            self.fit(X)
        else:
            self.default()


    def default(self):
        self.mean = np.matrix(np.zeros((1, self.n_var))) # TODO init to mean of the dataset
        self.covar = 100.0 * np.matrix(np.eye(self.n_var)) # TODO change 100


    def recompute_ss(self):
        """ need to have actualized _X, _sum, and _square_sum """ 
        self.n_points = self._X.shape[0]
        self.n_var = self._X.shape[1]
        if self.n_points <= 0:
            self.default()
            return

        kappa_n = self._kappa_0 + self.n_points
        nu = self._nu_0 + self.n_points 
        mu = np.matrix(self._sum) / self.n_points
        mu_mu_0 = mu - self._mu_0

        C = self._square_sum - self.n_points * (mu.transpose() * mu)
        Psi = (self._Psi_0 + C + self._kappa_0 * self.n_points
             * mu_mu_0.transpose() * mu_mu_0 / (self._kappa_0 + self.n_points))

        self.mean = ((self._kappa_0 * self._mu_0 + self.n_points * mu) 
                    / (self._kappa_0 + self.n_points))
        self.covar = (Psi * (kappa_n + 1)) / (kappa_n * (nu - self.n_var + 1))
        assert(np.linalg.det(self.covar) != 0)
        self.nu = nu - self.n_var + 1


    def inv_covar(self):
        """ memoize the inverse of the covariance matrix """
        if self._hash_covar != hash(self.covar):
            self._hash_covar = hash(self.covar)
            self._inv_covar = np.linalg.inv(self.covar)
        return self._inv_covar


    def fit(self, X):
        """ to add several points at once without recomputing """
        self.n_points = X.shape[0]
        self.n_var = X.shape[1]
        self._X = X
        self._sum = X.sum(0)
        self._square_sum = np.matrix(X).transpose() * np.matrix(X)
        self.recompute_ss()

    
    def add_point(self, x):
        """ add a point to this Gaussian cluster """
        if self.n_points <= 0:
            self._X = np.array([x])
            self._sum = self._X.sum(0)
            self._square_sum = np.matrix(self._X).transpose() * np.matrix(self._X)
        else:
            self._X = np.append(self._X, [x], axis=0)
            self._sum += x
            self._square_sum += np.matrix(x).transpose() * np.matrix(x)
        self.recompute_ss()


    def rm_point(self, x):
        """ remove a point from this Gaussian cluster """
        assert(self._X.shape[0] > 0)
        # Find the indice of the point x in self._X, be careful with
        indices = (abs(self._X - x)).argmin(axis=0)
        indices = np.matrix(indices)
        ind = indices[0,0]
        for ii in indices:
            if (ii-ii[0] == np.zeros(len(ii))).all(): # ensure that all coordinates match (finding [1, 1] in [[1, 2], [1, 1]] would otherwise return indice 0)
                ind = ii[0,0]
                break
        tmp = np.matrix(self._X[ind])
        self._sum -= self._X[ind]
        self._X = np.delete(self._X, ind, axis=0)
        self._square_sum -= tmp.transpose() * tmp
        self.recompute_ss()


    def pdf(self, x):
        """ probability density function for a multivariate Gaussian """
        size = len(x)
        assert(size == self.mean.shape[1])
        assert((size, size) == self.covar.shape)
        det = np.linalg.det(self.covar)
        assert(det != 0)
        norm_const = 1.0 / (math.pow((2*np.pi), float(size)/2) 
                * math.pow(det, 1.0/2))
        x_mu = x - self.mean
        inv = self.covar.I        
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.transpose()))
        return norm_const * result


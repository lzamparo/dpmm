# -*- coding: utf-8 -*-
from numpy import argsort, bincount, ones, where, zeros
from numpy.random import multivariate_normal, seed
from Gaussian import Gaussian
from kale.math_utils import sample

import numpy as np

def generate_data(V, D, alpha):
    """
    Generates a synthetic corpus of data points from a Dirichlet process
    mixture model with Gaussian mixture components. 
    
    Another one is:
    1) generate cluster assignments c_1, ..., c_N ~ CRP(N, α) (K clusters)
    2) generate parameters Φ_1, ...,Φ_K ~ G_0
    3) generate each datapoint y_i ~ F(Φ_{c_i})

    So we have P(y | Φ_{1:K}, β_{1:K}) = \sum_{j=1}^K β_j Norm(y | μ_j, S_j)
    
    Except we don't explicitly represent the β_{1:K}
    
    Arguments:

    V -- number of features (N_DV.shape[1])
    D -- number of data points (N_DV.shape[0])
    alpha -- concentration parameter for the Dirichlet process
    """

    T = D # maximum number of components

    phi_TV = {}      # component parameters, using dict of Gaussian objects
    z_D = zeros(D, dtype=int)   # component indicators
    N_DV = zeros((D, V), dtype=float)   # generated data

    for d in xrange(D):
        # draw a cluster assignment for this data point

        dist = bincount(z_D).astype(float)
        dist[0] = alpha
        [t] = sample(dist)
        t = len(dist) if t == 0 else t
        z_D[d] = t

        # if it's a new cluster, draw the parameters for that component 
        # and draw & add the point
        if t == len(dist):
            pt = multivariate_normal(np.asarray([0.0,0.0]), np.eye(V))
            phi_TV[t-1] = Gaussian(X=pt.reshape((1,V)))
        else:
            # draw a sample from the component
            pt = phi_TV[t-1].sample(size=1)
            phi_TV[t-1].add_point(pt.ravel())
        N_DV[d] = pt

    z_D = z_D - 1

    return phi_TV, z_D, N_DV

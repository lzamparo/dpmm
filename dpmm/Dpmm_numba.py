# -*- coding: utf-8 -*-
import itertools, random, sys

import numpy as np
from numpy.random import choice, uniform
from scipy import linalg
import math

from sklearn.cluster import MiniBatchKMeans
from Gaussian_numba import Gaussian

epsilon = 10e-8
BOOTSTRAP = False

import numba

"""
Dirichlet process mixture model (for N observations y_1, ..., y_N)
    1) generate a distribution G ~ DP(G_0, α)
    2) generate parameters θ_1, ..., θ_N ~ G
    [1+2) <=> (with B_1, ..., B_N a measurable partition of the set for which 
        G_0 is a finite measure, G(B_i) = θ_i:)
       generate G(B_1), ..., G(B_N) ~ Dirichlet(αG_0(B_1), ..., αG_0(B_N)]
    3) generate each datapoint y_i ~ F(θ_i)
Now, an alternative is:
    1) generate a vector β ~ Stick(1, α) (<=> GEM(1, α))
    2) generate cluster assignments c_i ~ Categorical(β) (gives K clusters)
    3) generate parameters Φ_1, ...,Φ_K ~ G_0
    4) generate each datapoint y_i ~ F(Φ_{c_i})
    for instance F is a Gaussian and Φ_c = (mean_c, var_c)
Another one is:
    1) generate cluster assignments c_1, ..., c_N ~ CRP(N, α) (K clusters)
    2) generate parameters Φ_1, ...,Φ_K ~ G_0
    3) generate each datapoint y_i ~ F(Φ_{c_i})

So we have P(y | Φ_{1:K}, β_{1:K}) = \sum_{j=1}^K β_j Norm(y | μ_j, S_j)
"""
@numba.jit
class DPMM:
    @numba.jit
    def _get_means(self):
        return np.array([g.mean for g in self.params.itervalues()])


    def _get_covars(self):
        return np.array([g.covar for g in self.params.itervalues()])

    
    def __init__(self, n_components=-1, alpha=1.0, do_sample_alpha=False, a=0.05, b=0.25):
        self.params = {0: Gaussian()}
        self.n_components = n_components
        self.means_ = self._get_means()
        self.alpha = alpha
        if do_sample_alpha:
            self.alpha_hyperpriors = (a,b)
            self.alpha_samples = []
    
    @numba.jit    
    def sample_alpha(self):
        """ Sample a new value for α based on the formulation in 
        West (http://web.cse.ohio-state.edu/~kulis/teaching/788_sp12/DP.learnalpha.pdf):
        1) sample x from (nu | alpha, k) ~ Beta(alpha + 1, n) 
        2) sample a new alpha from (α|x, k) ∼ π_x G(a + k, b − log(x)) + (1 − π_x)G(a + k − 1, b − log(x))
        where π_x = \frac{a + k-1}{a + k-1 + n(b - log(x))}
        """
        
        n = self.n_points
        k = self.n_components
        a,b = self.alpha_hyperpriors
        x = np.random.beta(self.alpha + 1, n)
        pi_x = (a + k - 1.0) / (a + k - 1.0 + n * (b - np.log(x)))
        return pi_x*(np.random.gamma(a + k, b - np.log(x))) + (1 - pi_x)*(np.random.gamma(a + k - 1.0, b - np.log(x)))

    def fit_conjugate_split_merge(self,X, do_sample_alpha=False):
        """ according to algorithm of Jain & Neal 2004 """        
        pass
    
    def split_merge_iteration(self,X):
        """ This is the cycling randomized split & merge according to algorithm of Jain & Neal 2004 """
        # (1) choose 2 data points
        d, e = choice(D, 2, replace=False) 
        
        # (2) grab all points in the component for d
        C_d = self.z[d]
        
        # (2) grab all points in the component for e
        C_e = self.z[e]
        # (2) form the union of the set of points d,e, but withhold the points themselves

        # (3) define the launch state: partition the points uniformly at random

        # (3) define the launch state: perform num_inner_itns restricted Gibbs sampling scans

        # (4) if z[d] == z[e] propose a split: c^{split} initialized from c^{launch}

        #       adopt the split if move is accepted

        # (5) if z[d] != z[e] propose a merge: c^{merge} initialized from c^{launch}

        #       adopt the merge if move is accepted 

        
        pass
    
    @numba.jit
    def fit_collapsed_Gibbs(self, X, do_sample_alpha=False, do_kmeans=False, max_iter=100):
        """ according to algorithm 3 of collapsed Gibbs sampling in Neal 2000:
        http://www.stat.purdue.edu/~rdutta/24.PDF """
        
        previous_means, previous_components = self.initialize_model(X,do_kmeans)

        n_iter = 0 # with max_iter hard limit, in case of cluster oscillations
        # while the clusters did not converge (i.e. the number of components or
        # the means of the components changed) and we still have iter credit 
        while (n_iter < max_iter 
                and (previous_components != self.n_components
                or abs((previous_means - self._get_means()).sum()) > epsilon)):
            previous_means = self._get_means()
            previous_components = self.n_components            
            self.gibbs_iteration(X, do_sample_alpha)
            n_iter += 1
            print "still sampling, %i clusters currently, with log-likelihood %f, alpha %f" % (self.n_components, self.log_likelihood(), self.alpha)

        self.means_ = self._get_means() 
    
    @numba.jit
    def initialize_model(self, X, do_kmeans=False):
        """ Initialize each component of the model in an appropriate way.  This is done one of three ways:
        - one component per point.  
        - randomly assigned labels for self.n_components > 0 components
        - from a short k-means run with k = round(alpha * log(X.shape[0])) """
        
        mean_data = np.matrix(X.mean(axis=0))
        self.n_points = X.shape[0]
        self.n_var = X.shape[1]
        self._X = X
        
        if self.n_components == -1:
            # initialize with 1 cluster for each datapoint
            self.params = dict([(i, Gaussian(X=np.matrix(X[i]), mu_0=mean_data)) for i in xrange(X.shape[0])])
            self.z = dict([(i,i) for i in range(X.shape[0])])
            self.n_components = X.shape[0]
            previous_means = (1.0 + 5*epsilon) * self._get_means()
            previous_components = self.n_components
        elif self.n_components != -1 and do_kmeans:
            # init with k-means
            batch_size = (np.floor(X.shape[0] / 10.0)).astype(int)
            mbk = MiniBatchKMeans(init='k-means++', n_clusters=self.n_components, batch_size=batch_size,
                                  n_init=10, max_no_improvement=10, verbose=0)
            mbk.fit(X)
            labels_from_kmeans = mbk.labels_  
            means_from_kmeans = mbk.cluster_centers_ 
            self.params = dict([(j, Gaussian(X=np.zeros((0, X.shape[1])), mu_0=m.reshape(mean_data.shape))) for j,m in enumerate(means_from_kmeans)])
            self.z = dict([(i, l) for i,l in enumerate(labels_from_kmeans)])
            previous_means = (1.0 + 5*epsilon) * self._get_means()
            previous_components = self.n_components
            for i in xrange(X.shape[0]):
                self.params[self.z[i]].add_point(X[i])
        else:
            # init randomly among self.n_component component labels
            self.params = dict([(j, Gaussian(X=np.zeros((0, X.shape[1])), mu_0=mean_data)) for j in xrange(self.n_components)])
            self.z = dict([(i, random.randint(0, self.n_components - 1)) 
                      for i in range(X.shape[0])])
            previous_means = (1.0 + 5*epsilon) * self._get_means()
            previous_components = self.n_components
            for i in xrange(X.shape[0]):
                self.params[self.z[i]].add_point(X[i])                

        print "Initialized collapsed Gibbs sampling with %i clusters" % (self.n_components)
        return previous_means, previous_components

    @numba.jit
    def gibbs_iteration(self, X, do_sample_alpha=False):
        """ Perform one full iteration of gibbs sampling """
        
        # randomize the order of points for the scan
        indices = [i for i in xrange(X.shape[0])]
        for i in np.random.permutation(indices):
            # remove X[i]'s sufficient statistics from z[i]
            self.params[self.z[i]].rm_point(X[i])
            # if it empties the cluster, remove it and decrease K
            if self.params[self.z[i]].n_points <= 0:
                self.params.pop(self.z[i])
                self.n_components -= 1

            tmp = []
            for k, param in self.params.iteritems():
                # compute P_k(X[i]) = P(X[i] | X[-i] = k)
                marginal_likelihood_Xi = param.pdf(X[i])
                # set N_{k,-i} = dim({X[-i] = k})
                # compute P(z[i] = k | z[-i], Data) = N_{k,-i}/(α+N-1)
                mixing_Xi = param.n_points / (self.alpha + self.n_points - 1)
                tmp.append(marginal_likelihood_Xi * mixing_Xi)
                
            # compute P*(X[i]) = P(X[i]|λ)
            base_distrib = Gaussian(X=np.zeros((0, X.shape[1])))
            prior_predictive = base_distrib.pdf(X[i])
            # compute P(z[i] = * | z[-i], Data) = α/(α+N-1)
            prob_new_cluster = self.alpha / (self.alpha + self.n_points - 1)
            tmp.append(prior_predictive * prob_new_cluster)

            # normalize P(z[i]) (tmp above)
            tmp = np.array(tmp)
            tmp /= tmp.sum()

            # sample z[i] ~ P(z[i])
            rdm = np.random.rand()
            total = tmp[0]
            k = 0
            while (rdm > total):
                k += 1
                total += tmp[k]
            # add X[i]'s sufficient statistics to cluster z[i]
            new_key = max(self.params.keys()) + 1
            if k == self.n_components: # create a new cluster
                self.z[i] = new_key
                self.n_components += 1
                self.params[new_key] = Gaussian(X=np.matrix(X[i]))
            else:
                self.z[i] = self.params.keys()[k]
                self.params[self.params.keys()[k]].add_point(X[i])
            assert(k < self.n_components)
       
        if do_sample_alpha:
            new_alpha = self.sample_alpha()
            self.alpha_samples.append(self.alpha)
            self.alpha = new_alpha

    def predict(self, X):
        """ produces and returns the clustering of the X data """
        if (X != self._X).any():
            self.fit_collapsed_Gibbs(X)
        mapper = list(set(self.z.values())) # to map our clusters id to
        # incremental natural numbers starting at 0
        Y = np.array([mapper.index(self.z[i]) for i in range(X.shape[0])])
        return Y

    @numba.jit
    def log_likelihood(self): # TODO! currently it's far from the full log-likelihood
        #logprior = self._bound_concentration()
        #logprior += self._bound_means()
        #logprior += self._bound_precisions()
        #logprior += self._bound_proportions(z)
        # TODO test the values (anyway it's just indicative right now)
        log_likelihood = 0.
        for n in xrange(self.n_points):
            log_likelihood -= (0.5 * self.n_var * np.log(2.0 * np.pi) + 0.5 
                        * np.log(np.linalg.det(self.params[self.z[n]].covar)))
            mean_var = np.matrix(self._X[n, :] - self.params[self.z[n]]._X.mean(axis=0)) # TODO should compute self.params[self.z[n]]._X.mean(axis=0) less often
            assert(mean_var.shape == (1, self.params[self.z[n]].n_var))
            log_likelihood -= 0.5 * np.dot(np.dot(mean_var, 
                self.params[self.z[n]].inv_covar()), mean_var.transpose())
            # TODO add the influence of n_components
        return log_likelihood


### BOOTSTRAP ###
def create_DPMM_and_fit(X):
    tmp = DPMM(n_components=-1)
    tmp.fit_collapsed_Gibbs(X)
    return tmp


def merge_gaussian(l):
    """ merge a list of Gaussian objects """
    # TODO should try without taking the number of data points 
    # assigned to each Gaussian, just merging means/len(l)...
    X_ = np.ndarray((0, l[0].n_var))
    for g in l:
        X_ = np.append(X_, g._X, axis=0)
    return Gaussian(X_)


def merge_models(l):
    """ (c)rude merging """
    # TODO this is going meta, the merging can be done with a clustering 
    # algorithm, why not a DP(G)MM? 
    # Currently using a nearest neighbor's search on means
    n_clusters = min([len(mixt.params.keys()) for mixt in l]) # TODO change
    n_mixt = len(l)
    print >> sys.stderr, "final n_clusters", n_clusters
    ret = DPMM(n_components=n_clusters)
    ret.n_points = X.shape[0]
    ret.n_var = X.shape[1]
    ret._X = X
    #means = []
    #mapper_means = [] # means indices to full (mixt_ind, gaussian object)
    #i = 0
    #for j,mixt in enumerate(l):
    #    for g in mixt.params.itervalues():
    #        means.append(g.mean)
    #        mapper_means.append(j, g)
    means = [np.squeeze(np.asarray(g.mean)) for mixt in l for g in mixt.params.itervalues()]
    full_gaussian = [g for mixt in l for g in mixt.params.itervalues()]
    from scipy.spatial import cKDTree
    kdt = cKDTree(means)
    done = []
    for i in xrange(n_clusters):
        min_ = 1E80
        indices_means = None
        for g in full_gaussian:
            if g in done: # do not merge clusters that we already merged
                continue
            q = kdt.query(np.squeeze(np.asarray(g.mean)), k=n_mixt)
            if q[0].sum() < min_:
                min_ = q[0].sum() # distances
                indices_means = q[1] # means/gaussian indices
        # here we can merge 2 (or more) clusters coming from the same mixture
        # (bootstrap element), TODO see if we should take only 1
        # (c.f. mapper_means commented code)
        doing_gaussian = [full_gaussian[k] for k in indices_means]
        done.extend(doing_gaussian)
        ret.params[i] = merge_gaussian(doing_gaussian)
    print >> sys.stderr, "final len(ret.params)", len(ret.params)
    not_merged = set(full_gaussian)-set(done)
    print >> sys.stderr, "not merged", [g.mean for g in not_merged]
    print >> sys.stderr, "number of points concerned", sum([g.n_points for g in not_merged]), "on total number of points", ret.n_points

    # recompute data points clusters assignment with the merged gaussian mixts
    ret.z = dict([(i, 0) for i in range(X.shape[0])])
    for i in xrange(X.shape[0]): 
        max_ = -1
        for k, param in ret.params.iteritems():
            marginal_likelihood_Xi = param.pdf(X[i])
            mixing_Xi = param.n_points * 1.0 / ret.n_points
            tmp = marginal_likelihood_Xi * mixing_Xi
            if tmp > max_:
                max_ = tmp
                ret.z[i] = k

    ret.Y = np.array([ret.z[i] for i in range(X.shape[0])])
    ret.means_ = ret._get_means() 
    return ret


def fit_bootstrap(X):
    n_obs = X.shape[0]
    from joblib import Parallel, delayed
    from multiprocessing import cpu_count
    n_jobs = cpu_count()
    ldpmm = Parallel(n_jobs=n_jobs)(delayed(create_DPMM_and_fit)(X[i*n_obs/n_jobs:(i+1)*n_obs/n_jobs]) for i in range(n_jobs))
    return merge_models(ldpmm)
### /BOOTSTRAP ###


if __name__ == "__main__":

    import pylab as pl
    import matplotlib as mpl    

    # Number of samples per component
    n_samples = 800

    # Generate random sample, two components
    np.random.seed(0)
    
    # Sample for a maximum of this many iterations
    max_iter = 100
    
    # 4, 2-dimensional Gaussians
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples/4., 2), C),
              .7 * np.random.randn(n_samples/8., 2) + np.array([-6, 3]),
              1.1 * np.random.randn(n_samples/8., 2) + np.array([3,-3]),
              1.2 * np.random.randn(n_samples/2., 2) - np.array([2,-6])]

    # 2, 2-dimensional Gaussians
    #C = np.array([[0., -0.1], [1.7, .4]])
    #X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              #.7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

    # 2, 10-dimensional Gaussians
    #C = np.eye(10)
    #for i in xrange(100):
    #    C[random.randint(0,9)][random.randint(0,9)] = random.random()
    #X = np.r_[np.dot(np.random.randn(n_samples, 10), C),
    #          .7 * np.random.randn(n_samples, 10) + np.array([-6, 3, 0, 5, -8, 0, 0, 0, -3, -2])]

    # 2, 5-dimensional Gaussians
    #C = np.eye(5)
    #for i in xrange(25):
    #    C[random.randint(0,4)][random.randint(0,4)] = random.random()
    #X = np.r_[np.dot(np.random.randn(n_samples, 5), C),
              #np.dot(np.random.randn(n_samples, 5), 0.5*C - 4), # 3rd Gaussian?
    #          .7 * np.random.randn(n_samples, 5) + np.array([-6, 3, 5, -8, -2])]

    if BOOTSTRAP:
        np.random.shuffle(X)
    from sklearn import mixture

    # Fit a mixture of gaussians with EM using five components
    gmm = mixture.GMM(n_components=6, covariance_type='full')
    gmm.fit(X)

    # Fit a dirichlet process mixture of gaussians using five components
    dpgmm = mixture.DPGMM(n_components=6, covariance_type='full')
    dpgmm.fit(X)

    dpmm = None
    if BOOTSTRAP:
        dpmm = fit_bootstrap(X)
    else:
        # n_components is the number of initial clusters (at random, TODO k-means init)
        # -1 means that we initialize with 1 cluster per point
        pre_alpha = 0.5
        n_components = pre_alpha * np.log(X.shape[0])
        dpmm = DPMM(n_components=n_components.astype(int),alpha=pre_alpha,do_sample_alpha=True) # -1, 1, 2, 5
        dpmm.fit_collapsed_Gibbs(X,do_sample_alpha=True,do_kmeans=True,max_iter=max_iter)

    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

    X_repr = X
    if X.shape[1] > 2:
        from sklearn import manifold
        X_repr = manifold.Isomap(n_samples/10, n_components=2).fit_transform(X)

    for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                      (dpmm, 'Dirichlet Process GMM (ours, Gibbs)'),
                                      (dpgmm, 'Dirichlet Process GMM (sklearn, Variational)')]):
        splot = pl.subplot(3, 1, 1 + i)
        Y_ = clf.predict(X)
        print Y_
        for j, (mean, covar, color) in enumerate(zip(
                clf.means_, clf._get_covars(), color_iter)):
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == j):
                continue

            pl.scatter(X_repr[Y_ == j, 0], X_repr[Y_ == j, 1], .8, color=color)

            if clf.means_.shape[len(clf.means_.shape) - 1] == 2: # hack TODO remove
                # Plot an ellipse to show the Gaussian component
                v, w = linalg.eigh(covar)
                u = w[0] / linalg.norm(w[0])
                angle = np.arctan(u[1] / u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                if i == 1:
                    mean = mean[0] # because our mean is a matrix
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color='k')
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.5)
                splot.add_artist(ell)

        pl.xlim(-10, 10)
        pl.ylim(-6, 6)
        pl.xticks(())
        pl.yticks(())
        pl.title(title)

    pl.savefig('dpgmm.png')


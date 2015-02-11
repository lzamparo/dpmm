from numpy import bincount, log, log2, seterr, unique, zeros
from scipy.special import gammaln

from kale.math_utils import log_sample, vi


def iteration(V, D, N_DV, N_D, alpha, beta, z_D, inv_z_T, active_topics, inactive_topics, N_TV, N_T, D_T):
    """
    Performs a single iteration of Radford Neal's Algorithm 3.
    """

    for d in xrange(D):
        
        # retain the previous cluster indicator of d
        old_t = z_D[d]

        # make sure z_D[d] is no longer part of the set
        # of points associated with old_t
        if inv_z_T is not None:
            inv_z_T[old_t].remove(d)

        # remove the data for d from the sum of the elements assigned to component old_t
        N_TV[old_t, :] -= N_DV[d, :]
        # remove sum along data features for component old_t (N_T = N_DV.sum(1))
        N_T[old_t] -= N_D[d]
        # decrease the appearances of old_t in z_D
        D_T[old_t] -= 1

        # compute partial log probability of assigning the data point to component
        seterr(divide='ignore')
        log_dist = log(D_T)
        seterr(divide='warn')

        # if this component was a singleton, keep the index.  Otherwise, activate a new component
        idx = old_t if D_T[old_t] == 0 else inactive_topics.pop()
        active_topics.add(idx)
        # log probability of assigning this point to the new component.
        log_dist[idx] = log(alpha)

        # compute log remaining log probability of assigning d over components
        # note: gammaln(x) := ln(abs(gamma(x)))
        for t in active_topics:
            log_dist[t] += gammaln(N_T[t] + beta)
            log_dist[t] -= gammaln(N_D[d] + N_T[t] + beta)
            tmp = N_TV[t, :] + beta / V
            log_dist[t] += gammaln(N_DV[d, :] + tmp).sum()
            log_dist[t] -= gammaln(tmp).sum()

        # sample from log_dist to get the component for d
        [t] = log_sample(log_dist)

        # assign component t as responsible for point d
        z_D[d] = t

        # assign point d as part of component t
        if inv_z_T is not None:
            inv_z_T[t].add(d)

        # adjust the sufficient statistics for component t 
        # to account for the addition of d
        N_TV[t, :] += N_DV[d, :]
        N_T[t] += N_D[d]
        D_T[t] += 1

        # accounting of active topics:
        if t != idx:
            active_topics.remove(idx)
            inactive_topics.add(idx)


def inference(N_DV, alpha, beta, z_D, num_itns, true_z_D=None):
    """
    Algorithm 3.
    """

    D, V = N_DV.shape

    T = D # maximum number of topics

    N_D = N_DV.sum(1) # document lengths

    active_topics = set(unique(z_D))
    inactive_topics = set(xrange(T)) - active_topics

    N_TV = zeros((T, V), dtype=int)
    N_T = zeros(T, dtype=int)

    for d in xrange(D):
        N_TV[z_D[d], :] += N_DV[d, :]
        N_T[z_D[d]] += N_D[d]

    D_T = bincount(z_D, minlength=T)

    for itn in xrange(num_itns):

        iteration(V, D, N_DV, N_D, alpha, beta, z_D, None, active_topics, inactive_topics, N_TV, N_T, D_T)

        if true_z_D is not None:

            v = vi(true_z_D, z_D)

            print 'Itn. %d' % (itn + 1)
            print '%d topics' % len(active_topics)
            print 'VI: %f bits (%f bits max.)' % (v, log2(D))

            if v < 1e-6:
                break

    return z_D

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from numpy import array, empty_like, unique, zeros
from numpy.random import seed

#import matplotlib as mpl
#mpl.use('pdf') # needed so that you can plot in a batch job with no X server (undefined $DISPLAY) problems

import numpy as np
from Dpmm import DPMM

from generative_process import generate_data
from kale.iterview import iterview
from kale.math_utils import sample
from pp_plot import pp_plot


def getting_it_right(algorithm, V, D, alpha, num_itns, s):
    """
    Runs Geweke's "getting it right" test.
    """

    seed(s)

    # generate forward samples via the generative process
    print 'Generating forward samples...'

    forward_samples = []

    for _ in iterview(xrange(num_itns)):
        forward_samples.append(generate_data(V, D, alpha)[1:])

    # generate reverse samples via the inference algorithm
    print 'Generating reverse samples...'

    reverse_samples = []
    phi_TV, z_D, data = generate_data(V, D, alpha)

    for _ in iterview(xrange(num_itns)):

        N_DV = data

        T = D # maximum number of topics
        n_components = alpha * np.log(data.shape[0])

        dpmm = DPMM(n_components=n_components.astype(int),alpha=alpha,do_sample_alpha=False)
        dpmm.fit_collapsed_Gibbs(data,do_kmeans=False,max_iter=100,do_init=True)        

        z_D = dpmm._get_assignments()

        z_D_copy = empty_like(z_D)
        z_D_copy[:] = z_D

        reverse_samples.append((z_D_copy, N_DV))

    print 'Computing test statistics...'

    # test statistics: number of topics, maximum topic size, mean
    # topic size, standard deviation of topic sizes

    # compute test statistics for forward samples

    forward_num_topics = []
    forward_max_topic_size = []
    forward_mean_topic_size = []
    forward_std_topic_size = []

    for z_D, _ in forward_samples:
        forward_num_topics.append(len(unique(z_D)))
        topic_sizes = []
        for t in unique(z_D):
            topic_sizes.append((z_D[:] == t).sum())
        topic_sizes = array(topic_sizes)
        forward_max_topic_size.append(topic_sizes.max())
        forward_mean_topic_size.append(topic_sizes.mean())
        forward_std_topic_size.append(topic_sizes.std())

    # compute test statistics for reverse samples

    reverse_num_topics = []
    reverse_max_topic_size = []
    reverse_mean_topic_size = []
    reverse_std_topic_size = []

    for z_D, _ in reverse_samples:
        reverse_num_topics.append(len(unique(z_D)))
        topic_sizes = []
        for t in unique(z_D):
            topic_sizes.append((z_D[:] == t).sum())
        topic_sizes = array(topic_sizes)
        reverse_max_topic_size.append(topic_sizes.max())
        reverse_mean_topic_size.append(topic_sizes.mean())
        reverse_std_topic_size.append(topic_sizes.std())

    # generate P-P plots
    cluster_title, cluster_file = "PP plot: number of clusters", "num_clusters_" + str(D) + "_pts_" + str(num_itns) + "_itns.pdf"
    pp_plot(array(forward_num_topics), array(reverse_num_topics), savefile=cluster_file, plot_title=cluster_title)
    cluster_max_size_title, cluster_max_size_file = "PP plot: max cluster size", "max_cluster_size_" + str(D) + "_pts_" + str(num_itns) + "_itns.pdf"
    pp_plot(array(forward_max_topic_size), array(reverse_max_topic_size), savefile=cluster_max_size_file, plot_title=cluster_max_size_title)
    cluster_mean_size_title, cluster_mean_size_file = "PP plot: mean cluster size", "mean_cluster_size_" + str(D) + "_pts_" + str(num_itns) + "_itns.pdf"
    pp_plot(array(forward_mean_topic_size), array(reverse_mean_topic_size), savefile=cluster_mean_size_file, plot_title=cluster_mean_size_title)
    cluster_min_size_title, cluster_min_size_file = "PP plot: min cluster size", "min_cluster_size_" + str(D) + "_pts_" + str(num_itns) + "_itns.pdf"
    pp_plot(array(forward_std_topic_size), array(reverse_std_topic_size), savefile=cluster_min_size_file, plot_title=cluster_min_size_title)


def main():

    from Dpmm import DPMM

    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('algorithm', metavar='<inference-algorithm>',
                   choices=['collapsed_gibbs',
                            'conjugate_split_merge'], default='collapsed_gibbs',
                   help='inference algorithm to test')
    p.add_argument('-V', type=int, metavar='<num-components>', default=2,
                   help='number of components')
    p.add_argument('-D', type=int, metavar='<data-pts>', default=1000,
                   help='number of data points')
    p.add_argument('--alpha', type=float, metavar='<alpha>', default=1.0,
                   help='concentration parameter for the DP')
    p.add_argument('--num-itns', type=int, metavar='<num-itns>', default=10000,
                   help='number of iterations')
    p.add_argument('--seed', type=int, metavar='<seed>',
                   help='seed for the random number generator')

    args = p.parse_args()

    getting_it_right(args.algorithm,
                     args.V,
                     args.D,
                     args.alpha,
                     args.num_itns,
                     args.seed)


if __name__ == '__main__':
    main()

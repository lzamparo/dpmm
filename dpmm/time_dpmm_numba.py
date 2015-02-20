# -*- coding: utf-8 -*-
import sys
import numpy as np
import contextlib,time
import tables
import cPickle as pickle
from Dpmm_numba import DPMM
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

### Job submit script for Gibbs sampling on reference population model

# read arguments
p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

p.add_argument('algorithm', metavar='<inference-algorithm>',
                choices=['gibbs','split_merge'],
                help='inference algorithm to test')
p.add_argument('-i', metavar='<inputfile>', default='/scratch/z/zhaolei/lzamparo/sm_rep1_data/reference_samples/ref_pops_topmodel_10.h5',
               help='input hdf5 file containing the data')
p.add_argument('-a', metavar='array', default='/reduced_samples/reference_pop_seed_6789_prop_0.100000/reference_pop',
               help='path to data node within the input file')
p.add_argument('--alpha', type=float, metavar='<alpha>', default=1.0,
               help='concentration parameter for the DP')
p.add_argument('--num-itns', type=int, metavar='<num-itns>', default=5000,
               help='number of iterations')
p.add_argument('--save', metavar='<save>',
               help='pickle the DPMM to this file')

args = p.parse_args()


@contextlib.contextmanager
def timeit():
  t=time.time()
  yield
  print(time.time()-t,"sec")

# load data
try:
        h5file = tables.open_file(args.i,'r')
        node = h5file.get_node(args.a)
        X = node.read()
except:
        print "Couldn't read data %s from file %s " %(args.s, args.i)
        sys.exit()
finally:
        h5file.close()

# initialize DPMM
pre_alpha = 0.5
n_components = pre_alpha * np.log(X.shape[0])
 
for _ in xrange(10):
  with timeit():
    dpmm = DPMM(n_components=n_components.astype(int),alpha=pre_alpha,do_sample_alpha=True)
    dpmm.fit_collapsed_Gibbs(X,do_sample_alpha=True,do_kmeans=True,max_iter=args.num_itns)

# save the dpmm to outfile
#pickle.dump(dpmm, open( args.save, "wb" ) )


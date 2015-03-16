# -*- coding: utf-8 -*-
import sys
import numpy as np
import tables
import cPickle as pickle
from Dpmm import DPMM
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

### Job submit script for Gibbs sampling on reference population model

# read arguments
p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

p.add_argument('algorithm', metavar='<inference-algorithm>',
                choices=['gibbs','split_merge'],
                help='inference algorithm to test')
p.add_argument('-i', metavar='<inputfile>', 
               help='input hdf5 file containing the data')
p.add_argument('-a', metavar='array', default='/reduced_samples/reference_double_sized_seed_54321/reference_pop',
               help='path to data node within the input file')
p.add_argument('--alpha', type=float, metavar='<alpha>', default=1.0,
               help='concentration parameter for the DP')
p.add_argument('--num-itns', type=int, metavar='<num-itns>', default=1000,
               help='number of iterations')
p.add_argument('--save', metavar='<save>',
               help='pickle the DPMM to this file')

args = p.parse_args()

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
pre_alpha = 1.0
n_components = pre_alpha * np.log(X.shape[0])

# dump args to stdout
for arg in vars(args):
        print arg, ": ", getattr(args, arg)

if args.algorithm == 'gibbs':
        dpmm = DPMM(n_components=n_components.astype(int),alpha=pre_alpha,do_sample_alpha=True,do_init=True)
        dpmm.fit_collapsed_Gibbs(X,do_sample_alpha=True,do_kmeans=True,max_iter=args.num_itns)
elif args.algorithm == 'split_merge':
        dpmm = DPMM(n_components=1,alpha=pre_alpha,do_sample_alpha=True)
        dpmm.fit_conjugate_split_merge(X,do_sample_alpha=True,do_kmeans=False,max_iter=args.num_itns,do_init=True)

# save the dpmm to outfile
pickle.dump(dpmm, open( args.save, "wb" ) )


# -*- coding: utf-8 -*-
import sys
import numpy as np
import tables
import cPickle as pickle
from Dpmm import DPMM
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from collections import defaultdict


### Job submit script for MCMC sampling continuation on reference population model
time_obj = datetime.now()
pkl_prefix = "-".join([str(s) for s in [time_obj.year,time_obj.month,time_obj.day,time_obj.hour,time_obj.minute]])
save_default = "dpmm-" + pkl_prefix + ".pkl"


# read arguments
p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

p.add_argument('algorithm', metavar='<inference-algorithm>',
                choices=['gibbs','split_merge'],
                help='inference algorithm to test')
p.add_argument('loadfile', metavar='<loadfile>',
               help='load the dpmm from this pkl file')
p.add_argument('-i', metavar='<inputfile>', default='/scratch/z/zhaolei/lzamparo/sm_rep1_data/reference_samples/ref_pops_topmodel_10.h5',
               help='input hdf5 file containing the data')
p.add_argument('-a', metavar='array', default='/reduced_samples/reference_double_sized_seed_54321/reference_pop',
               help='path to data node within the input file')
p.add_argument('--alpha', type=float, metavar='<alpha>', default=1.0,
               help='concentration parameter for the DP')
p.add_argument('--num-itns', type=int, metavar='<num-itns>', default=1000,
               help='number of iterations')
p.add_argument('--save', metavar='<save>', default=save_default,
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

# load DPMM
try:
        dpmm = pickle.load(open(args.loadfile, mode='rb'))
except:
        e = sys.exc_info()[0]
        print "Couldn't load a model from " % (args.loadfile)
        print e
        sys.exit()
        
# rebuild inv_z dict if required        
if not hasattr(dpmm,"inv_z"):
        inv_z = defaultdict(set)
        for pt in xrange(X.shape[0]):
                inv_z[dpmm.z[pt]].add(pt)
        dpmm.inv_z = inv_z
                

# dump args to stdout
for arg in vars(args):
        print arg, ": ", getattr(args, arg)

if args.algorithm == 'gibbs':
        dpmm.fit_collapsed_Gibbs(X,do_sample_alpha=True,do_kmeans=True,max_iter=args.num_itns,do_init=False)
elif args.algorithm == 'split_merge':
        dpmm.fit_conjugate_split_merge(X,do_sample_alpha=True,do_kmeans=False,max_iter=args.num_itns,do_init=False)

# save the dpmm to outfile
pickle.dump(dpmm, open( args.save, "wb" ) )


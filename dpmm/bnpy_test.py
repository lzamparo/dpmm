# -*- coding: utf-8 -*-

### bnpy experiments script for reduced cells data

import bnpy

k_inits = [5]
gamma_inits = [0.5,1.0,2.0,3.0]
outfile_3 = '/data/bnpy/output/ReducedCells_3layer/best_output.txt'
outfile_4 = '/data/bnpy/output/ReducedCells_4layer/best_output.txt'

# 3 layer jobs
for k_val in k_inits:
    for gamma_val in gamma_inits:
        name = "reduced_cells_10_3layer_gamma_" + str(gamma_val)
        three_layer_hmodel, three_layer_RInfo = bnpy.run('ReducedCells_3layer', 'DPMixtureModel', 'Gauss', 'moVB',
                             nLap=100, K=k_val, gamma0=gamma_val, Kfresh=2, mergePerLap=2, moves='birth,merge',  
                             nTask=5, jobname=name)

outfile = open(outfile_3,'w')
print >> outfile, "Best 3 layer model info"
for key in three_layer_RInfo.keys():
    print >> outfile, key, three_layer_RInfo[key]
outfile.close()


# 4 layer jobs
for k_val in k_inits:
    for gamma_val in gamma_inits:
        name = "reduced_cells_10_4layer_gamma_" + str(gamma_val)
        four_layer_hmodel, four_layer_RInfo = bnpy.run('ReducedCells_4layer', 'DPMixtureModel', 'Gauss', 'moVB',
                             nLap=100, K=k_val, gamma0=gamma_val, Kfresh=2, mergePerLap=2, moves='birth,merge',  
                             nTask=5, jobname=name)
 
outfile = open(outfile_4,'w')
print >> outfile, "Best 4 layer model info"
for key in four_layer_RInfo.keys():
    print >> outfile, key, four_layer_RInfo[key]
outfile.close()       

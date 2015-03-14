# -*- coding: utf-8 -*-

### bnpy test script for reduced cells data

import bnpy

#kwargs['Gauss'].keys()
#['min_covar', 'kappa', 'MMat', 'nu', 'ECovMat', 'sF']
#kwargs['DPMixtureModel'].keys()
#['gamma0']
#other kwargs to use: 'initname' (randombydist)


hmodel, RInfo = bnpy.run('ReducedCells', 'DPMixtureModel', 'Gauss', 'moVB',
                         nLap=100, K=20, gamma0=10.0, Kfresh=2, mergePerLap=2, moves='birth,merge',  
                         nTask=2, jobname='reduced_cells_20_birthmerge')


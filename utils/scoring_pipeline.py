import bnpy
import pandas as pd
import numpy as np
import tables
from sklearn.mixture import GMM
from scipy.stats import entropy as KL
 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def return_ctrl_data(data_file):
  ''' read training data for GMM control from h5 file, return '''
  input_h5 = tables.open_file(data_file)
  data_node = input_h5.get_node('/reduced_samples/reference_double_sized_seed_54321/reference_pop', classname='Array')
  data = data_node.read()
  input_h5.close()
  return data

def bnpy_dpmm_to_gmm(model_file):
  ''' load the a given bnpy model 
  model_file should look like $BNPYOUTDIR/dataname/jobname/taskid/ '''
  # sklearn GMM params with set_params
  hmodel = bnpy.load_model(model_file)
  beta = hmodel.allocModel.get_active_comp_probs()
  
  K = beta.shape[0]
  means = np.empty((K,10))
  covars = np.empty((K,10,10))
  # get means, covars
  for k in xrange(K):
    means[k,:] = hmodel.obsModel.get_mean_for_comp(k)
    covars[k,:,:] = hmodel.obsModel.get_covar_mat_for_comp(k)
  gmm = GMM(n_components=K, covariance_type='full')
  gmm.means_ = means
  gmm.covars_ = covars
  gmm.weights_ = beta
  return gmm
    

def generate_plate_to_gene(plates_file):
  ''' Construct {(plate,well): gene into} dict '''
  df = pd.read_csv(plates_file)
  plates_to_gene = {}
  
  # populate the lookup table of image number to (plate, well)
  for index, rec in df.iterrows():
      well = (int(rec['Row']) - 1) * 24 + int(rec['Col'])
      plates_to_gene[(str(rec['Plate']),str(well))] = (rec['ORF'],rec['Gene'],rec['Plate'],rec['Row'],rec['Col'],rec['Cells'])
  return plates_to_gene


def score_population(data,ref_model):
  ''' score the population given in the ndarray data under the ref_model GMM object '''
  # score each well with predict proba
  if data.shape[0] > 0:
    resp = ref_model.predict_proba(data)
    normed_resp = resp.sum(axis=0)
    normed_resp = normed_resp * (1.0/float(data.shape[0]))
    score = 0.5 * (KL(normed_resp,ref_model.weights_) + KL(ref_model.weights_, normed_resp))
    return score
  else:
    return 0.0
  
  
def score_each_well(dataset,plates_to_genes,ref_model):
  ''' calculate the score for each well in the data set, 
  append to a data frame, return the frame
  '''  
  
  scores = []
  input_h5 = tables.open_file(dataset,mode='r')
  for in_plate in input_h5.listNodes('/plates',classname="Group"):
    plate_name = in_plate._v_name
    wells = in_plate._f_listNodes(classname='Array')
    for well in wells:
      well_name = well._v_name
      try:
        data = well.read() 
        symmetric_kl_score = score_population(data, ref_model)
        orf,gene,p,r,c,cells = plates_to_genes[plate_name,well_name]
        score = {"ORF": [orf], "Gene": gene, "Plate": p, "Row": r, "Col": c, "Cells": cells, "Score": symmetric_kl_score}
        scores.append(pd.DataFrame(data=score,index=[0]))
      except:
        print "problem reading data from this well.  Or something."
        continue
      
  input_h5.close()    
  master_df = scores[0]
  for i in xrange(1,len(scores)):
    master_df = master_df.append(scores[i])        
  return master_df

# load  reduced dataset: 
def main():
  p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  p.add_argument('-m', type=str, metavar='<dpmm-file>', default='/data/bnpy/output/ReducedCells_3layer/reduced_cells_10_3layer_gamma_1.0/2/', help='dpmm model from which to extract params')
  p.add_argument('-d', type=str, metavar='<dataset>', default='/data/bnpy/reduced_full_data/sm_rep1_screen_reduced_3layer.h5', help='load this data set')
  p.add_argument('-p', type=str, metavar='<plates-to-genes>', default='/data/sm_rep1_screen/image_well_map.csv', help='load this plate to gene csv file')
  p.add_argument('-o', type=str, metavar='<outfile>', default='/data/bnpy/scored_lists/bnpy_3layer_list.csv', help='write the output dataframe here')
  p.add_argument('-c', type=str, metavar='<controldata>', help='fit the control data from here')
  args = p.parse_args()
  
  # get the well info dict
  plates_to_genes = generate_plate_to_gene(args.p)
  
  # get the gmm for scoring
  if args.m != "control":
    ref_model = bnpy_dpmm_to_gmm(args.m)
  else:
    print "control run, fitting control model..."
    lowest_bic = np.infty
    bic = []
    n_components_range = range(3, 9)
    ref_data = return_ctrl_data(args.c)
    cv_data = ref_data[np.random.choice(ref_data.shape[0],1000)]
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
        ref_model = GMM(n_components=n_components, covariance_type='full', n_init=5)
        ref_model.fit(cv_data)
        bic.append(ref_model.bic(cv_data))
        if bic[-1] < lowest_bic:
          lowest_bic = bic[-1]
          best_k = n_components    
    print "done, n_components = " + str(best_k) +" won with bic = " + str(lowest_bic)
    ref_model = GMM(n_components=best_k, covariance_type='full')
    ref_model.fit(ref_data)
    
    
  # get the df of scores
  scores = score_each_well(args.d, plates_to_genes, ref_model)
  scores.to_csv(path_or_buf=args.o,index=False)

if __name__ == "__main__":
  main()

 
 
 
 
 
  
 
 
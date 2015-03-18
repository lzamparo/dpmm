""" Process all the collapsed Gibbs sampling .o files, combine into a dataframe """

import sys, re, os
import numpy as np
import pandas as pd

# Extract the model name from each filename.
# Extract chain, layer from:    save :  /scratch/z/zhaolei/lzamparo/sm_rep1_data/dpmm_gibbs_pklfiles/gibbs_3layer_chain_1.pkl
# Extract K from:               still sampling, 10 clusters currently, with log-likelihood 193040.985131, alpha 85.610721
def extract_data(line_regex,chain_regex,filename):
    iters = []
    k_vals = []
    iter_count = 0
    for line in filename:
        if line.startswith('save'):
            line = line.strip()
            parts = line.split('/')
            match = chain_regex.match(parts[-1])
            layer, chain = match.groups()
        elif line.startswith('still'):
            match = line_regex.match(line.strip())
            if match is not None:
                k_vals.append(match.groups()[0])
                iter_count += 1
                iters.append(iter_count)
        else:
            continue
    return chain, layer, k_vals, iters        

input_dir = '/data/bnpy/output/gibbs_traces'

# Store the contents of each file as a DataFrame, store in this list to merge later
data_files = []

# compile a regex to extract the number of components on from this step
get_clusters = re.compile("still sampling, ([\d_]+) clusters")
get_model_and_chain = re.compile("gibbs\_(\d)layer\_chain\_([\d]+).pkl")

print "...Processing files"
o_files = os.listdir(input_dir)

# for each file: 
for o_file in o_files:
    # read a list of all files in the directory that match model output files
    trace_file = open(os.path.join(input_dir,o_file),'r')
    chain, layer, k_vals, iters = extract_data(get_clusters,get_model_and_chain,trace_file)
    trace_file.close()
    
    # build the one line df, store in list
    f_dict = {"Chain": [chain for i in xrange(len(k_vals))], "Layer Model": [layer for i in xrange(len(k_vals))] , "K": k_vals, "Iteration": iters}
    data_files.append(pd.DataFrame(data=f_dict))
    
print "...Done"
print "...rbinding DataFrames"
master_df = data_files[0]
for i in xrange(1,len(data_files)):
    master_df = master_df.append(data_files[i])
print "...Done"    
os.chdir(input_dir)
master_df.to_csv(path_or_buf="all_gibbs_chains.csv",index=False)


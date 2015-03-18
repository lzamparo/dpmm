# coding: utf-8
import os, sys
import pandas as pd

three_K = []
four_K = []

input_dir = sys.argv[1]
os.chdir(input_dir)

for dir in os.listdir('ReducedCells_3layer'):
    for subdir in os.listdir(os.path.join('ReducedCells_3layer',dir)):
        K_vals = open(os.path.join('ReducedCells_3layer',dir,subdir,'K.txt'),'r').readlines()
        for val in K_vals[5:]:
            three_K.append(val.strip())
                        
for dir in os.listdir('ReducedCells_4layer'):
    for subdir in os.listdir(os.path.join('ReducedCells_4layer',dir)):
        K_vals = open(os.path.join('ReducedCells_4layer',dir,subdir,'K.txt'),'r').readlines()
        for val in K_vals[5:]:
            four_K.append(val.strip())
            
# write out to data.frame
# build the one line df, store in list
data_files = []
for name,array in [("3",three_K),("4",four_K)]:
    f_dict = {"Layer Model": [name for i in xrange(len(array))] , "K": array, "Iteration": [i+1 for i in xrange(len(array))]}
    data_files.append(pd.DataFrame(data=f_dict))

print "...Done"
print "...rbinding DataFrames"
master_df = data_files[0]
for i in xrange(1,len(data_files)):
    master_df = master_df.append(data_files[i])
print "...Done"    
os.chdir(input_dir)
master_df.to_csv(path_or_buf="all_vi_component_traces.csv",index=False)

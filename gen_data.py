import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from design_space_bounds import parameter_bounds_small_design

def file_to_DataFrame(input_file):
    raw_data = pd.read_csv(input_file,index_col = 'index').head(5000)
    raw_data['place_freq'] = [int(x[:-1]) for x in raw_data['place_freq']]
    #raw_data['flowEffort'] = (raw_data['flowEffort']=='extreme').astype(int)
    #raw_data['place_global_uniform_density'] = raw_data['place_global_uniform_density'].astype(int)
    #assert set(raw_data['place_global_cong_effort'].values.tolist()) <= {'auto','high'}
    #raw_data['place_global_cong_effort'] = (raw_data['place_global_cong_effort']=='high').astype(int)
    #assert {x[-1] for x in raw_data['routing_overflow_H']}=={'%'}
    #raw_data['routing_overflow_H'] = [float(x[:-1]) for x in raw_data['routing_overflow_H']]
    #assert {x[-1] for x in raw_data['total_interal_power_pct']}=={'%'}
    #raw_data['total_interal_power_pct'] = [float(x[:-1]) for x in raw_data['total_interal_power_pct']]
    #assert {x[-1] for x in raw_data['total_leakage_power_pct']}=={'%'}
    #raw_data['total_leakage_power_pct'] = [float(x[:-1]) for x in raw_data['total_leakage_power_pct']]
    #assert {x[-1] for x in raw_data['routing_overflow_V']}=={'%'}
    #raw_data['routing_overflow_V'] = [float(x[:-1]) for x in raw_data['routing_overflow_V']]
    #assert {x[-1] for x in raw_data['total_switching_power_pct']}=={'%'}
    #raw_data['total_switching_power_pct'] = [float(x[:-1]) for x in raw_data['total_switching_power_pct']]
    #assert {x[-1] for x in raw_data['density']}=={'%'}
    #raw_data['density'] = [float(x[:-1]) for x in raw_data['density']]
    #assert set(raw_data['#EDA_FINISH'].values.tolist())=={'#YES'}
    #raw_data = raw_data.drop(['#EDA_FINISH'],axis = 1)  #  #EDA_FINISH is useless information
    return raw_data

def normalize(feature, min, max):
    feature = (feature - min) / (max - min)
    return feature

inputfile = 'Small_Design.csv'
nd_raw_data = file_to_DataFrame(inputfile).values    #numpy ndarray, 5000 x 38
N = nd_raw_data.shape[0]
feature = nd_raw_data[:,:4]
feature_name = ['place_freq', 'place_rcfactor', 'place_global_max_density', 'max_transition']

#for i in range(len(feature_name)):
#    feature[:, i] = normalize(feature[:, i], parameter_bounds_small_design[feature_name[i]][0], parameter_bounds_small_design[feature_name[i]][1])

#print(feature)

total_power = nd_raw_data[:,6:7]
#print(total_power)
total_cell_area = nd_raw_data[:,4:5]
#print(total_cell_area)
all_tns = -1 * nd_raw_data[:,5:6]
#print(all_tns)
#run_time_h = nd_raw_data[:,7:8]
#print(run_time_h)
mat_all = np.hstack((feature,total_cell_area))
mat_all = np.hstack((mat_all,total_power))
mat_all = np.hstack((mat_all,all_tns))
#mat_all = np.hstack((mat_all,run_time_h))
df_all = DataFrame(mat_all)
df_all.to_csv('small-design-parameter-tuning.csv', sep=',', header=False, index=False)
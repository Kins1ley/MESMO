import numpy as np
from GPmodel import GaussianProcess
import os
from singlemes import MaxvalueEntropySearch
from scipy.optimize import minimize as scipyminimize
from platypus import NSGAII, Problem, Real
import sobol_seq
from pygmo import hypervolume
from design_space_bounds import parameter_bounds_small_design, parameter_bounds_medium_design
from sklearn.preprocessing import RobustScaler

def find_neighbor(feature, x_best):

    distance = np.linalg.norm(feature - x_best, axis=1)
    #print(distance)
    index = np.argmin(distance)
    return index


paths='.'
d = 4
total_iterations = 20   #Problem 1
seed = 224                #Problem 1.5
np.random.seed(seed)
intial_number = 20       #Problem 1
bound = [0,1]           #Problem 2
sample_number = 1
Fun_bounds = [bound] * d

mat_raw = np.genfromtxt("small-design-parameter-tuning.csv", delimiter=',', dtype='float')
robust_scaler = RobustScaler()
mat_feature = robust_scaler.fit_transform(mat_raw[:,:4])
mat = np.hstack((mat_feature, mat_raw[:,4:]))

feature = mat[:,:4]
total_power = mat[:,5]
total_power_max = np.max(total_power)
#(total_power_max)
total_cell_area = mat[:,4]   
total_cell_area_max = np.max(total_cell_area)
#print(total_cell_area_max)
all_tns = mat[:,6]
all_tns_max = np.max(all_tns)
#(all_tns_max)
referencePoint = [float(total_cell_area_max), float(total_power_max), float(all_tns_max)]
print (referencePoint)

functions = [total_cell_area, total_power, all_tns]

design_index = np.random.randint(0, mat.shape[0] - 1)


GPs = []
Multiplemes = []
for i in range(len(functions)):
    GPs.append(GaussianProcess(d))

for k in range(intial_number):
    exist=True
    while exist:
        design_index = np.random.randint(0, mat.shape[0] - 1)
        x_rand = feature[design_index, :].tolist()
        #print(x_rand)
        if (any((x_rand == x).all() for x in GPs[0].xValues)) == False:
            exist = False
        for i in range(len(functions)):
            GPs[i].addSample(np.asarray(x_rand), functions[i][design_index])
           # print(functions[i][design_index])
for i in range(len(functions)):
    GPs[i].fitModel()
    Multiplemes.append(MaxvalueEntropySearch(GPs[i]))


input_output= open(os.path.join(paths,'input_output_small_design.txt'), "a")
for j in range(len(GPs[0].yValues)):
    input_output.write(str(GPs[0].xValues[j])+'---'+str([GPs[i].yValues[j] for i in range(len(functions))]) +'\n' )
input_output.close()

##################### main loop ##########
for l in range(total_iterations):

    for i in range(len(functions)):
        Multiplemes[i] = MaxvalueEntropySearch(GPs[i])
        Multiplemes[i].Sampling_RFM()
    max_samples = []
    for j in range(sample_number):
        for i in range(len(functions)):
            Multiplemes[i].weigh_sampling()
        cheap_pareto_front=[]


        def CMO(xi):
            xi=np.asarray(xi)
            y=[Multiplemes[i].f_regression(xi)[0][0] for i in range(len(GPs))]
            return y

        problem = Problem(d, len(functions))
        problem.types[:] = Real(bound[0], bound[1])
        problem.function = CMO
        algorithm = NSGAII(problem)
        algorithm.run(1500)
        cheap_pareto_front = [list(solution.objectives) for solution in algorithm.result]
        #########picking the max over the pareto: best case
        maxoffunctions = [-1 * min(f) for f in list(zip(*cheap_pareto_front))]
        max_samples.append(maxoffunctions)


    def mesmo_acq(x):
        multi_obj_acq_total = 0
        for j in range(sample_number):
            multi_obj_acq_sample = 0
            for i in range(len(functions)):
                multi_obj_acq_sample = multi_obj_acq_sample + Multiplemes[i].single_acq(x, max_samples[j][i])
            multi_obj_acq_total = multi_obj_acq_total + multi_obj_acq_sample
        return (multi_obj_acq_total / sample_number)


    # l-bfgs-b acquisation optimization
    x_tries = feature
    y_tries = [mesmo_acq(x) for x in x_tries]
    sorted_indecies = np.argsort(y_tries)
    i = 0
    x_best = x_tries[sorted_indecies[i]]
    while (any((x_best == x).all() for x in GPs[0].xValues)):
        i = i + 1
        x_best = x_tries[sorted_indecies[i]]
    y_best = y_tries[sorted_indecies[i]]
    x_seed = list(feature)
    index = find_neighbor(feature, x_best)
    feature_value = feature[index, :]
    result = 0
    i = 0
    for x_try in x_seed:
        result = scipyminimize(mesmo_acq, x0=np.asarray(x_try).reshape(1, -1), method='L-BFGS-B', bounds=Fun_bounds)
        if not result.success:
            continue
        index = find_neighbor(feature, result.x)
        feature_value = feature[index, :]
        if ((result.fun <= y_best) and (not (feature_value in np.asarray(GPs[0].xValues)))):
            #print('result.x', result.x)
            x_best = feature_value
            #print('x_best', x_best)
            y_best = result.fun

#---------------Updating and fitting the GPs-----------------
    for i in range(len(functions)):
        GPs[i].addSample(x_best, functions[i][int(np.where((feature == x_best).all(1))[0][0])])
        GPs[i].fitModel()
    ############################ write Input output into file ##################
    input_output= open(os.path.join(paths,'input_output_small_design.txt'), "a")
    #robust_scaler.inverse_transform(mat_feature)[np.where(feature==x_best)[0][0]]
    input_output.write(str(GPs[0].xValues[-1])+'---'+str([GPs[i].yValues[-1] for i in range(len(functions))]) +'\n' )
    input_output.close()

    ########################### write hypervolume into file##################    

    current_hypervolume= open(os.path.join(paths,'hypervolumes_small_design.txt'), "a") 
    simple_pareto_front_evaluations=list(zip(*[GPs[i].yValues for i in range(len(functions))]))
    print(simple_pareto_front_evaluations)
    print("hypervolume ", hypervolume(-1*(np.asarray(simple_pareto_front_evaluations))).compute(referencePoint))   # Problem 3
    current_hypervolume.write('%f \n' % hypervolume(-1*(np.asarray(simple_pareto_front_evaluations))).compute(referencePoint))
    current_hypervolume.close()



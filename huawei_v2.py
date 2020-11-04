import numpy as np
from GPmodel import GaussianProcess
import os
from singlemes import MaxvalueEntropySearch
from scipy.optimize import minimize as scipyminimize
from platypus import NSGAII, Problem, Real
import sobol_seq
from pygmo import hypervolume
from design_space_bounds import parameter_bounds_small_design, parameter_bounds_medium_design
from sklearn.preprocessing import MinMaxScaler
from config import conf
import matplotlib.pyplot as plt
import sys
import DeepSparseKernel as dsk

def find_neighbor(feature, x_best):

    distance = np.linalg.norm(feature - x_best, axis=1)
    #print(distance)
    index = np.argmin(distance)
    return index


def multi_output_gassian(train_x, train_y):
    num_shared_layer = conf["num_shared_layer"]
    num_non_shared_layer = conf["num_non_shared_layer"]
    hidden_shared = conf["hidden_shared"]
    hidden_non_shared = conf["hidden_non_shared"]
    l1 = conf["l1"]
    l2 = conf["l2"]
    scale = conf["scale"]
    max_iter = conf["max_iter"]
    K = conf["K"]
    activation = conf["activation"]
    # act_f = dsk.tanh
    if activation == "relu":
        act_f = dsk.relu
    elif activation == "erf":
        act_f = dsk.erf
    elif activation == "sigmoid":
        act_f = dsk.sigmoid
    else:
        act_f = dsk.tanh
    # dim, num_train = train_x.shape
    num_obj = train_y.shape[1]
    # num_test = test_x.shape[1]
    shared_layers_sizes = [hidden_shared] * num_shared_layer
    shared_activations = [dsk.tanh] * num_shared_layer
    non_shared_layers_sizes = [hidden_non_shared] * num_non_shared_layer
    non_shared_activations = [dsk.tanh] * num_non_shared_layer
    shared_nn = dsk.NN(shared_layers_sizes, shared_activations)
    non_shared_nns = []
    for i in range(num_obj):
        non_shared_nns += [dsk.NN(non_shared_layers_sizes, non_shared_activations)]
    modsk = dsk.MODSK(train_x, train_y, shared_nn, non_shared_nns, debug=True, max_iter=max_iter, l1=l1, l2=l2)

    return modsk


paths='.'
dim = 4
total_iterations = 1
seed = 0
np.random.seed(seed)
intial_number = 1
bound = [0,1]
sample_number = 1
Fun_bounds = [bound] * dim
xValues = []
yValues = []
mat = np.genfromtxt("small-design-parameter-tuning.csv", delimiter=',', dtype='float')
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
# train = mat[:200, :]
# test = mat[200:, :]
# train_x = train[:, :4].T
# train_y = train[:, 4:7]
# test_x = test[:, :4].T
# test_y = test[:, 4:7]

feature = mat[:,:4]
target = mat[:, 4:7]
num_target = target.shape[1]
print(num_target)
# total_power = mat[:,5]
# total_power_max = np.max(total_power)
# #(total_power_max)
# total_cell_area = mat[:,4]
# total_cell_area_max = np.max(total_cell_area)
# #print(total_cell_area_max)
# all_tns = mat[:,6]
# all_tns_max = np.max(all_tns)
# #(all_tns_max)
# referencePoint = [float(total_cell_area_max), float(total_power_max), float(all_tns_max)]
# functions = [total_cell_area, total_power, all_tns]

design_index = np.random.randint(0, mat.shape[0] - 1)

Multiplemes = []

for k in range(intial_number):
    exist = True
    while exist:
        design_index = np.random.randint(0, mat.shape[0] - 1)
        x_rand = feature[design_index, :].tolist()
        #print(x_rand)
        if (any((x_rand == x).all() for x in xValues)) == False:
            exist = False
        xValues.append(x_rand)
        yValues.append(target[design_index, :].tolist())

print(design_index)
print(np.asarray(xValues).T.shape)
print(np.asarray(yValues).shape)

GP_model = multi_output_gassian(np.asarray(xValues).T, np.asarray(yValues))
for i in range(target.shape[1]):
    Multiplemes.append(MaxvalueEntropySearch(GP_model, np.asarray(xValues), np.asarray(yValues)[:, i], dim))

input_output= open(os.path.join(paths,'input_output_small_design.txt'), "a")
for j in range(np.asarray(yValues).shape[0]):
    input_output.write(str(np.asarray(xValues)[j])+'---'+str(np.asarray(yValues)[j]) +'\n' )
input_output.close()

for l in range(total_iterations):

    for i in range(num_target):
        Multiplemes[i] = MaxvalueEntropySearch(GP_model, np.asarray(xValues), np.asarray(yValues)[:, i], dim)
        Multiplemes[i].Sampling_RFM()
    max_samples = []
    for j in range(sample_number):
        for i in range(num_target):
            Multiplemes[i].weigh_sampling()
        cheap_pareto_front=[]

        def CMO(xi):
            xi = np.asarray(xi)
            y = [Multiplemes[i].f_regression(xi)[0][0] for i in range(num_target)]
            return y

        problem = Problem(dim, num_target)
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
            for i in range(num_target):
                multi_obj_acq_sample = multi_obj_acq_sample + Multiplemes[i].single_acq(x, max_samples[j][i])
            multi_obj_acq_total = multi_obj_acq_total + multi_obj_acq_sample
        return (multi_obj_acq_total / sample_number)


    x_tries = feature
    y_tries = [mesmo_acq(x) for x in x_tries]
    sorted_indecies = np.argsort(y_tries)
    i = 0
    x_best = x_tries[sorted_indecies[i]].tolist()
    while (any((x_best == x).all() for x in xValues)):
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
        if ((result.fun <= y_best) and (not (feature_value in np.asarray(xValues)))):
            #print('result.x', result.x)
            x_best = feature_value
            #print('x_best', x_best)
            y_best = result.fun

    xValues.append(x_best.tolist())
    yValues.append(target[np.where((feature == x_best).all(1))[0][0], :].tolist())

    GP_model = multi_output_gassian(np.asarray(xValues).T, np.asarray(yValues))
    # for i in range(num_target):
    #
    #     GPs[i].addSample(x_best, functions[i][int(np.where((feature == x_best).all(1))[0][0])])
    #     GPs[i].fitModel()
    ############################ write Input output into file ##################
    input_output= open(os.path.join(paths,'input_output_small_design.txt'), "a")
    input_output.write(str(np.asarray(xValues)[-1])+'---'+str(np.asarray(yValues)[-1]) +'\n' )
    input_output.close()

    current_hypervolume = open(os.path.join(paths, 'hypervolumes_small_design.txt'), "a")
    simple_pareto_front_evaluations = list(zip(*[GPs[i].yValues for i in range(num_target)]))
    print(simple_pareto_front_evaluations)
    print("hypervolume ", hypervolume(-1 * (np.asarray(simple_pareto_front_evaluations))).compute(referencePoint))
    current_hypervolume.write(
        '%f \n' % hypervolume(-1 * (np.asarray(simple_pareto_front_evaluations))).compute(referencePoint))
    current_hypervolume.close()

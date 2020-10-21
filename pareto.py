# import numpy as np
# from benchmark_functions import branin, Currin
#
# a = np.random.rand(1000)
# b = np.random.rand(1000)
#
# c = np.vstack((a, b)).T
# print(c.shape)
# function  = [branin, Currin]
# #d = []
# d = np.zeros(1000)
# e = np.zeros(1000)
#
# for i in range(1000):
#     x_rand = list(c[i])
#     #print(x_rand)
#     d[i] = branin(x_rand, 2)
#     e[i] = Currin(x_rand, 2)
#     #print(d)
#     #print(e)
#
# d = np.vstack((d, e)).T
#
# result = np.hstack((c, d))
# print(result)
#
# x_rand = [6.96510926e-01,  9.68654776e-01]
# #print(branin(x_rand, 2))
# #print(Currin(x_rand, 2))
#
# np.savetxt('new.csv', result, delimiter = ',')
import numpy as np
from golden_hypervolume import pf, VolumePPA
from design_space_bounds import parameter_bounds_small_design
file = open("input_output_small_design.txt")

result = []
while 1:
    line = file.readline()
    line.replace('\n','')
    a = line.rfind('-')
    #print(type(line[a+1:]))
    line = line[a+2:-2]
    #line = eval(line[a+2:-1])
    #print(line)
    #result.append(np.array(line[a+1:]))
    if not line:
        break
    pass # do something
    line = line.split(',')
    for i in range(len(line)):
        line[i] = float(line[i])
    result.append(line)
file.close()
#print(result)
result = np.array(result)
print(type(result))
ppaPAL_PF = np.array(pf(result, 3))
mat = np.genfromtxt("small-design-parameter-tuning.csv", delimiter=',', dtype='float')

feature = mat[:,:4]
total_power = mat[:,5]
total_power_max = np.max(total_power)
print(total_power_max)
total_cell_area = mat[:,4]
total_cell_area_max = np.max(total_cell_area)
print(total_cell_area_max)
all_tns = mat[:,6]
all_tns_max = np.max(all_tns)
print(all_tns_max)

ppaVolPAL = VolumePPA(ppaPAL_PF, total_cell_area_max, total_power_max, all_tns_max)
print(ppaVolPAL)
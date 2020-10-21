# vito PF
import numpy as np
import sys
import csv
import copy


# maxdelay = 145.952
# maxarea = 6526.824
# maxpower = 28.36

# minimize first element, minimize second element
def pf(a, dim):
    pfs = []
    i = 0
    while i < len(a):
        # find a[i] in the pareto front
        j = 0
        while j < len(a):
            if dim == 2:
                if i != j:  # to compare
                    vj1 = a[j][0]
                    vj2 = a[j][1]
                    vi1 = a[i][0]
                    vi2 = a[i][1]
                    # if a[j] dominates a[i]
                    if (vj1 <= vi1 and vj2 <= vi2) and (vj1 < vi1 or vj2 < vi2):
                        i += 1
                        break
                    else:  # increase j to compare next
                        j += 1
                        if j == len(a):  # if no next, a[i] is in the PF
                            pfs.append(a[i])
                            i += 1
                            break
                else:  # increase j to compare next
                    j += 1
                    if i == len(a) - 1 and j == len(a):  # if no next, a[i] is in the PF
                        pfs.append(a[i])
                        i += 1
            if dim == 3:
                if i != j:  # to compare
                    vj1 = a[j][0]
                    vj2 = a[j][1]
                    vj3 = a[j][2]
                    vi1 = a[i][0]
                    vi2 = a[i][1]
                    vi3 = a[i][2]
                    # if a[j] dominates a[i]
                    if (vj1 <= vi1 and vj2 <= vi2 and vj3 <= vi3) and (vj1 < vi1 or vj2 < vi2 or vj3 < vi3):
                        i += 1
                        break
                    else:  # increase j to compare next
                        j += 1
                        if j == len(a):  # if no next, a[i] is in the PF
                            pfs.append(a[i])
                            i += 1
                            break
                else:  # increase j to compare next
                    j += 1
                    if i == len(a) - 1 and j == len(a):  # if no next, a[i] is in the PF
                        pfs.append(a[i])
                        i += 1
    pfs = np.array(pfs)
    pfs = pfs[pfs[:, 1].argsort()]
    return pfs


def Areaad(array, maxarea, maxdelay):
    a = 0
    for i in range(0, len(array) - 1):
        a += (array[i + 1, 1] - array[i, 1]) * (maxarea - array[i, 0])
    a += (maxdelay - array[-1, 1]) * (maxarea - array[-1, 0])
    return a


def Areapd(array, maxpower, maxdelay):
    a = 0
    for i in range(0, len(array) - 1):
        a += (array[i + 1, 1] - array[i, 1]) * (maxpower - array[i, 0])
    a += (maxdelay - array[-1, 1]) * (maxpower - array[-1, 0])
    return a


def VolumePPA(array, maxarea, maxpower, maxdelay):
    set_area = np.unique(array[:, 0])
    set_power = np.unique(array[:, 1])
    set_delay = np.unique(array[:, 2])

    points = array.tolist()
    for i in range(len(set_area)):
        for j in range(len(set_power)):
            for k in range(len(set_delay)):
                tmp = [set_area[i], set_power[j], set_delay[k]]
                accept = True
                for num in range(len(array)):
                    if tmp[0] <= array[num, 0] and tmp[1] <= array[num, 1] and tmp[2] <= array[num, 2]:
                        accept = False
                        break
                if accept:
                    points.append(tmp)

    vol = 0
    points = np.array(points)
    for i in range(len(points)):
        idx = np.where(set_area == points[i, 0])
        j = idx[0][0]
        if j == len(set_area) - 1:
            next_area = maxarea
        else:
            next_area = set_area[j + 1]

        idy = np.where(set_power == points[i, 1])
        j = idy[0][0]
        if j == len(set_power) - 1:
            next_power = maxpower
        else:
            next_power = set_power[j + 1]

        idz = np.where(set_delay == points[i, 2])
        j = idz[0][0]
        if j == len(set_delay) - 1:
            next_delay = maxdelay
        else:
            next_delay = set_delay[j + 1]

        vol += (next_area - points[i, 0]) * (next_power - points[i, 1]) * (next_delay - points[i, 2])

    return vol


# # max features in physical design space
# # def hypervolume(dim, repeat, datapath='./', resultpath='./'):
# dim = 3
# repeat = 1
# datapath='./'
# resultpath='./'
# matAll = np.genfromtxt((datapath + 'small-design-parameter-tuning.csv'), delimiter=',', dtype='float')
# matAll = matAll[:, :-1]
#
# maxArea_abs = np.max(matAll[:, -3])
# print(maxArea_abs)
# maxPower_abs = np.max(matAll[:, -2])
# print(maxPower_abs)
# maxDelay_abs = np.max(matAll[:, -1])
# print(maxDelay_abs)
# # for i in range(-4, -1):
# #    matAll[:, i] = np.divide(matAll[:, i], np.max(matAll[:, i]))
# # Get golden result for hypervolume
# if dim == 3:
#     ppaAll = matAll[:, -3:]
#     ppaAll_PF = np.array(pf(ppaAll, 3))
#     # savePF(ppaAll_PF, 'ppaAll_PF.dat')
#     ppaVolAll = VolumePPA(ppaAll_PF, maxArea_abs, maxPower_abs, maxDelay_abs)
#     print("Hypervolume of PPA (golden): ", ppaVolAll)
#     print('\n')
#
# else:
#     adAll = matAll[:, -3:]
#     adAll = np.delete(adAll, 1, 1)
#     adAll_PF = pf(adAll, 2)
#     # savePF(adAll_PF, 'adAll_PF.dat')
#     adAreaAll = Areaad(adAll_PF, maxArea_abs, maxDelay_abs)
#     print("Hypervolume of area-delay (golden): ", adAreaAll)
#
#     pdAll = matAll[:, -2:]
#     # pdAll[:, 0] = np.divide(pdAll[:, 0], maxPower_abs)
#     # pdAll[:, 1] = np.divide(pdAll[:, 1], maxDelay_abs)
#     pdAll_PF = np.array(pf(pdAll, 2))
#     # savePF(pdAll_PF, 'pdAll_PF.dat')
#     pdAreaAll = Areapd(pdAll_PF, maxPower_abs, maxDelay_abs)
#     print("Hypervolume of power-delay (golden): ", pdAreaAll)
#
# sumPAL, sumAlpha = 0, 0
# repetition = repeat
# maxadarea = 0
# maxpdarea = 0
# adareaPAL = [0] * repetition
# pdareaPAL = [0] * repetition
# adareaAlpha = [0] * repetition
# pdareaAlpha = [0] * repetition
# ppaVolAlpha = [0] * repetition
# ppaVolPAL = [0] * repetition
# for i in range(0, repetition):
#     # PAL
#     if dim == 2:
#         adPAL = np.genfromtxt((resultpath + 'result_ad' + str(i) + '.csv'), delimiter=',', dtype='float')
#         pdPAL = np.genfromtxt((resultpath + 'result_pd' + str(i) + '.csv'), delimiter=',', dtype='float')
#         sumPAL += len(adPAL) + len(pdPAL)
#         pdPAL_PF = pf(pdPAL, 2)
#         adPAL_PF = pf(adPAL, 2)
#         adareaPAL[i] = Areaad(adPAL_PF, maxArea_abs, maxDelay_abs)
#         pdareaPAL[i] = Areapd(pdPAL_PF, maxPower_abs, maxDelay_abs)
#         ##keep the best PF prediction for output
#         if np.max(adareaPAL) == Areaad(adPAL_PF, maxArea_abs, maxDelay_abs):
#             maxad_PAL = adPAL_PF
#         if np.max(pdareaPAL) == Areapd(pdPAL_PF, maxPower_abs, maxDelay_abs):
#             maxpd_PAL = pdPAL_PF
#
#     if dim == 3:
#         ppaPAL_Pset = np.genfromtxt((resultpath + 'result_apd' + str(i) + '.csv'), delimiter=',', dtype='float')
#         sumPAL += len(ppaPAL_Pset)
#         ppaPAL_PF = np.array(pf(ppaPAL_Pset, 3))
#         ppaVolPAL[i] = VolumePPA(ppaPAL_PF, maxArea_abs, maxPower_abs, maxDelay_abs)
#
# print("------------------Output Results:------------------")
# if dim == 2:
#     # print sumPAL * 1.0 / repetition + 150
#     print(adareaPAL)
#     print(np.mean(adareaPAL), np.std(adareaPAL), np.max(adareaPAL), np.argmax(adareaPAL))
#     print(pdareaPAL)
#     print(np.mean(pdareaPAL), np.std(pdareaPAL), np.max(pdareaPAL), np.argmax(pdareaPAL))
# else:
#     # print sumPAL * 1.0 / repetition + 250
#     print(np.mean(ppaVolPAL), np.std(ppaVolPAL), np.max(ppaVolPAL), np.argmax(ppaVolPAL), "\n")


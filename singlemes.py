# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:34:01 2018

@author: Syrine Belakaria
"""
import numpy as np
from scipy.stats import norm
from sklearn.kernel_approximation import RBFSampler
from sklearn.gaussian_process.kernels import RBF

class MaxvalueEntropySearch(object):
    def __init__(self, GPmodel, xValues, yValues, dim):
        # self.GPmodel = GPmodel
        # self.y_max = max(GPmodel.yValues)
        # self.d = GPmodel.dim
        self.GPmodel = GPmodel
        self.xValues = xValues
        self.yValues = yValues
        self.y_max = np.max(yValues)
        # print(self.y_max)
        self.dim = dim
        self.beta = 1e6
        self.kernel = RBF(length_scale=1, length_scale_bounds=(1e-3, 1e2))
        # # print(self.d)
        # print('d', self.dim)
        # print('xValues', self.xValues)
        # print('yValues', self.yValues)
        # print('dim', self.dim)
        # print('beta', self.beta)
        # print('length_scale', self.kernel.length_scale)

    def Sampling_RFM(self):
        self.rbf_features = RBFSampler(gamma=1/(2*self.kernel.length_scale**2), n_components=1000, random_state=1)
        X_train_features = self.rbf_features.fit_transform(np.asarray(self.xValues))

        A_inv = np.linalg.inv((X_train_features.T).dot(X_train_features) + np.eye(self.rbf_features.n_components)/self.beta)
        self.weights_mu = A_inv.dot(X_train_features.T).dot(self.yValues)
        weights_gamma = A_inv / self.beta
        self.L = np.linalg.cholesky(weights_gamma)

    def weigh_sampling(self):
        random_normal_sample = np.random.normal(0, 1, np.size(self.weights_mu))
        # print('random_normal_sample', random_normal_sample)
        self.sampled_weights = np.c_[self.weights_mu] + self.L.dot(np.c_[random_normal_sample])
        # print('sampled_weights', self.sampled_weights)
    def f_regression(self,x):

        X_features = self.rbf_features.fit_transform(x.reshape(1,len(x)))
        return -(X_features.dot(self.sampled_weights)) 

    def single_acq(self, mean, std, maximum):
        #mean, std = self.GPmodel.mix_predict(K=4, x=x, scale=1)
        #mean=mean[0]
        #std=std[0]
        if maximum < max(self.yValues)+5/self.beta:
            maximum=max(self.yValues)+5/self.beta

        normalized_max = (maximum - mean) / std
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        if (cdf==0):
            cdf=1e-30
        return   -(normalized_max * pdf) / (2*cdf) + np.log(cdf)


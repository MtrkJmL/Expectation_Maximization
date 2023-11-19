import numpy as np
from getLogLikelihood import getLogLikelihood
import math

def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    K = len(weights)
    N, D = X.shape
    gamma = np.zeros((N,K))
    norm = lambda cov: 1. / float((2 * np.pi) ** (float(D) / 2.) * np.sqrt(np.linalg.det(cov)))
    gaussian = lambda x, mean, cov: norm(cov) * np.exp(-0.5 * (np.dot((x - mean).T, np.dot(np.linalg.inv(cov), (x - mean)))))
    for n in range(N):
        sum=0
        for k in range(K):
            cov = covariances[:, :, k].copy()
            sum+= weights[k]*gaussian(X[n],means[k],cov)
        for j in range(K):
            cov = covariances[:, :, j].copy()
            gamma[n,j] = weights[j]*gaussian(X[n],means[j],cov)/sum
    logLikelihood=getLogLikelihood(means,weights,covariances,X)

    return [logLikelihood, gamma]

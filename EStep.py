import numpy as np
from getLogLikelihood import getLogLikelihood
import math

def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
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

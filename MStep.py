import numpy as np

def MStep(gamma, X):
    # Maximization step of the EM Algorithm

    N,K = gamma.shape
    _,D = X.shape
    N_new = gamma.sum(axis=0)
    weights = N_new/N

    means = np.zeros((K,D))
    for j in range(K):
        sum=0
        for n in range(N):
            sum+=gamma[n,j]*X[n]
        means[j] = sum/N_new[j]

    covariances = np.zeros((D,D,K))
    for j in range(K):
        sum= np.zeros((D,D))
        for n in range(N):
            sum= sum + gamma[n,j]*np.outer((X[n]-means[j]),(X[n]-means[j]).T)
        covariances[:,:,j]=sum/N_new[j]


    return weights, means, covariances

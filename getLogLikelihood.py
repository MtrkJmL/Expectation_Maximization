import numpy as np

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation

    K= len(weights)
    if len(X.shape)>1:
        N,D = X.shape
    else:
        N=1
        D=X.shape[0]
    norm = lambda cov: 1./float((2*np.pi)**(float(D)/2.)*np.sqrt(np.linalg.det(cov)))
    gaussian = lambda x,mean,cov: norm(cov)*np.exp(-0.5*(np.dot((x-mean).T,np.dot(np.linalg.inv(cov),(x-mean)))))
    logLikelihood = 0
    for n in range(N):
        insum=0
        for k in range(K):
            if N == 1:
                x = X
            else:
                x = X[n,:]
            cov = covariances[:,:,k].copy()
            insum+= weights[k]*gaussian(x,means[k],cov)
        logLikelihood+= np.log(insum)
    return logLikelihood


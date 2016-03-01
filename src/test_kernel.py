import numpy as np


state_range = [np.zeros(2), np.ones(2)]

def kernel(X, Y):
    if X.ndim == 1:
        X = X.reshape((1,-1))
        
    if Y.ndim == 1:
        Y = Y.reshape((1,-1))
    width = 0.1
    scale = ((state_range[1] - state_range[0]) * width)[None,:,None]
        
    # compute squared weighted distance distance 
    dsqr = -(((X[:,:,None] - Y.T[None,:,:])/scale)**2).sum(axis=1)
    return np.exp(dsqr).squeeze()

x0 = np.zeros(2)
x1 = np.ones(2)
X = np.array([x0,x1])
print kernel(X, X)
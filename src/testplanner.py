from recnamo.planner import linear_find_stoc_plan

from scipy import io
from scipy.sparse import eye
import scipy.sparse
import scipy.optimize

import numpy as np
import pickle

import theano

from itertools import izip

import matplotlib.pyplot as plt

matrices = io.loadmat('linear_models.mat')
with open('representation.data', 'rb') as f:
    phi= pickle.load(f) 
    
    
def plot_plan(x0, plan, Fs, phi):
    num = 40
    XX, YY = np.meshgrid(np.linspace(-1, 10, num, True),
                         np.linspace(-1, 10, num, True))
    points = np.concatenate([ p.reshape((-1,1)) for p in [XX, YY]], axis=1)
    points = np.hstack( (points,np.tile(np.array([[2., 2., 0.]]), (points.shape[0], 1))))
    
    x_T = x0.T
    for p in plan:
        x_T = sum([ prob*F.dot(x_T) for prob, F in izip(p, Fs)])
        values = phi(points).dot(x_T)
        plt.figure()
        if scipy.sparse.issparse(values):
            values = values.toarray()
        plt.pcolormesh(XX, YY, values.reshape((num, -1)))
        
    

    
s0 =np.array([0. ,0., 2., 2., 0.])
sg = np.array([2. ,0., 2., 2., 0.])
x0 = phi(s0)
xg = phi(sg)
Fs = matrices.values()[:-3] + [eye(phi.size)*0.99]
if not scipy.sparse.issparse(Fs[0]):
    Fs[-1] = Fs[-1].toarray()
    
if scipy.sparse.issparse(x0):
    x0 = x0.toarray()
    xg = xg.toarray()

# print [type(F) for F in Fs], Fs[0].shape

plan_len = 10
plan, fn, msg = linear_find_stoc_plan(x0, xg, phi, Fs, plan_len)
print plan
print msg
plan_dummy = [np.array([0, 0, 1.0, 0., 0,0, 0])]* plan_len
print fn( np.hstack(plan))[0], fn( np.hstack(plan_dummy))[0]

# f = lambda x: fn(x)[0]
# df = lambda x: fn(x)[1]
# 
# data = np.random.rand(10, np.hstack(plan).shape[0])
# data = data/(np.sum(data, axis=1)[:,None])
#
# print [scipy.optimize.check_grad(f, df, d) for d in data]

plot_plan(x0, plan, Fs, phi)
plot_plan(x0, plan_dummy, Fs, phi)
plt.show()
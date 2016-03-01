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


# theano.config.compute_test_value = 'warn'

matrices = io.loadmat('grid_models.mat')
# with open('representation.data', 'rb') as f:
#     phi= pickle.load(f) 
    
class ravel_proj(object):
    def __init__(self, dims):
        self.dims = dims
        self.size = np.prod(dims)
    def __call__(self, state):
        if state.ndim == 1:
            state = state.reshape((1,-1))
        size = state.shape[0]
        state = state.astype('int').T
        indices = np.ravel_multi_index(state, self.dims)
        phi_t = scipy.sparse.coo_matrix((np.ones(size), (np.arange(size), indices)),
                                shape = (size, np.prod(self.dims)))
        return phi_t.tocsr()
    
phi = ravel_proj( np.ones(2, dtype='int')*20)
    
    
def plot_plan(x0, plan, Fs, phi):
    num = 20
    XX, YY = np.meshgrid(np.linspace(0, 20, num, False),
                         np.linspace(0, 20, num, False))
    points = np.concatenate([ p.reshape((-1,1)) for p in [XX, YY]], axis=1)
    points = points.astype('int')
#     points = np.hstack( (points,np.tile(np.array([[2., 2., 0.]]), (points.shape[0], 1))))
    x_T = x0.T
    for count, p in enumerate(plan):
        p = np.exp(p)/ sum(np.exp(p))
        x_T = sum([ prob*F.dot(x_T) for prob, F in izip(p, Fs)])
        if (count %8) == 7:
            values = phi(points).dot(x_T)
            plt.figure()
            if scipy.sparse.issparse(values):
                values = values.toarray()
            plt.pcolormesh(XX, YY, values.reshape((num, -1)))
        
    

    
# s0 =np.array([0. ,0., 2., 2., 0.])
# sg = np.array([2. ,0., 2., 2., 0.])
s0 =np.array([0. ,0.], dtype='int')
sg = np.array([18 ,18], dtype='int')
x0 = phi(s0)
xg = phi(sg)
Fs = matrices.values()[:-3]
Fs = [F.toarray() for F in Fs]
Fs += [eye(phi.size)]
if not scipy.sparse.issparse(Fs[0]):
    Fs[-1] = Fs[-1].toarray()
    
if scipy.sparse.issparse(x0):
    x0 = x0.toarray()
    xg = xg.toarray()

# print [type(F) for F in Fs], Fs[0].shape

plan_len = 42
plan, fn, msg = linear_find_stoc_plan(x0, xg, phi, Fs, plan_len)
print plan
print msg

plan_dummy = [np.array([1.0, 0.0, 0.0, 0., 0])]* plan_len
print fn( np.hstack(plan))[0], fn( np.hstack(plan_dummy))[0]

# f = lambda x: fn(x)[0]
# df = lambda x: fn(x)[1]
# 
# data = np.random.rand(10, np.hstack(plan).shape[0])
# data = data/(np.sum(data, axis=1)[:,None])
#
# print [scipy.optimize.check_grad(f, df, d) for d in data]

plot_plan(x0, plan, Fs, phi)
# plot_plan(x0, plan_dummy, Fs, phi)
plt.show()
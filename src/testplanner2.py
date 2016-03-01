from recnamo.greedyplanner import linear_find_stoc_plan
from recnamo.gridnamo import gridnamo

from scipy import io
from scipy.sparse import eye
import scipy.sparse
import scipy.optimize
import scipy.sparse as sp

import numpy as np
import pickle

import theano

from itertools import izip

import matplotlib.pyplot as plt
import matplotlib.animation as animation

dim = 6
size = np.array((dim,dim), dtype='int')

walls = [(1,0), (1,1), (1,2), (1,3), (1,5),
         (2,0), (2,1), (2,2), (2,3), (2,5)]
walls = None

env = gridnamo([], size, walls = walls)
matrices = io.loadmat('grid_models.mat')
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
    
class factor_state(object):
    def __init__(self, dim):
        self.dim = dim
        self.size = sum(dim)
        
    def __call__(self, state):
        if state.ndim == 1:
            state = state.reshape((1,-1))
        row_index = np.tile(np.arange(state.shape[0]).reshape((1,-1)), reps=(2, 1)).reshape((-1,))
        col_index = np.vstack((state[:,0].reshape((-1,1)), state[:,1].reshape((-1,1)))).squeeze()
        s = sp.coo_matrix((np.ones(state.shape[0]*2), (row_index, col_index)),
                          shape = (state.shape[0], self.size))
        return s
    
# phi = ravel_proj( np.ones(2, dtype='int')*dim)
phi = factor_state([dim, dim])
    
    
def plot_plan_features(x0, plan, Fs, phi):
    num = dim
    XX, YY = np.meshgrid(np.linspace(0, dim, num, False),
                         np.linspace(0, dim, num, False))
    points = np.concatenate([ p.reshape((-1,1)) for p in [XX, YY]], axis=1)
    points = points.astype('int')
#     points = np.hstack( (points,np.tile(np.array([[2., 2., 0.]]), (points.shape[0], 1))))
    x_T = x0.T
    fig = plt.figure()
    anim_img = []
    values = phi(points).dot(x_T)
        
    if scipy.sparse.issparse(values):
        values = values.toarray()
    imgplot = plt.imshow(values.reshape((num, -1)))
    imgplot.set_interpolation('none')
    anim_img.append([imgplot])
    for count, p in enumerate(plan):
        x_T = sum([ prob*F.dot(x_T) for prob, F in izip(p, Fs)])
        values = phi(points).dot(x_T)
        
        if scipy.sparse.issparse(values):
            values = values.toarray()
            
        imgplot = plt.imshow(values.reshape((num, -1)))
        imgplot.set_interpolation('none')
        anim_img.append([imgplot])
        
    anim = animation.ArtistAnimation(fig, anim_img, interval=100, repeat_delay = 1000,
                               blit = True)
    plt.show()
    
def plot_plan_actions(s0, plan, env):
    env.set_state(s0)
    
    fig = plt.figure()
    anim_img = []
    
    imgplot = plt.imshow(env.get_state_img(env.get_state()))
    imgplot.set_interpolation('none')
    anim_img.append([imgplot])
    
    for count, p in enumerate(plan):
        a = np.argmax(p)
        if a<4:
            env.step(a)
        imgplot = plt.imshow(env.get_state_img(env.get_state()))
        imgplot.set_interpolation('none')
        anim_img.append([imgplot])
        
    anim = animation.ArtistAnimation(fig, anim_img, interval=100, repeat_delay = 1000,
                               blit = True)
    plt.show()

    
# s0 =np.array([0. ,0., 2., 2., 0.])
# sg = np.array([2. ,0., 2., 2., 0.])
s0 =np.array([0. ,0.], dtype='int')

# sg = [[0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [5,0]]
# w = [0, 1, 1, 1, 1, -2, 2]

sg = [[0,1],[5,0]]
w = [0, 2]

# R_terminal = [ (phi(np.array(s, 'int'))) * float(c) for s, c in zip(sg, w)]
# R_terminal = sum(R_terminal).T
# 
# R = [ -np.ones(R_terminal.shape) + phi(np.array([5, 0], 'int')).T  for i in xrange(5)]

x0 = phi(s0)
# xg = phi(sg)
Fs = [matrices['F'+str(i)] for i in xrange(4)]
R = [matrices['R'+str(i)].T for i in xrange(4)] + [np.zeros((phi.size,1))]

print R[0]
Fs = [F.toarray() if sp.issparse(F) else F for F in Fs]
Fs += [eye(phi.size)]
if not scipy.sparse.issparse(Fs[0]):
    Fs[-1] = Fs[-1].toarray()
    
if scipy.sparse.issparse(x0):
    x0 = x0.toarray()
#     R_terminal = R_terminal.toarray()# + np.ones((phi.size, 1))*-1
# print [type(F) for F in Fs], Fs[0].shape

plan_len = 20
plan = linear_find_stoc_plan(x0, R, np.zeros((phi.size,1)), phi, Fs, plan_len)
print plan
# print msg

# plan_dummy = [np.array([1.0, 0.0, 0.0, 0., 0])]* plan_len


plot_plan_actions(np.zeros(2, 'int'), plan, env)
plot_plan_features(x0, plan, Fs, phi)
plt.show()
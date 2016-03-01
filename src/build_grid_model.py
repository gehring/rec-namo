from recnamo.gridnamo import gridnamo

import numpy as np
from scipy import sparse as sp

from scipy import io
import scipy.linalg

dim = 6
size = np.array((dim,dim), dtype='int')

walls = [(1,0), (1,1), (1,2), (1,3), (1,5),
         (2,0), (2,1), (2,2), (2,3), (2,5)]

walls = None

env = gridnamo([], size, walls = walls)


# next_index = [list() for i in xrange(4)]


def solve_F(X, Xp):
    if sp.issparse(X):
        X = X.toarray()
    if sp.issparse(Xp):
        Xp = Xp.toarray()
        
    F = scipy.linalg.lstsq(X, Xp)[0]
    return F

def solve_R(X, R):
    if sp.issparse(X):
        X = X.toarray()
        
    R = scipy.linalg.lstsq(X, R)[0]
    return R
    
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
        
X = [ np.zeros((dim**2, 2)) for i in xrange(4)]
Xp = [ np.zeros((dim**2, 2)) for i in xrange(4)]
R = [ np.zeros(dim**2) for i in xrange(4)]
goal = np.array(([5,0]), dtype='int')
for i in xrange(dim**2):
    bs = sp.coo_matrix((np.ones(1), (np.zeros(1, dtype='int'), 
                                     np.ones(1, dtype='int')*i)),
                        shape = (1, dim**2))
    for j in xrange(4):
        env.set_state_from_binary(bs)
        X.append(env.get_state())
        X[j][i,:] = env.get_state()
        env.step(j)
        Xp[j][i,:] = env.get_state()
        R[j][i] = 1 if np.all(Xp[j][i,:] == goal) else 0
#         next_index[j].append(env.get_binary_vector(env.get_state()).col[0])
        
        
# Fs = [ sp.coo_matrix((np.ones(dim**2), (np.array(n, dtype='int'), np.arange(dim**2))),
#                      shape = (dim**2, dim**2))
#         for n in next_index]    

phi = factor_state([dim, dim])
Fs = [ solve_F(phi(X[j]), phi(Xp[j])) for j in xrange(4)]
R = [ solve_F(phi(X[j]),R[j]) for j in xrange(4)]

matrices = {'F'+str(i):F for i,F in enumerate(Fs)}
for i, r in enumerate(R):
    matrices['R' + str(i)] = r
io.savemat('grid_models.mat', matrices)

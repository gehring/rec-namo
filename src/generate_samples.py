from recnamo.recnamo import RectNamo, parse_world
from recnamo.representation import TileCoding, IndexToBinarySparse, RandomProjector

import numpy as np
from scipy import io
from scipy import optimize
from scipy import linalg
import scipy.sparse as sparse
import cvxopt.solvers
import cvxopt.cholmod
from cvxopt import spmatrix, matrix
import pickle

from itertools import chain, izip, product, ifilter

def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())
    return SP
 
def spmatrix_sparse_to_scipy(A):
    data = np.array(A.V).squeeze()
    rows = np.array(A.I).squeeze()
    cols = np.array(A.J).squeeze()
    return sparse.coo_matrix( (data, (rows, cols)) )

def hacky_nnls_solve(A,B, lamb, iter = 10, phis = None):
    X = sparse.eye(A.shape[0])
    ApI = A + lamb *sparse.eye(A.shape[0])
    alpha = 0.1
    beta = 0.01
    for i in xrange(iter):
        alpha = np.clip(beta/(1*(i+1)), 0.0, 1.0)
        Xp = X - alpha*((ApI).dot(X) - B)
        Xp.data = np.clip( Xp.data, 0.0, np.infty)
        Xp.eliminate_zeros()
        
        X = X * (1-alpha) + alpha*Xp
        
        score = 0
        if phis is not None:
            remainder = phis[0].dot(X) - phis[1]
            score = np.sum(remainder.data**2)/phis[0].shape[0]
        print X.data.size, score
    return X

def dense_scipy_solve(phi_t, phi_tp1):
    if sparse.issparse(phi_t):
        phi_tp1 = phi_tp1.toarray()
        phi_t = phi_t.toarray()
    print phi_t.shape
    X = linalg.lstsq(phi_t, phi_tp1)[0]
        
    return X

class action_rectnamo(object):
    
    actions = [ np.array([0.5, 0]),
               np.array([-0.5, 0]),
               np.array([0, 0.5]),
               np.array([0, -0.5])]
    def __init__(self, rectnamo):
        self.rectnamo = rectnamo
        self.num_actions = 1+ len(rectnamo.poly) + len(self.actions)
        
    def apply_action(self, i):
        s_t = self.rectnamo.state
        
        if i == 0:
            self.rectnamo.toggle_grab()
        elif i <= len(self.rectnamo.poly):
            agent_state = s_t[-3:-1]
            self.rectnamo.move(np.array(self.rectnamo.poly[i-1].centroid.coords).squeeze() - agent_state)
        elif i<= len(self.rectnamo.poly)+ len(self.actions):
            j = i - len(self.rectnamo.poly) - 1
            self.rectnamo.move(self.actions[j])
        
        s_tp1 = self.rectnamo.state
        return s_t, i, s_tp1
    
    @property
    def state(self):
        return self.rectnamo.state
    
    @state.setter
    def state(self, value):
        self.rectnamo.state = value
        
        
        
def grid_of_points(state_range, num_centers):
    if isinstance(num_centers, int):
        num_centers = [num_centers] * state_range[0].shape[0]
    points = [ np.linspace(start, stop, num, endpoint = True) 
                    for start, stop, num in izip(state_range[0],
                                                 state_range[1],
                                                 num_centers)]
    
    points = np.meshgrid(*points)
    points = np.concatenate([ p.reshape((-1,1)) for p in points], axis=1)
    return points

def generate_transition(state, env):
    samples = []
    for act in product(xrange(env.num_actions), xrange(env.num_actions)):
        env.state = state
        samples.extend([env.apply_action(a)  for a in act])         
    return samples

def generate_states(s_range, envi, num_samples):
    return ifilter(lambda s: envi.is_valid_state(s) , [s for s in grid_of_points(s_range, [num_samples]*(s_range[0].shape[0]-1) + [2])])

def generate_samples(state_generator, transition_generator):
    return chain( *[ transition_generator(state) for state in state_generator])

def random_state(s_range):
    s = np.random.rand(s_range[0].shape[0])* (s_range[1] - s_range[0]) + s_range[1]
    s[-1] = 0.0
    return s

samples = None

s_range, poly = parse_world('testworld.xml')
env = RectNamo(poly, s_range, agent_size= np.ones(2))
act_env = action_rectnamo(env)

s_range = [np.hstack((np.tile(s_range[0], len(poly) + 1), np.zeros(1))),
           np.hstack((np.tile(s_range[1], len(poly) + 1), np.ones(1)))]


# states =chain( generate_states(s_range, env, 5), ifilter(lambda s: env.is_valid_state(s),
#                                                    ( random_state(s_range) for i in xrange(800)))) 
# 
# 
# samples = list(generate_samples(states, lambda s: generate_transition(s, act_env)))
# 
# with open('sample_trans.data', 'wb') as f:
#     pickle.dump(samples, f)
    
if samples is None:
    with open('sample_trans.data', 'rb') as f:
        samples = pickle.load(f)
        
print 'separating actions'
samples = [filter(lambda x: x[1]==i, samples) for i in xrange(act_env.num_actions)]


print 'solving for F'
dim = s_range[0].shape[0]
input_indices = [np.array([i], 'int') for i in xrange(dim)] + \
                [np.array(list(index) + [dim-1]) for index in product(xrange(dim-1), xrange(dim-1))] + \
                [np.arange(dim)]
                
# input_indices = [np.arange(dim)]
                
ntiles = ([10] * (dim-1) + [2] + 
            [[10, 10, 2]]*((dim - 1)**2) + 
            [[10]*(dim-1) + [2]] )

# ntiles = ([[10]*(dim-1) + [2]] )
#             42, 5 nz
#             3200, 16 nz
#             20000, 1 nz
            
ntilings = ([10] * (dim-1) + [1] + 
            [2] * (dim-1)**2 +
            [10])

# ntilings = ([5])

                
phi = IndexToBinarySparse(TileCoding(input_indices, ntiles, ntilings, None, s_range, bias_term = True))
phi = RandomProjector(phi, n_dim = 1000)
print phi.size
Fs = []
for a_samples in samples:
    print 'processing action', a_samples[0][1]
    s_t, a, s_tp1 = zip(*a_samples)
    phi_t = phi(np.vstack(s_t))
    phi_tp1 = phi(np.vstack(s_tp1))* 0.99
    print 'encoding complete, now solving...'
#     A = phi_t.T.dot(phi_t) #+ sparse.eye(phi_t.shape[1])*0.1
#     B = phi_t.T.dot(phi_tp1)
#     A = scipy_sparse_to_spmatrix(A)
#     B = scipy_sparse_to_spmatrix(B)
    print 'A, B computed, calling solver'
#     Fcol = [ cvxopt.solvers.coneqp(P=A, 
#                                    q=matrix(B[:,i].toarray()), 
#                                    G=spmatrix(-1.0, range(A.size[0]), range(A.size[0])),
#                                    h=matrix(np.zeros(A.size[0]))) 
#                                     for i in xrange(A.size[0])]
#     Ft = spmatrix_sparse_to_scipy(cvxopt.cholmod.splinsolve(A, B))
#     Ft = hacky_nnls_solve(A, B, 0, iter=5000, phis = (phi_t, phi_tp1))
    Ft = dense_scipy_solve(phi_t,phi_tp1)
    print 'solution has non-zero elements', Ft.shape#, Ft.data.size
    Fs.append(Ft.T)
    
matrices = {'F'+str(i):F for i,F in enumerate(Fs)}
io.savemat('linear_models.mat', matrices)
with open('representation.data', 'wb') as f:
    pickle.dump(phi, f)

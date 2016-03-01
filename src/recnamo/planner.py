import theano.sparse
import theano
import theano.tensor
from theano import shared

from itertools import izip

import numpy as np

import scipy.sparse
from scipy.optimize import minimize



import pyOpt
from pyOpt import Optimization
from pyOpt import pySNOPT
from pyOpt import SLSQP
from pyOpt import PSQP
from pyOpt import CONMIN
from pyOpt import ALGENCAN

def get_location_cost_fn_sparse(x_g, Fs, x_0, plan, n):
    dcdphis = [x_g]
    for p in plan[:-1]:
        dcdphis.append(reduce( theano.sparse.add, [ (p[i]* (theano.sparse.true_dot(dcdphis[-1],Fs[i]))) for i in xrange(n)]))
    
    dcdphis.reverse()
    
    x_T = x_0.T
    dcdps = []
    for p, dcdphi in izip(plan, dcdphis):
        x_T = reduce( theano.sparse.add, [ (p[i]* (theano.sparse.true_dot(Fs[i],x_T))) for i in xrange(n)])
        for F in Fs:
            dcdps.append(theano.sparse.dot(dcdphi, theano.sparse.true_dot(F,x_T)))
        
    gradient = theano.tensor.concatenate(dcdps)
    return theano.sparse.dot(-x_g,x_T)[0,0],  gradient.T

def get_location_cost_fn_dense(x_g, Fs, x_0, plan, plan_flat, n):
    x_T = x_0.T
    for p in plan:
        x_T = reduce( theano.tensor.add, [ (p[i]* (theano.tensor.dot(Fs[i],x_T))) for i in xrange(n)])
    fn = theano.tensor.dot(-x_g,x_T)[0,0]
    return theano.printing.Print('cost')(fn),  theano.tensor.grad(fn, plan_flat)


def linear_combination_transition(Fs, x, flat_plan, index_offset, n):
    prob = theano.tensor.exp(flat_plan[index_offset:index_offset + n])
    prob = prob/ theano.tensor.sum(prob)
    return sum([ (prob[i]* (theano.tensor.dot(Fs[i],x))) for i in xrange(n)])
#     return reduce( theano.tensor.add, [ (flat_plan[index_offset + i]* (theano.tensor.dot(Fs[i],x))) for i in xrange(n)])


def linear_combination_transition_noexp(Fs, x, flat_plan, index_offset, n):
    prob = flat_plan[index_offset:index_offset + n]
    return sum([ (prob[i]* (theano.tensor.dot(Fs[i],x))) for i in xrange(n)])

def get_location_cost_fn_dense_scan(x_g, Fs, x_0, plan_flat, n):
    Fs = theano.shared( np.concatenate([ F.reshape(1,F.shape[0], -1) for F in Fs], axis = 0))
    outputs_info = theano.shared(x_0.reshape((-1,1)))
    x_T, updates = theano.scan( fn = lambda index, x_t, p_flat, F: linear_combination_transition(F, x_t, p_flat, index*n, n),
                     outputs_info = outputs_info,
                     sequences = [theano.tensor.arange(plan_flat.shape[0]/n)],
                     non_sequences= [plan_flat, Fs])
    x_T = x_T[-1]
    print 'b'
#     print type(x_T), x_T[-1].ndim, outputs_info.ndim
    
    fn = theano.tensor.dot(-x_g,x_T)[0,0]
    return theano.printing.Print('cost')(fn),  (theano.tensor.grad(fn, plan_flat))

def get_location(Fs, x_0, plan_flat, n):
    pass

class const_jac(object):
    def __init__(self, i, n):
        self.i = i
        self.n = n
    def __call__(self, p):
        i = self.i
        n = self.n
        j = np.zeros_like(p)
        j[i*n:(i+1)*n] = 1
        return j
    
class const_sum_one(object):
    def __init__(self, i, n):
        self.i = i
        self.n = n
    def __call__(self, p):
        i = self.i
        n = self.n
        return np.sum(p[i*n:(i+1)*n]) - 1
    

class obj_fun(object):
    def __init__(self, f, g):
        self.f = f
        self.g = g
        
    def __call__(self, x, *args):
        return self.f(x), np.array([fg(x) for fg in self.g]), 0
    
def get_pyopt_optimization(f, g_f, con, g_con, x0, T):
    opt_prob = Optimization('stoc planner', obj_fun(f, con))
    opt_prob.addObj('f')
    opt_prob.addVarGroup('flat_plan', 
                         x0.size, 
                         type='c', 
                         value = x0,
                         lower = 0.,
                         upper = 1.0)
    opt_prob.addConGroup('g', T, 'e')
    
#     opt = SLSQP()
#     opt = pySNOPT.SNOPT()
#     opt = PSQP()
#     opt = CONMIN()
    opt = ALGENCAN()
    
    return opt_prob, opt
    
    
    

    
def get_random_plan(T, n):
    p0 = np.abs(np.ones((T,n))/n + np.random.normal(loc = 0.0, scale = 0.01, size = (T, n)))
    p0 = p0/ (np.sum(p0, axis=1)[:,None])
    
#     p0 = [ 1.0] +  [0.0] * 4 + [ 1.0] +  [0.0] * 4 +[0.0] *4 + [1.0] +[0.0] *4 + [1.0]
#     p0 = np.abs(np.array(p0).reshape((-1,5)))#+ np.random.normal(loc = 0.0, scale = 0.05, size = (T, n)))
#     p0 = p0/ (np.sum(p0, axis=1)[:,None])
    return p0.flatten()
def linear_find_stoc_plan(x_0, x_g, phi, Fs, T):
    n = len(Fs)
    
    plan_flat = theano.tensor.vector(name='plan')
    plan_flat.tag.test_value = np.random.rand(T*n).astype('float32')
    plan = [plan_flat[i*n:(i+1)*n] for i in xrange(T)]
    
#     start_index = theano.tensor.iscalar('start_index')
#     horizon = theano.tensor.iscalar('T')
#     horizon.tag.test_value = T
    if scipy.sparse.issparse(Fs[0]):
        Fs = [F.tocsr() for F in Fs]
        Fs = [theano.sparse.CSR(F.data, F.indices, F.indptr, F.shape) for F in Fs]
        x_0 = theano.sparse.CSR(x_0.data, x_0.indices, x_0.indptr, x_0.shape)
        x_g = theano.sparse.CSR(x_g.data, x_g.indices, x_g.indptr, x_g.shape)
        f, df = get_location_cost_fn_sparse(x_g, Fs, x_0, plan,n)
    else:
        f, df = get_location_cost_fn_dense_scan(x_g, Fs, x_0, plan_flat, n)
#         f, df = get_location_cost_fn_dense(x_g, Fs, x_0, plan, plan_flat, n)
    
    
    
#     con = [const_sum_one(i,n) for i in range(T)]
#     g_con = [const_jac(i,n) for i in range(T)]
#     p0 = get_random_plan(T, n)
#     
#     func = theano.function([plan_flat], 
#                             [f, df],
#                             allow_input_downcast=True)
#     
#     f = lambda x: float(func(x)[0])
#     df = lambda x:func(x)[1].reshape((1,-1))
#     
#     opt_prob, opt = get_pyopt_optimization(f, df, con, g_con, p0, T)
#     print df(p0), type(df(p0))
#     ff, xs, msg = opt(opt_prob, sens_type = obj_fun(df, g_con))
#     print xs
#     print f(xs)
# 
#     const = [{'type' : 'eq',
#              'fun' : const_sum_one(i,n),
#             'jac' : const_jac(i, n)} 
#              for i in range(T)]
     
    options = {'gtol': 1e-16}
     
    func = theano.function([plan_flat], 
                           [f, df],
                           allow_input_downcast=True)
#     func = theano.function([plan_flat], 
#                            [f, df],
#                            allow_input_downcast=True)

    p0 = get_random_plan(T, n)
    score = np.zeros(n)
    for j in xrange(T):
        for i in xrange(5):
            p0[j*n:(j+1)*n] = -1e5
            p0[j*n+i] = 0.0
            score[i] = func(p0)[0]
        p0[j*n:(j+1)*n] = -1e5
        p0[j*n+np.argmin(score)] = 0.0
    print p0
    plan = [p0[i*n:(i+1)*n] for i in xrange(T)]
        
    def fn(x):
        res = func(x)
        return res[0].astype('float64'), res[1].astype('float64')
    print 'starting optimization'
#     print get_random_plan(T, n)
#     results = [minimize(fn, 
#                          get_random_plan(T, n).astype('float32'),
# #                         method = 'SLSQP',
#                          jac=True, 
#                          bounds = [(-1.0e2, 1.0e2)]*(n*T),
#                          options = options) for i in xrange(1)]
#     plan_flat = min(results, key = lambda x: x['fun'])
#      
#     print plan_flat['fun']
     
#     plan = [plan_flat['x'][i*n:(i+1)*n] for i in xrange(T)]
    return plan, func, None
    
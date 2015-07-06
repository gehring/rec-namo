import theano.sparse
import theano
import theano.tensor
from theano import shared

from itertools import izip

import numpy as np

import scipy.sparse
from scipy.optimize import minimize

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
    print sum([ (flat_plan[index_offset + i]* (theano.tensor.dot(Fs[i],x))) for i in xrange(n)]).type
    return sum([ (flat_plan[index_offset + i]* (theano.tensor.dot(Fs[i],x))) for i in xrange(n)])
#     return reduce( theano.tensor.add, [ (flat_plan[index_offset + i]* (theano.tensor.dot(Fs[i],x))) for i in xrange(n)])


def get_location_cost_fn_dense_scan(x_g, Fs, x_0, T, plan_flat, n):

    outputs_info = theano.shared(x_0.reshape((-1,1)))
    print outputs_info.type
    x_T, updates = theano.scan( fn = lambda index, x_t, p_flat: linear_combination_transition(Fs, x_t, p_flat, index*n, n),
                     outputs_info = outputs_info,
                     sequences = theano.tensor.arange(T),
                     non_sequences=plan_flat)
    x_T = x_T[:,-1]
    print 'b'
#     print type(x_T), x_T[-1].ndim, outputs_info.ndim
    
    fn = theano.tensor.dot(-x_g,x_T)[0,0]
    return theano.printing.Print('cost')(fn),  theano.tensor.grad(fn, plan_flat)

class const_jac(object):
    def __init__(self, i, n):
        self.i = i
        self.n = n
    def __call__(self, p):
        i = self.i
        n = self.n
        j = np.zeros_like(p)
        j[i*n:(i+1)*n] = p[i*n:(i+1)*n]
        return j
    
class const_sum_one(object):
    def __init__(self, i, n):
        self.i = i
        self.n = n
    def __call__(self, p):
        i = self.i
        n = self.n
        return np.sum(p[i*n:(i+1)*n]) - 1

def linear_find_stoc_plan(x_0, x_g, phi, Fs, T):
    n = len(Fs)
    
    plan_flat = theano.tensor.vector(name='plan')
    plan = [plan_flat[i*n:(i+1)*n] for i in xrange(T)]
    
#     start_index = theano.tensor.iscalar('start_index')
    horizon = theano.tensor.iscalar('T')
    if scipy.sparse.issparse(Fs[0]):
        Fs = [F.tocsr() for F in Fs]
        Fs = [theano.sparse.CSR(F.data, F.indices, F.indptr, F.shape) for F in Fs]
        x_0 = theano.sparse.CSR(x_0.data, x_0.indices, x_0.indptr, x_0.shape)
        x_g = theano.sparse.CSR(x_g.data, x_g.indices, x_g.indptr, x_g.shape)
        f, df = get_location_cost_fn_sparse(x_g, Fs, x_0, plan,n)
    else:
        f, df = get_location_cost_fn_dense_scan(x_g, Fs, x_0, horizon, plan_flat, n)
#         f, df = get_location_cost_fn_dense(x_g, Fs, x_0, plan, plan_flat, n)
    
    
    
    
    
    

    const = [{'type' : 'eq',
             'fun' : const_sum_one(i,n),
            'jac' : const_jac(i, n)} 
             for i in range(T)]
    
    func = theano.function([plan_flat, horizon], 
                           [f, df],
                           allow_input_downcast=True)
#     func = theano.function([plan_flat], 
#                            [f, df],
#                            allow_input_downcast=True)
    p0 = np.ones(n*T)/n
    print 'starting optimization'
    plan_flat = minimize(lambda x: func(x, T), 
                         p0,
                         jac=True, 
                         bounds = [(0.0, 1.0)]*p0.size, 
                         constraints = const)
    
    plan = [plan_flat['x'][i*n:(i+1)*n] for i in xrange(T)]
    return plan, func, plan_flat['message']
    
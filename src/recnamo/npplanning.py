import numpy as np
import scipy

from itertools import product, izip

from scipy.linalg import lu_factor, lu_solve

import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.spatial.distance import pdist, squareform

from scikits.sparse.cholmod import cholesky
import matplotlib.pyplot as plt

def build_np_models(kernel, trans_samples, ter_samples, ter_rew_samples, lamb):
    Xa, Ra, Xpa = zip(*trans_samples)
    Xa_term, Ra_term = zip(*ter_samples)
    Xa = [ np.vstack((xa, xa_term)) if xa_term.size > 0 else xa for xa, xa_term in izip(Xa, Xa_term) ]
    Ra = [ np.hstack((ra, ra_term)) if ra_term.size > 0 else ra for ra, ra_term in izip(Ra, Ra_term) ] 
    
    k = len(trans_samples) 
    
    # build the K_a,b matrices
    Kab = dict()
    for a,b in product(xrange(k), xrange(k)):
        if Xa_term[b].size > 0:
            Kab[(a,b)] = np.hstack((kernel(Xa[a], Xpa[b]), 
                                    np.zeros((Xa[a].shape[0], Xa_term[b].shape[0]))))
        else:
            Kab[(a,b)] = kernel(Xa[a], Xpa[b])
        
    
    # build the K_a, D_a matrices
    Ka = [kernel(Xa[i], Xa[i])  for i in xrange(k)]
    Dainv = [Ka[i] + lamb*scipy.eye(*Ka[i].shape) for i in xrange(k)]
    Da = [lu_factor(Dainv[i], overwrite_a = False) for i in xrange(k)]
        
    # build K_ter matrix
    Kterma = [ np.hstack((kernel(ter_rew_samples[0], Xpa[i]),
                          np.zeros((ter_rew_samples[0].shape[0], Xa_term[i].shape[0])))) if Xa_term[i].size > 0
                else kernel(ter_rew_samples[0], Xpa[i]) for i in xrange(k)]
    K_ter = kernel(ter_rew_samples[0], ter_rew_samples[0])
    D_ter = lu_factor(K_ter + lamb*scipy.eye(*K_ter.shape), overwrite_a = True)
    R_ter = ter_rew_samples[1]
    
    return kernel, Kab, Da, Dainv, Ra, Kterma, D_ter, R_ter, Xa
    
def sparse_build_np_models(kernel, trans_samples, ter_samples, ter_rew_samples, lamb):
    Xa, Ra, Xpa = zip(*trans_samples)
    Xa_term, Ra_term = zip(*ter_samples)
    Xa = [ np.vstack((xa, xa_term)) if xa_term.size > 0 else xa for xa, xa_term in izip(Xa, Xa_term) ]
    Ra = [ np.hstack((ra, ra_term)) if ra_term.size > 0 else ra for ra, ra_term in izip(Ra, Ra_term) ] 
    
    k = len(trans_samples) 
    
    # build the K_a,b matrices
    Kab = dict()
    KabT = dict()
    for a,b in product(xrange(k), xrange(k)):
        if Xa_term[b].size > 0:
            Kab[(a,b)] = np.hstack((kernel(Xa[a], Xpa[b]), 
                                    np.zeros((Xa[a].shape[0], Xa_term[b].shape[0]))))
        else:
            Kab[(a,b)] = kernel(Xa[a], Xpa[b])
        Kab[(a,b)] = csr_matrix(Kab[(a,b)] * (np.abs(Kab[(a,b)]) > 1e-3))
        KabT[(a,b)] = Kab[(a,b)].T.tocsr()
    
    # build the K_a, D_a matrices
    Ka = [kernel(Xa[i], Xa[i])  for i in xrange(k)]
    Dainv = [csc_matrix(Ka[i] * (np.abs(Ka[i]) > 1e-3)) + lamb*scipy.sparse.eye(*Ka[i].shape) for i in xrange(k)]
#    print np.linalg.matrix_rank(Dainv[2].toarray()), Dainv[2].shape, np.linalg.cond(Dainv[2].toarray())
#    print np.linalg.eig(Dainv[2].toarray())[0]
#    print [np.linalg.eig(Dainv[i].toarray())[0].min() for i in xrange(3)]
#    plt.spy(Dainv[2].toarray())
#    plt.show()  
#    print Dainv[2].shape

    index = (squareform(pdist(Xa[2])) == 0.0).nonzero()    
    
#    print (squareform(pdist(Xa[2])) == 0.0).nonzero()
#    print Xa[2][index[0][:5],:]
#    print Xa[2][index[1][:5],:]
#    
#    splu(Dainv[0])    
#    cholesky(Dainv[0])
#    splu(Dainv[1])
#    cholesky(Dainv[1])
#    splu(Dainv[2])
    cholesky(Dainv[2])
        
    
    Da= [cholesky(Dainv[i]) for i in xrange(k)]
#    Da = [splu(Dainv[i]) for i in xrange(k)]
        
    # build K_ter matrix
    Kterma = [ np.hstack((kernel(ter_rew_samples[0], Xpa[i]),
                          np.zeros((ter_rew_samples[0].shape[0], Xa_term[i].shape[0])))) if Xa_term[i].size > 0
                else kernel(ter_rew_samples[0], Xpa[i]) for i in xrange(k)]
    K_ter = kernel(ter_rew_samples[0], ter_rew_samples[0])
    D_ter = cholesky(csc_matrix(K_ter*(np.abs(K_ter) > 1e-3)) + lamb*scipy.sparse.eye(*K_ter.shape))
#    D_ter = splu(csc_matrix(K_ter*(np.abs(K_ter) > 1e-3)) + lamb*scipy.sparse.eye(*K_ter.shape))
    R_ter = ter_rew_samples[1]
    
    return kernel, Kab, KabT, Da, Dainv, Ra, Kterma, D_ter, R_ter, Xa
    
@profile        
def non_param_improve(plan, 
                      gamma, 
                      kernel, 
                      Kab, 
                      Da, 
                      Dainv,
                      Ra, 
                      Kterma, 
                      Dterm, 
                      Rterm, 
                      Xa, 
                      x_1,
                      alphas = None,
                      betas = None,
                      forward = True):
    H = plan.shape[0]
    new_plan = plan * gamma   
    k = len(Dainv)
    if alphas is None:
        alphas = [ np.zeros((H, Xa[a].shape[0])) for a in xrange(k)]

    if betas is None:    
        betas = [ np.zeros((H, Xa[a].shape[0])) for a in xrange(k)]
    
    for a in xrange(k):
        alphas[a][H-1, :] = lu_solve( Da[a], Ra[a] + Kterma[a].T.dot(lu_solve( Dterm, Rterm, trans = 1)), trans = 1)
        betas[a][0,:] = lu_solve(Da[a], kernel(Xa[a], x_1))
        
    
    va = np.empty(k)
    if forward:
        for t in xrange(H-2, -1, -1):
            for a in xrange(k):
                a_prime = sum([ plan[t+1,b] * Kab[b, a].T.dot(alphas[b][t+1,:]) for b in xrange(k)])
                alphas[a][t,:] = lu_solve(Da[a], Ra[a] + a_prime, trans = 1)

        for a in xrange(k):
            va[a] = alphas[a][0,:].dot(Dainv[a].dot(betas[a][0,:]))
        old_val = plan[0,:].dot(va)    
        
        for t in xrange(0, H):
            for a in xrange(k):
                if t > 0:
                    b_prime = sum( [Kab[a,b].dot(betas[b][t-1,:]) * new_plan[t-1,b] for b in xrange(k)])
                    betas[a][t] = lu_solve(Da[a], b_prime, trans = 0)
                va[a] = alphas[a][t,:].dot(Dainv[a].dot(betas[a][t,:]))
                
            a_best = np.argmax(va)
            new_plan[t,a_best] += (1-gamma)
    else:
        for t in xrange(1, H-1):
            for a in xrange(k):
                b_prime = sum( [Kab[a,b].dot(betas[b][t-1,:]) * plan[t-1,b] for b in xrange(k)])
                betas[a][t] = lu_solve(Da[a], b_prime, trans = 0)
                
        for t in xrange(H-1,-1, -1):
            for a in xrange(k):
                if t< H-1:
                    a_prime = sum([ new_plan[t+1,b] * Kab[b, a].T.dot(alphas[b][t+1,:]) for b in xrange(k)])
                    alphas[a][t,:] = lu_solve(Da[a], Ra[a] + a_prime, trans = 1)
                va[a] = alphas[a][t,:].dot(Dainv[a].dot(betas[a][t,:]))
                
            a_best = np.argmax(va)
            new_plan[t,a_best] += (1-gamma)
            
        
        for a in xrange(k):
            va[a] = alphas[a][0,:].dot(Dainv[a].dot(betas[a][0,:]))
        old_val = new_plan[0,:].dot(va) 
    
    return new_plan, old_val, alphas, betas
    
@profile        
def sparse_non_param_improve(plan, 
                              gamma, 
                              kernel, 
                              Kab,
                              KabT,
                              Da, 
                              Dainv,
                              Ra, 
                              Kterma, 
                              Dterm, 
                              Rterm, 
                              Xa, 
                              x_1,
                              alphas = None,
                              betas = None,
                              forward = True):
    H = plan.shape[0]
    new_plan = plan * gamma   
    k = len(Dainv)
    if alphas is None:
        alphas = [ np.zeros((H, Xa[a].shape[0])) for a in xrange(k)]

    if betas is None:    
        betas = [ np.zeros((H, Xa[a].shape[0])) for a in xrange(k)]
    
    for a in xrange(k):
#        alphas[a][H-1, :] = Da[a].solve( Ra[a] + Kterma[a].T.dot( Dterm.solve(Rterm, trans = 'T')), trans = 'T')
#        betas[a][0,:] = Da[a].solve( kernel(Xa[a], x_1).squeeze(), trans='N')
        alphas[a][H-1, :] = Da[a]( Ra[a] + Kterma[a].T.dot( Dterm(Rterm)))
        betas[a][0,:] = Da[a]( kernel(Xa[a], x_1).squeeze())
        
    
    va = np.empty(k)
    if forward:
        for t in xrange(H-2, -1, -1):
            for a in xrange(k):
                a_prime = sum([ plan[t+1,b] * KabT[b, a].dot(alphas[b][t+1,:]) for b in xrange(k)])
#                alphas[a][t,:] = Da[a].solve( Ra[a] + a_prime, trans = 'T')
                alphas[a][t,:] = Da[a]( Ra[a] + a_prime)

        for a in xrange(k):
            va[a] = alphas[a][0,:].dot(Dainv[a].dot(betas[a][0,:]))
        old_val = plan[0,:].dot(va)    
        
        for t in xrange(0, H):
            for a in xrange(k):
                if t > 0:
                    b_prime = sum( [Kab[a,b].dot(betas[b][t-1,:]) * new_plan[t-1,b] for b in xrange(k)])
#                    betas[a][t] = Da[a].solve( b_prime, trans = 'N')
                    betas[a][t] = Da[a]( b_prime)
                va[a] = alphas[a][t,:].dot(Dainv[a].dot(betas[a][t,:]))
                
            a_best = np.argmax(va)
            new_plan[t,a_best] += (1-gamma)
    else:
        for t in xrange(1, H-1):
            for a in xrange(k):
                b_prime = sum( [Kab[a,b].dot(betas[b][t-1,:]) * plan[t-1,b] for b in xrange(k)])
#                betas[a][t] = Da[a].solve( b_prime, trans = 'N')
                betas[a][t] = Da[a]( b_prime)

        for a in xrange(k):
            va[a] = alphas[a][0,:].dot(Dainv[a].dot(betas[a][0,:]))
        old_val = plan[0,:].dot(va)    
        
        for t in xrange(H-1,-1, -1):
            for a in xrange(k):
                if t< H-1:
                    a_prime = sum([ new_plan[t+1,b] * KabT[b, a].dot(alphas[b][t+1,:]) for b in xrange(k)])
#                    alphas[a][t,:] = Da[a].solve( Ra[a] + a_prime, trans = 'T')
                    alphas[a][t,:] = Da[a]( Ra[a] + a_prime)
                va[a] = alphas[a][t,:].dot(Dainv[a].dot(betas[a][t,:]))
                
            a_best = np.argmax(va)
            new_plan[t,a_best] += (1-gamma)
    
    return new_plan, old_val, alphas, betas

def get_random_plan(T, n):
    p0 = np.abs(np.ones((T,n))/n)# + np.random.normal(loc = 0.0, scale = 0.01, size = (T, n)))
    p0 = p0/ (np.sum(p0, axis=1)[:,None])
    return p0

def check_convergence(p0, p1):
    converged = np.allclose(p0, p1)
    return converged

@profile
def find_stoc_plan(x_1, H, num_actions, models, gamma, forward = True, sparse = False):
    
    if sparse:
        improve_step = sparse_non_param_improve
    else:
        improve_step = non_param_improve
    
    
    p0 = get_random_plan(H, num_actions)
    p1, best_val, alphas, betas = improve_step(p0, gamma, *models, x_1 = x_1, forward = forward)
    i = 0
    while not (check_convergence(p0, p1) or i>4):
        p0 = p1
        p1, val, alphas, betas = improve_step(p0, 
                                                   gamma, 
                                                   *models, 
                                                   x_1 = x_1, 
                                                   alphas = alphas, 
                                                   betas = betas,
                                                   forward = forward)
        if best_val != 0:
            norm_val = best_val
        elif val != 0:
            norm_val = val
        else:
            norm_val = 1.0
            
        if np.abs(val - best_val)/np.abs(norm_val) <1e-2:
            i += 1
        else:
            i=0
        print i, val
        best_val = val
        if np.isnan(best_val):
            break
            
    return p1, alphas, betas
        
                            

class NonParametricModel(object):
    def __init__(self, kernel, trans_samples, ter_samples, lamb):
        rew_models, trans_models, ter_model = build_np_models(kernel, 
                                                              trans_samples, 
                                                              ter_samples, 
                                                              lamb)
        
        
    
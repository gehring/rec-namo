import __builtin__

try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile



from recnamo.domain import MountainCar
from recnamo.npplanning import build_np_models, find_stoc_plan, sparse_build_np_models

import numpy as np

from itertools import product, izip

from scipy.linalg import lu_solve

from matplotlib import pyplot as plt

import pylab as pl
from matplotlib import collections  as mc
from matplotlib.animation import FuncAnimation


def generate_domain():
    domain = MountainCar(random_start=True)
    return domain


domain = generate_domain()
state_range = domain.state_range
policy = domain.get_pumping_policy()


def generate_data(domain):
#    state_range = domain.state_range
#    num = 20
#    XX, YY = np.meshgrid(np.linspace(state_range[0][0], state_range[1][0], num, True),
#                          np.linspace(state_range[0][1], state_range[1][1], num, True))
#    points = np.concatenate([ p.reshape((-1,1)) for p in [XX, YY]], axis=1)
#      
#    actions = domain.discrete_actions
#    samples = []
#    for s in points:
#         for i in xrange(3):
#             domain.reset()
#             domain.state = s
#             r_t, s_tp1 = domain.step(actions[i])
#             samples.append((s, i, r_t, s_tp1))
#             if s_tp1 is not None:
##                 continue
#                 for j in xrange(3):
#                     domain.state = s_tp1
#                     r_tp1, s_tp2 = domain.step(actions[j])
#                     samples.append((s_tp1, j, r_tp1, s_tp2))
#                     if s_tp2 is not None:
#                         continue
#                         for k in xrange(3):
#                             domain.state = s_tp2
#                             r_tp2, s_tp3 = domain.step(actions[k])
#                             samples.append((s_tp2, k, r_tp2, s_tp3))
#                             if s_tp3 is not None:
#                                 for l in xrange(3):
#                                     domain.state = s_tp3
#                                     r_tp3, s_tp4 = domain.step(actions[l])
#                                     samples.append((s_tp3, k, r_tp3, s_tp4))

    num_traj = 20
    samples = []
    actions = domain.discrete_actions
    for i in xrange(num_traj):
        s_t = domain.reset()
        while s_t is not None:
            if np.random.rand(1) < 0.05:
                a = 1#np.random.randint(3)
            else:
                a = policy(s_t)
            r_t, s_tp1 = domain.step(actions[a])
            samples.append((s_t, a, r_t, s_tp1))
            s_t = s_tp1
    domain.random_start= False        
    s_t = domain.reset()
    while s_t is not None:
        a = policy(s_t)
        r_t, s_tp1 = domain.step(actions[a])
        samples.append((s_t, a, r_t, s_tp1))
        s_t = s_tp1
                
    sample_a = [ list() for i in xrange(3)]
    term_sample_a = [ list() for i in xrange(3)]
    for s_t, a_t, r_t, s_tp1 in samples:
        if s_tp1 is not None:
            sample_a[a_t].append((s_t, r_t, s_tp1))
        else:
            term_sample_a[a_t].append((s_t, r_t))
            
    for i in xrange(3):
        X, R, Xp = zip(*sample_a[i])
        sample_a[i] = (np.array(X), np.zeros_like(np.array(R)), np.array(Xp))
    #         sample_a[i] = (np.array(X), np.array(R), np.array(Xp))
    
        if len(term_sample_a[i]) > 0:
            X, R = zip(*term_sample_a[i])
            term_sample_a[i] = (np.array(X), np.ones_like(np.array(R)))
    #         term_sample_a[i] = (np.array(X), np.array(R))
        else:
            term_sample_a[i] = (np.array([[]]), np.array([]))
         
        
    term_rew_samples = (sample_a[0][0], np.zeros_like(sample_a[0][1]))
    return sample_a, term_sample_a, term_rew_samples, samples


        
def apply_plan(plan, domain):
    pass


def plot_fn(y, D, data, kernel):
    num = 80
    l = (state_range[1] - state_range[0])*0.8
    s_range = [state_range[0] - l/2, state_range[1] + l/2]
    XX, YY = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], num, True),
                         np.linspace(s_range[0][1], s_range[1][1], num, True))
    points = np.concatenate([ p.reshape((-1,1)) for p in [XX, YY]], axis=1)
    
    k = kernel(data, points)
    alpha = lu_solve(D, y, trans=1)
    val = alpha.dot(k)
    plt.pcolormesh(XX, YY, val.reshape((num, -1)))
    plt.colorbar()
    plt.show()
    





def plot_samples(samples, traj):
    S,A,R,Sp = zip(*samples)
    lines = [ [x0, x1] for x0,x1 in zip(S, Sp) if x1 is not None]
    fig, ax = pl.subplots()
    
    sample_color = [(0,0,1,1)]*len(lines)
    
    traj_color = [(1,0,0,1)]*len(traj)
#     print traj, len(traj)
    lc = mc.LineCollection(lines + traj, color = sample_color+traj_color, linewidths=2)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    
def update_scatter(i, sizes, colors, scat):
    scat.set_sizes(sizes[i])
    scat.set_edgecolors(colors[i])
#    print i
  
def animate_betas(betas, Xa):
    fig2 = plt.figure()
    img = []
    max_beta = np.max( [b.max() for b in betas])
    min_beta = np.min( [b.min() for b in betas])
    sizes = []
    colors = []
    points = []
    for a in range(3):
        points.append(Xa[a])
    points = np.vstack(points)    
    
    for i in xrange(betas[0].shape[0]):
        psize = []
        for a in range(3):
            psize.append( betas[a][i,:].squeeze())
        s = np.hstack(psize)
        mn, mx= s.min(), s.max()#min_beta, max_beta
        if mx - mn == 0.0:
            mx += 0.1
        s = ((s - mn)/(mx - mn))
        c = s[:, None] * np.array([1,0,0,1])[None,:]
        c += (1-s)[:, None] * np.array([0,0,1,1])[None,:]
        s *= 100 + 0.1
        sizes.append(s)
        colors.append(c)
    scat = plt.scatter(points[:,0], points[:,1], s=sizes[0], c = colors[0] )
    anim = FuncAnimation(fig2, update_scatter, frames = len(sizes),
                            interval=3000/betas[0].shape[0], repeat_delay = 1000,
                            fargs=(sizes, colors, scat))
    return anim

def kernel(X, Y):
    if X.ndim == 1:
        X = X.reshape((1,-1))
        
    if Y.ndim == 1:
        Y = Y.reshape((1,-1))
    width = 0.02
    scale = ((state_range[1] - state_range[0]) * width)[None,:,None]
        
    # compute squared weighted distance distance 
    dsqr = -(((X[:,:,None] - Y.T[None,:,:])/scale)**2).sum(axis=1)
    return np.exp(dsqr).squeeze()

trans_samples, ter_samples, ter_rew_samples, samples = generate_data(domain)



#Kab = models[1]
#
#K = Kab[(0,1)]
#if sparse:
#    Da = models[3]
#    print type(K), type(K.T)
#    print float(K.nnz)/ (K.shape[0] * K.shape[1])
#    print 'U', float(Da[0].L.nnz)/(Da[0].L.shape[0]* Da[0].L.shape[1])

# for i in xrange(3):
#     plot_fn(y_a[i], D_a[i], X_a[i], kernel)
@profile
def cat(lamb=0.1, gamma = 0.5, forward = True):
    
    sparse = True
    if sparse:
        models = sparse_build_np_models(kernel, 
                                 trans_samples, 
                                 ter_samples, 
                                 ter_rew_samples, 
                                 lamb)
    else:
        models = build_np_models(kernel, 
                             trans_samples, 
                             ter_samples, 
                             ter_rew_samples, 
                             lamb)    
    
    
    Xa = models[-1] 
    domain.random_start = False
    state = domain.reset()
    plan, alphas, betas = find_stoc_plan(state, 200, 3, models, gamma, forward = forward, sparse = sparse)
    # plan[:,:] = np.array([0,0,1])[None,:]
    
    traj = []
    actions = domain.discrete_actions
    s_t = state
    for p in plan:
        a = actions[np.argmax(p)]
#        print np.argmax(p), p.max()
        r_t, s_tp1 = domain.step(a)
        if s_tp1 is None:
            break
        else:
            traj.append((s_t, s_tp1))
        s_t = s_tp1
      
      
#    plot_samples(samples, [])      
    plt.ioff()
    plot_samples(samples, traj)
    plt.title('$\lambda=$'+str(lamb) + ', $\gamma=$'+str(gamma)+', forward='+str(forward))
    anim = animate_betas(betas, Xa)
    plt.title('$\lambda=$'+str(lamb) + ', $\gamma=$'+str(gamma)+', forward='+str(forward))
  
    return anim



a4 = cat(lamb = 0.01, gamma=0.5, forward = True)
a5 = cat(lamb = 0.01, gamma=0.5, forward = False)
#a3 = cat(lamb = 0.5, gamma=0.5, forward = True)
#a2 = cat(lamb = 0.5, gamma=0.5, forward = False)
#a0 = cat(lamb = 1.0, gamma=0.5, forward = True)
#a1 = cat(lamb = 1.0, gamma=0.5, forward = False)
#
#
#cat(lamb = 0.1, gamma=0.0, forward = True)
#cat(lamb = 0.1, gamma=0.0, forward = False)
#cat(lamb = 0.5, gamma=0.0, forward = True)
#cat(lamb = 0.5, gamma=0.0, forward = False)
#cat(lamb = 1.0, gamma=0.0, forward = True)
#cat(lamb = 1.0, gamma=0.0, forward = False)
plt.show()



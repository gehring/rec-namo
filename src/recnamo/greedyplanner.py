import numpy as np

import theano.tensor as T
import theano

from itertools import izip

def linear_combination_transition(Fs, x, flat_plan, index_offset, n):
    prob = flat_plan[index_offset:index_offset + n]
    return sum([ (prob[i]* (T.dot(Fs[i],x))) for i in xrange(n)])

def linear_combination_left_transpose(Fs, x, flat_plan, index_offset, n):
    prob = flat_plan[index_offset:index_offset + n]
    return sum([ (prob[i]* (T.dot(x,Fs[i]))) for i in xrange(n)])

def compute_expected_goal(Fs):
    pass


def numpy_greedy_iteration(Fs, plan, x0, R, R_terminal, blend = 0.5):
    expected_goal = [R_terminal]
    for p in plan[:0:-1]:
        x = sum([(p[i]* (Fs[i].T.dot(expected_goal[-1]) + R[i])) for i in xrange(len(Fs))])
        expected_goal.append(x)
    expected_goal.reverse()
    new_plan = []
    for xg, p in izip(expected_goal, plan):
        expected_outcome = np.array([ xg.T.dot(F.dot(x0)) for F in Fs]).squeeze()
        # argmax with random tie break
#         i = np.argmax(expected_outcome)
#         i = np.random.choice(np.argwhere(expected_outcome == np.amax(expected_outcome)).flatten(),1)[0]
        i = np.argwhere(expected_outcome == np.amax(expected_outcome)).squeeze()

#         print x0.nonzero()
        p_new = np.zeros(len(Fs))
        p_new[i] = 1.0/i.size
#         p_new[i] = 1.0
        
        p_new = blend*p + (1-blend)*p_new
        new_plan.append(p_new)
        
        x0 = sum([(p_new[i]* Fs[i].dot(x0)) for i in xrange(len(Fs))])
#     val = expected_outcome.dot(new_plan[-1])
#     print val, expected_goal[0].T.dot(x0)[0,0]


    # calculate total val:
    expected_goal = [R_terminal]
    for p in plan[:0:-1]:
        x = sum([(p[i]* (Fs[i].T.dot(expected_goal[-1]) + R[i])) for i in xrange(len(Fs))])
        expected_goal.append(x)
    expected_goal.reverse()
    expected_outcome = np.array([ expected_goal[0].T.dot(F.dot(x0)) for F in Fs]).squeeze()
    val = expected_outcome.dot(new_plan[0])
    return new_plan, val

def get_random_plan(T, n):
    p0 = np.abs(np.ones((T,n))/n)# + np.random.normal(loc = 0.0, scale = 0.01, size = (T, n)))
    p0 = p0/ (np.sum(p0, axis=1)[:,None])
    return p0.flatten()

def check_convergence(plan0, plan1):
    converged = len(plan0) ==  len(plan1)
    for p0, p1 in izip(plan0, plan1):
        converged &= np.allclose(p0, p1)
        if not converged:
            break
    return converged

def linear_find_stoc_plan(x_0, R, R_terminal, phi, Fs, T):
    x_0 = x_0.T
    
    n = len(Fs)
    p0 = get_random_plan(T, n)
    plan0 = [p0[i*n:(i+1)*n] for i in xrange(T)]
    plan1, best_val = numpy_greedy_iteration(Fs, plan0, x_0,  R, R_terminal)
    i = 0
    while not (check_convergence(plan0, plan1) or i>20):
        plan0 = plan1
        plan1, val = numpy_greedy_iteration(Fs, plan0, x_0,  R, R_terminal)
        if np.abs(val - best_val) <1e-9:
            i += 1
        else:
            i=0
        print i, val
        best_val = val
            
    return plan1
    
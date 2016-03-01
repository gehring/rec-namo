import numpy as np 
import scipy as sp 
import scipy.sparse.linalg 
from scipy.sparse.linalg import splu

Adense = np.matrix([[ 1.,  0.,  0.], 
        [ 0.,  1.,  0.], 
        [ 0.,  0.,  1.]]) 
As =  sp.sparse.csc_matrix(Adense) 
x = np.random.randn(3) 
b = As.dot(x) 

Asinv = splu(As)

print x 
print Asinv.solve(b)
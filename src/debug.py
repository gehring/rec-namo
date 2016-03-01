from recnamo.gridnamo import gridnamo
import numpy as np

env = gridnamo([], np.ones(2)*10)

s = np.array( [[1,0], [2,0], [1,0], [1,1], [2,2]], dtype = 'int')
bs = env.get_binary_vector(s)
ss = env.get_state_vector(bs)
print bs.col
print ss
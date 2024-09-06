#%%
import numpy as np
import torch
#%% numpy
#create vector
nv = np.array([[1,2,3,4]])
print(nv), print(' ')
#transpose
print(nv.T), print(' ')
# transpose the tronpose
nvT = nv.T
print(nvT.T)

print('Matrix')

# create for a matrix
nm = np.array([[1,2,3,4],[5,6,7,8]])
print(nm), print(' ')
# transpose
print(nm.T), print(' ')
# transpose the transpose
nmT = nm.T
print(nmT.T)

#%% torch
# create tensor
tv = torch.tensor([[1,2,3,4]])
print(tv), print(' ')
# transpose
print(tv.T), print(' ')
# tranpose the transpose
tvT = tv.T
print(tvT.T)

print('Matrix')

# create matrix
tm = torch.tensor([[1,2,3,4],[5,6,7,8]])
print(tm), print(' ')
#tranpose
print(tm.T), print(' ')
#tranpose to tranpose
tmT = tm.T
print(tmT.T)

#%%
print(f'Variable nv is of type {type(nv)}')
print(f'Variable nm is of type {type(nm)}')
print(f'Variable tv is of type {type(tv)}')
print(f'Variable tm is of type {type(tm)}')
'''
Variable nv is of type <class 'numpy.ndarray'>
Variable nm is of type <class 'numpy.ndarray'>
Variable tv is of type <class 'torch.Tensor'>
Variable tm is of type <class 'torch.Tensor'>
'''

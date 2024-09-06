#%%
import numpy as np
import matplotlib.pyplot as plt
#%% entropy
# probability of an event happening
x = [.25,.75]
H = 0
for p in x:
    H +=-(p*np.log(p))
print(f'Entropy: {H}')

# Binary cross-entropy
H = -(p*np.log(p)+ (1-p)*np.log(1-p))
print(f'Binary cross-entropy: {H}')

#%% cross-entropy
p = [1,0] # sum 1
q = [.25,.75] # sum 1

H=0
for i in range(len(p)):
    H +=-(p[i]*np.log(q[i]))
print(f'Cross Entropy: {H}')

# also correct, written out for N=2 events
H = -(p[0]*np.log(q[0])+ p[1]*np.log(q[1]))
print(f'Cross-entropy: {H}')

# simplification
H = -np.log(q[0])
print(f'Manually simplified Cross Entropy: {H}')

#%% torch
import torch
import torch.nn.functional as F

q_tensor = torch.Tensor(q)
p_tensor = torch.Tensor(p)
print(F.binary_cross_entropy(q_tensor, p_tensor))

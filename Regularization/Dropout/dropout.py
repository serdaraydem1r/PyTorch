#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
# define a dropout instance and make some data
prob = .5
dropout = nn.Dropout(p=prob)
x = torch.ones(10)

# let's see what dropout returns
y = dropout(x)
print(x)
print('-'*10)
print(y)
print('-'*10)
print(torch.mean(y))

#%%
# dropout is turned off when evaluating the model
dropout.eval()
y = dropout(x)
print(y)
print(torch.mean(y))

#%%
# annoyingly, F.dropout() is not deactivated in eval mode:
dropout.eval()
y = F.dropout(x, p=prob)
print(y)
print(torch.mean(y))

# but you can manually switch it off
dropout.eval()
y = F.dropout(x, training=False)
print(y)
print(torch.mean(y))
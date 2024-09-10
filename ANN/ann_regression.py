#%% library  import
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

#%%
# create data
N = 30
x = torch.randn(N, 1)
y = x + torch.randn(N, 1)/2

# plot
plt.figure(figsize=(10,10))
plt.plot(x,y,'s')
plt.show()
#%%
# basic build model
ANNreg = nn.Sequential(
    nn.Linear(1,1), # input layer
    nn.ReLU(), # activation function
    nn.Linear(1,1) # output layer
)

#%%
learning_rate = .05
lossfun = nn.MSELoss()
# optimizer (the flavor of gradient descent to implement)
optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learning_rate)

# train model
numepochs = 500
losses = torch.zeros(numepochs)
for epochi in range(numepochs):
    # forward pass
    yHat = ANNreg(x) # tahmin edilen değerler ileriye yayılım
    # compute loss
    loss = lossfun(yHat, y) # tahmin edilenler ile loss hesapla
    losses[epochi] = loss
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#%%
# show the losses

predictions = ANNreg(x)
testloss = (predictions-y).pow(2).mean()

plt.figure(figsize=(10,10))
plt.plot(losses.detach(),'o',markerfacecolor='blue',linewidth=.1)
plt.plot(numepochs,testloss.detach(),'ro')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Final Loss = %g'%testloss.item())
plt.show()
print(testloss.item())
#%%
plt.figure(figsize=(10,10))
plt.plot(x,y,'bo',label='Real Data')
plt.plot(x,predictions.detach(),'rs',label='Predicted Data')
plt.title(f'Prediction Data r = {np.corrcoef(y.T,predictions.detach().T)[0,1]:.2f}')
plt.legend()
plt.show()
#%%






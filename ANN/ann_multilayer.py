#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize

from ANN.ann_learningrate import ANNclassify, createANNmodel, totalacc
from ANN.ann_regression import lossfun, optimizer, losses, predictions

#%%
# create data
nPerClust = 100
blur = 1

A=[1,3]
B=[1,-2]

# generate data
a = [A[0]+np.random.rand(nPerClust)*blur, A[1]+np.random.rand(nPerClust)*blur]
b = [B[0]+np.random.rand(nPerClust)*blur, B[1]+np.random.rand(nPerClust)*blur]

# true labels
labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

# concatanate into a matriz
data_np = np.hstack((a,b)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

# show data
fig = plt.figure(figsize(5,5))
plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'rs')
plt.title('The qwerties')
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.show()

#%%
def ANNmodel(learningRate):
    # model architecture
    ANNclassify = nn.Sequential(
        nn.Linear(2,16), # input layer
        nn.ReLU(), # activation unit
        nn.Linear(16,1), # hidden layer
        nn.ReLU(),
        nn.Linear(1,1), # output unit
        nn.Sigmoid(), # bunu kullanmadan lossfun kullanmak önerilir
    )
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=learningRate) # gradyan dexcent
    return ANNclassify,lossfun,optimizer

#%%
# a function that trains the model

# a fixed parameter
numepochs = 1000

def trainTheModel(ANNmodel):
    # initialize losses
    losses = torch.zeros(numepochs)

    # loop over epochs
    for epochi in range(numepochs):
        # forward pass
        yHat = ANNmodel(data)
        # compute loss
        loss = lossfun(yHat,labels)
        losses[epochi] = loss
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # final forward pass
        predictions = ANNmodel(data)
        totalacc = 100*torch.mean(((predictions>.5)==labels).float())
        return totalacc,losses,predictions
#%%
# create everthing
ANNclassify,lossfun,optimizer = ANNmodel(.01)

# run it
losses, predictions, totalacc = trainTheModel(ANNclassify)

# report accuracy
print(f'Final Accuracy: {totalacc}')

# show the losses
plt.figure(figsize(7,7))
plt.plot(losses.detach(),'o',markerfacecolar='w',linewidth=.1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
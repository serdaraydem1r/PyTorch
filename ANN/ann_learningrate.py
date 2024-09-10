#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ANN.ann_regression import optimizer, predictions

#%%
# create data
nPerClust = 100
blur = 1
A = [ 2.1 ,3 ] # veri bulutlarının x,y merkez kordinatları
B = [  2.75,2.1  ]

# generate data
a = [ A[0]+np.random.rand(nPerClust)*blur,
      A[1]+np.random.rand(nPerClust)*blur ]
b = [ B[0]+np.random.rand(nPerClust)*blur,
      B[1]+np.random.rand(nPerClust)*blur ]

# true labels
labels_np = np.vstack((np.zeros((nPerClust,1)),
                       np.ones((nPerClust,1))))
# concatanate into a matrix
data_np = np.hstack((a,b)).T
# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()
# show the data
fig = plt.figure(figsize = (5,5))
plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
plt.title("ANN Classification")
plt.xlabel("dimension 1")
plt.ylabel("dimension 2")
plt.show()
#%% Functions to build and train the model
def createANNmodel(learningRate):
    ANNclassify = nn.Sequential(
        nn.Linear(2,1),
        nn.ReLU(),
        nn.Linear(1,1),
        #nn.Sigmoid() # BCEWithLogitsLoss() hesaplayacak, pytorchda bunu öneriyor
    )
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=learningRate)
    return ANNclassify, lossfun, optimizer

#%% functions that trains the model
numepochs = 1000
def trainModel(ANNmodel):
    # losses
    losses = torch.zeros(numepochs)
    for epochi in range(numepochs):
        yHat = ANNmodel(data)

        loss = lossfun(yHat,labels)
        losses[epochi] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = ANNmodel(data)
        totalacc = 100*torch.mean(((predictions>0)==labels).float())
        return losses, totalacc, predictions
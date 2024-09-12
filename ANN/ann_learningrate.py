#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpmath import sigmoid

from ANN.ann_regression import optimizer, predictions, lossfun, losses

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

        predictions = sigmoid(ANNmodel(data))
        totalacc = 100*torch.mean(((predictions>0)==labels).float())
        return losses, totalacc, predictions

#%%
ANNclassify,lossfun,optimizer = createANNmodel(.01)

losses,predictions,totalacc = trainModel(ANNclassify)

print(f"Final Accuracy: {totalacc}")

plt.figure(figsize = (5,5))
plt.plot(losses.detach(),'o',markerfacecolor='blue',linewidth = .1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#%%
learningrates = np.linspace(.001,.1,40)
# initialize results output
accByLR = []
allLosses = np.zeros((len(learningrates),numepochs))

# loop through learnin rates
for i, lr in enumerate(learningrates):
    ANNclassify,lossfun,optimizer = createANNmodel(lr)
    losses,predictions,totalacc = trainModel(ANNclassify)
    accByLR.append(totalacc)
    allLosses[i,:] = losses.detach()
#%%
fig,ax = plt.subplots(1,2,figsize = (12,4))

ax[0].plot(learningrates,accByLR,'s-')
ax[0].set_xlabel('Learning Rate')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy vs Learning Rate')

ax[1].plot(allLosses.T)
ax[1].set_xlabel('Epoch Number')
ax[1].set_ylabel('Loss')
ax[1].set_title('Loss by learning rate')
plt.show()

#%%
sum(torch.tensor(accByLR)>70)/len(accByLR)
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

#%%
# Create Data
nPerClust = 200

th = np.linspace(0,4*np.pi,nPerClust)
r1 = 4
r2 = 5

# generate data
a = [ r1*np.cos(th)+np.random.rand(nPerClust)*3,
      r1*np.sin(th)+np.random.rand(nPerClust) ]
b = [ r2*np.cos(th)+np.random.rand(nPerClust),
      r2*np.sin(th)+np.random.rand(nPerClust)*3]

# true labels
labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

# concatanate into a matrix
data_np = np.hstack((a,b)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

# show the data
fig = plt.figure(figsize=(5,5))
plt.plot(data[np.where(labels==0)[0],0], data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0], data[np.where(labels==1)[0],1],'ko')
plt.title('Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

#%%
# Separate the data into DataLoaders
train_data, test_data,train_labels, test_labels = train_test_split(data,labels,test_size=0.2)

# the convert them into Pytorch Datasets(note: already converted to tensor)
train_data = TensorDataset(train_data,train_labels)
test_data = TensorDataset(test_data,test_labels)

# finally translate into dataloader objects
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

#%%
# Create the model
class theModelClass(nn.Module):
      def __init__(self,dropoutRate):
            super().__init__()

            # layers
            self.input = nn.Linear(2,128)
            self.hidden = nn.Linear(128,128)
            self.output = nn.Linear(128,1)

            # parameters
            self.dr = dropoutRate
      # forward pass
      def forward(self, x):
            # pass the data through the input layer
            x = F.relu(self.input(x))

            # dropout after input layer
            x = F.dropout(x,p=self.dr,training=self.training)

            # pass the data through the hidden layer
            x = F.relu(self.hidden(x))

            # dropout after hidden layer
            x = F.dropout(x,p=self.dr,training=self.training)

            # output layer
            x = self.output(x)
            return x
#%%
# test the model
tmpnet = theModelClass(.5)
# run some random data through
tmpdata = torch.randn((10,2))
yhat = tmpnet(tmpdata)
yhat
#%%
# functions to create and train the net
def createNewModel(dropoutRate):
      # grab an instance of the model class
      ANNQC = theModelClass(dropoutRate)

      # loss function
      lossfun = nn.BCEWithLogitsLoss()

      # optimizer
      optimizer = torch.optim.SGD(ANNQC.parameters(), lr=0.01)

      return ANNQC, lossfun, optimizer

#%%
# train the model

numepochs = 1000

def trainTheModel(ANNQC, lossfun, optimizer):
      trainAcc = []
      testAcc = []

      # loop over epochs
      for epochi in range(numepochs):
            # switch training mode on
            ANNQC.train()
            batchAcc = []
            for X, y in train_loader:
                  # forward pass and loss
                  yHat = ANNQC(X)
                  loss = lossfun(yHat,y)

                  # backprop
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()

                  # compute training accuracy just for this batch
                  batchAcc.append(100*torch.mean(((yHat>.5)==y).float()).item())
            trainAcc.append(np.mean(batchAcc))

            # test accuracy
            ANNQC.eval()
            X,y = next(iter(test_loader))
            yHat = ANNQC(X)
            testAcc.append( 100*torch.mean(((yHat>.5)==y).float()).item())
      return trainAcc, testAcc

#%%
# Test the Model
# create a model
dropoutrate = .0
ANNQC, lossfun, optimizer = createNewModel(dropoutrate)
trainAcc, testAcc = trainTheModel(ANNQC, lossfun, optimizer)

#%%
# create a 1D smoothing filter
def smooth(x,k=5):
      return np.convolve(x,np.ones(k)/k,mode='same')

#%%
fig = plt.figure(figsize=(10,5))
plt.plot(smooth(trainAcc),'bs-')
plt.plot(smooth(testAcc),'ro-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train','Test'])
plt.title('Dropout rate = %g'%dropoutrate)
plt.show()

#%%

dropoutRates = np.arange(10)/10
results = np.zeros((len(dropoutRates),2))

for i in range(len(dropoutRates)):
      ANNQC, lossfun, optimizer = createNewModel(dropoutRates[i])
      trainAcc, testAcc = trainTheModel(ANNQC, lossfun, optimizer)

      # store accuracies from last 100 epochs
      results[i,0] = np.mean(trainAcc[-100:])
      results[i,1] = np.mean(testAcc[-100:])
#%%
fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(dropoutRates,results,'o-')
ax[0].set_xlabel('Dropout Proportion')
ax[0].set_ylabel('Accuracy (%)')
ax[0].legend(['Train','Test'])

ax[1].plot(dropoutRates,-np.diff(results,axis=1),'o-')
ax[1].plot([0,.9],[0,0],'k--')
ax[1].set_xlabel('Dropout Proportion')
ax[1].set_ylabel('Train-test difference (acc%)')

plt.show()
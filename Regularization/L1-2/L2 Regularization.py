#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from Regularization.Dropout.ann_dropout import trainAcc
from Regularization.Dropout.dropout import dropout

#%%
iris = sns.load_dataset('iris')
data = torch.tensor(iris[iris.columns[0:4]].values).float()
labels = torch.zeros(len(data),dtype=torch.long)
labels[iris.species=='versicolor']=1
labels[iris.species=='virginica']=2
#%%
train_data, test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.2)
# convert them into PyTorch Datasets
train_dataDataset = torch.utils.data.TensorDataset(train_data,train_labels)
test_datasetDataset = torch.utils.data.TensorDataset(test_data,test_labels)

# create dataloader objects
train_loader = DataLoader(train_dataDataset,batch_size=64,shuffle=True,drop_last=True)
test_loader = DataLoader(test_datasetDataset, batch_size=test_datasetDataset.tensors[0].shape[0],drop_last=False)
#%%
# Model
def createNewModel(L2lambda):
    ANNiris = nn.Sequential(
        nn.Linear(4,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,3)
    )
    # loss function
    lossfun = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(ANNiris.parameters(), lr=.005, weight_decay=L2lambda)
    return ANNiris, lossfun, optimizer

#%%
# train model
numepochs = 1000
def trainTheModel(ANNiris, lossfun, optimizer):
    trainAcc = []
    testAcc = []
    losses = []
    for epoch in range(numepochs):
        batchAcc = []
        batchLoss = []
        for X,y in train_loader:
            # forward pass nd loss
            yHat = ANNiris(X)
            loss = lossfun(yHat,y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAcc.append(100*torch.mean((torch.argmax(yHat,axis=1)==y).float()).item())
            batchLoss.append(loss.item())

        trainAcc.append(np.mean(batchAcc))
        losses.append(np.mean(batchLoss))

        ANNiris.eval()
        X,y = next(iter(test_loader))
        predlabels = torch.argmax(ANNiris(X),axis=1)
        testAcc.append(100*torch.mean((predlabels==y).float()).item())
        ANNiris.train()
    return trainAcc, testAcc, losses

#%%
# create a model
L2lambda = .01
ANNiris, lossfun, optimizer = createNewModel(L2lambda)
trainAcc, testAcc, losses = trainTheModel(ANNiris, lossfun, optimizer)
# plot the result
fig, ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(losses,'k^-')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_title(' Losses with L2 $\lambda$=' + str(L2lambda))

ax[1].plot(trainAcc,'ro')
ax[1].plot(testAcc,'bs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_xlabel('Epoch')
ax[1].set_title('Accuracy with L2 $\lambda$=' + str(L2lambda))
ax[1].legend(['Train','Test'])
plt.show()



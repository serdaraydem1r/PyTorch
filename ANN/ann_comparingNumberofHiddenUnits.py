#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
#%% import data
iris = sns.load_dataset('iris')
print(iris.head())
sns.pairplot(iris,hue='species')
plt.show()
#%% organize the data
# convert from pandas dataframe to tensor
data = torch.tensor(iris[iris.columns[0:4]].values).float()
# transform species to number
labels = torch.zeros(len(data),dtype=torch.long)
# labels[iris.species=='setosa'] = 0 # dont need
labels[iris.species=='versicolor'] = 1
labels[iris.species=='virginica'] = 2
labels
#%% Functions to create and train the model
# note the input into the function!
def createIrisModel(nHidden):
    # model architecture (with number of units soft-coded!)
    ANNiris = nn.Sequential(
        nn.Linear(4,nHidden),
        nn.ReLU(),
        nn.Linear(nHidden,nHidden),
        nn.ReLU(),
        nn.Linear(nHidden,3)
    )
    # loss function
    lossfun = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(ANNiris.parameters(), lr=.01)

    return ANNiris, lossfun, optimizer

#%%
# a function to train the model
def trainTheModel(ANNiris):
    # loop over epochs
    for epochi in range(numepochs):
        # forward pass
        yHat = ANNiris(data)

        # compute loss
        loss = lossfun(yHat,labels)


        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # final forward pass
    predictions = ANNiris(data)
    predlabels = torch.argmax(predictions,axis=1)
    return 100*torch.mean((predlabels==labels).float())

numepochs = 150
numhiddens = np.arange(1,129)
accuracies = []
for nunits in numhiddens:
    ANNiris, lossfun, optimizer = createIrisModel(nunits)
    acc = trainTheModel(ANNiris)
    accuracies.append(acc)

#%% Report Accuracy
fig,ax = plt.subplots(1,figsize=(13,6))
ax.plot(accuracies,'ko-',markerfacecolor='w',markersize=9)
ax.plot(numhiddens[[0,-1]],[33,33],'--',color=[.8,.8,.8])
ax.plot(numhiddens[[0,-1]],[67,67],'--',color=[.8,.8,.8])
ax.set_ylabel('Accuracy')
ax.set_xlabel('Number of hidden units')
ax.set_title('Accuracy vs Number of hidden units')
plt.show()
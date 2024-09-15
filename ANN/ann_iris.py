#%%
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
#%% Create ANN model
# model architecture
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
optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.1)
#%%
numepochs = 1000
# initialize losses
losses = torch.zeros(numepochs)
ongoinAcc = []

# loop over epochs
for epochi in range(numepochs):
    # forward pass
    yHat = ANNiris(data)
    # compute loss
    loss = lossfun(yHat, labels)
    losses[epochi] = loss
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # compute accuracy
    matches = torch.argmax(yHat,axis=1)==labels # booleans (false/true)
    matchesNumeric = matches.float() # convert to numbers (0/1)
    accuracyPct = 100*torch.mean(matchesNumeric)
    ongoinAcc.append(accuracyPct)

# final forward pass
predictions = ANNiris(data)
predlabels = torch.argmax(predictions,axis=1)
totalacc = 100*torch.mean((predlabels == labels).float())

#%%
print('Final Accuracy: %g%%' %totalacc)

# plot the result
fig, ax = plt.subplots(1,2,figsize=(13,4))
ax[0].plot(losses.detach())
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_title('Losses')

ax[1].plot(ongoinAcc)
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_title('Accuracy')
plt.show()

#%%

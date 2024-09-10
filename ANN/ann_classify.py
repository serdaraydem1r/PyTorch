#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#%%
# create data
nPerClust = 100
blur = 1
A = [ 4, 5 ] # veri bulutlarının x,y merkez kordinatları
B = [ 3, 3 ]

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
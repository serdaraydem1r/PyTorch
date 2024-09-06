#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
X = [1,2,4,6,5,4,0,-4,5,-2,6,10,-9,1,3,-6]
n = len(X)

# mean
popmean = np.mean(X)
# compute a sample mean
sample = np.random.choice(X,size=10,replace=True)
sampmean = np.mean(sample)

print(popmean)
print(sampmean)
#%%
# compute lost of sample means

n = 10000
sampleMeans = np.zeros(n)
for i in range(n):
    sample = np.random.choice(X,size=10,replace=True)
    sampleMeans[i] = np.mean(sample)

plt.hist(sampleMeans,bins=50,density=True)
plt.plot([popmean,popmean],[0,.3],'m--')
plt.ylabel('Count')
plt.xlabel('Sample Mean')
plt.show()


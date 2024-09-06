#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
#%%
n1 = 30 # samples in datasets1
n2 = 40 # samples in dataset2
mu1 = 1 # population mean in dataset1
mu2 = 2 # population mean in dataset2

# generate data
data1 = mu1+np.random.rand(n1)
data2 = mu2+np.random.rand(n2)

# plot them
plt.plot(np.zeros(n1),data1,'ro',markerfacecolor='w',markersize=14)
plt.plot(np.ones(n2),data2,'bs',markerfacecolor='w',markersize=14)
plt.xlim([-1,2])
plt.xticks([0,1],labels=['Group 1', 'Group 2'])
plt.show()

#%% t-test
t,p = stats.ttest_ind(data1,data2)
print(t)
print(p)

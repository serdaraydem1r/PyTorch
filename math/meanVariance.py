#%%
import numpy as np

x = [1,2,4,6,5,4,0]
n = len(x)
mean1 = np.mean(x)
mean2 = np.sum(x)/n
print(mean1)
print(mean2)

# variance
var1 = np.var(x)
var2 = (1/(n-1))*np.sum((x-mean1)**2)
print(var1)
print(var2)
var3 = np.var(x,ddof=1)
print(var3)


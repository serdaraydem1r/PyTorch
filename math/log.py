#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
x = np.linspace(.0001,1,200)
logx = np.log(x)
fig = plt.figure(figsize=(10,6))
plt.rcParams.update({'font.size': 16})

plt.plot(x,logx,'ks-',markerfacecolor='w')
plt.xlabel('x')
plt.ylabel('log(x)')
plt.title('log(x)')
plt.show()

#%%

x1 = np.linspace(.0001,1,200)
logx1 = np.log(x1)
expx1 = np.exp(x1)

plt.plot(x,x1,color=[0.8,0.8,0.8])
plt.plot(x,np.exp(logx1),color=[0.8,0.8,0.8])

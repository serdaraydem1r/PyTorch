#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# function
x = np.linspace(-2*np.pi, 2*np.pi, 401)
fx = np.sin(x) * np.exp(-x**2*.05)
# derivative
df = np.cos(x)*np.exp(-x**2*.05) + np.sin(x)*(-.1*x)*np.exp(-x**2*.05)
# plot
plt.figure(figsize=(10,10))
plt.plot(x,fx, x,df)
plt.legend(['f(x)','df'])
plt.show()

#%%
# function
def fx(x):
    return np.sin(x)*np.exp(-x**2*.05)
def derivative(x):
    return np.cos(x)*np.exp(-x**2*.05) - np.sin(x)*(-.1*x)*np.exp(-x**2*.05)
#%%
localmin = np.random.choice(x,1)
learning_rate = .01
training_epochs = 1000

for i in range(training_epochs):
    grad = derivative(localmin)
    localmin = localmin - learning_rate * grad
plt.figure(figsize=(10,10))
plt.plot(x,fx, x,df,'---')
plt.plot(localmin,derivative(localmin),'bo')
plt.plot(localmin,fx(localmin),'ro')

plt.xlim(x[[0,-1]])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)','df','f(x) min'])
plt.show()

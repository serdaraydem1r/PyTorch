#%%
'''
f(x) = 3x**2 - 3x+4
'''
import numpy as np
import matplotlib.pyplot as plt
#%%
# function
def fx(x):
    return 3*x**2 - 3*x+4

# derivative function
def derivative(x):
    return 6*x - 3
#%%
# define a range for x
x = np.linspace(-2,2,2001)
#plotting
plt.figure(figsize=(6,6))
plt.plot(x,fx(x),x,derivative(x))
plt.xlim(x[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)=3x^2-3x+4',"f(x)'=6*x-3"])
plt.show()

#%%
# random starting point
localmin = np.random.choice(x,1)
# learning parameters
learning_rate = .01
training_epochs = 1000
# run through training
for i in range(training_epochs):
    grad = derivative(localmin)
    localmin = localmin - learning_rate * grad
    print('local min:', localmin)

#%%
plt.figure(figsize=(6,6))
plt.plot(x,fx(x),x,derivative(x))
plt.plot(localmin,derivative(localmin),'bo')
plt.plot(localmin,fx(localmin),'ro')
plt.xlim(x[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)=3x^2-3x+4',"f(x)'=6*x-3",f'Global min: {localmin}'])
plt.show()



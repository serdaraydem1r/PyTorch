#%%
'''
Exercise: compute the mean of var
'''

def exercise():
    var = []
    a = int(input("Enter a number: "))
    for i in range (1,a+1):
        var.append(i)
        print(sum(var)/len(var))
exercise()

#%%
import numpy as np

numbers = [1,2,3,4,5,6,7,8,9,10]
print(np.mean(numbers))

#%%
print(type(np.linspace(1,7,5)))

#%%
numbers_np = np.array(numbers)
print(type(numbers))
print(type(numbers_np))

#%%
numbers_np.min()
numbers_np.max()

#%%
import numpy as np
import pandas as pd

# create random data
var1 = np.random.rand(100)*5+20
var2 = np.random.rand(100)>0

#variable labels
labels = ['Temp(C)','Ice cream']

# put them together into a dictionary
D = {labels[0]:var1, labels[1]:var2}

#import the dictionary into a pandas dataframe
df = pd.DataFrame(D)
print(df.head())
print(df.tail())
print(df.describe())
print(df.corr())
print(df.count())
#%%
'''
Exercise: create a pandas dataframe with:
    int going from 0 to 10, their square, and their log
'''
a = np.array(range(11))
b = np.square(a)
c = np.log(b)

labels = ['Numbers','Square','Log']
D1 = {labels[0]:a, labels[1]:b, labels[2]:c}
df = pd.DataFrame(D1)
print(df.head())
print(df.tail())
print(df.describe())
print(df.corr())
print(df.count())

#%% functions

def func():
   print("Hello")
func()

def func2(input1,input2):
    print(input1+input2)
func2(10,2)

def func3(i,j):
    return i*j
result = func3(10,-2)
print(result)

def func4(i,j):
    result=i*j
    return result
print(func4(10,2))

def func5(i,j):
    prod = i*j
    summ = i+j
    return prod,summ
print(func5(10,2))

#%%
# lambda function
'''
function name = lambda x(girdi) : x**2-1(işlemler)
'''
functionLambda = lambda i,j:i*j
print(functionLambda(10,2))

#%%
'''
Exercise: create function that computes a factorial
'''
import math
def factorial(n):
    for i in range(1,n+1):
        print(math.factorial(i))
print(factorial(5))
def faktoriyel1(n):
    sonuc = 1
    for i in range(2, n + 1):
        sonuc *= i
    return sonuc
sayi = 5
print(f"{sayi}! = {faktoriyel1(sayi)}")

#%%
'''
Exercise: write function that flips a coin N times, and reports the average
'''
import numpy as np
def flip(n):
    tura = 0
    for i in range(n):
        flip = np.random.choice(0,1)
        tura += flip
    return tura / n
average_tura = flip(5)
print(average_tura)




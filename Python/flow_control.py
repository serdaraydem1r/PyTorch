caketype = 'bananacake'
print('Probably carrot cake ') if caketype[0]=='c' else print('Wrong')

'''
Exercise: Create a function that compuse the 'dot product' between two vector
    1- check the both inputs are numpy arrays( isinstance() )
    2- check that the two inputs are the same lenght
    3a- break out of the function and give a helpful error message
    3b- compute and return the dot product of the two inputs
'''
import numpy as np
def dot_product(a,b):
    #1- check that both vectors are numpy arrays
    if not isinstance(a,np.ndarray) or not isinstance(b,np.ndarray):
        raise Exception('Inputs must both be numpy arrays')
    #2- chechk taht both vectors have the same lenght
    if not len(a)==len(b):
        raise Exception('Length of inputs must equal length of outputs')
    #3- return the dot product
    return sum(a*b)
print(dot_product(np.array([1,2,3]),np.array([3,2,1])))
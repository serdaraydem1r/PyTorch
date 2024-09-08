#%%
import numpy as np
import sympy as sym
from IPython.display import display

#%%
x = sym.Symbol('x')
fx = 2*x**2
gx = 4*x**3 - 3*x**4

df = sym.diff(fx)
dg = sym.diff(gx)

manual = df*gx + fx*dg
thewrongway = df*dg

viasympy = sym.diff(fx*gx)
print('The Functions:')
display(fx)
display(gx)
print(' ')

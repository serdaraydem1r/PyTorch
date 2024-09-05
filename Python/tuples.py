#%%
atuple = (3,'asdas',3.14) # immutable
alist = [3,'asdas',3.14] # mutable
print(type(atuple))
print(type(alist))
#%%
print(alist)
alist[0] = 2
print(alist)

print(atuple)
atuple[0]=2 #TypeError: 'tuple' object does not support item assignment
print(atuple)



#%% key / value pairs

# [] list
# () tuple
# {} dictionaries
d = dict()
print(type(d))

#%%
d["name"]="BlaBla"
d["age"]=26
print(d)
#%%
print(d["age"])
print(d.keys())
print(d.values())
print(d.items())

#%%
def exam():
    a = int(input("Enter a number: "))
    b = int(input("Enter another number: "))
    d = dict(firstNum=a, secondNum=b, product=a*b)
    print(d.values())
    print(d.items())
    print(d.keys())

exam()


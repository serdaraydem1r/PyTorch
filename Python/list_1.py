#%%
aList = [1,2,3,4,5,6,7,8,9]
strList = ['hi','my','name','is','FSA']
mixList = [1,2.45,'test']
listList = [1,2,3,['hi','my','name','is','FSA'],[50.32,4,5.01]]

#%%
print(4 in aList) # belirli eleman listede var mı?
print(40 not in aList)

print(aList+strList) # listeler birleşebilir.
print(aList*3) # 3 kez listeyi yazdırır.

#%%
len(aList)
type(aList)
aList.append(10)
aList.copy()
aList.count(10)
aList.index(10)
aList.pop(3)
aList.reverse()
aList.sort()
print(aList.count(10))

#%%
lst = [1,2,3,4,5]
print(lst)
lst.append(6)
print(lst)

lst.insert(2,9)
print(lst)

lst.remove(9)
print(lst)

lst.insert(4,'hello')
print(lst)
#%%
lst.remove('hello')
print(lst)
lst.append(3)
print(lst)
lst.append(5)
print(lst)
lst.sort()
print(lst)
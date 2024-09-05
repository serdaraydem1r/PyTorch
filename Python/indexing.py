#%%
alist = [4,6,2,-6,'blabla',10.01]
print(alist[3]) # -6
print(alist[-1]) # 10.01
print(alist[-2]) # blabla

#%%
# get the attribute of Penguin from this list
listList = [4,'hi',[5,4,3],'yo',{'Squirrel':'cute','Penguin':'Yummy'}]
print(listList[4]['Penguin'])

#%% slicing
'''
a[start:stop:skip]
'''
a = list(range(5,11))
print(a)
print(a[1:5:2])
print(a[::-1]) # reverse list

#%% exercise
# use slicing to write "FSA is a nice guy" from the list
text = '345dfyug ecin a si ASF 24234d'
print(text[-8:4:-1])
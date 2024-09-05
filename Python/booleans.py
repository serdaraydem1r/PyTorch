#%%
t = True
f = False

def exam():
    a = int(input("Enter a number: "))
    b = int(input("Enter b number: "))
    c = int(input("Enter c number: "))
    result = c**2 == a**2 + b**2
    if result == True :
        print("Teorem karşılandı")
    else:
        print("Teorem Karşılanmadı")

exam()

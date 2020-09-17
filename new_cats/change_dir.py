import os

def changeName(path):
    i = 0
    for filename in os.listdir(path):
        if i<10:
            num = '00'+str(i)
        elif i<100:
            num = '0'+str(i)
        elif i<1000:
            num = str(i)
        os.rename(path+filename,filename[:3]+'.'+filename[3:])
        i+=1
changeName('./train/')

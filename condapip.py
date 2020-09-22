
from subprocess import call

with open('requirements.txt',encoding='utf-8-sig',mode='r') as file:
    for library_name in file.readlines():
        call("conda install -y " + library_name, shell=True)
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:38:31 2017

@author: thoth
"""

#table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
#print(type(table))
#for name, phone in table.items():
#     print('{0} ==> {1:5d}'.format(name.ljust(10), phone))

import json
#x = [1, 'simple', 'list']
#print(type(x))


with open('data\data2.json','w+') as f:
    try:
        #x = json.load(f)
#        for line in f:
#            if line.strip():
#                x = json.loads(line)
                
        #or
        x = [json.loads(line) for line in f if line.strip()]
        print(x)
    except Exception as inst:
        print(type(inst))
        print(inst)
        x = [1, 'simple', 'list']
    x.append('tth4')    
    json.dump(x,f)
f.closed

#with open('data\data3.txt','r') as f:
#    #read_data = f.read()
#    for line in f:
#        print(line, end='')
#        
#    f.write('\nxi gua like to eat zebra')
#f.closed
#print(read_data)
#print(x2)



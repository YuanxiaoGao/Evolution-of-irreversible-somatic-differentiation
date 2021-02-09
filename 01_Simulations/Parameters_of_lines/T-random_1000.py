#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:51:45 2019

@author: gao
"""

#--------T vs X0-----------------
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
#-----------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=400)

#--------------------choose random lines----------------------------------------------------------------------------------------
'''initial parameter setting: generate the T function datas; we first choose num_lines T function, in each function
we need 2*num_seg random values for contructing the T step function.
	num_lines : number of T functions
'''
num_lines=1000
num_para=7                 # a,x0,b,x1,cs,cg,z
                     
def one_ran():
	
	a=random.uniform(-2, 2)	
	x0=random.uniform(0, 1)	
	b=random.uniform(0, 1)
	x1=random.uniform(0, 1)
	cs=random.uniform(0, 5)
	cg=random.uniform(0, 5)
	z=random.uniform(0, 1)
	
	if x0<x1:
		one=[a,x0,b,x1,cs,cg,z]
	else:
		one=[a,x1,b,x0,cs,cg,z]

	return one


def multi_ran(num_lines):
	ran_data=[one_ran() for k in range(num_lines)]   # all sorted T sequences

	return ran_data

line_data=multi_ran(num_lines)

#---save list
with open('./v37_%s_v0.pickle'%num_lines, 'wb') as f:
    pickle.dump(line_data, f)

#---------------------define T function-------------------------------------------------------

def Time(a,x0,b,x1,x):
	if x<=x0:
		t=1
	elif x>x0 and x<=x1:
		t=1-b+b*((x1-x)/(x1-x0))**a
	elif x>x1:
		t=1-b
	return t

#--------------------draw samples--------------------------------------
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111,aspect='equal')

x = np.linspace(0, 1, 100)

for line in line_data:
#		label, x0,a,b,x1,c
	time=[]
	for item in x:
		time.append(Time(np.power(10,line[0]),line[1],line[2],line[3],item))
	ax.plot(x,time,color="gray",linewidth=0.5, alpha=0.5)

#------remove ticks and top and right frames
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#------Decorate the spins--------
arrow_length = 300 # In points

	# X-axis arrow
ax.annotate('', xy=(-0.05, -0.05), xycoords=('axes fraction', 'data'), 
            xytext=(arrow_length, 0), textcoords='offset points',
            arrowprops=dict(arrowstyle='<|-', fc='black'))

	# Y-axis arrow
ax.annotate('', xy=(-0.045, -0.049), xycoords=('data', 'axes fraction'), 
            xytext=(0, arrow_length), textcoords='offset points',
            arrowprops=dict(arrowstyle='<|-', fc='black'))

#------x and y labels and limits--------
plt.xlim(-0.05,1.07)
plt.ylim(-0.05,1.07)

plt.xlabel('Soma percentage',fontsize=14)
plt.ylabel('Cell doubling time',fontsize=14)


fig.subplots_adjust(wspace=0.6)             # space between two panels
plt.show()

fig.savefig('../../03_Data/Parameters_of_1000_lines/v37_%s_v1.pdf'%num_lines, bbox_inches = 'tight')   # save figures






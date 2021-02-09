#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:15:37 2018

@author: gao
"""

#------------------------------------------------------------------------------------------------------------
import numpy as np
from growth_cluster import Growth
import sys
import pickle

#------------------------------------------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=280)

#------------------------------------------------------------------------------------------------------------
''' inital conditions;
	a-yaxes has the value from 0 to 58;
	b-xaxes has the value from 0 to 1;
'''
num_lines=200
grid_num=51	 	                              # how many pixels grids (grid_num) in the final B-C map
division_times=16

#--------read tuple to get the specific T function in this file-----------------------	
with open('v37_%s_v1.pickle'%num_lines, 'rb') as f:
	line_data = pickle.load(f)

line_data=np.array(line_data)                        # a,x0,b,x1,cs,cg,z
line_data[:,0]=np.power(10,line_data[:,0])           # change the longa into a 

line_data1=line_data[:,[0,1,2,3,6]]                  # get the a,x0,b,x1,z

       #-------- sys.argv
n=int(sys.argv[1])                             
c_cluster=int(sys.argv[2])                             # range of 0-20
line_id=int(sys.argv[3])                            # range from 0 to num_lines


		# ------set a, x0 and b values
c_range=np.array([0,10])                        # range of log(a)
grid_c=np.linspace(c_range[0],c_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points
c=grid_c[c_cluster]

#		# ------set line four paras		

cur_line=line_data1[line_id]
cur_line1=np.insert(cur_line,[4],c)
para=cur_line1                                       #para=np.array([a,x0,b,x1])

		# ------set simulation times and dps
m=300                    	                           # how many simulation times to get the growth rate of one dp
size=np.int(11)          	         		           # how many discrete dp = [size*(size+1)/2]^2 -- how many colors in the final figure
left,right=-10,10     	         		               # boundaries to find root -- growth rate

#------------------------------------------------------------------------------------------------------------
'''Construct initial dps' table: [s1, s2, s3, g1, g2, g3]'''
	#-------- table is a list including 6 parameters [[s1,s2,s3],[g1,g2,g3]]
               #-------- table size which reflected in the later cbar (how many colors appear)
q=np.linspace(0.0,1.0,size)                   # take sequence values between 0-1

initial_para=[]
for i in range(size):
	for j in range(size):
		for k in range(size):
			for l in range(size):
				if 1-q[i]-q[j]>=0 and 1-q[k]-q[l]>=0:
					initial_para.append(np.array([[q[i],1-q[i]-q[j],q[j]],[q[k],1-q[k]-q[l],q[l]]]))

#------------------------------------------------------------------------------------------------------------
'''store all dps and corresponding growth rate \lambda: [s1, s2, s3, g1, g2, g3, \lambda]'''
def dp_lambda(para):
	growth_table=[]
	for item in initial_para:
		growth_value=Growth(m,n,item,para,left,right)
		para_1d=item.flatten()                                    #flatten s1,s2,s3,s4,s5,s6
		growth_table.append(np.insert(para_1d,6,growth_value))    # insert lambda at the end
	return growth_table

#------------------------------------------------------------------------------------------------------------
''' find max dp for constant a & b;       Return np.ndarray containing all max DPs in constant a&b'''
def optimal_dp_lambda(para):
	array_data=np.vstack(dp_lambda(para))				   # change list to np.ndarray

	growth_max=np.nanmax(array_data[:,6])			   	   # maximum lambda

	index_growth_max=np.where(array_data==growth_max)      # index for maximum

	dp_growth_max=array_data[index_growth_max[0]]          #  dp for maximum  :np.ndarray

	if np.shape(dp_growth_max)[0]>1:
		if all(abs(dp_growth_max[:,5]-1.0)<10**(-8)):      # exclude all max dps with g3=1
			return np.array([np.nan,np.nan,np.nan,0,0,1.0,np.nan])
	return dp_growth_max

##-----------------------------------------------------------------------------------------------------
'''output result'''
result=optimal_dp_lambda(para)

with open('data/%s_%s_%s.txt'%(n,c_cluster,line_id), 'wb') as f:
    np.savetxt(f, result, fmt='%1.8f')                     # demical number


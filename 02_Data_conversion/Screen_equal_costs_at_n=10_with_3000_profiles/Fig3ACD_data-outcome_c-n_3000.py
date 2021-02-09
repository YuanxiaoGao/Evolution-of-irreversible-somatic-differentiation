#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:40:40 2018

@author: gao
"""

''' We read the rae data and generated the npy files.

'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import pickle
#import os

#-------------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=300)

#-------------------------------------------------------------------------------
"build para[b,c] grids"
num_lines=3000
grid_num=11                                            # scatter points in x axes i.e. number of b between 0 to 10
division_times=16                                      # n value --division times

		#-------- initial conditions also for optimal_map
c_range=np.array([0,10])                          # range of log(a)
grid_c=np.linspace(c_range[0],c_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points

#---read tuple to get the specific T function in this file--------------------	
with open('../../03_Data/Screen_equal_costs_at_n=10_with_3000_profiles/v37_3000_v0.pickle', 'rb') as f:
     line_data = pickle.load(f)

line_data=np.array(line_data)                        # a,x0,b,x1,z
'''n=14 fixed and cost from 0 to 10'''
#-------------------------------------------------------------------------------
"read max dps from data into result []"

result=[[] for i in range(1,division_times)]             # to store each figure's data

optimal_matrix=[np.zeros(shape=(grid_num,division_times-1))*np.nan]

for n in range(14,15):                        # how many figures or cell divisions
	for c_cluster in range(0,grid_num):                  # 1,---,30
		st_num=0
		
		for line_id in range(0,num_lines):            # 0,---,29

			result1=np.loadtxt('data/%d_%d_%d.txt'%(n,c_cluster,line_id))

			if np.size(result1)>7:                           # more than one dps are optimal,choose one randomly
				index=np.random.randint(len(result1))
				result2=result1[index]
	
			else:
				result2=result1
	
			#------ this change all s values even for the single optimal dps
			if abs(result2[5]-1.0)<10**(-8):
				result3=np.array([np.nan,np.nan,np.nan,0.0,0.0,1.0,result2[6]])
			else:
				result3=result2
				
			st_value=result3[0]+0.5*result3[1]
			if st_value==1.0:
				label=1
				st_num=st_num+1
			else:
				label=0                    # for nan-st lines
	
			#------ insert b and r grid number in the first two place for later drawing 		

			result4=np.array([n,c_cluster,line_id,label,st_value,result3[3],result3[4],result3[5],line_data[line_id][0],line_data[line_id][1],line_data[line_id][2],line_data[line_id][3] ,line_data[line_id][5] ])
#				[n, cs_id= cg_id, line_id,label,st, g1, g2,g3---a, x0, b, x1, z]
#				 0,  1,              2,      3,  4, 5,  6 , 7,  8,  9, 10, 11,12			
			
			result[n-1].append(result4)
		ratio_st0=st_num/num_lines
		ratio_st1=float("{0:.2f}".format(ratio_st0))*100
		optimal_matrix[0][c_cluster][n-1]=ratio_st1

result5=[np.array(i) for i in result]
result6=np.array(result5)                                        #  (15, 3000, 13) division_times, lines*cost_num, all parameters

#-------------------------------------------------------------------------------
'''save the data_outcome'''

np.save('../../03_Data/Screen_equal_costs_at_n=10_with_3000_profiles/data_n10_c_%s.npy'%num_lines,result6)




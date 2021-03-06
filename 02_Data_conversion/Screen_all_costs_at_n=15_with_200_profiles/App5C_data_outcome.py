#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:05:13 2019

@author: gao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:33:56 2019

@author: gao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:40:40 2018

@author: gao
"""
''' IN this version, T growth time is a step function with respect to s/(s+g)
	Here, b=T is the time when s/(s+g)>l; b [0,1]
	otherwise T=1 '''

'''remember to set
	grid_num
	b_interval
	c_interval

'''
import numpy as np
import pickle
#-------------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=300)

#-------------------------------------------------------------------------------
'''read data '''
#division_times=16
division_times=np.array([15])

num_lines=200
grid_num=26

#---read tuple to get the specific T function in this file	
with open('../../03_Data/Parameters_of_200_lines/v37_200_v1.pickle', 'rb') as f:
     line_data = pickle.load(f)

line_data=np.array(line_data)                        # a,x0,b,x1,z

#cs_range=np.array([0,5])                       # x1 range
#grid_cs=np.linspace(cs_range[0],cs_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points
#
#cg_range=np.array([0,5])                       # x1 range
#grid_cg=np.linspace(cg_range[0],cg_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points

#-------------------------------------------------------------------------------
"read max dps from data into result []"

num_division=len(division_times)
result=[[] for i in range(0,num_division)]                       # to store each figure's data ofr n=5,10,15
#result=[]

#for n_th in range(num_division):                                 # how many figures or cell divisions
for n_th in range(0,1):                                 # how many figures or cell divisions

	n=division_times[n_th]
	for cs_n in range(0,grid_num):                               # how many figures or cell divisions
		for cg_n in range(0,grid_num):                           # how many figures or cell divisions	
			for line_id in range(0,num_lines):                   # x axes points = b
				result1=np.loadtxt('./data/%d_%d_%d_%d.txt'%(n,cs_n,cg_n,line_id))
		
		#				#----- if more than one dps to be max
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
				else:
					label=0                    # for nan-st lines

				result4=np.array([n,cs_n,cg_n,line_id,label,st_value,result3[3],result3[4],result3[5],line_data[line_id][0],line_data[line_id][1],line_data[line_id][2],line_data[line_id][3] ,line_data[line_id][4]    ])
#				[n, cs_id, cg_id, line_id,label,st, g1, g2,g3---a, x0, b, x1, z]
#				 0,  1,    2,      3,      4,   5, 6 ,  7, 8,  9, 10, 11,12, 13				
				result[n_th].append(result4)


				
result5=[np.array(i) for i in result]

result6=np.array(result5)


#print(type(result6))
#print(np.shape(result6))
#
#3*22050

#-------------------------------------------------------------------------------
'''save the data_outcome'''

np.save('../../03_Data/Screen_all_costs_at_n=15_with_200_profiles/n15_cs_cg_grids26_lines%s.npy'%num_lines,result6)


#'''read data test'''
#read_data=np.load('/Users/gao/Desktop/deve/Simulation/v34_cost_model/1_comp/1000_sample/data_outcome_%s.npy'%num_lines)
#
#np.shape(read_data[0])
#
#type(read_data)
#
#read_data[0]
#



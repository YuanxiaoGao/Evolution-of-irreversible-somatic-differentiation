#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:40:40 2018

@author: gao
"""

''' IN this version, T growth time is a step function with respect to s/(s+g)
	Here, b=T is the time when s/(s+g)>l; b [0,1]
	otherwise T=1 ;
	r here is the reproduce suppression [0,1)
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib as mpl
import pickle

from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import ticker, cm
from  matplotlib.colors import LogNorm
import matplotlib.colors as mpc
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd




def PlotSizeCostsHeatmap(Data, ColorMap, FileName2Write, Write2File):
#	cmap = plt.get_cmap('terrain')
	ColorMap.set_bad('white')
	Colobar_Properties = dict(ticks=[0.0, 0.25, 0.5, 0.75, 1])
	#ax2 = sns.heatmap(opt, vmin=0.005,vmax=max_legend, cmap = cmap, norm=mpc.LogNorm(), cbar_kws = Colobar_Properties)
	ax2 = sns.heatmap(Data, vmin=0.0,vmax=max_legend, cmap = ColorMap, cbar_kws = Colobar_Properties)
	
	ax2.set_aspect(1.0/ax2.get_data_ratio())
	ax2.invert_yaxis()
	
	ax2.set_ylabel('Germ-role cell differentiation \n cost, $c_g$' ,fontsize=15)
	ax2.set_xlabel(r'Maturity size, $2^n$',fontsize=15)
	
	# artifical x and y ticks	
	y_ticks1=[i*c_range[1] for i in [0,0.25,0.5,0.75,1.0]]
	y_ticks2=[str(i) for i in y_ticks1]
	
	
	x_tick_list=['$2^{'+str(i)+'}$' for i in [1,3,5,7,9,11,13,15]]
	ax2.set_xticks(np.asarray([0,2,4,6, 8, 10, 12, 14])+0.5)
	ax2.set_xticklabels(x_tick_list)
	ax2.set_yticks(np.asarray([0.0*size_num,.25*size_num,.5*size_num,.75*size_num,1.0*size_num])+0.5, minor=False)
	ax2.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(y_ticks2))
	ax2.tick_params(axis='y', rotation=0)

	ax2.patch.set_edgecolor('black')  
	ax2.patch.set_linewidth('2')  
	
	# We change the fontsize of minor ticks label 
	ax2.tick_params(axis='both', which='major', labelsize=12)
	if Write2File == 1:
		figure = ax2.get_figure()
		figure.savefig(FileName2Write ,bbox_inches='tight')   # save figures
	plt.show()
	return 0
#-------------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=300)

""" flags """
Write2File = 1
ReloadData = 1

#-------------------------------------------------------------------------------
"build para[b,c] grids"
num_lines=200

grid_num=21                                            # scatter points in x axes i.e. number of b between 0 to 10
division_times=16                                      # n value --division times

		#-------- initial conditions also for optimal_map
c_range=np.array([0,10])                          # range of log(a)
grid_c=np.linspace(c_range[0],c_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points

if ReloadData == 1:
	#---read tuple to get the specific T function in this file--------------------	
	DataLinesLocation = '../../03_Data/Parameters_of_200_lines/v37_200_v1.pickle'
	with open(DataLinesLocation, 'rb') as f:
	     line_data = pickle.load(f)
	
	line_data=np.array(line_data)                        # a,x0,b,x1,z
	
	#-------------------------------------------------------------------------------
	"read max dps from data into result []"
	
	result=[[] for i in range(1,division_times)]             # to store each figure's data
	
	optimal_matrix=[np.zeros(shape=(grid_num,division_times-1))*np.nan]
	
	for n in range(1,division_times):                        # how many figures or cell divisions
		for c_cluster in range(0,grid_num):                  # 1,---,30
			st_num=0
			
			for line_id in range(0,num_lines):            # 0,---,29
				DataLocation =  '../../03_Data/__Screen_costs_size_at_cs=2cg/%d_%d_%d.txt'%(n,c_cluster,line_id)
				result1=np.loadtxt(DataLocation)
	
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
#			ratio_st1=float("{0:.2f}".format(ratio_st0))*100
			ratio_st1=float("{0:.2f}".format(ratio_st0))
			optimal_matrix[0][c_cluster][n-1]=ratio_st1
	
	result5=[np.array(i) for i in result]
	result6=np.array(result5)                                        #  (15, 3000, 13) division_times, lines*cost_num, all parameters

##-------------------------------------------------------------------------------
max_legend=1.0
size_num=grid_num-1

Points = 400
OriginalCMap = plt.get_cmap('terrain')
Colors = OriginalCMap(np.linspace(0, 1, Points))
HighCutoff = 300
#LowCutoff = 50 	#v1
LowCutoff = 0
Colors = Colors[LowCutoff:HighCutoff]
ModifiedCMap = ListedColormap(Colors)
PalName = "linterrain_v2"

Data_ISD = pd.DataFrame(optimal_matrix[0])
Data_ISD[Data_ISD == 0] = np.nan

#Data_RSD = pd.DataFrame(cs_cg_matrix_RSD[0])
#Data_RSD[Data_RSD == 0] = np.nan
#
#Data_NSD = pd.DataFrame(cs_cg_matrix_NSD[0])
#Data_NSD[Data_NSD == 0] = np.nan



PlotSizeCostsHeatmap(Data_ISD, ModifiedCMap, 'heatmap_ncg_cs=2cg_v3_'+PalName+'.pdf', Write2File)
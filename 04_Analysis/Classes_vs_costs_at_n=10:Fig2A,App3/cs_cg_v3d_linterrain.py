#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:07:30 2019

@author: gao
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib as mpl
#from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mpc
import seaborn as sns
import pandas as pd

#-------------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=300)

""" flag """
Write2File = 1

#-------------------------------------------------------------------------------

#---------------------read data-------------------------------------------------------
'''read data '''

division_times=np.array([10])
num_division=len(division_times)

num_lines=200
grid_num=26

DataLocation = '../../03_Data/Screen_all_costs_at_n=10_with_200_profiles/n_cs_cg_grids26_lines200.npy'

read_data=np.load(DataLocation)
#				[n, cs_id, cg_id, line_id,label,st, g1, g2,g3---a, x0, b, x1, z]
#				 0,  1,    2,      3,      4,   5, 6 ,  7, 8,  9, 10, 11,12, 13				
# numpy.ndarray ---- size=(#division_time,grid_num*grid_num*num_lines,14)   here is 3*22050*14    	
	
#np.shape(read_data)
#type(read_data)

#c_id=13
#data=read_data[0]
#data1_id=np.where(data[:,1]==c_id)
#data1_data=data[data1_id]
#data2_id=np.where(data1_data[:,2]==c_id)
#data2_data=data1_data[data2_id]
#st_id=np.where(data2_data[:,4]==1)
#st_data=data2_data[st_id]
#st_line_id=st_data[:,3]

#np.array([ 16.,  20.,  31.,  50.,  52.,  63.,  69.,  80.,  90.,  96., 100., 116., 156., 176., 190.])

#---------------------find the matrix of cs vs cg-------------------------------------------------------
                                    # partition number of c
#c_tick=np.linspace(0.0, 10, num=grid_num, endpoint=True)   

       #-------- sys to local
#cs_range=np.array([0,10])                       # x1 range
#grid_cs=np.linspace(cs_range[0],cs_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points
#   
#cg_range=np.array([0,10])                       # x1 range
#grid_cg=np.linspace(cg_range[0],cg_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points
#grid_cg[13]

#  size grid_num * grid_num
cs_cg_matrix=[np.zeros(shape=(grid_num,grid_num))*np.nan for n in range(num_division)]
cs_cg_matrix_RSD=[np.zeros(shape=(grid_num,grid_num))*np.nan for n in range(num_division)]
cs_cg_matrix_NSD=[np.zeros(shape=(grid_num,grid_num))*np.nan for n in range(num_division)]
line_samp_matrix=np.zeros(shape=(grid_num,grid_num))*np.nan 



for n_th in range(num_division):                                 # how many figures or cell divisions
#	n=division_times[n_th]
	data_cur=read_data[n_th]
	
	for cs_n in range(0,grid_num):                               # how many figures or cell divisions
		
		cs_id=np.where(data_cur[:,1]==cs_n)
		data_cur1=data_cur[cs_id]
				
		for cg_n in range(0,grid_num):                           # data in a cs and cg grid

			cg_id=np.where(data_cur1[:,2]==cg_n)
			data_cur2=data_cur1[cg_id] 
			
			st_line=(data_cur2[:,4]==1).sum()                  # number of st lines out of the  num_lines
			NSD_line=(data_cur2[:,8]==1).sum()                 # number of NSD lines out of the  num_lines
			RSD_line=sum(np.logical_and((data_cur2[:,8]<1),(data_cur2[:,4]<1)))
			if st_line>0:
#				ratio=100.0*st_line/num_lines
				ratio=st_line/num_lines
			else:
				ratio=0
			
			cs_cg_matrix[n_th][cs_n][cg_n]=ratio
			cs_cg_matrix_NSD[n_th][cs_n][cg_n] = NSD_line/num_lines
			cs_cg_matrix_RSD[n_th][cs_n][cg_n] = RSD_line/num_lines
			if n_th==num_division-1:
				line_samp_matrix[cs_n][cg_n]=num_lines
				

max_legend=1.0
size_num=grid_num-1

def PlotGSCostsHeatmap(Data, ColorMap, FileName2Write, Write2File):

#	cmap = plt.get_cmap('terrain')
	ColorMap.set_bad('white')
	Colobar_Properties = dict(ticks=[0.0, 0.25, 0.5, 0.75, 1])
	#ax2 = sns.heatmap(opt, vmin=0.005,vmax=max_legend, cmap = cmap, norm=mpc.LogNorm(), cbar_kws = Colobar_Properties)
	ax2 = sns.heatmap(Data, vmin=0.0, vmax=max_legend, cmap = ColorMap, cbar_kws = Colobar_Properties)
	
#	bottom, top = ax2.get_ylim()
#	ax2.set_ylim(bottom + 0.5, top - 0.5)
	
	ax2.set_aspect(1.0/ax2.get_data_ratio())
	ax2.invert_yaxis()
	
	ax2.set_ylabel(r'Soma-role differentiation cost, $c_{s}$', fontsize=15)
	ax2.set_xlabel(r'Germ-role differentiation cost, $c_{g}$', fontsize=15)
	
	# artifical x and y ticks	
	ax2.set_xticks(np.asarray([0.0*size_num,.20*size_num,.40*size_num,.6*size_num,.8*size_num,1*size_num])+0.5, minor=False)
	ax2.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(['0','2', '4', '6', '8','10']))
	ax2.set_yticks(np.asarray([0.0*size_num,.20*size_num,.40*size_num,.6*size_num,.8*size_num,1*size_num])+0.5, minor=False)
	ax2.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(['0','2', '4', '6', '8','10']))
	ax2.tick_params(axis='y', rotation=0)
	
	ax2.patch.set_edgecolor('black')  
	ax2.patch.set_linewidth('2')  
	
	# We change the fontsize of minor ticks label 
	ax2.tick_params(axis='both', which='major', labelsize=12)
	if Write2File == 1:
		figure = ax2.get_figure()
		figure.savefig(FileName2Write ,bbox_inches='tight')   # save figures
	plt.show()

""" Colors palettes """

#""" nice rgb set """
##PalName = 'Pal1'
##CLR_ISD = '#05d282'
##CLR_RSD = '#d21b05'
##CLR_NSD = '#0556d2'
#
#""" original set """
#PalName = 'Pal2'
#CLR_ISD = "#1b9e77"
#CLR_RSD = "#d95f02"
#CLR_NSD = 'k'
#
#RGB_ISD = mpc.to_rgb(CLR_ISD)
#RGB_RSD = mpc.to_rgb(CLR_RSD)
#RGB_NSD = mpc.to_rgb(CLR_NSD)
#
#Points = 400
#
#List_ISD = np.ones((Points, 4))
#List_RSD = np.ones((Points, 4))
#List_NSD = np.ones((Points, 4))
#
#for i in [0,1,2]:
#	List_ISD[:, i] = np.linspace(1, RGB_ISD[i], Points)
#	List_RSD[:, i] = np.linspace(1, RGB_RSD[i], Points)
#	List_NSD[:, i] = np.linspace(1, RGB_NSD[i], Points)
#
#CMAP_ISD = ListedColormap(List_ISD)
#CMAP_RSD = ListedColormap(List_RSD)
#CMAP_NSD = ListedColormap(List_NSD)

#vals[:, 0] = np.linspace(90/256, 1, Points)
#vals[:, 1] = np.linspace(39/256, 1, Points)
#vals[:, 2] = np.linspace(41/256, 1, Points)
#newcmp = ListedColormap(vals)

Points = 400
OriginalCMap = plt.get_cmap('terrain')
Colors = OriginalCMap(np.linspace(0, 1, Points))
HighCutoff = 300
#LowCutoff = 50 	#v1
LowCutoff = 0
Colors = Colors[LowCutoff:HighCutoff]
ModifiedCMap = ListedColormap(Colors)
PalName = "linterrain_v2"

Data_ISD = pd.DataFrame(cs_cg_matrix[0])
Data_ISD[Data_ISD == 0] = np.nan

Data_RSD = pd.DataFrame(cs_cg_matrix_RSD[0])
Data_RSD[Data_RSD == 0] = np.nan

Data_NSD = pd.DataFrame(cs_cg_matrix_NSD[0])
Data_NSD[Data_NSD == 0] = np.nan


PlotGSCostsHeatmap(Data_ISD, ModifiedCMap, 'cs_cg_ISD_'+PalName+'.pdf', Write2File)
PlotGSCostsHeatmap(Data_RSD, ModifiedCMap, 'cs_cg_RSD_'+PalName+'.pdf', Write2File)
PlotGSCostsHeatmap(Data_NSD, ModifiedCMap, 'cs_cg_NSD_'+PalName+'.pdf', Write2File)

#import pandas as pd
#
#Costs = np.round(0.4*np.arange(26), decimals = 2)
#Data_ISD = pd.DataFrame(cs_cg_matrix[0], columns = Costs, index = Costs)
#Data_RSD = pd.DataFrame(cs_cg_matrix_RSD[0], columns = Costs, index = Costs)
#Data_NSD = pd.DataFrame(cs_cg_matrix_NSD[0], columns = Costs, index = Costs)
#
#Data_ISD.to_csv('../../fig/Table_ISD.txt')
#Data_RSD.to_csv('../../fig/Table_RSD.txt')
#Data_NSD.to_csv('../../fig/Table_NSD.txt')



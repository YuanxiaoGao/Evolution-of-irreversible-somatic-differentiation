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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import pickle
import matplotlib.colors as mpc
import seaborn as sns
import pandas as pd

#import os

#-------------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=300)

""" flag """
Write2File = 1
ReloadData = 1

#-------------------------------------------------------------------------------
"build para[b,c] grids"
num_lines=200

grid_num=51                                            # scatter points in x axes i.e. number of b between 0 to 10
division_times=16                                      # n value --division times

		#-------- initial conditions also for optimal_map
c_range=np.array([0,10])                          # range of log(a)
grid_c=np.linspace(c_range[0],c_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points


#---read tuple to get the specific T function in this file--------------------	
DataLinesLocation = '../../03_Data/Parameters_of_200_lines/v37_200_v1.pickle'
with open(DataLinesLocation, 'rb') as f:
     line_data = pickle.load(f)

line_data=np.array(line_data)                        # a,x0,b,x1,z


#-------------------------------------------------------------------------------
"read max dps from data into result []"

if ReloadData == 1:
	result=[[] for i in range(1,division_times)]             # to store each figure's data
	
	optimal_matrix=[np.zeros(shape=(grid_num,division_times-1))*np.nan]
	optimal_matrix_RSD=[np.zeros(shape=(grid_num,division_times-1))*np.nan]
	optimal_matrix_NSD=[np.zeros(shape=(grid_num,division_times-1))*np.nan]
	
	for n in range(1,division_times):                        # how many figures or cell divisions
		for c_cluster in range(0,grid_num):                  # 1,---,30
			st_num=0
			RSD_num = 0
			NSD_num = 0
			
			for line_id in range(0,num_lines):            # 0,---,29
				
				DataLocation = '../../03_Data/__Screen_costs_size_at_cs=cg/%d_%d_%d.txt'%(n,c_cluster,line_id)
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
				
				""" compute NSD and RSD counts """
				if result3[5]==1.0:
					NSD_num+=1
				
				if (label == 0) and (result3[5] < 1.0):
					RSD_num+=1
					
				#------ insert b and r grid number in the first two place for later drawing 		
	
				result4=np.array([n,c_cluster,line_id,label,st_value,result3[3],result3[4],result3[5],line_data[line_id][0],line_data[line_id][1],line_data[line_id][2],line_data[line_id][3] ,line_data[line_id][5] ])
	#				[n, cs_id= cg_id, line_id,label,st, g1, g2,g3---a, x0, b, x1, z]
	#				 0,  1,              2,      3,  4, 5,  6 , 7,  8,  9, 10, 11,12			
				
				result[n-1].append(result4)
			ratio_st0=st_num/num_lines
			ratio_st1=float("{0:.2f}".format(ratio_st0))*100
#			optimal_matrix[0][c_cluster][n-1]=ratio_st1
			optimal_matrix[0][c_cluster][n-1]=ratio_st0
			
			ratio_NSD=NSD_num/num_lines
			optimal_matrix_NSD[0][c_cluster][n-1]=ratio_NSD
			
			ratio_RSD=RSD_num/num_lines
			optimal_matrix_RSD[0][c_cluster][n-1]=ratio_RSD
			
	
	result5=[np.array(i) for i in result]
	result6=np.array(result5)                                        #  (15, 3000, 13) division_times, lines*cost_num, all parameters

##-------------------------------------------------------------------------------
#'''save the data_outcome'''
#
#np.save('/Users/gao/Desktop/deve/Simulation/v37-final-figure/v2_n_c_heatmaps/v2_n_c_line200-grids51/data_outcome/data_n_c_%s.npy'%num_lines,result6)


#-----------------blue--------------------------------------------------------------
"draw figures"
#import math

ratio_max=np.max(optimal_matrix)
#max_legend=math.ceil(ratio_max/10)*10	
max_legend=1.0

#---------- plot figure
fig = plt.figure(figsize=(6.1,6.1))
#st=fig.suptitle(r"$Ratio_{max}=%s$" %(ratio_max), fontsize=16,y=0.9)
      # -------draw axes
grid = AxesGrid(fig, 111,                  # similar to subplot(111)
				nrows_ncols=(1, 1), 	       # creates 2x2 grid of axes
				axes_pad=0.5,       	       # pad between axes in inch.
				direction='row',   	       # default : draw figure in row direction
				add_all='True',    	       # default: add axes to figures
				share_all=True,    	       # xaxis & yaxis of all axes are shared if True
				aspect='True',   	          # ration of xaxis and y axis : {'equal', 'auto'} or float
				label_mode="L",    	       #location of tick labels thaw will be displayed.
				#“1” (only the lower left axes), “L” (left most and bottom most axes), or “all”.

				cbar_mode="single",        #[None|single|each]
				cbar_location="right",
				cbar_pad=0.2,
				cbar_size='5%',            # default: size of the colorbar
				)
plt.rcParams["axes.grid"] = False          # remove white lines

#---- Use a bray areas for the nan -- (germ only goes to germ)
color0=['#a50026','#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4','#313695']
color1=color0[::-1]							

cmap = LinearSegmentedColormap.from_list('mycmap', color1) #'#2166ac','yellow','#b2182b'
		
#optimal_matrix[0]=np.ma.masked_where(optimal_matrix[0]==0,optimal_matrix[0])									
#cmap.set_bad('lightgray',1.)	                 # color for g3=1 i.e. s_t=nan												
#cmap.set_bad('black')	                          # color for g3=1 i.e. s_t=nan												

size_num=grid_num-1
#i_titlt=0
for opt, ax in zip(optimal_matrix,grid):

	# draw figure
	im = ax.imshow(opt, origin='lower',vmin=0,vmax=max_legend,cmap=cmap)
	ax.set_aspect(round((division_times-1)/grid_num,2))                           # set the figue shape into square or other by changing the value

	# set title
#		title=['s1','s2','s3','g1','g2','g3']
#		ax.set_title(title[i_titlt], fontsize=12)
#		i_titlt=i_titlt+1
#		title=[]

	# x and y label
	ax.set_xlabel('Maturity size (power index)',fontsize=20)
	ax.set_ylabel('Type switching cost',fontsize=20)

	# artifical x and y ticks
	y_ticks1=[i*c_range[1] for i in [0,0.25,0.5,0.75,1.0]]
	y_ticks2=[str(i) for i in y_ticks1]
	
	ax.set_xticks([i for i in range(0,division_times-1)], minor=False)
	ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(['%s'%i for i in range(1,division_times)]))
	ax.set_yticks([0.0*size_num,.25*size_num,.5*size_num,.75*size_num,1.0*size_num], minor=False)
	ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(y_ticks2))

	# We change the fontsize of minor ticks label 
	ax.tick_params(axis='both', which='major', labelsize=12)
	ax.tick_params(axis='both', which='minor', labelsize=12)
	
grid.cbar_axes[0].colorbar(im,cmap=cmap)#,  boundaries=None)
grid.cbar_axes[0].set_ylabel('Fraction of of ISD', rotation=90,fontsize=16, y=0.55, labelpad=8)

     # -- tick label size and tick length
font_size = 12                                          # Adjust as appropriate.
grid.cbar_axes[0].tick_params(labelsize=font_size,length=5,direction='in',pad=5)	

plt.show()
#if Write2File == 1:
#	fig.savefig('../fig/Fig2B_n_c51_v2.pdf' ,bbox_inches='tight')   # save figures

""" Plot maturity size vs switching costs heatmaps """

def PlotSizeCostsHeatmap(Data, ColorMap, FileName2Write, Write2File):
#	cmap = plt.get_cmap('terrain')
	ColorMap.set_bad('white')
	Colobar_Properties = dict(ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
	#ax2 = sns.heatmap(opt, vmin=0.005,vmax=max_legend, cmap = cmap, norm=mpc.LogNorm(), cbar_kws = Colobar_Properties)
	ax2 = sns.heatmap(Data, vmin=0.005,vmax=max_legend, cmap = ColorMap, cbar_kws = Colobar_Properties)
	
	ax2.set_aspect(1.0/ax2.get_data_ratio())
	ax2.invert_yaxis()
	
	ax2.set_ylabel('Cell differentiation cost, $c$' ,fontsize=15)
	ax2.set_xlabel(r'Maturity size, $2^n$',fontsize=15)
	
	# artifical x and y ticks	
#	y_ticks1=[i*c_range[1] for i in [0,0.25,0.5,0.75,1.0]]
#	y_ticks2=[str(i) for i in y_ticks1]
	
	ax2.patch.set_edgecolor('black')  
	ax2.patch.set_linewidth('2')  
	
	x_tick_list=['$2^{'+str(i)+'}$' for i in [1,3,5,7,9,11,13,15]]
	ax2.set_xticks(np.asarray([0,2,4,6, 8, 10, 12, 14])+0.5)
	ax2.set_xticklabels(x_tick_list)
	ax2.set_yticks(np.asarray([0, 10, 20, 30, 40, 50])+0.5)
	ax2.set_yticklabels(np.asarray([0, 2, 4, 6, 8, 10]))
#	ax2.set_yticks(np.asarray([0.0*size_num,.25*size_num,.5*size_num,.75*size_num,1.0*size_num])+0.5, minor=False)
#	ax2.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(y_ticks2))
	ax2.tick_params(axis='y', rotation=0)
	
	# We change the fontsize of minor ticks label 
	ax2.tick_params(axis='both', which='major', labelsize=12)
	if Write2File == 1:
		figure = ax2.get_figure()
		figure.savefig(FileName2Write ,bbox_inches='tight')   # save figures
	plt.show()
	return 0



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

Data_RSD = pd.DataFrame(optimal_matrix_RSD[0])
Data_RSD[Data_RSD == 0] = np.nan

Data_NSD = pd.DataFrame(optimal_matrix_NSD[0])
Data_NSD[Data_NSD == 0] = np.nan




PlotSizeCostsHeatmap(Data_ISD, ModifiedCMap, 'c_vs_n_ISD_v5_'+PalName+'.pdf', Write2File)
PlotSizeCostsHeatmap(Data_NSD, ModifiedCMap, 'c_vs_n_NSD_v5_'+PalName+'.pdf', Write2File)
PlotSizeCostsHeatmap(Data_RSD, ModifiedCMap, 'c_vs_n_RSD_v5_'+PalName+'.pdf', Write2File)



#cmap = plt.get_cmap('terrain')
#cmap.set_bad('black')
#Colobar_Properties = dict(ticks=[0.001, 0.01, 0.1, 1])
##ax2 = sns.heatmap(opt, vmin=0.005,vmax=max_legend, cmap = cmap, norm=mpc.LogNorm(), cbar_kws = Colobar_Properties)
#ax2 = sns.heatmap(optimal_matrix[0], vmin=0.005,vmax=max_legend, cmap = cmap, norm=mpc.LogNorm(), cbar_kws = Colobar_Properties)
#
#ax2.set_aspect(1.0/ax.get_data_ratio())
#ax2.invert_yaxis()
#
#ax2.set_ylabel('Type switching cost, $c_s = c_g$' ,fontsize=15)
#ax2.set_xlabel(r'Maturity size, $2^n$',fontsize=15)
#
## artifical x and y ticks	
#y_ticks1=[i*c_range[1] for i in [0,0.25,0.5,0.75,1.0]]
#y_ticks2=[str(i) for i in y_ticks1]
#
#
#x_tick_list=['$2^{'+str(i)+'}$' for i in [1,3,5,7,9,11,13,15]]
#ax2.set_xticks(np.asarray([0,2,4,6, 8, 10, 12, 14])+0.5)
#ax2.set_xticklabels(x_tick_list)
#ax2.set_yticks([0.0*size_num,.25*size_num,.5*size_num,.75*size_num,1.0*size_num], minor=False)
#ax2.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(y_ticks2))
#ax2.tick_params(axis='y', rotation=0)
#
## We change the fontsize of minor ticks label 
#ax2.tick_params(axis='both', which='major', labelsize=12)
#if Write2File == 1:
#	figure = ax2.get_figure()
#	figure.savefig('../fig/Fig2B_v3.pdf' ,bbox_inches='tight')   # save figures







#
##-----------------black--------------------------------------------------------------
#"draw figures"
#import math
#
#ratio_max=np.max(optimal_matrix)
##max_legend=math.ceil(ratio_max/10)*10	
#max_legend=25
#
##---------- plot figure
#fig = plt.figure(figsize=(6.1,6.1))
##st=fig.suptitle(r"$Ratio_{max}=%s$" %(ratio_max), fontsize=16,y=0.9)
#      # -------draw axes
#grid = AxesGrid(fig, 111,                  # similar to subplot(111)
#				nrows_ncols=(1, 1), 	       # creates 2x2 grid of axes
#				axes_pad=0.5,       	       # pad between axes in inch.
#				direction='row',   	       # default : draw figure in row direction
#				add_all='True',    	       # default: add axes to figures
#				share_all=True,    	       # xaxis & yaxis of all axes are shared if True
#				aspect='auto',   	          # ration of xaxis and y axis : {'equal', 'auto'} or float
#				label_mode="L",    	       #location of tick labels thaw will be displayed.
#				#“1” (only the lower left axes), “L” (left most and bottom most axes), or “all”.
#
#				cbar_mode="single",        #[None|single|each]
#				cbar_location="right",
#				cbar_pad=0.2,
#				cbar_size='5%',            # default: size of the colorbar
#				)
#plt.rcParams["axes.grid"] = False          # remove white lines
#
##---- Use a bray areas for the nan -- (germ only goes to germ)
#color0=['#a50026','#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4','#313695']
#color1=color0[::-1]							
#
#cmap = LinearSegmentedColormap.from_list('mycmap', color1) #'#2166ac','yellow','#b2182b'
#		
#optimal_matrix[0]=np.ma.masked_where(optimal_matrix[0]==0,optimal_matrix[0])									
##cmap.set_bad('lightgray',1.)	                 # color for g3=1 i.e. s_t=nan												
#cmap.set_bad('black')	                          # color for g3=1 i.e. s_t=nan												
#
#size_num=grid_num-1
##i_titlt=0
#for opt, ax in zip(optimal_matrix,grid):
#
#	# draw figure
#	im = ax.imshow(opt, origin='lower',vmin=0,vmax=max_legend,cmap=cmap)
#	ax.set_aspect(round((division_times-1)/grid_num,2))                           # set the figue shape into square or other by changing the value
#
#	# set title
##		title=['s1','s2','s3','g1','g2','g3']
##		ax.set_title(title[i_titlt], fontsize=12)
##		i_titlt=i_titlt+1
##		title=[]
#
#	# x and y label
#	ax.set_xlabel('Maturity size (power index)',fontsize=20)
#	ax.set_ylabel('Type switching cost',fontsize=20)
#
#	# artifical x and y ticks
#	y_ticks1=[i*c_range[1] for i in [0,0.25,0.5,0.75,1.0]]
#	y_ticks2=[str(i) for i in y_ticks1]
#	
#	ax.set_xticks([i for i in range(0,division_times-1)], minor=False)
#	ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(['%s'%i for i in range(1,division_times)]))
#	ax.set_yticks([0.0*size_num,.25*size_num,.5*size_num,.75*size_num,1.0*size_num], minor=False)
#	ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(y_ticks2))
#
#	# We change the fontsize of minor ticks label 
#	ax.tick_params(axis='both', which='major', labelsize=12)
#	ax.tick_params(axis='both', which='minor', labelsize=12)
#	
#grid.cbar_axes[0].colorbar(im,cmap=cmap)#,  boundaries=None)
#grid.cbar_axes[0].set_ylabel('Fraction of of ISD', rotation=90,fontsize=16, y=0.55, labelpad=8)
#
#     # -- tick label size and tick length
#font_size = 12                                          # Adjust as appropriate.
#grid.cbar_axes[0].tick_params(labelsize=font_size,length=5,direction='in',pad=5)	
#
#plt.show()
#fig.savefig('/Users/gao/Desktop/deve/Simulation/v37-final-figure/v2_n_c_heatmaps/v2_n_c_line200-grids51/fig/Fig2B_n_c51_black.pdf' ,bbox_inches='tight')   # save figures

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
from matplotlib import ticker, cm
from  matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mpc
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd



def PlotGSCostsHeatmap(Data, ColorMap, FileName2Write, Write2File):

#	cmap = plt.get_cmap('terrain')
	ColorMap.set_bad('white')
	Colobar_Properties = dict(ticks=[0.0, 0.25, 0.5, 0.75, 1])
	#ax2 = sns.heatmap(opt, vmin=0.005,vmax=max_legend, cmap = cmap, norm=mpc.LogNorm(), cbar_kws = Colobar_Properties)
	ax2 = sns.heatmap(Data, vmin=0.0,vmax=max_legend, cmap = ColorMap, cbar_kws = Colobar_Properties)
	
	ax2.set_aspect(1.0/ax2.get_data_ratio())
	ax2.invert_yaxis()

	
	ax2.patch.set_edgecolor('black')  
	ax2.patch.set_linewidth('2') 
	
	ax2.set_ylabel(r'Soma-role differentiation cost, $c_{s}$ ' ,fontsize=15)
	ax2.set_xlabel(r'Germ-role differentiation cost, $c_{g}$',fontsize=15)
	
	# artifical x and y ticks	
	ax2.set_xticks(np.asarray([0.0*size_num,.20*size_num,.40*size_num,.6*size_num,.8*size_num,1*size_num])+0.5, minor=False)
	ax2.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(['0','2', '4', '6', '8','10']))
	ax2.set_yticks(np.asarray([0.0*size_num,.20*size_num,.40*size_num,.6*size_num,.8*size_num,1*size_num])+0.5, minor=False)
	ax2.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(['0','2', '4', '6', '8','10']))
	ax2.tick_params(axis='y', rotation=0)
	
	# We change the fontsize of minor ticks label 
	ax2.tick_params(axis='both', which='major', labelsize=12)
	if Write2File == 1:
		figure = ax2.get_figure()
		figure.savefig(FileName2Write ,bbox_inches='tight')   # save figures
	plt.show()



#-------------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=300)

#-------------------------------------------------------------------------------
""" flag """
Write2File = 1
#---------------------read data-------------------------------------------------------
'''read data '''

division_times=np.array([5])
num_division=len(division_times)

num_lines=200
grid_num=26


DataLocation = '../../03_Data/Screen_all_costs_at_n=5_with_200_profiles/n5_cs_cg_grids26_lines200.npy'
read_data=np.load(DataLocation)
#				[n, cs_id, cg_id, line_id,label,st, g1, g2,g3---a, x0, b, x1, z]
#				 0,  1,    2,      3,      4,   5, 6 ,  7, 8,  9, 10, 11,12, 13				
# numpy.ndarray ---- size=(#division_time,grid_num*grid_num*num_lines,14)   here is 3*22050*14    	
	
#np.shape(read_data)
#type(read_data)

#---------------------find the matrix of cs vs cg-------------------------------------------------------
                                    # partition number of c
c_tick=np.linspace(0.0, 10, num=grid_num, endpoint=True)   

       #-------- sys to local
cs_range=np.array([0,10])                       # x1 range
grid_cs=np.linspace(cs_range[0],cs_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points
   
cg_range=np.array([0,10])                       # x1 range
grid_cg=np.linspace(cg_range[0],cg_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points
   

#  size grid_num * grid_num
cs_cg_matrix=[np.zeros(shape=(grid_num,grid_num))*np.nan for n in range(num_division)]
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

			if st_line>0:
#				ratio=100.0*st_line/num_lines
				ratio=st_line/num_lines
			else:
				ratio=0

			cs_cg_matrix[n_th][cs_n][cg_n]=ratio
			if n_th==num_division-1:
				line_samp_matrix[cs_n][cg_n]=num_lines
				
##-------------------------------------------------------------------------------
#'''save the data_outcome''' # cs_cg_matrix contains the % s-irr, which can later compare to the size effects!!!
#			
#np.save('/Users/gao/Desktop/deve/Simulation/v36_costs_exhaustive/1_small_sample/linear_cs_cg/data_outcome_analysis/matrix_val_comp.npy',cs_cg_matrix)
#		
#
##test=np.load('/Users/gao/Desktop/deve/Simulation/v36_costs_exhaustive/1_small_sample/linear_cs_cg/data_outcome_analysis/matrix_val_comp.npy')
#		
				
#---------- plot figure--------------------------------------------------------------
#---------- plot figure--------------------------------------------------------------
				

max_legend=1.0
size_num=grid_num-1
#
#for n_th in range(num_division):                                 # how many figures or cell divisions
##for n_th in range(2,3):                                 # how many figures or cell divisions
#
#	n=division_times[n_th]                              # which division times n 
#
#	matrix=cs_cg_matrix[n_th]                           # for cs cg grid to put the pertentage of s_irr
#	
#	# to make the matrix to be a set to fit the following plot		
#	final_matrix=[]   
#	final_matrix.append(matrix) 
#	
#	
#	fig = plt.figure(figsize=(6,6))
##	st=fig.suptitle(r'$n=$%s, max=%s'%(n,ratio_max[n_th]), fontsize=16,y=0.9)
#	      # -------draw axes
#	grid = AxesGrid(fig, 111,                  # similar to subplot(111)
#					nrows_ncols=(1, 1), 	       # creates 2x3 grid of axes
#					axes_pad=0.5,       	       # pad between axes in inch.
#					direction='row',   	       # default : draw figure in row direction
#					add_all='True',    	       # default: add axes to figures
#					share_all=True,    	       # xaxis & yaxis of all axes are shared if True
#					aspect='auto',   	          # ration of xaxis and y axis : {'equal', 'auto'} or float
#					label_mode="L",    	       #location of tick labels thaw will be displayed.
#					#“1” (only the lower left axes), “L” (left most and bottom most axes), or “all”.
#	
#					cbar_mode="None",        #[None|single|each]
#					cbar_location="right",
#					cbar_pad=0.2,
#					cbar_size='5%',            # default: size of the colorbar
#					)
#	plt.rcParams["axes.grid"] = False          # remove white lines
#	
#	#---- Use a boundary norm instead
#	
##	cmap = LinearSegmentedColormap.from_list('mycmap', color1) #'#2166ac','yellow','#b2182b'
#	cmap =cm.jet
#	cmap.set_bad('black')	                 # color for g3=1 i.e. s_t=nan												
#	
#	
#	size_num=grid_num-1
#	#i_titlt=0
#	for opt, ax in zip(final_matrix,grid):
#	
#		# draw figure
#		im = ax.imshow(opt,origin='lower',cmap=cmap,norm=LogNorm(vmin=1, vmax=100))#,norm=mpl.colors.LogNorm()) #,extent=[0,1,15,-1,15])
#		ax.set_aspect(1)                          # set the figue shape into square or other by changing the value
#		# x and y label
#		ax.set_ylabel(r'Soma-role differentiation cost $c_{s}$ ' ,fontsize=20)
#		ax.set_xlabel(r'Germ-role differentiation cost $c_{g}$',fontsize=20)
#		# artifical x and y ticks	
#		ax.set_xticks([0.0*size_num,.20*size_num,.40*size_num,.6*size_num,.8*size_num,1*size_num], minor=False)
#		ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(['0','2', '4', '6', '8','10']))
#		ax.set_yticks([0.0*size_num,.20*size_num,.40*size_num,.6*size_num,.8*size_num,1*size_num], minor=False)
#		ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(['0','2', '4', '6', '8','10']))
#		
##		# We change the fontsize of minor ticks label 
##		ax.tick_params(axis='both', which='major', labelsize=12)
##		ax.tick_params(axis='both', which='minor', labelsize=12)
##		cbaxes = fig.add_axes([0.93, 0.1, 0.03, 0.8]) 
#		divider = make_axes_locatable(ax)
#		cax = divider.append_axes("right", size="4%", pad=0.1)
#		cbar=plt.colorbar(im,cax = cax)
#		
#		cbar.ax.set_ylabel('Percentage of ISD', rotation=90,fontsize=16, y=0.55, labelpad=1)
#		font_size = 10 
#		cbar.ax.tick_params(labelsize=font_size,length=5,direction='in',pad=5)	
##	grid.cbar_axes[0].colorbar()
##	grid.cbar_axes[0].set_ylabel('Fraction of ISD', rotation=90,fontsize=16, y=0.55, labelpad=8)
##
##	
##	     # -- tick label size and tick length
##	font_size = 12                                          # Adjust as appropriate.
##	grid.cbar_axes[0].tick_params(labelsize=font_size,length=5,direction='in',pad=5)	
#	plt.show()
#	
##	fig.savefig('/Users/gao/Desktop/deve/Simulation/v37-final-figure/v3_n_cs_cg_heatmaps/comp_26_26_line200_n10/appendix_cs_cg_n5/fig/f_n5_cg_cs_=%s_jet.pdf'%n, bbox_inches='tight')   # save figures


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

#Data_RSD = pd.DataFrame(cs_cg_matrix_RSD[0])
#Data_RSD[Data_RSD == 0] = np.nan
#
#Data_NSD = pd.DataFrame(cs_cg_matrix_NSD[0])
#Data_NSD[Data_NSD == 0] = np.nan

PlotGSCostsHeatmap(Data_ISD, ModifiedCMap, 'heatmap_cscg_n=5_v3_'+PalName+'.pdf', Write2File)


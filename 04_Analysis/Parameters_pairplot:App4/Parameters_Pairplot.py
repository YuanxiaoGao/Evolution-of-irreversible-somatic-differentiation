#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:33:31 2019
@author: gao
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
#-------------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=300)

""" flag """
Write2File = 1

#---------------------read data-------------------------------------------------------
'''read data '''
division_times=16
num_lines=1000
grid_num=11

DataLocation = '../../03_Data/Screen_equal_costs_at_n=10_with_3000_profiles/data_n10_c_3000.npy'
read_data=np.load(DataLocation, allow_pickle=True)
#read_data=np.load('../data_n10_c_3000.npy', allow_pickle=True)

#				[n, cs_id= cg_id, line_id,label,st, g1, g2,g3---a, x0, b, x1, z]
#				 0,  1,              2,      3,  4, 5,  6 , 7,  8,  9, 10, 11,12			
# numpy.ndarray ---- size=(#division_time,grid_num*grid_num*num_lines,14)   here is (15, 3000, 13) 	

#---------------------which c data we choose, all or particuly-------------------------------------------------------
'''set the cs cg value''' # cs=15_id cg=5_id
#--choose a small grid-------------------------
c_range=np.array([0,10])                             # x1 range
grid_c=np.linspace(c_range[0],c_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points
  
c_id=5    # totally 26
#cost1=float("{0:.2f}".format(grid_c[c_id]))
n_id_list=[14]  
#n_id_list=[7]  
n_sample_num=len(n_id_list)                                            # = division times 

#-------------------------------------------------------------------------------------
#------collect the mean and variance value---------------------------
#irre_line_list=[]                                       # irre lines of the n, c_id, label,st, x0,a,b,x1
#run_num=0                                                # only run once foe sample
#for n_id in n_id_list:
#
#	desire_data=read_data[n_id-1]                        # find n_id division times column
#	c_grid_id=np.where(desire_data[:,1]==c_id)           # find c_id cost row
#	desire_data1=desire_data[c_grid_id]                  # 100*13
#	 	
#	desire_data2=desire_data1[:,[0,1,3,4,9,8,10,11]]     # rearrange the data in: n, c_id, label,st, x0,a,b,x1	
#	
#	# find non-stlines 
#	line_non=np.where(desire_data2[:,2]!=1.0)
#	line=desire_data2[line_non]
#	line_non_data=line[:,[4,5,6,7]]
#	line_non_data[:,1]=0.25*(line_non_data[:,1]+2) 	             # normalise alpha values
#	
#	# ----find out the lines leading s_irre
#	line_irre_id=np.where(desire_data2[:,2]==1.0)
#	line_irre=desire_data2[line_irre_id]                 # rearrange the data in: n, c_id, label=1,st, x0,a,b,x1	
#	
#	line_irre_4para=line_irre[:,[4,5,6,7]]
#	line_irre_4para[:,1]=0.25*(line_irre_4para[:,1]+2) 	     # normalise alpha values
#	num_line=np.shape(line_irre_4para)[0]
#	
#	if num_line>0:
#		irre_line_list.append(line_irre_4para)
#
##-----------check if all sample have irreversible lines 
#num_samp=len(irre_line_list) 
#if num_samp!=n_sample_num:
#	print("Some n sample don't have any irrreversible lines!!!!!!")
#
#np.shape(irre_line_list[0])
#type(irre_line_list)	
#len(irre_line_list)	





""" alternative processing, YP """
""" data extraction """
MatSize = 10
dfData = pd.DataFrame(read_data[MatSize-1])
##	 column names are based on the comment above:
##				[n, cs_id= cg_id, line_id,label,st, g1, g2,g3---a, x0, b, x1, z]
##				 0,  1,              2,      3,  4, 5,  6 , 7,  8,  9, 10, 11,12
dfData.columns = ['Maturity.size', 'Differentiation.cost', 'Line.ID', 'Label', 'Prob.Soma2Soma',
			   'g_ss', 'g_gs', 'g_gg', 'log10(alpha)', 'x_0', 'b', 'x_1', 'Size.costs']

dfData = dfData[dfData['Differentiation.cost']==5.0]
dfData = dfData.reset_index(drop = True)

dfData['Class'] = 'none'

SetClass = 'C_ISD'
for i in np.arange(len(dfData)):
	if (dfData['Prob.Soma2Soma'][i]==1.0):
		dfData.loc[i, 'Class'] = SetClass

SetClass = 'B_RSD'
for i in np.arange(len(dfData)):
	if (dfData['Prob.Soma2Soma'][i]<1.0)and(dfData['g_gg'][i]<1.0):
		dfData.loc[i, 'Class'] = SetClass

SetClass = 'A_NSD'
for i in np.arange(len(dfData)):
	if (dfData['g_gg'][i]==1.0):
		dfData.loc[i, 'Class'] = SetClass

PlotData = dfData[['log10(alpha)', 'x_0', 'b', 'x_1', 'Class']]
PlotData = PlotData.sort_values(by=['Class'])


CLR_ISD = "#1b9e77"
CLR_RSD = "#d95f02"
CLR_NSD = 'k'
newPal   = dict(B_RSD = CLR_RSD, C_ISD = CLR_ISD, A_NSD = CLR_NSD)
Fig = sns.pairplot(PlotData, hue = 'Class', markers = ['.', 'o', 'o'], palette = newPal, plot_kws=dict(alpha = 0.7))

if Write2File == 1:
	Fig.savefig('Parameters_full_pairplot_v2.pdf',bbox_inches='tight')   # save figures






#""" Linear discriminant analysis to distinguish the three classes """
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#
#lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))




		
#ISD_Data = dfData[dfData['Prob.Soma2Soma']==1.0]

#""" plot the figure """
#fig, ax = plt.subplots(1,1)
#
#CLR_ISD = '#2c7fb8'
#size_bkg=5
#size_focal=15
#
#plt.scatter(dfData['x_1'], dfData['alpha'], s=size_bkg, c='k', alpha=0.1,label="sth")
#plt.scatter(ISD_Data['x_1'], ISD_Data['alpha'], s=size_focal, c=CLR_ISD, alpha=1,label="sth")
#
#ax.set_yticks([-2, -1, 0, 1, 2])
#ax.set_yticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$'])
#
#ax.set_aspect(1.0/ax.get_data_ratio())
#
#plt.xlabel("Saturation threshold, "+r'$x_1$', fontsize=15)
#plt.ylabel("Contribution pattern, "+r"$\alpha$", fontsize=15)
#plt.show()
#
#if Write2File == 1:
#	fig.savefig('../../4para_correlation/Fig_alpha_vs_x1_updated.pdf',bbox_inches='tight')   # save figures


#
#""" data for all three classes """
#RSD_Data = dfData[(dfData['Prob.Soma2Soma']<1.0)&(dfData['g_gg']<1.0)]
#NSD_Data = dfData[dfData['g_gg']==1.0]
#""" plot the figure """
#fig, ax = plt.subplots(1,1)
#
#CLR_ISD = "#1b9e77"
#CLR_RSD = "#d95f02"
#size_bkg=5
#size_focal=15
#
##ParY = 'alpha'
##ParX = 'x_1'
#
#ParY = 'b'
#ParX = 'x_0'
#
#plt.scatter(NSD_Data[ParX], NSD_Data[ParY], s=size_bkg, c='k', alpha=0.1,label="sth")
##plt.scatter(RSD_Data[ParX], RSD_Data[ParY], s=size_focal, c=CLR_RSD, alpha=0.7,label="sth")
#plt.scatter(RSD_Data[ParX], RSD_Data[ParY], s=size_bkg, c='k', alpha=0.1,label="sth")
#plt.scatter(ISD_Data[ParX], ISD_Data[ParY], s=size_focal, c=CLR_ISD, alpha=0.7,label="sth")
#
#if ParY == 'alpha':
#	ax.set_yticks([-2, -1, 0, 1, 2])
#	ax.set_yticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$'])
#	Ylabel = "Contribution pattern, "+r"$\alpha$"
#elif ParY == 'b':
#	Ylabel = "Maximal benefit, "+r"$b$"
#elif ParY == 'x_0':
#	Ylabel = "Contribution threshold, "+r'$x_0$'
#elif ParY == 'x_1':
#	Ylabel = "Saturation threshold, "+r'$x_1$'
#
#if ParX == 'alpha':
#	ax.set_xticks([-2, -1, 0, 1, 2])
#	ax.set_xticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$'])
#	Xlabel = "Contribution pattern, "+r"$\alpha$"
#elif ParX == 'b':
#	Xlabel = "Maximal benefit, "+r"$b$"
#elif ParX == 'x_0':
#	Xlabel = "Contribution threshold, "+r'$x_0$'
#elif ParX == 'x_1':
#	Xlabel = "Saturation threshold, "+r'$x_1$'
#	
#ax.set_aspect(1.0/ax.get_data_ratio())
#
#plt.xlabel(Xlabel, fontsize=15)
#plt.ylabel(Ylabel, fontsize=15)
#plt.show()
#
#if Write2File == 1:
#	fig.savefig('../Correlation_ISD_'+str(ParX)+'_vs_'+str(ParY)+'.pdf',bbox_inches='tight')   # save figures
#
#
#
#










## ---------------draw figures --------------------------------------------
#fig = plt.figure(figsize=(13,6))
#st=fig.suptitle(r" $n=14$, $cost=%s $"%(int(cost1)),fontsize=16,y=0.9)
#
#ax = plt.subplot(111)
#ax.set_aspect(4.5) 	
#
##--- plot scatter ------------------	 
#x=np.linspace(0.3,10.3,num=4,endpoint=True)	                # for the first n 
#
#num_samp=len(irre_line_list) 
#alpha_value=0.9
#size_value=50
#
##---------plot dots
#line_current=irre_line_list[0]                         # to find the current n 
#num_point=np.shape(line_current)[0]                     # number of lines under this n
#
#	#----assort lines 
##line_current=line_current[line_current[:,2].argsort()]
#
#colors = cm.nipy_spectral(np.linspace(0, 1, num_point))
#step_gap=0.025
#x_final=[x+j*step_gap for j in range(num_point)]
#
#for x_final, y_final, col in zip(list(x_final),list(line_current), colors):	
#	plt.scatter(x_final, y_final, color=col,s=size_value,  alpha=alpha_value)   #vmin=0, vmax=1, cmap=cm,
#				 
##remove ticks and top and right frames
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#
##------Decorate the spins--------
#arrow_length = 750 # In points
#
#	# X-axis arrow
#ax.annotate('', xy=(0.0, 0.0), xycoords=('axes fraction', 'data'), 
#            xytext=(arrow_length, 0), textcoords='offset points',
#            arrowprops=dict(arrowstyle='<|-', fc='black'))
#
#	# Y-axis arrow
#ax.annotate('', xy=(-0.05, -0.02), xycoords=('data', 'axes fraction'), 
#            xytext=(0, 280), textcoords='offset points',
#            arrowprops=dict(arrowstyle='<|-', fc='black'))
#
##------x and y labels and limits--------
#plt.xlim(-0.05,13)
#plt.ylim(0.0,1.05)
#
## artifical x and y ticks
#x_ticks1=x+step_gap*( num_point-1)/2
#x_tick_label=["Timing","Pattern","Degree","Saturation"]
#
#ax.set_xticks([i for i in x_ticks1], minor=False)
#ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(x_tick_label))
#ax.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="k")
#ax.tick_params(axis="y", labelsize=12, labelrotation=0, labelcolor="k")
#		
#plt.show()		
#fig.savefig('/Users/gao/Desktop/deve/Simulation/v37-final-figure/v2_n_c_heatmaps/4para_correlation/Fig3_4parasn14_c5_4para.pdf',bbox_inches='tight')   # save figures

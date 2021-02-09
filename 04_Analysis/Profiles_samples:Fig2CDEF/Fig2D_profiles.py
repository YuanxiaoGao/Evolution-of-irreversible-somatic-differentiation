#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:33:31 2019

@author: gao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:21:48 2019

@author: gao
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statistics
from mpl_toolkits.axes_grid1 import make_axes_locatable

#--------------T function-----------------------------------
def Time(a,x0,b,x1,x):
	if x<=x0:
		t=1
	elif x>x0 and x<=x1:
		t=1-b+b*((x1-x)/(x1-x0))**a
	elif x>x1:
		t=1-b
	return t

def ExtractData(data0, cg_id, cs_id):
	"""
	Extract subsets of lines with RSD/ISD/NSD strategies 
	corresponding to (cg_id, cs_id) node
	"""
	c_grid_id=np.where(data0[:,2]==cg_id)           # find n_id cost row
	desire_data1=data0[c_grid_id]                  # 200*13
	
	cs_grid_id=np.where(desire_data1[:,1]==cs_id)           # find n_id cost row
	desire_data20=desire_data1[cs_grid_id]                  # 200*13
	
	#np.shape(desire_data1)
	desire_data2=desire_data20[:,[0,1,4,5,10,9,11,12]]     # rearrange the data in: n, c_id, label, st, x0,a,b,x1	
	
	#---------------------figuresss------------------------------------------------------	
#	cg_cost=grid_c[cg_id]	
#	cs_cost=grid_c[cs_id]	
		
	data=desire_data2
	
#	st_line=(data[:,2] ==1 ).sum()                       # number of st lines
	#	s_ratio=int(round(100.0*st_line/num_lines,1))        # ratio of st
	#	s_ratio_list.append(s_ratio)
	
	st_id=np.where(data[:,2] ==1)[0]                     # collecnt st line data
	NSD_id = np.where(desire_data20[:,8]==1)
	RSD_id = np.where(np.logical_and((desire_data20[:,8]<1),(desire_data20[:,4]<1)))
	
	ISD_data=data[st_id]
	RSD_data=data[RSD_id]
	NSD_data=data[NSD_id]
	return [ISD_data, RSD_data, NSD_data, data]


#-------------------------------------------------------------------------------

Write2File = 1

# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=300)

#---------------------read data-------------------------------------------------------
'''read data '''
division_times=16

num_lines=200
grid_num=26

DataLocation = '../../03_Data/Screen_all_costs_at_n=10_with_200_profiles/n_cs_cg_grids26_lines200.npy'

read_data=np.load(DataLocation)
#read_data=np.load('../n_cs_cg_grids26_lines200.npy')


#				[n, cs_id, cg_id, line_id,label,st, g1, g2,g3---a, x0, b, x1, z]
#				 0,  1,    2,      3,      4,   5, 6 ,  7, 8,  9, 10, 11,12, 13				
# numpy.ndarray ---- size=(#division_time,grid_num*grid_num*num_lines,14)   here is (15, 3000, 13) 	

data0=read_data[0]



#--choose a small grid-------------------------

c_range=np.array([0,10])                             # x1 range
grid_c=np.linspace(c_range[0],c_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points
grid_c[3]




""" Templates for different panels: """


""" Panel 1. RSD prevalence """
cg_id,cs_id = 1, 1          # c_g = 0.4,  c_s = 0.4
cg_value = str(np.round(0.4*cg_id, decimals = 1))
cs_value = str(np.round(0.4*cs_id, decimals = 1))
FigureName = 'Fig_2_profiles_cg'+cg_value+'_cs'+cs_value+'.pdf'

""" Panel 2. ISD prevalence """
cg_id,cs_id = 3, 15          # c_g = 1.2, c_s = 6.0
cg_value = str(np.round(0.4*cg_id, decimals = 1))
cs_value = str(np.round(0.4*cs_id, decimals = 1))
FigureName = 'Fig_2_profiles_cg'+cg_value+'_cs'+cs_value+'.pdf'

""" Panel 3. NSD prevalence """
cg_id,cs_id = 22, 3          # c_g = 6.0,  c_s = 1.2
cg_value = str(np.round(0.4*cg_id, decimals = 1))
cs_value = str(np.round(0.4*cs_id, decimals = 1))
FigureName = 'Fig_2_profiles_cg'+cg_value+'_cs'+cs_value+'.pdf'

##""" Panel 3. NSD prevalence - 2 """
##cg_id,cs_id = 15, 3          # c_g = 6.0,  c_s = 1.2
##cg_value = str(np.round(0.4*cg_id, decimals = 1))
##cs_value = str(np.round(0.4*cs_id, decimals = 1))
##FigureName = '../../fig/Fig_2_profiles_cg'+cg_value+'_cs'+cs_value+'.pdf'

""" Panel 4. Some equal costs """
cg_id,cs_id = 12, 12          # c_g = 6.0,  c_s = 1.2
cg_value = str(np.round(0.4*cg_id, decimals = 1))
cs_value = str(np.round(0.4*cs_id, decimals = 1))
FigureName = 'Fig_2_profiles_cg'+cg_value+'_cs'+cs_value+'.pdf'






print(grid_c[cg_id],grid_c[cs_id])
#cs_id=5          # c_s=1.2

##c_id_list=[29] 
#n_id=14                                             # = division times 
#desire_data=read_data[n_id-1]                        # find n_id division times column

#c_grid_id=np.where(data0[:,2]==cg_id)           # find n_id cost row
#desire_data1=data0[c_grid_id]                  # 200*13
#
#cs_grid_id=np.where(desire_data1[:,1]==cs_id)           # find n_id cost row
#desire_data20=desire_data1[cs_grid_id]                  # 200*13
#
##np.shape(desire_data1)
#desire_data2=desire_data20[:,[0,1,4,5,10,9,11,12]]     # rearrange the data in: n, c_id, label,st, x0,a,b,x1	
#
##---------------------figuresss------------------------------------------------------	
#cg_cost=grid_c[cg_id]	
#cs_cost=grid_c[cs_id]	
#	
#data=desire_data2
#
#st_line=(data[:,2] ==1 ).sum()                       # number of st lines
##	s_ratio=int(round(100.0*st_line/num_lines,1))        # ratio of st
##	s_ratio_list.append(s_ratio)
#
#st_id=np.where(data[:,2] ==1)[0]                     # collecnt st line data
#st_data=data[st_id]


st_data, RSD_data, NSD_data, data = ExtractData(data0, cg_id, cs_id)

	#---------------------figuresss------------------------------------------------------	

#c_id_list1=[26] 
##c_id_list=[29] 
#n_id1=8                                             # = division times 
#desire_data=read_data[n_id1-1]                        # find n_id division times column
#
#for c_id1 in c_id_list1:
#
#	c_grid_id1=np.where(desire_data[:,1]==c_id1)           # find n_id cost row
#	desire_data2=desire_data[c_grid_id1]                  # 200*13
#	np.shape(desire_data2)
#	desire_data2=desire_data2[:,[0,1,3,4,9,8,10,11]]     # rearrange the data in: n, c_id, label,st, x0,a,b,x1	
#
#	#---------------------figuresss------------------------------------------------------	
#	cost1=grid_c[c_id1]		
#	data1=desire_data2
#	
#	st_line1=(data[:,2] ==1 ).sum()                       # number of st lines
##	s_ratio=int(round(100.0*st_line/num_lines,1))        # ratio of st
##	s_ratio_list.append(s_ratio)
#	
#	st_id1=np.where(data1[:,2] ==1)[0]                     # collecnt st line data
#	st_data1=data[st_id1]
		
# ---------------draw figures --------------------------------------------
fig = plt.figure(figsize=(6,6))
		
ax = plt.subplot(111,aspect='equal')

col=["#5e4fa2",'#3288bd','#abdda4','#ff7f00','#e41a1c']	 
	 
x = np.linspace(0, 1, 100)
#for line in data:
##		n, c_id, label,st, x0,a,b,x1
#	time=[]
#	for item in x:
#		time.append(Time(np.power(10,line[5]),line[4],line[6],line[7],item))
#	ax.plot(x,time,color="k",alpha=0.15)

for line in NSD_data:
#		n, c_id, label,st, x0,a,b,x1
	time=[]
	for item in x:
		time.append(Time(np.power(10,line[5]),line[4],line[6],line[7],item))
	ax.plot(x,time,color="k" , linewidth=2,alpha=0.15)

for line in RSD_data:
#		n, c_id, label,st, x0,a,b,x1
	time=[]
	for item in x:
		time.append(Time(np.power(10,line[5]),line[4],line[6],line[7],item))
	ax.plot(x,time,color="#d95f02" , linewidth=2, alpha=0.4)

for line in st_data:
#		n, c_id, label,st, x0,a,b,x1
	time=[]
	for item in x:
		time.append(Time(np.power(10,line[5]),line[4],line[6],line[7],item))
	ax.plot(x,time,color="#1b9e77", linewidth=2, alpha=0.4)  #  fec44f   9ecae1
	

#	#------remove ticks and top and right frames
#	ax.spines['top'].set_visible(False)
#	ax.spines['right'].set_visible(False)
#	
#	#------Decorate the spins--------
#	arrow_length = 350 # In points
#	
#		# X-axis arrow
#	ax.annotate('', xy=(-0.05, -0.055), xycoords=('axes fraction', 'data'), 
#	            xytext=(arrow_length, 0), textcoords='offset points',
#	            arrowprops=dict(arrowstyle='<|-', fc='black'))
#	
#		# Y-axis arrow
#	ax.annotate('', xy=(-0.046, -0.049), xycoords=('data', 'axes fraction'), 
#	            xytext=(0, arrow_length), textcoords='offset points',
#	            arrowprops=dict(arrowstyle='<|-', fc='black'))
#	
#------x and y labels and limits--------
plt.xlim(-0.05,1.07)
plt.ylim(-0.05,1.07)

FTS = 20
#	plt.xlabel('Fraction of soma-role cells, $x$',fontsize=FTS)
#	plt.ylabel(r'Celll doubling time, $F_{comp}$',fontsize=FTS)

plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='both', which='both', length=0)
		
plt.show()
if Write2File == 1:
	fig.savefig(FigureName,bbox_inches='tight')   # save figures		
#fig.savefig('/Users/gao/Desktop/deve/Simulation/v37-final-figure/v3_n_cs_cg_heatmaps/comp_26_26_line200_n10/fig_v7/Fig2A_CG3_CS17_zoom-in2.pdf',bbox_inches='tight')   # save figures

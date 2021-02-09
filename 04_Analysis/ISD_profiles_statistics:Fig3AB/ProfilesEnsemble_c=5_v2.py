#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:33:31 2019

@author: gao
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import pickle
#-------------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=300)

""" flag """
Write2File = 1


#---------------------read data-------------------------------------------------------
'''read data '''
#division_times=16

num_lines=1000
grid_num=11

DataLocation = '../../03_Data/Screen_maturity_sizes_at_c=5_with_1000_profiles/data_n_c5_1000.npy'
read_data=np.load(DataLocation, allow_pickle=True)
#read_data=np.load('../../data_n_c5_1000.npy', allow_pickle=True)

#				[n, cs_id= cg_id, line_id,label,st, g1, g2,g3---a, x0, b, x1, z]
#				 0,  1,              2,      3,  4, 5,  6 , 7,  8,  9, 10, 11,12			
# numpy.ndarray ---- size=(#division_time,grid_num*grid_num*num_lines,14)   here is (1, 3000, 13) 	
'''cost=5; but n from 5 to 15'''
#np.shape(desire_data)
#read_data[10]
#--------------T function-----------------------------------
def Time(a,x0,b,x1,x):
	if x<=x0:
		t=1
	elif x>x0 and x<=x1:
		t=1-b+b*((x1-x)/(x1-x0))**a
	elif x>x1:
		t=1-b
	return t

#---------------------read data-------------------------------------------------------

'read the raw 1000 lines'

ParametersLocation = '../../03_Data/Parameters_of_1000_lines/v37_1000_v1.pickle'
with open(ParametersLocation, 'rb') as f:
	line_data = pickle.load(f)
	
line_data=np.array(line_data)                                         # a,x0,b,x1,cs,cg,z
#line_data[:,0]=np.power(10,line_data[:,0])                            # change the longa into a 

line_data1=line_data[:,[1,0,2,3]]                                   # get the x0,a,b,x1
np.shape(line_data1)
line_data1[:,1]=0.25*(line_data1[:,1]+2) 	                          # normalise alpha values

np.shape(line_data1)
line_data1[:,1]


#---------------------calculate the confedente area-------------------------------------------------------
'''functuon'''  

z=[0.05,0.5,0.95]                                                                 # 95%
def confident(raw_data,z):                                            # by default the raw_data is a np.array
	min_ele=np.quantile(raw_data,z[0])
	mid_ele=np.quantile(raw_data,z[1])
	max_ele=np.quantile(raw_data,z[2])

	return np.array([min_ele,mid_ele,max_ele])

'calculte each features confidence interval'

confidence_list=[]
for i in range(4):
	foc_data=line_data1[:,i]
	mean_inte=confident(foc_data,z)
	confidence_list.append(mean_inte)



#-------------choose a small grid-----------------------------------
c_range=np.array([0,10])                             # x1 range
grid_c=np.linspace(c_range[0],c_range[1],num=grid_num,endpoint=True) # split log(a) into grid_num points
  
c_id=5          # cost is fixed is 5
#n_id_list=[6]                                              # = division times 
n_id_list=[7,15]                                              # = division times 

graident_n=[]                                           # y_all, each y in n_list
confi_n=[]                                              # y_all_confi, each y in n_list
#np.shape(confi_n[1])
#confi_n[0]

s_ratio_list=[]

run_num=0                                            # only run once foe sample
for n_id in n_id_list:

	desire_data=read_data[n_id-1]                        # find n_id division times column
	c_grid_id=np.where(desire_data[:,1]==c_id)           # find c_id cost row
	desire_data1=desire_data[c_grid_id]                  # 100*13
	 	
	desire_data2=desire_data1[:,[0,1,3,4,9,8,10,11]]     # rearrange the data in: n, c_id, label,st, x0,a,b,x1	
	
	#---------------------figuresss------------------------------------------------------	
	cost=grid_c[c_id]		
	data=desire_data2
	
	st_line=(data[:,2] ==1 ).sum()                       # number of st lines
	s_ratio=int(round(100.0*st_line/num_lines,1))        # ratio of st
	s_ratio_list.append(s_ratio)
	
	st_id=np.where(data[:,2] ==1)[0]                     # collecnt st line data
	st_data=data[st_id]
	
	x = np.linspace(0, 1, 100)
	
	# collect the mean and variance of all-lines
	# only run once
	if run_num==0:
		
		y_all=[]
		y_confi=[]
		for x_value in x:
			y_fixed=[]                            # a squenece of vertical y values
			for line in data:
				y_fixed.append(Time(np.power(10,line[5]),line[4],line[6],line[7],x_value))  
			y_mean=np.mean(y_fixed) 
			if len(y_fixed)==1:
				y_var=0.0
			else:
				y_var=np.std(y_fixed)
			
			y_all.append(np.array([y_mean,y_var]))
			low_mid_high=confident(y_fixed,z)
			y_confi.append(low_mid_high)            # min med max
		y_all=np.array(y_all)
		y_confi=np.array(y_confi)
		graident_n.append(y_all)
		confi_n.append(y_confi)
	
	if st_line==0:
		graident_n.append(np.zeros(shape=(num_lines,2))*np.nan)
	
	else:                # st_line!=0:
		# collect the mean and variance of st-lines
		y=[]                                       # contains 100 sublist
		y_confi_1=[]
                                            
		for x_value in x:
			y_fixed=[]
			for line in st_data:
				y_fixed.append(Time(np.power(10,line[5]),line[4],line[6],line[7],x_value))  
			y_mean=np.mean(y_fixed) 
			if len(y_fixed)==1:
				y_var=0.0
			else:
				y_var=np.std(y_fixed)
			
			y.append(np.array([y_mean,y_var]))
			low_mid_high=confident(y_fixed,z)
			y_confi_1.append(low_mid_high)            # min med max

		y=np.array(y)
		y_confi_1=np.array(y_confi_1)
		graident_n.append(y)
		confi_n.append(y_confi_1)
		
	run_num=run_num+1

# ---------------draw figures --------------------------------------------
fig = plt.figure(figsize=(5,5))
#st=fig.suptitle(r" $cost=%s $"%("{:.2f}".format(cost)),fontsize=18,y=0.95)
		
ax = plt.subplot(111,aspect='equal')
#col1=['#d53e4f','#fdae61','#e6f598','#66c2a5','#5e4fa2']
#col=col1[::-1]

#col=['#377eb8','#e41a1c']
col=['#24d09d','#126c51']
alpha_value=0.3 
linewidth_value=3

'draw the mean and std lines'
#for i in range(len(graident_n)):
#	y=graident_n[i]
#	
#	# draw mean and var fig for two samples
#	if i==0:
#		ax.plot(x,y[:,0],linewidth=linewidth_value,c='k',label="Samples")
#		ax.fill_between(x,y[:,0]-y[:,1],y[:,0]+y[:,1],
#					 facecolor='k', alpha=alpha_value)
#	else:
##		ax.plot(x,y[:,0],linewidth=2,c=col[i-1],label=r"$n=%s$,$S_{irr}=%s \% %$"%(n_id_list[i-1],s_ratio_list[i-1]))
#		ax.plot(x,y[:,0],linewidth=linewidth_value,c=col[i-1],label=r"$n=%s$"%(n_id_list[i-1]))
#		ax.fill_between(x,y[:,0]-y[:,1],y[:,0]+y[:,1],
#					 facecolor=col[i-1], alpha=alpha_value)
'draw the cofident lines'
for i in range(len(confi_n)):
#for i in range(2):

	y=confi_n[i]
#	print(y)
	
	# draw mean and var fig for two samples
	if i==0:
		ax.plot(x,y[:,1],linewidth=linewidth_value,c='grey',label="All profiles")
		ax.fill_between(x,y[:,0],y[:,2],
					 facecolor='k', alpha=0.2)
	else:
#		ax.plot(x,y[:,0],linewidth=2,c=col[i-1],label=r"$n=%s$,$S_{irr}=%s \% %$"%(n_id_list[i-1],s_ratio_list[i-1]))
		ax.plot(x,y[:,1],linewidth=linewidth_value,c=col[i-1],label=r"ISD ($n=%s$)"%(n_id_list[i-1]))
		ax.fill_between(x,y[:,0],y[:,2],
					 facecolor=col[i-1], alpha=alpha_value)
	
	ax.legend(frameon=False,loc=1,prop={'size': 12})
	#remove ticks and top and right frames
#	ax.spines['top'].set_visible(False)
#	ax.spines['right'].set_visible(False)

	#------x and y labels and limits--------
	plt.xlim(-0.05,1.07)
	plt.ylim(-0.05,1.07)
	
	LabelFont = 15
	
	plt.xlabel(r'Fraction of soma-role cells, $x$',fontsize=LabelFont)
	plt.ylabel(r'Cell doubling time effect, $F_{comp}$',fontsize=LabelFont)
	
	ax.tick_params(axis='both', which='major', labelsize=12)
#	ax.tick_params(axis='both', which='minor', labelsize=8)
	#plt.title("Contribution mode",fontsize=14)
		
plt.show()
if Write2File == 1:
	fig.savefig('confi_Fig2c_cost-fixed%s_1000_s_v3.pdf'%(grid_c[c_id]),bbox_inches='tight')   # save figures


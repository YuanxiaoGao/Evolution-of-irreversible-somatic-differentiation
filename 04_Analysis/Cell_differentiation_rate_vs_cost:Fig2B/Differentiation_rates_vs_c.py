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
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import statistics
import pickle

def PlotQuantileProfile(X_array, Upper, Lower, Middle, CLR, Alpha):
	DeltaX = (X_array[1]-X_array[0])/2.0
	for i in np.arange(len(X_array)):
		LocalX = [X_array[i]-DeltaX, X_array[i]+DeltaX]
		plt.fill_between(LocalX, Upper[i], Lower[i], facecolor = CLR, alpha = Alpha, edgecolor = 'None')
		plt.plot(LocalX, [Middle[i], Middle[i]], linewidth = 3, color = CLR)


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

"""				[n, cs_id= cg_id, line_id,label,st, g1, g2,g3---a, x0, b, x1, z]  """
"""				 14,  1,              2,      3,  4, 5,  6 , 7,  8,  9, 10, 11,12 """
""" numpy.ndarray ---- size=(#division_time,grid_num*grid_num*num_lines,14)   here is (15, 3000, 13) here only [13] exist, which have the size 11000,13	"""


DataLocation = '../../03_Data/Screen_equal_costs_at_n=10_with_3000_profiles/data_n10_c_3000.npy'

read_data=np.load(DataLocation, allow_pickle=True)
m_size = 10
Data_1 = read_data[m_size-1]  


RSD_diffrate = []
RSD_low = []
RSD_high = []
ISD_diffrate = []
ISD_low = []
ISD_high = []
NSD_diffrate = []

""" loop over costs """
c_id_list=[1,2,3,4,5,6,7,8,9,10]
for c_id in c_id_list:
	c_id=np.where(Data_1[:,1]==c_id)
	Data_2=Data_1[c_id]

	""" select different classes of developmental strategies """
	ISD_inds = np.where(Data_2[:,3]==1)
	RSD_inds = np.where(np.logical_and((Data_2[:,7]<1),(Data_2[:,3]<1)))
	NSD_inds = np.where(Data_2[:,7]==1)

	ISD_data = Data_2[ISD_inds]
	RSD_data = Data_2[RSD_inds]
	NSD_data = Data_2[NSD_inds]
	
	ISD_diffrate.append( np.median(1-ISD_data[:,4] + ISD_data[:,5] + 1.0/2.0*ISD_data[:,6]) )
	RSD_diffrate.append( np.median(1-RSD_data[:,4] + RSD_data[:,5] + 1.0/2.0*RSD_data[:,6]) )
#	RSD_low.append( np.min(1-RSD_data[:,4] + RSD_data[:,5] + 1.0/2.0*RSD_data[:,6]) )
#	RSD_high.append( np.max(1-RSD_data[:,4] + RSD_data[:,5] + 1.0/2.0*RSD_data[:,6]) )
	RSD_low.append( np.quantile(1-RSD_data[:,4] + RSD_data[:,5] + 1.0/2.0*RSD_data[:,6], 0.05) )
	RSD_high.append( np.quantile(1-RSD_data[:,4] + RSD_data[:,5] + 1.0/2.0*RSD_data[:,6], 0.95) )
	NSD_diffrate.append( np.median(NSD_data[:,5] + 1.0/2.0*NSD_data[:,6]) )

	ISD_low.append( np.quantile(1-ISD_data[:,4] + ISD_data[:,5] + 1.0/2.0*ISD_data[:,6], 0.05) )
	ISD_high.append( np.quantile(1-ISD_data[:,4] + ISD_data[:,5] + 1.0/2.0*ISD_data[:,6], 0.95) )


#	print(np.mean(RSD_diffrate))



LabelsFontSize = 17
TicksFontSize = 12

plt.style.use('default')
fig, ax = plt.subplots(1,1)


#X_array = np.linspace(0, 10, num=11)
X_array = c_id_list
CLR_NSD = "k"
CLR_RSD = "#d95f02"
CLR_ISD = "#1b9e77"

#plt.plot(X_array, ISD_diffrate, color = CLR_ISD, linewidth = 3)

#PlotQuantileProfile(X_array, ISD_diffrate, ISD_diffrate, ISD_diffrate, CLR_ISD, 0.3)
PlotQuantileProfile(X_array, RSD_high, RSD_low, RSD_diffrate, CLR_RSD, 0.3)
PlotQuantileProfile(X_array, NSD_diffrate, NSD_diffrate, NSD_diffrate, CLR_NSD, 0.3)
PlotQuantileProfile(X_array, ISD_high, ISD_low, ISD_diffrate, CLR_ISD, 0.3)

#plt.plot(X_array, RSD_diffrate, color = CLR_RSD, linewidth = 3)
#plt.fill_between(X_array, RSD_low, RSD_high, color = CLR_RSD, alpha = 0.3 )
#plt.plot(X_array, NSD_diffrate, color = CLR_NSD, linewidth = 3)

#plt.ylim(-0.1, 1.1)
#plt.xlim(1,10)
ax.set_aspect(1.0/ax.get_data_ratio())

plt.xticks(np.arange(1, 11), fontsize = TicksFontSize)
plt.yticks(fontsize = TicksFontSize)

plt.xlabel('Cell differentiation cost, '+r'$c_s = c_g$', fontsize = LabelsFontSize)
plt.ylabel('Cell differentiation rate,'+'\n'+r'$g_{ss}+\frac{1}{2} g_{gs}+s_{gg}+\frac{1}{2} s_{gs}$', fontsize = LabelsFontSize)

#plt.legend(["ISD", "RSD", "NSD"], fontsize = 12)
#leg = ax.get_legend()
#leg.legendHandles[0].set_color(CLR_ISD)
#leg.legendHandles[1].set_color(CLR_RSD)
#leg.legendHandles[2].set_color(CLR_NSD)


if Write2File == 1:
	plt.savefig('Dif_Rates_no_leg_v2.png', dpi=300, bbox_inches='tight')
plt.show()
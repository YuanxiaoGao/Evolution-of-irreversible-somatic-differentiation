#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 11:00:47 2018

@author: gao
"""


#-*-encoding:utf-8 -*-
################################################################
# 2018-06-25												   #
################################################################

""" code description:
Aim: Implement cell division.
In each division step, cells divide into germ or soma cells depending on the six probabilities.
At a given composition, the time to division is calculated at a given function.
Here, our aim is to obtain
	(1) time duration to division
	(2) next cell composition
at a given current composition.


Parameters and Variables:
	Para: parameters for the gworth time function --- to be determined
		[...] (np array: float elements)

	DP: six developement probabilities for cells
		[S->S+S, S->G+S, S->G+G, G->S+S, G->G+S, G->G+G] (np array: float elements)

	Comp:
		[# of Soma cells, # of Germ cells] (np array: int elements)

	T: time to division (float)
 ---------------- """
import numpy as np

'''arguments
    m: running times for developmental trajectory (int)
    n: division times (int)
    para: np.array, [0]: benfits->b [1]:c is the costs for soma (np.array)(float,float[0,1])
    left,right: initial root interval (folat)
'''
'''golbal variables'''


#------------------------------------------------------------------------------------------------------------
'''Division return next cell composition and the costs of task switching;
here we set the g_cost=1 and the s_cost/g_cost =c=para[4]
'''
def Division(Comp,DP,para): 
	
	Nsoma=np.array([2,1,0],dtype=int)                    # soma->2somas; soma->1germ and 1soma; soma->2germs
	Ngerm=np.array([0,1,2],dtype=int)
	
	Ms=np.random.multinomial(Comp[0],DP[0])              # cost [0,c, 2c]
#	print(Ms,Ms)
	Mg=np.random.multinomial(Comp[1],DP[1])              # cost [2c, c, 0]
	s=sum((Ms+Mg)*Nsoma)
	g=sum((Ms+Mg)*Ngerm)
	
	cost_s=np.array([0,1,2])
	cost_g=np.array([2,1,0])
	
	cost=(sum(Ms*cost_s)*para[4]+sum(Mg*cost_g)*para[4])/sum(Comp)
	
	return [[s,g],cost]

#------------------------------------------------------------------------------------------------------------
'''GrowthTime from the cell composition ; T_cost is in the Division() function in the last step'''
''' T function is a exponential function T=e**-a(x-b); a,x0,b,x1,cs,cg,z
	And we assume null time is 1.
    para[0]=a: exponent;
	para[1]=x0; upper threshold; 
	para[2]=b; minimum time;
	para[3]=x1; bottom threshold;
	para[4]=c; [0,10]
	

	para[6]=z; size exponent;
'''

def GrowthTime(Comp,para,n):                             # calculate time of each division

	x=Comp[0]/(Comp[0]+Comp[1])                          # soma cell fraction

	if x<=para[1]:                                       # lower threshold b
		T_onestep=1.0
		
	elif x>para[1] and x<=para[3]:
		T_onestep=(1-para[2])+para[2]*((para[3]-x)/(para[3]-para[1]))**para[0]
		
	elif x>para[3]:
		T_onestep=1-para[2]

	return T_onestep

#------------------------------------------------------------------------------------------------------------
'''Trajectory: return the final cell coposition and whole time after a random trajectory'''
def Trajectory(n,DP,para):                               # one trajectory; return composition and time
	
	Comp=np.array([0,1],dtype=int)
	T=np.int(0)
	for i in range(n):
		t=GrowthTime(Comp,para,n)
		com_cos=Division(Comp,DP,para)
		Next_comp=com_cos[0]
		T_cost=com_cos[1]
		T=T+t*(1+T_cost)                          # two parts of time: cell composition and cost
		Comp=Next_comp
		
	return [Comp,T]

#------------------------------------------------------------------------------------------------------------
'''Growth rate'''
def Growth(m,n,DP,para,left,right):
	T_data=list()
	for runs in range(m):
		T_table=Trajectory(n,DP,para)
		T_data.append(np.array([T_table[0][1],T_table[1]])) 
	return FindRoot(T_data,left,right,m)

#------------------------------------------------------------------------------------------------------------
'''FindRoot return the roots in a given interval'''
def FindRoot(T_data,left,right,m):                         # binary methid finding the roots
    initial_left=Expression(left,T_data,m)
    initial_right=Expression(right,T_data,m)
    if abs(initial_left)<10**(-8):                         # left boundary point is root
        return left
    elif abs(initial_right)<10**(-8):				       # right boundary point is root
        return right
    elif np.sign(initial_left)*np.sign(initial_right)>0:   # no roots in the initial interval
        return np.nan
    else:
        left_sign=np.sign(initial_left)
        while abs(left-right)>10**(-14):     # find the root by loop

            mean=(left+right)/2
            mean_sign=np.sign(Expression(mean,T_data,m))
            if (left_sign*mean_sign)>0:
                left=mean
                left_sign=mean_sign
            else:
                right=mean
        return left

#------------------------------------------------------------------------------------------------------------
'''equation'''
def Expression(x,T_data,m):
    exponential=0.0
    for runs in range(m):
        exponential+=(T_data[runs][0])*np.exp(-(x)*(T_data[runs][1]))
    return (1/m)*exponential-1

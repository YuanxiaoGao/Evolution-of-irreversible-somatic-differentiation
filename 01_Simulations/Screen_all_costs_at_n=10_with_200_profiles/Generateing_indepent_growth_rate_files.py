#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:52:22 2018

@author: gao
"""
#------- make sure grid_num hve the same value in all .py files for one version

import numpy as np

division_times=np.array([10])
num_lines=200
grid_num=26
#num_para=4            # a,x0,b,x1,z


fp = open("run.sh", "w")
for n in division_times:  
#	print(n)
	for cs_cluster in range(0,grid_num):  
		for cg_cluster in range(0,grid_num):  
			for line_id in range(0,num_lines):                    

				jobname = "Growth_rate_at_cg_cs_%d_%d_%d_%d" % (n,cs_cluster,cg_cluster,line_id)
				fname = "script/Growth_rate_at_cg_cs_%d_%d_%d_%d.sh" % (n,cs_cluster,cg_cluster,line_id)
				fp.write("sbatch %s\n" % fname)
				bashfp = open(fname, "w")
		
				bashfp.write("#!/bin/sh\n")
				bashfp.write("#SBATCH --time=00-01:00:00\n")
				bashfp.write("#SBATCH --job-name %s\n" % jobname)
				bashfp.write("#SBATCH -o out/%s\n" % jobname)
				bashfp.write("#SBATCH -e err/%s\n" % jobname)
				bashfp.write("python Growth_rate_at_cg_cs.py  %d %d %d %d\n" % (n,cs_cluster,cg_cluster,line_id) )
		

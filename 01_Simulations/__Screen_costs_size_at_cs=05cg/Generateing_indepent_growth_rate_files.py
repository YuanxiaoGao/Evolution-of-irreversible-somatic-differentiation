#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:52:22 2018

@author: gao
"""

num_lines=200
grid_num=51		 	       # how many pixels grids (grid_num-1) in the final B-C map
division_times=16

fp = open("run.sh", "w")
for n in range(1,division_times):                           # how many figures or cell divisions
	for c_cluster in range(0,grid_num):                 # how many grids in each figures --grid_num**2
		for line_id in range(0,num_lines):
			jobname = "Growth_rate_at_cg_cs_%d_%d_%d" % (n,c_cluster,line_id)
			fname = "script/Growth_rate_at_cg_cs_%d_%d_%d.sh" % (n,c_cluster,line_id)
			fp.write("sbatch %s\n" % fname)
			bashfp = open(fname, "w")


			bashfp.write("#!/bin/sh\n")
			bashfp.write("#SBATCH --time=00-03:00:00\n")
			bashfp.write("#SBATCH --job-name %s\n" % jobname)
			bashfp.write("#SBATCH -o out/%s\n" % jobname)
			bashfp.write("#SBATCH -e err/%s\n" % jobname)
			bashfp.write("python Growth_rate_at_cg_cs.py  %d %d %d\n" % (n,c_cluster,line_id) )





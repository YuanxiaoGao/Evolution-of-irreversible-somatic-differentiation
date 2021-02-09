#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:52:22 2018

@author: gao
"""

#------- make sure grid_num hve the same value in all .py files for one version
num_lines=1000
division_times=16

fp = open("run.sh", "w")
for n in range(5,division_times):                           # how many figures or cell divisions
	for c_cluster in range(5,6):                 # how many grids in each figures --grid_num**2
		for line_id in range(0,num_lines):
			jobname = "Growth_rate_at_cg_cs_%d_%d_%d" % (n,c_cluster,line_id)
			fname = "script/Growth_rate_at_cg_cs_%d_%d_%d.sh" % (n,c_cluster,line_id)
			fp.write("sbatch %s\n" % fname)
			bashfp = open(fname, "w")


			bashfp.write("#!/bin/sh\n")
			bashfp.write("#SBATCH --time=00-01:00:00\n")
			bashfp.write("#SBATCH --job-name %s\n" % jobname)
			bashfp.write("#SBATCH -o out/%s\n" % jobname)
			bashfp.write("#SBATCH -e err/%s\n" % jobname)
			bashfp.write("python Growth_rate_at_cg_cs.py  %d %d %d\n" % (n,c_cluster,line_id) )



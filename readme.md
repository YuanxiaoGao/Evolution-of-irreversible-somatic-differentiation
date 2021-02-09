This repository contains code and data used in the manuscript "Evolution of irreversible somatic differentiation" by Yuanxiao Gao, Hye Jin Park, Arne Traulsen, and Yuriy Pichugin. You are welcome to have a look at our preprint on the bioRxiv: https://www.biorxiv.org/content/10.1101/2021.01.18.427219v1

The content of repository:

\01_Simulations -  the code used to run simulations. 

Generateing_indepent_growth_rate_files.py (and similar file names) generates slurm script that calls the simulations.
Growth_rate_at_cg_cs.py (and similar file names) contains the protocal of simulation
Growth_rate.py contains the numerical model itself

Warning: These calculations will run for several days on a cluster with about 1000 nodes. They are not for a personal machine.

\02_Data_conversion - contains the archived raw data, which is about 50'000 small files for each dataset. This folder also contain the code converting these file collections into a more accessible format.

\03_Data - contains converted datasets used to produce figures in the manuscript.
Note 1: Archives starting with double underline __ contain raw data - about 50'000 of small files. Unzip to use.
Note 2: Datasets in folders \Parameters_of_X_lines are computed directly by the code in \01_Simulations and do not need processing.

\04_Analysis - code used to produce figures in the manuscript. 
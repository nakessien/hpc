#!/bin/bash 
module load python/3.6.3-intel2018u3 
cd test_gpu
bash ../analysis_scripts/collect_data.sh > data_gpu.tsv 
cd ..
cd test_cpu 
bash ../analysis_scripts/collect_data.sh > data_cpu.tsv 
cd ..
python analysis_scripts/merge.py 

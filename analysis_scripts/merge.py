# coding: utf-8
import pandas as pd
import numpy as np

# reading results
data_cpu = pd.read_csv('test_cpu/data_cpu.tsv', sep = '\t')#.drop(labels='JobID',axis='columns')
data_gpu = pd.read_csv('test_gpu/data_gpu.tsv', sep = '\t')#.drop(labels='JobID',axis='columns')

# converting real times to timedelta datatype
data_gpu.RealTime = pd.to_timedelta(data_gpu.RealTime)
data_cpu.RealTime = pd.to_timedelta(data_cpu.RealTime)

# adding time batch per epoch to dataframes
data_cpu['TimePerBatchPerEpoch(ms)'] = data_cpu.RealTime / (data_cpu.Epochs *data_cpu.TrSize / data_cpu.BtchSize)/np.timedelta64(1,'ms')
data_gpu['TimePerBatchPerEpoch(ms)'] = data_gpu.RealTime / (data_gpu.Epochs *data_gpu.TrSize / data_gpu.BtchSize)/np.timedelta64(1,'ms')

# changing index for join
data_cpu = data_cpu.set_index(['TrSize','Epochs','BtchSize'])
data_gpu = data_gpu.set_index(['TrSize','Epochs','BtchSize'])

# join
data_all = data_gpu.join(data_cpu,lsuffix='gpu',rsuffix='cpu')

# saving
data_all.to_csv('data_all.tsv',sep='\t')

#!/usr/bin/env python3.6
import pandas as pd
import numpy as np
from sys import argv,stdout

size = int(argv[1])

maximum = 200
minimum = 0

window = 15
saturation = 5

s = np.random.normal(
    loc = (maximum+minimum)/2,
    scale = saturation*np.sqrt(window)*(maximum-minimum)/2,
    size = size
    )

s = pd.Series(s).rolling(window=window).mean()
s = s.loc[~pd.isna(s)]

s = s.clip(minimum,maximum)

df = pd.DataFrame(columns=['values'])

df.values = s

df.to_csv(stdout)



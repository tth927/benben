#%%
import pandas as pd
import numpy as np

df = pd.read_csv('C:\MyWorkspace\python_pj\DataCamp_02_CookieCat\datasets\cookie_cats.csv', 
    nrows=5)
print('original')
print(df)

print('sample')
# replace=True, will duplicate the same sample, else jz reshuffle
bs_sample = df.sample(frac=1, replace=True)
print(bs_sample)
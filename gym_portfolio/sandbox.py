import pandas as pd 
import numpy as np

df = pd.read_pickle('/home/ernestas/Desktop/stuff/py/cbaf/awesome.pkl')
print(df['action'][0])
print(df['reward'][0])
print(df.loc[df['reward'].idxmax()])
print(df.loc[df['reward'].idxmin()])


# arr = []
# for s in range(len(df['state_features'][0])):
#   arr.append(df['state_features'][0][s])
# arr = np.array(arr)


import numpy as np
import pandas as pd

from functools import reduce
from tqdm import tqdm
import qlib
from qlib.data import D
import copy
import datetime as dt
qlib.init()

print('Reading...')
label_names = ['LABEL0']
data = pd.read_pickle('./data/day_csi300_till_20201228.pkl').loc(axis=0)['2007-01-01':,:]
ret = data[label_names]
stocks = ret.reset_index().groupby('instrument')['datetime'].agg(['min', 'max'])
print(stocks)
print('Read over')
N = 15 # 5 min
assert 240 % N == 0 
assert N != 240
num_bar  =  240 //N
print(f'num_bar {num_bar}')
res = dict()
for code, (start_time, end_time) in tqdm(stocks.iterrows()):
    # print(code)
    df = D.features([code], ['$high', '$low', '$close', '$open','$volume','$money'],
                    start_time='%s 09:31:00'%str(start_time)[:10],
                    end_time='%s 15:00:00'%str(end_time)[:10],
                    freq='1min', disk_cache=0)
    if df.empty:
        print('WARN: %s not found'%code)
        continue
    df = df.loc[code]
    if N > 1:
        # df = df.groupby(df.index.ceil('%smin'%N)).agg(
        #     {'$high': 'max', '$low': 'min', '$close': 'last', '$open':'first','$volume': 'sum','$money':'sum'})
        if N == 30:
            index_arr = np.array([])
            for day in np.unique(df.index.date):
                index_arr = np.append(index_arr, \
                np.array([dt.datetime(day.year, day.month, day.day, 10, 30, 0, 0)] * N + \
                [dt.datetime(day.year, day.month, day.day, 11, 30, 0, 0)] * N + \
                [dt.datetime(day.year, day.month, day.day, 14, 0, 0, 0)] * N + \
                [dt.datetime(day.year, day.month, day.day, 15, 0, 0, 0)] * N))
            df_index = pd.to_datetime(index_arr)
            df = df.groupby(df_index).agg(
            {'$high': 'max', '$low': 'min', '$close': 'last', '$open':'first','$volume': 'sum','$money':'sum'})
        else:
            df = df.groupby(df.index.ceil('%smin'%N)).agg(
                {'$high': 'max', '$low': 'min', '$close': 'last', '$open':'first','$volume': 'sum','$money':'sum'})
    
    X = df.values.reshape(-1, num_bar, len(df.columns)) # [*,*,h/l/c/o/vol/vwap]
    close = copy.deepcopy(X[:, -1:, 2]) # last close
    volume = np.mean(X[:, :, 4], axis=1, keepdims=True) # mean volume
    for i in range(4):
        X[:, :, i] = copy.deepcopy(X[:, :, i] / close)
    X[:, :, 5] = copy.deepcopy(X[:, :, 5] / (X[:, :, 4] + 1e-12)) # vwap = money / volume
    X[:, :, 5] = copy.deepcopy(X[:, :, 5] / close) # vwap norm by last close

    X[:, :, 4] = copy.deepcopy(X[:, :, 4] / (volume + 1e-12)) #volume norm by mean volume
    
    new_df = pd.DataFrame(X.reshape(-1, num_bar * len(df.columns)),
                          index=pd.to_datetime(df.index[::num_bar].date))
    columns = ['$high', '$low', '$close', '$open','$volume','$vwap']
    new_df.columns = reduce(lambda a, b: a + b, [
        ['%s_%d'%(s[1:],d) for s in columns]
        for d in range(1, num_bar+1)
    ])
    new_df = new_df[new_df.columns.values.reshape(-1, len(columns)).T.flatten().tolist()]
    res[code] = copy.deepcopy(new_df)

df = pd.concat(res, axis=0)
df.index.names = ['instrument', 'datetime']
df.index = df.index.swaplevel() #['instrument', 'datetime'] -> ['datetime', 'instrument' ] 
df = df.reindex(ret.index)

# df['LABEL0'] = ret
ret = ret.reindex(df.index)
for item in label_names:
    df[item] = ret[item].values

df = df.sort_index()
print(df)
df.to_pickle('../data/hft_%dm_csi300.pkl'%N)

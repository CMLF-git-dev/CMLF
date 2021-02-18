import numpy as np
import pandas as pd

import qlib
qlib.init()
from qlib.data import D

def load_dataset(market='csi300'):

    # features
    fields = []
    names = []

    fields += ['$open/$close']  # NOTE: Ref($open, 0) != $open
    fields += ['Ref($open, %d)/$close' % d for d in range(1, 60)]
    names  += ['OPEN%d'%d for d in range(60)]

    fields += ['$high/$close']
    fields += ['Ref($high, %d)/$close' % d for d in range(1, 60)]
    names  += ['HIGH%d'%d for d in range(60)]

    fields += ['$low/$close']
    fields += ['Ref($low, %d)/$close' % d for d in range(1, 60)]
    names  += ['LOW%d'%d for d in range(60)]

    fields += ['$close/$close']  # 1
    fields += ['Ref($close, %d)/$close' % d for d in range(1, 60)]
    names  += ['CLOSE%d'%d for d in range(60)]

    fields += ['$vwap/$close']
    fields += ['Ref($vwap, %d)/$close' % d for d in range(1, 60)]
    names  += ['VWAP%d'%d for d in range(60)]

    # fields += ['Log($volume/$volume)']  # 1
    # fields += ['Log(Ref($volume, %d)/$volume)' % d for d in range(1, 60)]
    # names  += ['VOLUME%d'%d for d in range(60)]

    fields += ['$volume/$volume']  # 1
    fields += ['Ref($volume, %d)/$volume' % d for d in range(1, 60)]
    names  += ['VOLUME%d'%d for d in range(60)]

    # labels
    labels = ['Ref($vwap, -2)/Ref($vwap, -1)-1']
    label_names = ['LABEL0']

    ## load features
    print('loading features...')
    df = D.features(D.instruments(market), fields, start_time='2007-01-01')
    df.columns = names
    print('load features over')
    ## load labels
    if len(labels):
        print('loading labels...')
        df_labels = D.features(D.instruments('all'), labels, start_time='2007-01-01')
        df_labels.columns = label_names
        df[label_names] = df_labels
        print('load labels over')

    return df, names, label_names

if __name__ == '__main__':
    print('loading...')
    df, _, _ = load_dataset('csi300')
    print('loading over')
    df.index = df.index.swaplevel()
    df = df.sort_index()
    
    df.to_pickle('./data/day_csi300.pkl')

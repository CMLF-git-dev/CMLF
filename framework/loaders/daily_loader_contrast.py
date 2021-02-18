import os
import numpy as np
import pandas as pd
import gc 
from common.utils import pprint, robust_zscore, download_http_resource
from loaders.dataset_contrast import Dataset
from tqdm import tqdm
import copy
from sacred import Ingredient

DATA_SERVER = os.environ.get('DATA_SERVER', 'http://10.150.144.154:10020')


data_ingredient = Ingredient('daily_loader_contrast')

EPS = 1e-12

@data_ingredient.config
def data_config():
    dset = 'day_csi800' # dataset
    label_id = 0 # LABEL$i
    log_transform = False

    train_start_date = '2008-01-01'
    train_end_date = '2014-12-31'
    valid_start_date = '2015-01-01'
    valid_end_date = '2016-12-31'
    test_start_date = '2017-01-01'
    test_end_date = '2019-06-18'

    batch_size = 300
    daily_batch = True
    pre_n_day = 1
    train_shuffle = True
    DATA_PATH = None

class DataLoader(object):
    """Data Loader

    Parameters
    ----------
    dset : str
        dataset filename, will be load from DATA_PATH/{dset}.pkl
    label_id : int
        label index
    batch_size : int
        batch size for NN model
    daily_batch : bool, optional
        whether use daily batch, i.e., ensure batch data from the same day
    log_transform : bool, optional
        whether use `log1p(abs(x))*sign(x)` transform to ensure norm-like distribution
    negative_sample: int, optional
    """

    @data_ingredient.capture
    def __init__(self, dset, label_id, batch_size,
                 daily_batch=True, log_transform=False, pre_n_day=10, train_shuffle=True, DATA_PATH=None, negative_sample=5):
        self.dset_daily = dset[0]
        self.dset_highfreq = dset[1]
        self.label_id = label_id
        self.batch_size = batch_size
        self.daily_batch = daily_batch
        self.log_transform = log_transform
        self.train_shuffle = train_shuffle
        self.pre_n_day = pre_n_day
        self.DATA_PATH = DATA_PATH
        self.negative_sample = negative_sample
        self._raw_df_daily = self._init_data(self.dset_daily)
        self._raw_df_highfreq = self._init_data(self.dset_highfreq)

    def _init_data(self, dset):
        fname = os.path.join(self.DATA_PATH, dset+'.pkl')
        if not os.path.exists(fname):
            try:
                download_http_resource(DATA_SERVER+'/'+dset+'.pkl', fname)
            except:
                pprint('ERROR: cannot find dataset `%s`'%dset)
                raise

        # load daily data
        df = pd.read_pickle(fname)
        print(f'DATA_PATH : {fname}')
        # NOTE: ensure the multiindex like <datetime, instrument>
        if df.index.names[0] == 'instrument':
            df.index = df.index.swaplevel()
            df = df.sort_index()

        df['LABEL'] = df['LABEL%s'%self.label_id].groupby(level='datetime').apply(robust_zscore)
        df['LABEL_cls'] = df['LABEL%s' % self.label_id].map(lambda x: 0 if x < -0.0005 else 2 if x > 0.0005 else 1)
        # drop 'money' and 'pre_close' cols in hft_m_csi_800
        if dset[:3] != 'day':  # high frequency
            if df.columns.str.contains('money|pre_close').any():
                mask = ~df.columns.str.contains('money|pre_close')
                
                df = df.iloc[:,mask]
            if self.pre_n_day > 1:
                pprint('processing previous n day...')
                mask = ~df.columns.str.contains('LABEL')
                feature = df.iloc[:,mask]
                label = df['LABEL']
                del df
                gc.collect()
                ins_box = []
                for ins in tqdm(feature.index.get_level_values('instrument').unique()):
                    _df = feature.xs(ins,level='instrument') #select the specific instrument df
                    _df = pd.concat([_df.shift(i) for i in reversed(range(self.pre_n_day))],axis=1)
                    #-> feature_1_pre_n_day...featrue_m_pre_n_day...feature_1_today...feature_m_today
                    _df['instrument'] = ins
                    ins_box.append(_df)
                feature = pd.concat(ins_box).reset_index()
                feature = feature.set_index(['datetime','instrument']).sort_index(level='datetime')
                label = label.reindex(feature.index)
                df = pd.concat([feature,label],axis=1)
                pprint(f'col {df.columns}')
                pprint(f'num_col {len(df.columns)}')
                pprint('finished !')
            return df

        elif dset[:3] == 'day':    # daily
            assert self.pre_n_day <= 60 #pre_n_day: daily_seq_len 
            col = []
            for i in ['OPEN', 'CLOSE', "HIGH", 'LOW', 'VOLUME', 'VWAP']:
                col.extend([i + str(j) for j in reversed(range(self.pre_n_day))]) # day_csi800 needs reserved feature cols for rnn
            _df = copy.deepcopy(df[col])
            _df['LABEL'] = df['LABEL'].values
            self.label_cls = df['LABEL_cls'].values
            pprint(f'num_col {len(_df.columns)}')
            return _df

    def _load_data(self, train_start_date, train_end_date, valid_start_date,
                   valid_end_date, test_start_date, test_end_date, raw_df):
        # string to datetime
        dates = (train_start_date, train_end_date, valid_start_date,
                 valid_end_date, test_start_date, test_end_date)
        (train_start_date, train_end_date, valid_start_date,
         valid_end_date, test_start_date, test_end_date) = [pd.Timestamp(x) for x in dates]

        # slice data
        train = raw_df.loc[train_start_date:train_end_date].dropna(subset=['LABEL'])
        valid = raw_df.loc[valid_start_date:valid_end_date].dropna(subset=['LABEL'])
        test = raw_df.loc[test_start_date:test_end_date]
        pprint('train: {} samples, from {:%Y-%m-%d} to {:%Y-%m-%d}'.format(
            len(train), train_start_date, train_end_date))
        pprint('valid: {} samples, from {:%Y-%m-%d} to {:%Y-%m-%d}'.format(
            len(valid), valid_start_date, valid_end_date))
        pprint('test : {} samples, from {:%Y-%m-%d} to {:%Y-%m-%d}'.format(
            len(test), test_start_date, test_end_date))

        # preprocess
        pprint('Preprocess Start')
        mask = ~train.columns.str.contains('^LABEL|^WEIGHT|^GROUP')
        x_train = train.iloc[:, mask]
        y_train = train['LABEL']
        x_valid = valid.iloc[:, mask]
        y_valid = valid['LABEL']
        x_test = test.iloc[:, mask]
        y_test = test['LABEL']
        del train
        del valid
        del test
        gc.collect()
        if self.log_transform:
            x_train = np.log1p(np.abs(x_train)) * np.sign(x_train)
            x_valid = np.log1p(np.abs(x_valid)) * np.sign(x_valid)
            x_test = np.log1p(np.abs(x_test)) * np.sign(x_test)

        mean = x_train.mean()
        std = x_train.std()

        # NOTE: fillna(0) before zscore will cause std != 1
        x_train = (x_train - mean).div(std + EPS).fillna(0)
        x_valid = (x_valid - mean).div(std + EPS).fillna(0)
        x_test = (x_test - mean).div(std + EPS).fillna(0)
        return x_train, x_valid, x_test, y_train, y_valid, y_test

    @data_ingredient.capture
    def load_data(self, train_start_date, train_end_date, valid_start_date,
                   valid_end_date, test_start_date, test_end_date):

        x_train_daily, x_valid_daily, x_test_daily, y_train_daily, y_valid_daily, y_test_daily = self._load_data(
            train_start_date, train_end_date, valid_start_date,
            valid_end_date, test_start_date, test_end_date, self._raw_df_daily)

        x_train_highfreq, x_valid_highfreq, x_test_highfreq, y_train_highfreq, y_valid_highfreq, y_test_highfreq = self._load_data(
            train_start_date, train_end_date, valid_start_date,
            valid_end_date, test_start_date, test_end_date, self._raw_df_highfreq)

        train_set = Dataset(x_train_daily, x_train_highfreq, y_train_daily, batch_size=self.batch_size,
                            daily_batch=self.daily_batch, shuffle=self.train_shuffle, pre_n_day=self.pre_n_day)
        valid_set = Dataset(x_valid_daily, x_valid_highfreq, y_valid_daily, batch_size=self.batch_size,
                            daily_batch=True, shuffle=False, pre_n_day=self.pre_n_day)
        test_set  = Dataset(x_test_daily, x_test_highfreq, y_test_daily, batch_size=self.batch_size,
                            daily_batch=True, shuffle=False, pre_n_day=self.pre_n_day)

        return train_set, valid_set, test_set

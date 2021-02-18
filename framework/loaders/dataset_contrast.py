import numpy as np
import pandas as pd
import math
from common.utils import pprint
from tqdm import tqdm 
import gc
class Dataset(object):
    """Dataset

    Parameters
    ----------
    feature : pd.DataFrame
        features DataFrame with rows as samples and cols as features
    label : pd.DataFrame or pd.Series
        labels DataFrame/Series with rows as samples and cols as labels
    batch_size : int
        batch size
    daily_batch : bool, optional
        whether ensure samples from the same day in the same batch
    shuffle : bool, optional
        whether shuffle samples
    """

    def __init__(self, feature_daily, feature_high_freq, label, batch_size,
                 daily_batch=True, shuffle=False, pre_n_day=0):

        assert isinstance(feature_daily, pd.DataFrame)
        assert isinstance(label, (pd.Series, pd.DataFrame))
        assert len(feature_daily) == len(label) # ensure the same number of time stamps
        assert len(feature_daily) == len(feature_high_freq)

        self.feature_daily = feature_daily
        self.feature_highfreq = feature_high_freq
        label = label.reindex(feature_daily.index)
        self.label = label
        # NOTE: always assume first index level is date
        self.count = label.groupby(level=0).size().values # [799,799,...,798,800] num of component stocks each day 

        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.daily_batch = daily_batch

        # calc size
        if self.daily_batch:
            self.indices = np.roll(np.cumsum(self.count), 1) # flatten then roll ahead 1 
            self.indices[0] = 0
            self.nb_batch = self.count // self.batch_size + (self.count % self.batch_size > 0) # batch_size : num of stocks in one batch
            # self.nb_batch -> [799,799,...,798,800] // batch_size
            self.length = self.nb_batch.sum()
        else:
            self.length = len(feature_daily)

    @property
    def index(self):
        return self.feature_daily.index

    def _iter_daily(self):
        indices = np.arange(len(self.nb_batch))
        if self.shuffle:
            np.random.shuffle(indices) # NOTE: only shuffle batches from different days
            pprint(f'day idx: {indices}') # !!! need to be set_random_seed in main
        for i in indices: # the ith daily batch
            for j in range(self.nb_batch[i]): # j_th batch in a day
                size = min(self.count[i] - j*self.batch_size, self.batch_size)
                start = self.indices[i] + j*self.batch_size
                yield (self.feature_daily.iloc[start:start+size].values,
                       self.feature_highfreq.iloc[start:start+size].values,
                       self.label.iloc[start:start+size].values)

    def _iter_random(self):
        indices = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in indices[::self.batch_size]:
            yield (self.feature.iloc[i:i+self.batch_size].values,
                   self.label.iloc[i:i+self.batch_size].values)

    def __iter__(self):
        if self.daily_batch:
            return self._iter_daily()
        return self._iter_random()

    def __len__(self):
        return self.length

    def __add__(self, other):
        feature = pd.concat([self.feature, other.feature], axis=0)
        label = pd.concat([self.label, other.label], axis=0)
        return Dataset(feature, label, self.batch_size,
                       self.daily_batch, self.shuffle)

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False
    
import loaders as loaders_module
import models as models_module

from common.utils import (
    pprint, set_random_seed, create_output_path,
    robust_zscore, count_num_params
)
from common.functions import K, rmse, mae # BUG: sacred cannot save source files used in ingredients

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('hft_pred', ingredients=[
    models_module.contrastive_all_2_encoder.model_ingredient,
    models_module.contrastive_all_2_stage.model_ingredient,
    loaders_module.daily_loader_contrast.data_ingredient,
])

create_output_path('my_runs')
ex.observers.append(FileStorageObserver("my_runs","my_runs/resource","my_runs/source","my_runs/templete"))


@ex.config
def run_config():
    seed = 0
    output_path = None
    loader_name = 'daily_loader'
    model_name = 'mlp'
    comt = 'rnn_60_1.0'
    run_on = False

@ex.main
def main(_run, seed, loader_name, model_name, output_path, comt, run_on):
    _run = SummaryWriter(comment=comt) if run_on else None
    # seed
    set_random_seed(seed)
    # path
    output_path = create_output_path(output_path)
    pprint('output path:', output_path)
    model_path = output_path+'/model.bin'
    pred_path = output_path+'/pred_%s_%d.pkl' %(model_name,seed)
    pprint('create loader `%s` and model `%s`...'%(loader_name, model_name))
    loader = getattr(loaders_module, loader_name).DataLoader()

    pprint('load data...')
    train_set, valid_set, test_set = loader.load_data()

    model = getattr(models_module, model_name).Model()
    pprint('total params:', count_num_params(model))

    pprint('training...')
    model.fit(train_set, valid_set, _run)
    model.save(model_path)
    ex.add_artifact(model_path)

    pprint('inference...')
    pprint('validation set:')
    pred = pd.DataFrame(index=valid_set.index)
    pred['label'] = valid_set.label
    pred['score'] = model.predict(valid_set)
    pred.to_pickle(pred_path+'.valid')
    ex.add_artifact(pred_path+'.valid')

    robust_ic = pred.groupby(level='datetime').apply(
        lambda x: robust_zscore(x.label).corr(robust_zscore(x.score)))
    rank_ic = pred.groupby(level='datetime').apply(
        lambda x: x.label.corr(x.score, method='spearman'))
    pprint('Robust IC:', robust_ic.mean(), ',', robust_ic.mean()/robust_ic.std())
    pprint('Rank IC:', rank_ic.mean(), ',', rank_ic.mean()/rank_ic.std())

    ori_label = pd.read_pickle("'../data/%s.pkl'%loader.dset")['LABEL0']
    pred['ori_label'] = ori_label.reindex(pred.index)
    _mean_1 = pred['ori_label'].mean()
    _std_1 = pred['ori_label'].std()
    pred_score_1 = pred.score *  _std_1 +  _mean_1
    RMSE_1 = rmse(pred_score_1, pred.ori_label)
    MAE_1 = mae(pred_score_1, pred.ori_label)
    pprint('RMSE_1:', RMSE_1)
    pprint('MAE_1:', MAE_1)

    _mean_2 = pred.score.mean()
    _std_2 = pred.score.std()
    pred_score_2 = pred.score *  _std_2 +  _mean_2
    RMSE_2 = rmse(pred_score_2, pred.ori_label)
    MAE_2 = mae(pred_score_2, pred.ori_label)
    pprint('RMSE_2:', RMSE_2)
    pprint('MAE_2:', MAE_2)



    pprint('testing set:')
    pred = pd.DataFrame(index=test_set.index)
    pred['label'] = test_set.label
    pred['score'] = model.predict(test_set)
    pred.to_pickle(pred_path)
    ex.add_artifact(pred_path)

    robust_ic = pred.groupby(level='datetime').apply(
        lambda x: robust_zscore(x.label).corr(robust_zscore(x.score)))
    rank_ic = pred.groupby(level='datetime').apply(
        lambda x: x.label.corr(x.score, method='spearman'))
    pprint('Robust IC:', robust_ic.mean(), ',', robust_ic.mean()/robust_ic.std())
    pprint('Rank IC:', rank_ic.mean(), ',', rank_ic.mean()/rank_ic.std())

    _mean_1 = pred['ori_label'].mean()
    _std_1 = pred['ori_label'].std()
    pred_score_1 = pred.score *  _std_1 +  _mean_1
    RMSE_1 = rmse(pred_score_1, pred.ori_label)
    MAE_1 = mae(pred_score_1, pred.ori_label)
    pprint('RMSE_1:', RMSE_1)
    pprint('MAE_1:', MAE_1)

    _mean_2 = pred.score.mean()
    _std_2 = pred.score.std()
    pred_score_2 = pred.score *  _std_2 +  _mean_2
    RMSE_2 = rmse(pred_score_2, pred.ori_label)
    MAE_2 = mae(pred_score_2, pred.ori_label)
    pprint('RMSE_2:', RMSE_2)
    pprint('MAE_2:', MAE_2)





    pprint('done.')


if __name__ == '__main__':

    ex.run_commandline()

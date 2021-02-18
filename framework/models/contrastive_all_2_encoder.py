import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from common.utils import pprint, AverageMeter
from common.functions import get_loss_fn, get_metric_fn

from sacred import Ingredient

model_ingredient = Ingredient('contrastive_all_2_encoder')


# try:
#     from tensorboardX import SummaryWriter
#     is_tensorboard_available = True
# except Exception:
#     is_tensorboard_available = False


@model_ingredient.config
def model_config():
    # architecture
    input_shape = [6, 60]
    rnn_type = 'LSTM'  # LSTM/GRU
    rnn_layer = 2
    hid_size = 64
    dropout = 0
    # optimization
    optim_method = 'Adam'
    optim_args = {'lr': 1e-3}
    loss_fn = 'mse'
    eval_metric = 'corr'
    verbose = 500
    max_steps = 50
    early_stopping_rounds = 5
    negative_sample = 5


class Model(nn.Module):
    @model_ingredient.capture
    def __init__(self,
                 input_shape,
                 rnn_type='LSTM',
                 rnn_layer=2,
                 hid_size=64,
                 dropout=0,
                 optim_method='Adam',
                 optim_args={'lr': 1e-3},
                 loss_fn='mse',
                 eval_metric='corr',
                 negative_sample=5):

        super().__init__()

        # Architecture
        self.hid_size = hid_size
        self.input_size = input_shape[0][0]    #feature num
        self.input_day = input_shape[0][2]
        self.input_daily_length = input_shape[0][1]
        self.input_highfreq_length = input_shape[1][1]
        self.dropout = dropout
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type
        self.negative_sample = negative_sample

        self._build_model()

        # Optimization
        self.optimizer = getattr(optim, optim_method)(self.parameters(),
                                                      **optim_args)
        self.loss_fn = get_loss_fn(loss_fn)
        self.metric_fn = get_metric_fn(eval_metric)

        if torch.cuda.is_available():
            self.cuda()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except:
            raise ValueError('unknown rnn_type `%s`' % self.rnn_type)
        self.net_daily_1 = nn.Sequential()
        self.net_daily_1.add_module('fc_in_daily_1', nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net_daily_1.add_module('act_daily_1', nn.Tanh())
        self.net_highfreq_1 = nn.Sequential()
        self.net_highfreq_1.add_module('fc_in_highfreq_1', nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net_highfreq_1.add_module('act_highfreq_1', nn.Tanh())
        self.rnn_highfreq_1 = klass(input_size=self.hid_size,
                                    hidden_size=self.hid_size,
                                    num_layers=self.rnn_layer,
                                    batch_first=True,
                                    dropout=self.dropout)

        self.net_daily_2 = nn.Sequential()
        self.net_daily_2.add_module('fc_in_daily_2', nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net_daily_2.add_module('act_daily_2', nn.Tanh())
        self.net_highfreq_2 = nn.Sequential()
        self.net_highfreq_2.add_module('fc_in_highfreq_2',
                                       nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net_highfreq_2.add_module('act_highfreq_2', nn.Tanh())
        self.rnn_highfreq_2 = klass(input_size=self.hid_size,
                                    hidden_size=self.hid_size,
                                    num_layers=self.rnn_layer,
                                    batch_first=True,
                                    dropout=self.dropout)

        # self.rnn_daily = klass(input_size=self.hid_size + self.input_size,
        self.rnn_daily_1 = klass(input_size=self.hid_size*2,
                                 hidden_size=self.hid_size,
                                 num_layers=self.rnn_layer,
                                 batch_first=True,
                                 dropout=self.dropout)
        self.fc_out_1 = nn.Linear(in_features=self.hid_size, out_features=self.hid_size)  # point contrast weight
        self.fc_out_2 = nn.Linear(in_features=self.hid_size, out_features=1)                # output fc
        self.fc_out_3 = nn.Linear(in_features=self.hid_size, out_features=self.hid_size*2)  # trend contrast weight

    def forward(self, data_daily, data_highfreq):
        data_highfreq = data_highfreq.view(-1, self.input_day, self.input_size, self.input_highfreq_length)
        arr_1 = []
        arr_2 = []
        for i in range(self.input_day):
            input = data_highfreq[:, i, :, :]  # [batch, input_day, input_size, seq_len] -> [batch, input_size, seq_len]
            input = input.permute(0, 2, 1)  # [batch, input_size, seq_len] -> [batch, seq_len, input_size ]
            # pprint(f'input_shape {input.shape}')
            out_1, _ = self.rnn_highfreq_1(self.net_highfreq_1(input))
            out_1 = out_1[:, -1, :].unsqueeze(1)  # [batch, seq_len, hid_size] -> [batch, 1, hid_size]
            arr_1.append(out_1)
            out_2, _ = self.rnn_highfreq_2(self.net_highfreq_2(input))
            out_2 = out_2[:, -1, :].unsqueeze(1)  # [batch, seq_len, hid_size] -> [batch, 1, hid_size]
            arr_2.append(out_2)
        day_reps_1 = torch.cat(arr_1, dim=1)  # arr: [batch, 1, hid_size] * input_day -> [batch, input_day, input_size]
        day_reps_2 = torch.cat(arr_2, dim=1)
        data_daily = data_daily.view(-1, self.input_size, self.input_day)
        data_daily = data_daily.permute(0, 2, 1)    # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        data_daily_1 = self.net_daily_1(data_daily)     # [batch, seq_len, input_size] -> [batch, seq_len, hidden_size]
        data_daily_2 = self.net_daily_2(data_daily)

        # Point contrast
        context = self.fc_out_1(data_daily_1)
        point_contrast_loss = 0
        for i in range(self.input_day):
            daily_input = context[:, i, :]          # [batch, seq_len, hidden_size] -> [batch, hidden_size]
            highfreq_input = day_reps_1[:, i, :]      # [batch, seq_len, hidden_size] -> [batch, hidden_size]
            highfreq_input = self._generate_highfreq_data(highfreq_input)

            highfreq_input = torch.reshape(highfreq_input, (-1, 1+self.negative_sample, self.hid_size))
            daily_input = torch.unsqueeze(daily_input, 1)   # [batch, 1, hidden_size]
            dot_product = torch.mean(highfreq_input * daily_input, -1)
            log_l1 = torch.nn.functional.log_softmax(dot_product, dim=1)[:, 0]
            point_contrast_loss += -torch.mean(log_l1)

        pred_1, _ = self.rnn_daily_1(torch.cat((data_daily_2 + day_reps_2, data_daily_1 + day_reps_1), dim=2))
        out = self.fc_out_2(pred_1[:, -1, :])

        # Trend contrast
        new_data_daily = torch.reshape(torch.cat((data_daily_2 + day_reps_2, data_daily_1 + day_reps_1), dim=2),
                                       (-1, self.hid_size*2 * self.input_day))
        new_data_daily = self._generate_data(new_data_daily, size=self.hid_size*2)
        new_data_daily = torch.reshape(new_data_daily, (-1, self.input_day+self.negative_sample, self.hid_size*2))
        next = new_data_daily[:, -1-self.negative_sample:, :]
        context_trend = self.fc_out_3(pred_1[:, -2, :])
        context_trend = torch.unsqueeze(context_trend, 1)
        dot_product_trend = torch.mean(next * context_trend, -1)
        log_l1 = torch.nn.functional.log_softmax(dot_product_trend, dim=1)[:, 0]
        trend_contrast_loss = -torch.mean(log_l1)

        return 0.05 * point_contrast_loss + trend_contrast_loss, out[..., 0]

    @model_ingredient.capture
    def fit(self,
            train_set,
            valid_set,
            run=None,
            max_steps=100,
            early_stopping_rounds=10,
            verbose=100):
        best_score = np.inf
        stop_steps = 0
        best_params = copy.deepcopy(self.state_dict())

        for step in range(max_steps):

            pprint('Step:', step)
            if stop_steps >= early_stopping_rounds:
                if verbose:
                    pprint('\tearly stop')
                break
            stop_steps += 1
            # training
            self.train()
            train_loss = AverageMeter()
            train_eval = AverageMeter()
            for i, (data_daily, data_highfreq, label) in enumerate(train_set):
                data_daily = torch.tensor(data_daily, dtype=torch.float)
                data_highfreq = torch.tensor(data_highfreq, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    data_daily, data_highfreq, label = data_daily.cuda(), data_highfreq.cuda(), label.cuda()
                loss_contrast, pred = self(data_daily, data_highfreq)
                loss = self.loss_fn(pred, label)
                loss = loss + loss_contrast
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_ = loss.item()
                eval_ = self.metric_fn(pred, label).item()
                train_loss.update(loss_, len(data_daily))
                train_eval.update(eval_)

                if verbose and i % verbose == 0:
                    pprint('iter %s: train_loss %.6f, train_eval %.6f' %
                           (i, train_loss.avg, train_eval.avg))
            # evaluation
            self.eval()
            valid_loss = AverageMeter()
            valid_eval = AverageMeter()
            for i, (data_daily, data_highfreq, label) in enumerate(valid_set):
                data_daily = torch.tensor(data_daily, dtype=torch.float)
                data_highfreq = torch.tensor(data_highfreq, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    data_daily, data_highfreq, label = data_daily.cuda(), data_highfreq.cuda(), label.cuda()
                with torch.no_grad():
                    loss_contrast, pred = self(data_daily, data_highfreq)
                loss = self.loss_fn(pred, label)
                loss = loss + loss_contrast
                valid_loss_ = loss.item()
                valid_eval_ = self.metric_fn(pred, label).item()
                valid_loss.update(valid_loss_, len(data_daily))
                valid_eval.update(valid_eval_)

            if run is not None:
                run.add_scalar('Train/Loss', train_loss.avg, step)
                run.add_scalar('Train/Eval', train_eval.avg, step)
                run.add_scalar('Valid/Loss', valid_loss.avg, step)
                run.add_scalar('Valid/Eval', valid_eval.avg, step)
            if verbose:
                pprint("current step: train_loss {:.6f}, valid_loss {:.6f}, "
                       "train_eval {:.6f}, valid_eval {:.6f}".format(
                    train_loss.avg, valid_loss.avg, train_eval.avg,
                    valid_eval.avg))
            if valid_eval.avg < best_score:
                if verbose:
                    pprint(
                        '\tvalid update from {:.6f} to {:.6f}, save checkpoint.'
                            .format(best_score, valid_eval.avg))
                best_score = valid_eval.avg
                stop_steps = 0
                best_params = copy.deepcopy(self.state_dict())
        # restore
        self.load_state_dict(best_params)

    def _generate_data(self, data, size):
        # [batch_size, previous_day+1(true)] -> [batch_size, (previous_day+1(true)+negative_sample)*feature_size]
        new_data = data.clone()
        data_lastday = data.clone()[:, -size:]
        for i in range(self.negative_sample):
            random_list_1 = torch.randperm(data_lastday.size(0))
            random_list_2 = torch.randperm(data_lastday.size(0))
            random_list = torch.where(random_list_1 - torch.arange(data_lastday.size(0)) == 0, random_list_2,
                                      random_list_1)
            random_lastday = data_lastday[random_list]
            new_data = torch.cat((new_data, random_lastday), 1)
        return new_data

    def _generate_highfreq_data(self, data):
        # [batch_size, previous_day+1(true)] -> [batch_size, (previous_day+1(true)+negative_sample)*feature_size]
        new_data = data.clone()
        for i in range(self.negative_sample):
            random_list_1 = torch.randperm(data.size(0))
            random_list_2 = torch.randperm(data.size(0))
            random_list = torch.where(random_list_1 - torch.arange(data.size(0)) == 0, random_list_2,
                                      random_list_1)
            random_data = data[random_list]
            new_data = torch.cat((new_data, random_data), 1)
        return new_data

    def predict(self, test_set):
        self.eval()
        preds = []
        for i, (data_daily, data_highfreq, _) in enumerate(test_set):
            data_daily = torch.tensor(data_daily, dtype=torch.float)
            data_highfreq = torch.tensor(data_highfreq, dtype=torch.float)

            if torch.cuda.is_available():
                data_daily = data_daily.cuda()
                data_highfreq = data_highfreq.cuda()
            with torch.no_grad():
                preds.append(self(data_daily, data_highfreq)[1].cpu().numpy())
        return np.concatenate(preds)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path, strict=True):
        self.load_state_dict(torch.load(model_path), strict=strict)

import torch
import numpy as np
# import tensorflow as tf
import torch.nn.functional as F
import torch.nn as nn
EPS = 1e-12
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef

#################### ops wrapper ####################
class K(object):
    """backend kernel"""

    @staticmethod
    def sum(x, axis=0, keepdims=True):
        if isinstance(x, np.ndarray):
            return x.sum(axis=axis, keepdims=keepdims)
        # if isinstance(x, tf.Tensor):
        #     return tf.reduce_sum(x, axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.sum(dim=axis, keepdim=keepdims)
        raise NotImplementedError('unsupported data type %s'%type(x))

    @staticmethod
    def mean(x, axis=0, keepdims=True):
        if isinstance(x, np.ndarray):
            return x.mean(axis=axis, keepdims=keepdims)
        # if isinstance(x, tf.Tensor):
        #     return tf.reduce_mean(x, axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.mean(dim=axis, keepdim=keepdims)
        raise NotImplementedError('unsupported data type %s'%type(x))
    @staticmethod
    def cos_sim(x1,x2,dim=1):
        if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
            return np.dot(x1,x2)/(np.linalg.norm(x1)*(np.linalg.norm(x2)))
        # if isinstance(x, tf.Tensor):
        #     return tf.reduce_mean(x, axis=axis, keepdims=keepdims)
        if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            return F.cosine_similarity(x1,x2,dim=dim)
        raise NotImplementedError('unsupported data type %s %s'%type(x1,x2))

    @staticmethod
    def std(x, axis=0, keepdims=True):
        if isinstance(x, np.ndarray):
            return x.std(axis=axis, keepdims=keepdims)
        # if isinstance(x, tf.Tensor):
        #     m = tf.reduce_mean(x, axis=axis, keepdims=keepdims)
        #     devs_squared = tf.square(x - m)
        #     return tf.sqrt(tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims))
        if isinstance(x, torch.Tensor):
            return x.std(dim=axis, unbiased=False, keepdim=keepdims)
        raise NotImplementedError('unsupported data type %s'%type(x))

    @staticmethod
    def median(x, axis=0, keepdims=True):
        # NOTE: numpy will average when size is even,
        # but tensorflow and pytorch don't average
        if isinstance(x, np.ndarray):
            return np.median(x, axis=axis, keepdims=keepdims)
        # if isinstance(x, tf.Tensor):
        #     return tf.contrib.distributions.percentile(x, 50, axis=axis, keep_dims=keepdims)
        if isinstance(x, torch.Tensor):
            return torch.median(x, dim=axis, keepdim=keepdims)[0]
        raise NotImplementedError('unsupported data type %s'%type(x))

    @staticmethod
    def shape(x):
        if isinstance(x, np.ndarray):
            return x.shape
        # if isinstance(x, tf.Tensor):
        #     return tf.shape(x)
        if isinstance(x, torch.Tensor):
            return list(x.shape)
        raise NotImplementedError('unsupported data type %s'%type(x))

    @staticmethod
    def cast(x, dtype='float'):
        if isinstance(x, np.ndarray):
            return x.astype(dtype)
        # if isinstance(x, tf.Tensor):
        #     return tf.cast(x, dtype)
        if isinstance(x, torch.Tensor):
            return x.type(getattr(torch, dtype))
        raise NotImplementedError('unsupported data type %s'%type(x))

    @staticmethod
    def sigmoid(x):
        if isinstance(x, np.ndarray):
            return 1 / (1 + np.exp(-x))
        # if isinstance(x, tf.Tensor):
        #     return tf.sigmoid(x)
        if isinstance(x, torch.Tensor):
            return torch.sigmoid(x)
        raise NotImplementedError('unsupported data type %s'%type(x))

    @staticmethod
    def maximum(x, v):
        if isinstance(x, np.ndarray):
            return np.maximum(x, v)
        # if isinstance(x, tf.Tensor):
        #     return tf.maximum(x, v)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min=v)
        raise NotImplementedError('unsupported data type %s'%type(x))

    @staticmethod
    def clip(x, min_val, max_val):
        if isinstance(x, np.ndarray):
            return np.clip(x, min_val, max_val)
        # if isinstance(x, tf.Tensor):
        #     return tf.clip_by_value(x, min_val, max_val)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min_val, max_val)
        raise NotImplementedError('unsupported data type %s'%type(x))

# Add Static Methods
def generic_ops(method):
    def wrapper(x, *args):
        if isinstance(x, np.ndarray):
            return getattr(np, method)(x, *args)
        # if isinstance(x, tf.Tensor):
        #     return getattr(tf, method)(x, *args)
        if isinstance(x, torch.Tensor):
            return getattr(torch, method)(x, *args)
        raise NotImplementedError('unsupported data type %s'%type(x))
    return wrapper

for method in ['abs', 'log', 'sqrt', 'exp', 'log1p', 'tanh',
               'cosh', 'squeeze', 'reshape', 'zeros_like']:
    setattr(K, method, staticmethod(generic_ops(method)))
#################### loss functions ####################
def zscore(x, axis=0):
    mean = K.mean(x, axis=axis)
    std = K.std(x, axis=axis)
    return (x - mean) / (std + EPS)

def rmse(pred, label):
    return np.sqrt(np.mean((pred - label)**2)) *100

def mae(pred, label):
    return np.mean(np.abs(pred - label)) * 100

def robust_zscore(x, axis=0):
    med = K.median(x, axis=axis)
    mad = K.median(K.abs(x-med), axis=axis)
    x = (x - med) / (mad*1.4826 + EPS)
    return K.clip(x, -3, 3)

# def batch_corr(x, y, axis=0, keepdims=True):
#     x = zscore(x, axis=axis)
#     y = zscore(y, axis=axis)
#     return K.mean(x*y, axis=axis, keepdims=False)
def batch_corr(x, y, axis=0, dim=0,keepdims=False):
    x = zscore(x, axis=axis)
    y = zscore(y, axis=axis)
    return K.mean(x*y,axis=dim,keepdims=keepdims)

def robust_batch_corr(x, y, axis=0, dim=0,keepdims=False):
    x = robust_zscore(x, axis=axis)
    y = robust_zscore(y, axis=axis)
    return K.mean(x*y,axis=dim,keepdims=keepdims)

def standard_mse(pred, label):
    loss = (pred - label)**2
    return K.mean(loss, keepdims=False)

def modified_mse(pred, label):
    pred = zscore(pred) # zscore before mse
    return K.mean((pred - label)**2,keepdims=False)

def modified_mse_v2(pred, label, dim=0):
    pred = zscore(pred) # zscore before mse
    return K.mean((pred - label)**2, axis=dim)

def modified_mse_v3(pred, label, dim=0):
    pred = zscore(pred) # zscore before mse
    label = zscore(label)
    return K.mean((pred - label)**2, axis=dim)

def logcosh(pred, label, dim=0):
    loss = K.log(K.cosh(pred - label))
    return K.mean(loss, axis=dim)

def standard_cos_sim(pred, label, axis=1):
    return K.cos_sim(pred, label,dim=axis)

def standard_cross_entropy(pred, label, reduce=True):
    y = K.cast(label > 0, 'float')
    p = pred # alias
    loss = K.maximum(p, 0) - p * y + K.log1p(K.exp(-K.abs(p)))
    if reduce:
        return K.mean(loss, keepdims=False)
    return loss

def bce(pred, label):
    label = label.gt(0).to(label)
    return nn.BCEWithLogitsLoss()(pred, label)

def cross_entropy(pred, label):
    return nn.CrossEntropyLoss()(pred, label)

def standard_mae(pred, label):
    loss = K.abs(pred - label)
    return K.mean(loss, keepdims=False)

def modified_mae(pred, label):
    loss = K.abs(pred - label)
    loss = loss * K.abs(label)
    return K.mean(loss, keepdims=False)

def modified_mae_v2(pred, label):
    pred = zscore(pred)
    label = zscore(label)
    loss = K.abs(pred - label)
    return K.mean(loss, keepdims=False)

def modified_cos_sim(pred, label, dim=1):
    pred = zscore(pred) # zscore before mse
    return K.cos_sim(pred, label, dim=dim)

def modified_cross_entropy(pred, label):
    w = K.abs(label) + 1 # normalized label ~ [-3, 3]
    w /= K.sum(w, axis=0)
    loss = standard_cross_entropy(pred, label, reduce=False)
    return K.sum(w * loss, keepdims=False)

def hinge(pred, label):
    return K.mean(torch.clamp(1 - pred * label, min=0), keepdims=False)

def acc(pred, label):
    """calculate accuracy

    NOTE:
    - this method is not differentiable
    - we assume the predictions are logits other than probs
    """
    label = label.gt(0).to(label)
    return pred.gt(0).float().eq(label).float().mean()

def multi_acc(pred, label, avg='weighted'):
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    pred = np.argmax(pred,axis=1)
    acc = precision_score(label, pred, average=avg).mean()
    # print('acc:',acc)
    return acc

def multi_acc_macro(pred, label):
    acc = multi_acc(pred, label, avg='macro')
    # print('acc:',acc)
    return acc

def multi_acc_micro(pred, label):
    acc = multi_acc(pred, label, avg='micro')
    # print('acc:',acc)
    return acc

def binary_acc(pred, label):
    acc = multi_acc(pred, label, avg='binary')
    return acc

def correlation(pred, label):
    return matthews_corrcoef(label, pred)

def neg_wrapper(func):
    def wrapper(*args, **kwargs):
        return -1*func(*args, **kwargs)
    return wrapper

def get_loss_fn(loss_fn):
    # reflection: legacy name
    if loss_fn == 'mse':
        return modified_mse
    if loss_fn == 'mse_v2':
        return modified_mse_v2
    if loss_fn == 'mse_v3':
        return modified_mse_v3
    if loss_fn == 'cos':
        return modified_cos_sim
    if loss_fn == 'std_cos':
        return standard_cos_sim
    if loss_fn == 'std_mae':
        return standard_mae
    if loss_fn == 'mae':
        return modified_mae
    if loss_fn == 'mae_v2':
        return modified_mae_v2
    if loss_fn == 'std_bce':
        return standard_cross_entropy
    if loss_fn == 'bce':
        return bce
    if loss_fn == 'ce':
        return cross_entropy
    if loss_fn == 'logcosh':
        return logcosh
    if loss_fn == 'hinge':
        return hinge
    if loss_fn == 'zscore':
        return zscore
    # return function by name
    try:
        return eval(loss_fn) # dangerous eval
    except:
        pass
    # return negative function by name
    try:
        return neg_wrapper(eval(loss_fn.replace('neg_', '')))
    except:
        raise NotImplementedError('loss function %s is not implemented'%loss_fn)

def get_metric_fn(eval_metric):
    # reflection: legacy name
    if eval_metric == 'corr':
        return neg_wrapper(batch_corr)
    if eval_metric == 'robust_corr':
        return neg_wrapper(robust_batch_corr)
    # return function by name
    if eval_metric == 'acc':
        return acc
    if eval_metric == 'multi_acc':
        return multi_acc
    if eval_metric == 'multi_acc_macro':
        return multi_acc_macro
    if eval_metric == 'multi_acc_micro':
        return multi_acc_micro
    if eval_metric == 'binary_acc':
        return binary_acc
    try:
        return eval(eval_metric) # dangerous eval
    except:
        pass
    # return negative function by name
    try:
        return neg_wrapper(eval(eval_metric.replace('neg_', '')))
    except:
        raise NotImplementedError('metric function %s is not implemented'%eval_metric)

def test():
    # test api
    for func in ['mse', 'neg_batch_corr']:
        get_loss_fn(func)
    # test calculation
    x = np.random.randn(10).astype(np.float32)
    y = np.random.randn(10).astype(np.float32)
    # x_tf = tf.convert_to_tensor(x)
    # y_tf = tf.convert_to_tensor(y)
    x_pt = torch.tensor(x)
    y_pt = torch.tensor(y)
    for func in [batch_corr, standard_mse, standard_mae,
                 standard_cross_entropy, modified_mse,
                 modified_mae, modified_cross_entropy,]:
        print('test %s...'%func.__name__)
        r = func(x, y)
        r_pt = func(x_pt, y_pt).numpy()
        # with tf.Session() as sess:
        #     r_tf = func(x_tf, y_tf).eval()
        assert np.all(np.isclose(r, r_pt)), 'numpy != pytorch'
        # assert np.all(np.isclose(r, r_tf)), 'numpy != tensorflow'
        print('passed.')

if __name__ == '__main__':

    test()
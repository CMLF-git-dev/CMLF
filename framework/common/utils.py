import os
import sys
import json
import uuid
import numpy
import numpy as np
import torch
import shutil
import random
import hashlib
import datetime
import requests
import contextlib
import collections


# import tensorflow

#################### Class ####################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#################### Methods ####################
def pprint(*args):
    time = '[' + str(datetime.datetime.utcnow() + datetime.timedelta(hours=8))[:19] + '] -'
    print(time, *args, flush=True)


def get_hash(size=8):
    return uuid.uuid4().hex[:size]


def create_output_path(path=None):
    if path is None:
        path = '/tmp/' + get_hash()
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pprint('WARN: output path %s already exist' % path)
    return path


def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # tensorflow.set_random_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['LGB_RANDOM_SEED'] = str(seed)


def robust_zscore(x):
    # MAD based robust zscore
    x = x - x.median()
    mad = x.abs().median()
    x = numpy.clip(x / mad / 1.4826, -3, 3)
    return x


def zscore(x, axis=0):
    mean = numpy.mean(x, axis=axis)
    std = numpy.std(x, axis=axis)
    return (x - mean) / (std + EPS)

def count_num_params(model):
    # pytorch
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # tensorflow
    # return int(numpy.sum([
    #     numpy.product([xi.value for xi in x.get_shape()])
    #     for x in tensorflow.global_variables()]))


def rmse(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))


def mae(pred, label):
    return np.mean(np.abs(pred - label))


def hash_args(*args):
    string = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(string.encode()).hexdigest()


@contextlib.contextmanager
def save_multiple_parts_file(filename, format='gztar'):
    if filename.startswith('~'):
        filename = os.path.expanduser(filename)
    file_path = os.path.abspath(filename)
    if os.path.exists(file_path):
        raise FileExistsError('file exists: {}, cannot create the directory.'.format(file_path))
    os.makedirs(file_path)
    yield file_path
    tar_file = shutil.make_archive(file_path, format=format, root_dir=file_path)
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    os.rename(tar_file, file_path)


@contextlib.contextmanager
def load_multiple_parts_file(filename, format='gztar'):
    temp_dir = create_output_path()
    file_path = os.path.join(temp_dir, os.path.basename(filename))
    shutil.copyfile(filename, file_path + '.tar.gz')
    try:
        os.makedirs(file_path)
        shutil.unpack_archive(file_path + '.tar.gz', format=format, extract_dir=file_path)
        yield file_path
    except Exception as e:
        print('Exception:', e)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def download_http_resource(url, fpath):
    dirname = os.path.dirname(fpath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fpath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params' % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params
import os
import numpy as np
import torch

from .flickr import Flickr
from .youth import Youth
from torch.utils.data import DataLoader
import logging

from .graphs import Graph
from .flickr import Flickr

__data_args = {
    'flickr': {'class': 2, 'shape': [3, 6, 300, 25, 2], 'feeder': Flickr},
    'youth': {'class': 2, 'shape': [3, 6, 300, 25, 2], 'feeder': Youth},
}


def create(dataset, trainset, evalset, **kwargs):
    g = Graph(dataset, **kwargs)
    try:
        data_args = __data_args[dataset]
        num_class = data_args['class']
        # train_set, eval_set = data_args['train'], data_args['eval']
    except:
        logging.info('')
        logging.error('Error: {} dataset does NOT exist!'.format(dataset))
        raise ValueError()

    feeders = {
        'train': data_args['feeder'](phase=trainset),
        'eval': data_args['feeder'](phase=evalset),
    }

    class_weights = feeders['train'].class_weights
    data_shape = feeders['train'].datashape
    return feeders, data_shape, num_class, g.A, g.parts, class_weights

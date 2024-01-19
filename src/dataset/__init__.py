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


def create(dataset, **kwargs):
    g = Graph(dataset, **kwargs)
    try:
        data_args = __data_args[dataset]
        num_class = data_args['class']
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()

    feeders = {
        'train': data_args['feeder'](phase='train'),
        'eval': data_args['feeder'](phase='test'),
    }

    class_weights = feeders['train'].class_weights
    data_shape = feeders['train'].datashape
    return feeders, data_shape, num_class, g.A, g.parts, class_weights

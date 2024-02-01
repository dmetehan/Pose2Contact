import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, phase, path, annot_file_name, subset='binary', fold='fold0'):
        self.subset = subset
        if subset == 'signature':
            path = os.path.join(path, fold)
        self.data = self.read_data(os.path.join(path, phase, annot_file_name))
        self.convert_to_flickr()
        self.remove_ambiguous()
        self.format_data()
        self.fill_no_dets()
        # TODO: Add normalization of the coordinates to [0, 1]
        self.datashape = [2, 17, 3]
        self.class_weights = self.calc_class_weights()

    def calc_class_weights(self):
        # TODO: Write class weights to config file?
        labels = [y for _, y in self.data]
        return [1, 4]
        # return [labels.count(1)/len(labels), labels.count(0)/len(labels)]  # reversed instance ratio

    def remove_ambiguous(self): self.data = [self.data[d] for d in range(len(self.data)) if self.data[d]['contact_type'] != '1']  # 0:no contact, 1:ambiguous, 2:contact

    def format_data(self):
        if self.subset == 'binary':
            self.data = [(np.array(self.data[d]['preds']), int(int(self.data[d]['contact_type']) > 0)) for d in range(len(self.data))]
        elif self.subset == 'signature':
            self.data = [(np.array(self.data[d]['preds']), self.onehot_sig(self.data[d]['signature'])) for d in range(len(self.data))]

    @staticmethod
    def onehot_sig(signature, res=21):
        mat = torch.zeros((res, res), dtype=int)
        for adult, child in signature:
            mat[adult, child] = 1
        return mat.flatten()

    @staticmethod
    def read_data(annot_file_path):
        logging.info(f'Reading data and annotations from: {annot_file_path}')
        with open(annot_file_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        :param index:
        :return: a tuple of data [(2,17,3) numpy array] and a label [int]
        """
        return self.data[index]

    def convert_to_flickr(self):
        pass

    def fill_no_dets(self):
        pass

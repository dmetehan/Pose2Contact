import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, phase, root_folder, annot_file_name, subset='binary', fold='fold0', **kwargs):
        self.subset = subset
        self.phase = phase
        logging.info(f'{self.subset} dataset {phase} set on fold {fold}')
        if subset == 'signature':
            root_folder = os.path.join(root_folder, fold)
        self.data = self.read_data(os.path.join(root_folder, phase, annot_file_name))
        self.convert_to_flickr()
        self.remove_ambiguous()
        self.format_data(normalize=False)
        if self.subset == 'binary':
            # self.fill_no_dets()
            self.remove_no_dets()
        self.datashape = [2, 17, 3]

    def remove_ambiguous(self):
        self.data = [self.data[d] for d in range(len(self.data)) if self.data[d]['contact_type'] != '1']  # 0:no contact, 1:ambiguous, 2:contact

    def format_data(self, normalize=False):
        if self.subset == 'binary':
            self.data = [(np.array(self.data[d]['preds']),
                          int(int(self.data[d]['contact_type']) > 0),
                          (self.data[d]['subject'], self.data[d]['frame'])) for d in range(len(self.data))]
        elif self.subset == 'signature':
            self.data = [(np.array(self.data[d]['preds']) if not normalize else self.normalize_preds(self.data[d]['preds']),
                          (self.onehot_segmentation(self.data[d]['seg21_adult'], self.data[d]['seg21_child'], res=21),
                           self.onehot_segmentation(self.data[d]['seg6_adult'], self.data[d]['seg6_child'], res=6),
                           self.onehot_sig(self.data[d]['signature21x21'], res=21),
                           self.onehot_sig(self.data[d]['signature6x6'], res=6)),
                          (self.data[d]['subject'], self.data[d]['frame'])) for d in range(len(self.data))
                         if self.phase in ['train', 'trainval']
                         or self.onehot_sig(self.data[d]['signature21x21'], res=21).sum() > 0]  # removing no contact cases from val, test sets

    @staticmethod
    def normalize_preds(preds):
        preds = np.array(preds)
        max_values = preds[:, :, :2].max(axis=1).max(axis=0)
        min_values = preds[:, :, :2].min(axis=1).min(axis=0)
        scale_values = max_values - min_values
        preds[:, :, :-1] = (preds[:, :, :-1] - min_values) / scale_values
        return preds

    @staticmethod
    def onehot_segmentation(adult_seg, child_seg, res=21):
        mat = torch.zeros(res + res, dtype=torch.int8)
        for adult in adult_seg:
            mat[adult] = 1
        for child in child_seg:
            mat[res + child] = 1
        return mat.flatten()

    @staticmethod
    def onehot_sig(signature, res=21):
        mat = torch.zeros(res, res, dtype=torch.int8)
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
        :return: a tuple of data [(2,17,3) numpy array] and a label [int] and a tuple metadata (subject, frame)
        """
        return self.data[index]

    def convert_to_flickr(self):
        pass

    def fill_no_dets(self):
        pass

    def remove_no_dets(self):
        pass

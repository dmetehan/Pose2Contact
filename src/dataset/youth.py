import os
import json
import logging
import numpy as np
from torch.utils.data import Dataset


class Youth(Dataset):

    def __init__(self, phase, path="data/youth", annot_file_name="pose_detections.json"):
        # TODO: read set_splits and filter data
        self.data = self.read_data(os.path.join(path, "all", annot_file_name))
        self.convert_to_flickr()
        self.remove_ambiguous()
        self.format_data()
        self.fill_no_dets()
        self.datashape = [2, 17, 3]
        self.class_weights = self.calc_class_weights()
        # TODO: Add normalization of the coordinates to [0, 1]

    def calc_class_weights(self):
        labels = [y for _, y in self.data]
        return [1, 1]
        # return [labels.count(1)/len(labels), labels.count(0)/len(labels)]  # reversed instance ratio

    def fill_no_dets(self): self.data = [(np.zeros((2, 17, 3)), item[1]) if len(item[0]) == 0
                                         else ((np.pad(item[0], [(0, 1), (0, 0), (0, 0)]), item[1])
                                               if len(item[0]) == 1 else item) for item in self.data]

    def convert_to_flickr(self): self.data = [{key: self.data[key][item] for key in self.data} for item in self.data['preds']]

    def remove_ambiguous(self): self.data = [self.data[d] for d in range(len(self.data)) if self.data[d]['contact_type'] != '1']  # 0:no contact, 1:ambiguous, 2:contact

    def format_data(self): self.data = [(np.array(self.data[d]['preds']), int(int(self.data[d]['contact_type']) > 0)) for d in range(len(self.data))]

    @staticmethod
    def read_data(annot_file_path):
        logging.info(f'Reading data and annotations from: {annot_file_path}')
        with open(annot_file_path, 'r') as f:
            data = json.load(f)
        logging.info(f'Removing ambiguous contact instances.')
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        :param index:
        :return: a tuple of data [(2,17,3) numpy array] and a label [int]
        """
        return self.data[index]

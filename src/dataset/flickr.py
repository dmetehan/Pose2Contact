import os
import json
import logging
import numpy as np
from torch.utils.data import Dataset


class Flickr(Dataset):

    def __init__(self, phase, path="data/flickr", annot_file_name="pose_detections.json"):
        self.data = self.read_data(os.path.join(path, phase, annot_file_name))
        self.datashape = [2, 17, 3]
        self.class_weights = self.calc_class_weights()

    def calc_class_weights(self):
        labels = [y for _, y in self.data]
        return [1, 4]
        # return [labels.count(1)/len(labels), labels.count(0)/len(labels)]  # reversed instance ratio

    @staticmethod
    def remove_ambiguous(data): return [data[d] for d in range(len(data)) if data[d]['contact_type'] != '1']  # 0:no contact, 1:ambiguous, 2:contact

    @staticmethod
    def format_data(data): return [(np.array(data[d]['preds']), int(int(data[d]['contact_type']) > 0)) for d in range(len(data))]

    def read_data(self, annot_file_path):
        logging.info(f'Reading data and annotations from: {annot_file_path}')
        with open(annot_file_path, 'r') as f:
            data = json.load(f)
        logging.info(f'Removing ambiguous contact instances.')
        data = self.remove_ambiguous(data)
        data = self.format_data(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        :param index:
        :return: a tuple of data [(2,17,3) numpy array] and a label [int]
        """
        return self.data[index]

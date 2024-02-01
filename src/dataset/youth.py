import os
import json
import logging
import numpy as np
import pandas as pd

from .custom_dataset import CustomDataset


class Youth(CustomDataset):

    def __init__(self, phase, path="data/youth", annot_file_name="pose_detections.json"):
        # TODO: read set_splits and filter data
        self.prepare_sets(path, annot_file_name)
        super().__init__(phase, path, annot_file_name)

    def prepare_sets(self, path, annot_file_name):
        if os.path.exists(os.path.join(path, 'train')) and os.path.exists(os.path.join(path, 'test')):
            return
        logging.info(f'Preparing train and test sets for the YOUth datset.')
        with open(os.path.join(path, 'all', 'set_splits.json')) as f:
            set_splits = json.load(f)
            set_splits['trainval'] = set_splits['train'] + set_splits['val']
            data = pd.DataFrame(self.read_data(os.path.join(path, "all", annot_file_name)))
            for _set in ['train', 'val', 'trainval', 'test']:
                data_subset = data[data['crop_path'].str.contains('|'.join(set_splits[_set]))]
                set_path = os.path.join(path, _set)
                os.makedirs(set_path)
                data_subset.to_json(os.path.join(set_path, "pose_detections.json"))

    def fill_no_dets(self): self.data = [(np.zeros((2, 17, 3)), item[1]) if len(item[0]) == 0
                                         else ((np.pad(item[0], [(0, 1), (0, 0), (0, 0)]), item[1])
                                               if len(item[0]) == 1 else item) for item in self.data]

    def convert_to_flickr(self): self.data = [{key: self.data[key][item] for key in self.data} for item in self.data['preds']]

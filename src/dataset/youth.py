import os
import json
import logging
import numpy as np
import pandas as pd

from .custom_dataset import CustomDataset


class Youth(CustomDataset):

    def __init__(self, phase, root_folder, subset="binary", annot_file_name="pose_detections_identity_fixed.json", fold='fold0', **kwargs):
        # raise ValueError("Download pose_detections_identity_fixed.json from drive and recalculate the folds!")
        self.fold = fold
        self.subset = subset
        path = os.path.join(root_folder, subset)
        if subset == 'binary':
            self.prepare_sets(path, annot_file_name)
        elif subset == 'signature':
            self.prepare_folds(path, annot_file_name, "all_signature.json")
        super().__init__(phase, path, annot_file_name, subset, fold, **kwargs)

    def prepare_folds(self, path, data_file_name, annot_file_name):
        if os.path.exists(os.path.join(path, self.fold)):
            return
        logging.info(f'Preparing folds for the YOUth contact signature dataset.')
        folds_file_name = 'folds.json'
        if not os.path.exists(os.path.join(path, 'all', folds_file_name)):
            self.divide_data_into_folds(path, data_file_name, folds_file_name)
        with open(os.path.join(path, 'all', folds_file_name)) as f:
            folds = json.load(f)
            set_splits = {'train': [], 'val': [], 'test': []}
            fold_division = {0: {'test': 0, 'val': 0, 'train': [1, 2, 3, 4]},
                             1: {'test': 0, 'val': 1, 'train': [2, 3, 4]},
                             2: {'test': 0, 'val': 2, 'train': [1, 3, 4]},
                             3: {'test': 0, 'val': 3, 'train': [1, 2, 4]},
                             4: {'test': 0, 'val': 4, 'train': [1, 2, 3]}}
            for f in range(len(folds)):
                set_splits['test'] = folds[fold_division[f]['test']]
                set_splits['val'] = folds[fold_division[f]['val']]
                for train_fold in fold_division[f]['train']:
                    set_splits['train'] += folds[train_fold]
                data = pd.DataFrame(self.read_data(os.path.join(path, "all", data_file_name)))
                data['subject'] = data['crop_path'].apply(lambda x: x.split('/')[-2])
                data['frame'] = data['crop_path'].apply(lambda x: x.split('/')[-1])
                annots = self.read_data(os.path.join(path, "all", annot_file_name))
                for _set in ['train', 'test', 'val']:
                    # Removing the frames with no pose detections!
                    data_subset = data[data['subject'].isin(set_splits[_set])]

                    reg_mapper = self.comb_regs(path, res=6)

                    def add_signature(x):
                        subj, frame = x['crop_path'].split('/')[-2:]
                        x['seg21_adult'] = [elem['adult'] for elem in annots[subj][frame]] if frame in annots[subj] else []
                        x['seg21_child'] = [elem['child'] for elem in annots[subj][frame]] if frame in annots[subj] else []
                        x['seg6_adult'] = [reg_mapper(elem['adult']) for elem in annots[subj][frame]] if frame in annots[subj] else []
                        x['seg6_child'] = [reg_mapper(elem['child']) for elem in annots[subj][frame]] if frame in annots[subj] else []
                        x['signature21x21'] = [(elem['adult'], elem['child'])
                                               for elem in annots[subj][frame]] if frame in annots[subj] else []
                        x['signature6x6'] = [(reg_mapper(elem['adult']), reg_mapper(elem['child']))
                                             for elem in annots[subj][frame]] if frame in annots[subj] else []
                        return x

                    data_subset = data_subset.apply(add_signature, axis=1)
                    set_path = os.path.join(path, f'fold{f}', _set)
                    os.makedirs(set_path)
                    data_subset.to_json(os.path.join(set_path, "pose_detections_identity_fixed.json"))

    def divide_data_into_folds(self, path, pose_detections_file_name, folds_file_name):
        data = pd.DataFrame(self.read_data(os.path.join(path, "all", pose_detections_file_name)))
        data['subject'] = data['crop_path'].apply(lambda x: x.split('/')[-2])
        data['frame'] = data['crop_path'].apply(lambda x: x.split('/')[-1])

        def get_frame_count(fold, annotations):
            return np.sum([len(annotations[annotations['subject'] == subj]) for subj in fold])

        def get_contact_pair_count(fold, annotations):
            return np.sum([np.sum([len(annotations[(annotations['subject'] == subj) & (annotations['frame'] == frame)]['contact_type'].item()) for frame in annotations[annotations['subject'] == subj]['frame']]) for subj in fold])

        def check_fold_diff(all_folds, annotations):
            frame_counts = sorted([get_frame_count(fold, annotations) for fold in all_folds])
            print(frame_counts, frame_counts[-1] - frame_counts[0])
            sample_diff = frame_counts[-1] - frame_counts[0]

            contact_pair_counts = sorted([get_contact_pair_count(fold, annotations) for fold in all_folds])
            print(contact_pair_counts, contact_pair_counts[-1] - contact_pair_counts[0])
            contact_diff = contact_pair_counts[-1] - contact_pair_counts[0]

            return sample_diff, contact_diff

        count = 0
        min_sample_diff, min_contact_diff = 9999999, 9999999
        min_folds = None
        while count < 500:
            folds = np.random.choice(list(data['subject'].unique()), (5, 19), replace=False)
            sample_diff, contact_diff = check_fold_diff(folds, data)
            if sample_diff < min_sample_diff and contact_diff < min_contact_diff:
                min_sample_diff, min_contact_diff = sample_diff, contact_diff
                min_folds = folds
            print(count)
            count += 1
        print(min_sample_diff)
        print(min_contact_diff)
        print(min_folds)
        with open(os.path.join(path, 'all', folds_file_name), 'w') as f:
            json.dump(min_folds.tolist(), f)

    @staticmethod
    def comb_regs(path, res=21):
        assert res in [6, 21]
        if res == 21:
            return lambda x: x
        else:
            mapping = {}
            with open(os.path.join(path, "all", "combined_regions_6.txt"), 'r') as f:
                for i, line in enumerate(f):
                    for reg in list(map(int, map(str.strip, line.strip().split(',')))):
                        mapping[reg] = i
            return lambda x: mapping[x]

    def prepare_sets(self, path, annot_file_name):
        if os.path.exists(os.path.join(path, 'train')) and os.path.exists(os.path.join(path, 'test')):
            return
        logging.info(f'Preparing train and test sets for the YOUth dataset.')
        with open(os.path.join(path, 'all', 'set_splits.json')) as f:
            set_splits = json.load(f)
            set_splits['trainval'] = set_splits['train'] + set_splits['val']
            data = pd.DataFrame(self.read_data(os.path.join(path, "all", annot_file_name)))
            for _set in ['train', 'val', 'trainval', 'test']:
                data_subset = data[data['crop_path'].str.contains('|'.join(set_splits[_set]))]
                set_path = os.path.join(path, _set)
                os.makedirs(set_path)
                data_subset.to_json(os.path.join(set_path, "pose_detections_identity_fixed.json"))

    def fill_no_dets(self):
        self.data = [(np.zeros((2, 17, 3)), item[1]) if len(item[0]) == 0
                     else ((np.pad(item[0], [(0, 1), (0, 0), (0, 0)]), item[1])
                           if len(item[0]) == 1 else item) for item in self.data]

    def convert_to_flickr(self):
        self.data = [{key: self.data[key][item] for key in self.data} for item in self.data['preds']]

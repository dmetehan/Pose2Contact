import os
import json
import logging
import numpy as np

from .custom_dataset import CustomDataset


class Flickr(CustomDataset):

    def __init__(self, phase, path="data/flickr", annot_file_name="pose_detections.json", **kwargs):
        super().__init__(phase, path, annot_file_name)

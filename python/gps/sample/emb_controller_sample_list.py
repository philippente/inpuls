""" This file defines the controller sample class. """

import numpy as np
import copy

from gps.sample.emb_controller_sample import EmbControllerSample


class EmbControllerSampleList(object):
    """
    Class that stores the controller samples.
    Note: must be serializable for easy saving, no C++ references!
    """

    def __init__(self):
        self.sample_list = []

    def add_raw_sample(self, idx, controller):
        """
        set a sample set.
        """
        self.sample_list.append(EmbControllerSample(idx, controller))

    def add_sample(self, emb_controller_sample):
        self.sample_list.append(emb_controller_sample)

    def get_sample(self, idx=None):
        if idx is None:
            idx = range(len(self.sample_list))
        #print("read controller, length: ", len(self.sample_list))
        return self.sample_list[idx]

    def get_list(self):
        return copy.deepcopy(np.array(self.sample_list))

    def get_length(self):
        return len(self.sample_list)

    def reset(self):
        self.sample_list = []



""" This file defines the sample class. """

class EmbControllerSample(object):
    """
    Class that handles the representation of a trajectory and stores a
    single trajectory.
    Note: must be serializable for easy saving, no C++ references!
    """
    def __init__(self, idx, controller):
        self.idx = idx
        self.controller = controller

    def get_idx(self):
        return self.idx

    def get_controller(self):
        return self.controller

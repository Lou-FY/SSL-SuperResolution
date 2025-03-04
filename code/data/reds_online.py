import os
from data.videodata_online import VIDEODATA_ONLINE as VIDEODATA


class REDS_ONLINE(VIDEODATA):
    def __init__(self, args, name='REDS_Online', train=True):
        super(REDS_ONLINE, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_lq = os.path.join(self.apath, 'blurdown_x4')
        print("DataSet gt path:", self.dir_lq)

    def _set_filesystem1(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_lq = os.path.join(self.apath, 'LR')
        print("DataSet gt path:", self.dir_lq)


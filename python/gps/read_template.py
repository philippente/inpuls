""" This file defines the main object that runs experiments. """

import matplotlib as mpl
mpl.use('Qt4Agg')

import logging
import sys

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def main():
    from gps import __file__ as gps_filepath

    BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
    EXP_DIR = BASE_DIR + '/../experiments/jaco_tf_example/'

    data_files_dir = EXP_DIR + 'data_files/'

    data_logger = DataLogger()

    itr = 1
    cond = 0
    print('Reading states (iteration = ' + str(itr) + ', condition = ' + str(cond) + ') ...')


    print('\n')


    #train_sample_lists = data_logger.unpickle(data_files_dir + ('_samplelist_itr%02d.pkl' % itr))
    lqr_sample_lists = data_logger.unpickle(data_files_dir + ('pol_lqr_sample_itr_%02d.pkl' % itr))
    badmm_sample_lists = data_logger.unpickle(data_files_dir + ('pol_badmm_sample_itr_%02d.pkl' % itr))

    print('lqr sample states ' + str(lqr_sample_lists[cond].get_X().shape) + ':')
    print(lqr_sample_lists[cond].get_X())
    print('\n')
    print('badmm sample states ' + str(badmm_sample_lists[cond].get_X().shape) + ':')
    print(badmm_sample_lists[cond].get_X())
    print('\n')


if __name__ == "__main__":
    main()

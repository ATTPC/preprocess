#!/usr/bin python3
"""pyclean
Authors M.P. Kuchera, J. Bradt

This python script parses through a data file and removes noise

This code:
1) Does a nearest neighbor comparison to eliminate statistical noise
2) Does a circular Hough transform to find the center of the spiral or curved track in the micromegas pad plane
3) Does a linear Hough transform on (z,r*phi) to find which points lie along the spiral/curve
4) Writes points and their distance from the line to an HDF5 file

This distance can be used to make cuts on how agressively you want to clean the data.
"""

import numpy as np
import argparse
import pytpc
from pytpc.cleaning import HoughCleaner
from pytpc.fitting.mixins import PreprocessMixin
from pytpc.padplane import generate_pad_plane
from pytpc.utilities import tilt_matrix, rot_matrix
from pytpc.constants import degrees
import sys
import h5py
import yaml
import logging
import logging.config

logger = logging.getLogger(__name__)


def event_iterator(input_evtid_set, output_evtid_set):
    unprocessed_events = input_evtid_set - output_evtid_set
    num_input_evts = len(input_evtid_set)
    num_events_remaining = len(unprocessed_events)
    num_events_finished = len(output_evtid_set)
    if num_events_remaining == 0:
        logger.warning('All events have already been processed.')
        raise StopIteration()
    elif num_events_finished > 0:
        logger.info('Already processed %d events. Continuing from where we left off.', num_events_finished)

    for i in unprocessed_events:
        if i % 100 == 0:
            logger.info('Processed %d / %d events', i, num_input_evts)
        yield i
    else:
        raise StopIteration()


class EventCleaner(HoughCleaner, PreprocessMixin):
    def __init__(self, config):
        super().__init__(config)
        self.tilt = config['tilt'] * degrees
        self.vd = np.array(config['vd'])
        self.clock = config['clock']
        self.pad_rot_angle = config['pad_rot_angle'] * degrees
        self.padrotmat = rot_matrix(self.pad_rot_angle)
        self.beampads = []
        self.untilt_mat = tilt_matrix(self.tilt)
        self.pads = generate_pad_plane(self.pad_rot_angle)

    def process_event(self, evt):
        raw_xyz = evt.xyzs(pads=self.pads, peaks_only=True, return_pads=True, cg_times=True, baseline_correction=True)
        xyz = self.preprocess(raw_xyz, rotate_pads=False)

        cleaning_data = xyz[['u', 'v', 'w']].values

        labels, mindists, nn_counts, (cu, cv) = self.clean(cleaning_data)

        clean_xyz = np.column_stack((raw_xyz, nn_counts, mindists))

        center = np.array([cu, cv, 0])
        center[1] -= np.tan(self.tilt) * 1000.
        cx, cy, _ = self.untilt_mat @ center

        return clean_xyz, (cx, cy)


def setup_logging(config):
    try:
        log_conf = config['logging_config']
        logging.config.dictConfig(log_conf)
    except KeyError:
        logger.exception('Log config failed')


def main():
    parser = argparse.ArgumentParser(description='A script to clean data and write results to an HDF5 file')
    parser.add_argument('config', help='Path to a config file')
    parser.add_argument('input', help='The input evt file')
    parser.add_argument('output', help='The output HDF5 file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    setup_logging(config)

    inFile = pytpc.HDFDataFile(args.input, 'r')

    cleaner = EventCleaner(config)

    with h5py.File(args.output, 'a') as outFile:
        gp = outFile.require_group('clean')
        logger.info('Finding set of event IDs in input')
        input_evtid_set = {k for k in inFile.evtids()}
        num_input_evts = len(input_evtid_set)
        logger.info('Input file contains %d events', num_input_evts)

        output_evtid_set = {int(k) for k in gp}

        for evt_index in event_iterator(input_evtid_set, output_evtid_set):
            try:
                evt = inFile[evt_index]
            except Exception:
                logger.exception('Failed to read event with index %d from input', evt_index)
                continue

            try:
                clean_xyz, center = cleaner.process_event(evt)
            except Exception:
                logger.exception('Cleaning failed for event with index %d', evt_index)
                continue

            try:
                dset = gp.create_dataset('{:d}'.format(evt.evt_id), data=clean_xyz, compression='gzip', shuffle=True)
                dset.attrs['center'] = center
            except Exception:
                logger.exception('Writing to HDF5 failed for event with index %d', evt_index)
                continue


if __name__ == '__main__':
    import signal

    def handle_signal(signum, stack_frame):
        logger.critical('Received signal %d. Quitting.', signum)
        sys.stdout.flush()
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGQUIT, handle_signal)

    main()

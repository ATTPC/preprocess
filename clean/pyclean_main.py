import pyclean
import argparse
import pytpc
import sys
import numpy as np
import h5py
from math import sin, cos
from pytpc.constants import pi, degrees
import pytpc.simulation

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def event_iterator(num_input_evts, num_output_evts):
    num_events_remaining = num_input_evts - num_output_evts
    if num_events_remaining == 0:
        logger.warning('All events have already been processed.')
        raise StopIteration()
    elif num_output_evts > 0:
        logger.info('Already processed %d events. Continuing from where we left off.', num_output_evts)

    first_evt = num_output_evts

    for i in range(first_evt, num_input_evts):
        if i % 100 == 0:
            logger.info('At event %d / %d', i, num_input_evts)
        yield i
    else:
        raise StopIteration()


def main():
    parser = argparse.ArgumentParser(
        description='A script to clean data and write results to an HDF5 file')
    parser.add_argument('input', help='The input evt file')
    parser.add_argument('output', help='The output HDF5 file')
    args = parser.parse_args()

    v_drift = pytpc.simulation.drift_velocity_vector(-5.2, 9000, 1.75, 0.10472)
    tmat = pytpc.utilities.tilt_matrix(-0.10472)
    inFile = pytpc.HDFDataFile(args.input, 'r')
    pad_plane = pytpc.generate_pad_plane(rotation_angle=-108 * degrees)
    un_tmat = pytpc.utilities.tilt_matrix(0.10472)
    un_rotate_ang = 108 * degrees
    rot = [cos(un_rotate_ang), -sin(un_rotate_ang)
           ], [sin(un_rotate_ang), cos(un_rotate_ang)]

    with h5py.File(args.output, 'a') as outFile:
        gp = outFile.require_group('clean')
        logger.info('Finding length of input file...')
        num_input_evts = len(inFile)
        logger.info('Input file contains %d events', num_input_evts)

        num_output_evts = len(gp)

        for evt_index in event_iterator(num_input_evts, num_output_evts):
            try:
                evt = inFile[evt_index]
                raw_xyz = evt.xyzs(pads=pad_plane, peaks_only=True, return_pads=True,
                                   cg_times=True, baseline_correction=True)
                uvw = pytpc.evtdata.calibrate(raw_xyz, v_drift, 12.5)
                uvw = np.dot(tmat, uvw[:, :3].T).T
                clean_uvw, center_uv = pyclean.clean(uvw)
                nearest_neighbors = pyclean.nearest_neighbor_count(uvw, 40)
                clean_xyz = np.column_stack((raw_xyz, nearest_neighbors, clean_uvw[:, -1]))
                gp = outFile.require_group('clean')
                deset = gp.create_dataset('{:d}'.format(evt.evt_id), data=clean_xyz, compression='gzip', shuffle=True)
                cc = [center_uv[0], center_uv[1], 0]
                cc = np.dot(un_tmat, cc)
                cc = np.dot(rot, cc[:2].T).T
                deset.attrs['center'] = cc
            except(KeyError) as err:
                print(err)
                continue
    return 0


if __name__ == '__main__':
    import signal

    def handle_signal(signum, stack_frame):
        logger.critical('Received signal %d. Quitting.', signum)
        sys.stdout.flush()
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGQUIT, handle_signal)

    main()

"""
\file pyclean.py
\author M.P. Kuchera

\brief This python script parses through a data file and removes noise

This code:
1) Does a nearest neighbor comparison to eliminate statistical noise
2) Does a circular Hough transform to find the center of the spiral or curved track in the micromegas pad plane
3) Does a linear Hough transform on (z,r*phi) to find which points lie along the spiral/curve
4) Writes points and their distance from the line to an HDF5 file

This distance can be used to make cuts on how agressively you want to clean the data.

Note: distances from line != distance from the spiral path.
This is because the relationship between (z,r*phi) is only approximately linear. Use care when using this value.
E.g. for 46Ar experiment, 8mm was a good cutoff for this paramemter, giving quite clean results.
"""

import numpy as np
import math
from math import sin, cos
import argparse
import pytpc
import sys
import h5py
from pytpc.constants import degrees
import pytpc.simulation

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logging.captureWarnings(True)


def nearest_neighbor_count(data, radius):
    return np.sum((np.sqrt(np.sum((data[:, None, :3] - data[:, :3])**2, -1))
                   <= radius) & ~np.eye(len(data), dtype='bool'), axis=0)


def hough_circle(xyzs):
    xyz_order = xyzs[np.argsort(xyzs[:, 2])]

    nbins = 200  # number of bins in r and theta for Hough space discretization
    ths = np.linspace(0, np.pi, nbins)

    npts = len(xyz_order) - 5
    sqdels = xyz_order[5:, :2]**2 - xyz_order[:-5, :2]**2
    deltas_tiled = np.tile(xyz_order[5:, :2] - xyz_order[:-5, :2], (nbins, 1))

    Hrad = np.empty((nbins * npts, 2))

    Hrad[:, 0] = np.repeat(ths, npts)
    Hrad[:, 1] = (np.tile(np.sum(sqdels, -1), nbins) /
                  (2 * (deltas_tiled[:, 0] * np.cos(Hrad[:, 0])
                        + deltas_tiled[:, 1] * np.sin(Hrad[:, 0]))))

    Hrad = Hrad[1:]

    # uncomment this line for viewing Hough space histogram (computationally slower)
    # countsRad, xedgesRad, yedgesRad, ImageRad = plt.hist2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-500,500]],cmap=plt.cm.jet)

    # comment out this line when viewing Hough space histogram (this would be
    # redundant with previous line)
    countsRad, xedgesRad, yedgesRad = np.histogram2d(
        Hrad[:, 0], Hrad[:, 1], nbins, range=[[0, math.pi], [-500, 500]])

    # max bin of histogram
    iRad, jRad = np.unravel_index(countsRad.argmax(), countsRad.shape)
    # convert max bin to theta, radius values
    tRad = iRad * math.pi / nbins
    rRad = jRad * 1000 / nbins - 500

    # convert theta, r to postions in xy space
    ax = rRad * math.cos(tRad)
    by = rRad * math.sin(tRad)

    return ax, by


def min_distance(point, array):
    d = []
    for a in array:
        # print("a = ", a)
        d.append(math.sqrt((a[0] - point[0])**2 + (a[1] - point[1])**2))
        # i += 1
    return min(d)


def find_good_points(counts, thh, rad):
    N = 100

    ind = np.argpartition(counts.flatten(), -N)[-N:]
    indices = np.vstack(np.unravel_index(ind, counts.shape)).T

    r_signal = [0, 0]
    r_signal2 = [0, 0]
    r_temp = thh

    nbins = 500
    tRad = indices[:, 0] * math.pi / nbins
    rRad = indices[:, 1] * 4000 / nbins - 2000

    test = np.zeros(indices.shape)
    test[:, 0] = indices[:, 0] * math.pi / nbins
    test[:, 1] = indices[:, 1] * 4000 / nbins - 2000

    x = np.linspace(50, 1600, 100)
    y = np.zeros(100)

    sig_i = []
    sig_i2 = []

    dist = np.zeros(thh[:, 1].shape)
    dist += 1000  # initialize all values to 1000

    for idx, rr in enumerate(rRad):
        mindist = 1000
        if(idx > 1):
            d = test[:idx]
            current = (tRad[idx], rRad[idx])
            mindist = min_distance(current, d)
        # want Hough lines sufficiently far away
        if(mindist > 25):
            y_temp = (rRad[idx] - x * math.cos(tRad[idx])) / \
                math.sin(tRad[idx])
            y = np.vstack((y, y_temp))
            b = math.sin(tRad[idx])
            c = -rr
            a = math.cos(tRad[idx])

            y2 = (c / b) - a * x / b

            idxx = []
            idxx2 = []
            indd = 0
            for rr in thh:
                yy = (a * (-b * rr[0] + a * rr[1]) - b * c) / (a**2 + b**2)
                xx = (b * (b * rr[0] - a * rr[1]) - a * c) / (a**2 + b**2)
                d = math.sqrt((xx - rr[0])**2 + (yy - rr[1])**2)

                if(d < 25):
                    idxx.append(indd)
                    if ((indd in sig_i) is False):
                        sig_i.append(indd)
                        r_signal = np.vstack((r_signal, rr))
                indd = indd + 1
            if(b == 0):
                mm = 1000
            else:
                mm = a / b
            A = np.vstack([thh[idxx, 0], np.ones(len(thh[idxx, 0]))]).T
            if(A.any()):
                m, yint = np.linalg.lstsq(A, thh[idxx, 1])[0]
                if(m > 0):
                    indd2 = 0
                    for rr in thh:
                        yy = (rr[0] * m + rr[1] * m * m + yint) / (m * m + 1)
                        xx = (rr[0] + rr[1] * m - yint * m) / (m * m + 1)
                        d = math.sqrt((xx - rr[0])**2 + (yy - rr[1])**2)
                        if(d < dist[indd2]):
                            dist[indd2] = d
                        if(d < 8):
                            idxx2.append(indd2)
                            if ((indd2 in sig_i2) is False):
                                sig_i2.append(indd2)
                                r_signal2 = np.vstack((r_signal2, rr))
                        indd2 = indd2 + 1
    # print(len(r_signal2))
    if(len(r_signal2) > 2):
        r_signal2 = r_signal2[1:]

        possible_inner = np.zeros(rad.shape)
        possible_inner[:, 0] = rad[:, 0]
        possible_inner[:, 1] = thh[:, 1] - math.pi * rad[:, 1]
        possible_outer = np.zeros(rad.shape)
        possible_outer[:, 0] = rad[:, 0]
        possible_outer[:, 1] = thh[:, 1] + math.pi * rad[:, 1]

        ordered = r_signal2[np.argsort(r_signal2[:, 0])]
        A_end = np.vstack([ordered[:20, 0], np.ones(len(ordered[:20, 0]))]).T
        m_end, yint_end = np.linalg.lstsq(A_end, ordered[:20, 1])[0]
        A_begin = np.vstack(
            [ordered[-20:, 0], np.ones(len(ordered[-20:, 0]))]).T
        m_begin, yint_begin = np.linalg.lstsq(A_begin, ordered[-20:, 1])[0]

        indd = 0
        for rr in possible_inner:
            try:
                if(rr[0] > (ordered[2, 0] - 50) and rr[0] < (ordered[2, 0] + 50)):
                    yy = (rr[0] * m_end + rr[1] * m_end *
                          m_end + yint_end) / (m_end * m_end + 1)
                    xx = (rr[0] + rr[1] * m_end - yint_end *
                          m_end) / (m_end * m_end + 1)
                    d = math.sqrt((xx - rr[0])**2 + (yy - rr[1])**2)
                    if(d < dist[indd]):
                        dist[indd] = d
                        if(d < 22):
                            idxx2.append(indd)
                            if ((indd in sig_i2) is False):
                                sig_i2.append(indd)
                                indd = indd + 1
            except IndexError:
                print("IndexError")
        indd = 0
        for rr in possible_outer:
            try:
                if(rr[0] > (ordered[-1, 0] - 50) and rr[0] < (ordered[-1, 0] + 50)):
                    yy = (rr[0] * m_begin + rr[1] * m_begin *
                          m_begin + yint_begin) / (m_begin * m_begin + 1)
                    xx = (rr[0] + rr[1] * m_begin - yint_begin *
                          m_begin) / (m_begin * m_begin + 1)
                    d = math.sqrt((xx - rr[0])**2 + (yy - rr[1])**2)
                    if(d < dist[indd]):
                        dist[indd] = d
                        if(d < 22):
                            idxx2.append(indd)
                            if ((indd in sig_i2) is False):
                                sig_i2.append(indd)
                                r_signal2 = np.vstack((r_signal2, rr))

                                indd = indd + 1
            except IndexError:
                print("IndexError")
    return sig_i2, rRad, tRad, dist


def hough_line(xy):
    """ Finds lines in data

    """
    nbins = 500  # number of bins for r, theta histogram
    max_val = 2000
    th = np.linspace(0, math.pi, nbins)
    Hrad = np.empty((nbins * len(xy), 2))

    Hrad[:, 0] = np.repeat(th, len(xy))
    Hrad[:, 1] = np.tile(xy[:, 0], nbins) * np.cos(Hrad[:, 0]) + \
        np.tile(xy[:, 1], nbins) * np.sin(Hrad[:, 0])

    # Hrad = Hrad[1:]

    # countsRad, xedgesRad, yedgesRad, ImageRad = plt.hist2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-500,500]],cmap=plt.cm.jet)
    # countsRad, xedgesRad, yedgesRad, ImageRad = plt.hist2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-2000,2000]],cmap=plt.cm.jet)
    countsRad, xedgesRad, yedgesRad = np.histogram2d(
        Hrad[:, 0], Hrad[:, 1], nbins, range=[[0, math.pi], [-max_val, max_val]])
    iRad, jRad = np.unravel_index(countsRad.argmax(), countsRad.shape)

    tRad = iRad * math.pi / nbins
    rRad = jRad * max_val * 2 / nbins - max_val

    return rRad, tRad, countsRad


def clean(xyz):
    a, b = hough_circle(xyz)
    # print("center found at ", a, b)

    rad_z = np.zeros((len(xyz), 2))
    th_z = np.zeros((len(xyz), 2))

    idx = 0
    for xy in xyz:
        r_xy = np.sqrt((xy[0] - a)**2 + (xy[1] - b)**2)
        rad_z[idx, :] = [xy[2], r_xy]
        th_xy = np.arctan((xy[1] - b) / (xy[0] - a))
        th_z[idx, :] = [xy[2], th_xy]
        idx += 1

    r_th = np.zeros(th_z.shape)

    r_th[:, 0] = th_z[:, 0]
    r_th[:, 1] = th_z[:, 1] * rad_z[:, 1]

    r, t, counts = hough_line(r_th)
    sig_i, rRad, tRad, dist = find_good_points(counts, th_z, rad_z)

    # append distances to data and return!
    clean_xyz = np.column_stack((xyz, dist))
    center = [a, b]

    return clean_xyz, center


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


class EventCleaner(object):
    def __init__(self):
        self.v_drift = pytpc.simulation.drift_velocity_vector(-5.2, 9000, 1.75, 0.10472)
        self.tmat = pytpc.utilities.tilt_matrix(-0.10472)
        self.un_tmat = pytpc.utilities.tilt_matrix(0.10472)
        self.un_rotate_ang = 108 * degrees
        self.pad_plane = pytpc.generate_pad_plane(rotation_angle=-108 * degrees)
        self.rot = np.array([[cos(self.un_rotate_ang), -sin(self.un_rotate_ang)],
                             [sin(self.un_rotate_ang), cos(self.un_rotate_ang)]])

    def process_event(self, evt):
        raw_xyz = evt.xyzs(pads=self.pad_plane, peaks_only=True, return_pads=True,
                           cg_times=True, baseline_correction=True)
        uvw = pytpc.evtdata.calibrate(raw_xyz, self.v_drift, 12.5)
        uvw = np.dot(self.tmat, uvw[:, :3].T).T
        clean_uvw, center_uv = clean(uvw)
        nearest_neighbors = nearest_neighbor_count(uvw, 40)
        clean_xyz = np.column_stack((raw_xyz, nearest_neighbors, clean_uvw[:, -1]))

        cc = [center_uv[0], center_uv[1], 0]
        cc = np.dot(self.un_tmat, cc)
        cc = np.dot(self.rot, cc[:2].T).T

        return clean_xyz, cc


def main():
    parser = argparse.ArgumentParser(
        description='A script to clean data and write results to an HDF5 file')
    parser.add_argument('input', help='The input evt file')
    parser.add_argument('output', help='The output HDF5 file')
    args = parser.parse_args()

    inFile = pytpc.HDFDataFile(args.input, 'r')

    cleaner = EventCleaner()

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
                clean_xyz, cc = cleaner.process_event(evt)
                deset = gp.create_dataset('{:d}'.format(evt.evt_id), data=clean_xyz, compression='gzip', shuffle=True)
                deset.attrs['center'] = cc
            except(KeyError) as err:
                print(err)
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

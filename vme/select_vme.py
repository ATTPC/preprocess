import numpy as np
import pandas as pd
import h5py
from scipy.signal import argrelextrema
import argparse

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def find_peaks(signal, noise_threshold):
    mkd = np.ma.masked_less(signal, noise_threshold).filled(np.nan)

    pks = argrelextrema(mkd, comparator=np.greater_equal, order=5, mode='clip')[0]
    adj = pks[np.where(np.diff(pks) == 1)]
    if len(adj) > 0:
        adj_chunks = [np.append(a, a.max() + 1) for a in np.split(adj, np.where(np.diff(adj) > 1)[0] + 1)]
        broad_pks = np.round(np.array([a.mean() for a in adj_chunks]))
        narrow_pks = np.setdiff1d(pks, np.concatenate(adj_chunks))
        pks = np.concatenate((broad_pks, narrow_pks))

    return pks


def cfd_trigger(sig, delay, frac, noise_thresh):
    cfd = sig - frac * np.roll(sig, delay)
    # When finding the zero crossings, omit actual zeros since they will otherwise be
    # counted twice because of the way np.sign works.
    zero_crossings = np.logical_and(np.diff(np.sign(cfd)) != 0, cfd[:-1] != 0)
    return np.logical_and(zero_crossings, np.abs(sig[:-1]) > noise_thresh).nonzero()[0] + 1


def coincidence_cut(coinc):
    return np.logical_and(np.any(coinc[:, :10], axis=1), coinc[:, 10])


def saturation_cut(ic):
    return ic.min(axis=1) > 0


def trigger_cut(trig):
    return np.any(trig < 250, axis=1)


def select_events(ic, goodidx):
    pkdata = {}
    dropped_evts_maxes = 0
    dropped_evts_cfd = 0
    for i, (evtid, sig) in enumerate(zip(goodidx, ic[goodidx])):
        if i % 1000 == 0:
            logger.info('At event %d / %d', i, len(goodidx))
        sig = sig.astype('float64')
        sig = -sig + np.median(sig)
        pks = find_peaks(sig, 600)
        cfd_zeros = cfd_trigger(sig, -10, 0.4, 500)
        if len(pks) != 1:
            dropped_evts_maxes += 1
            continue
        else:
            if len(cfd_zeros) != 1:
                logger.warn('Evt %d: CFD found %d peaks instead of 1. Dropping event.', evtid, len(cfd_zeros))
                dropped_evts_cfd += 1
                continue
            pkdata[evtid] = {'height': sig[int(pks[0])], 'pk_pos': pks[0], 'cfd_pos': cfd_zeros[0]}

    logger.info('Dropped %d events that had more than one peak', dropped_evts_maxes)
    logger.warn('Dropped %d events due to bad CFD filter', dropped_evts_cfd)
    return pd.DataFrame.from_dict(pkdata, orient='index')


def main():
    parser = argparse.ArgumentParser('A script to select good events based on VME data.')
    parser.add_argument('vme_data', help='Path to an HDF5 file containing the VME data')
    parser.add_argument('outpath', help='Path to an HDF5 file to write')
    args = parser.parse_args()

    with h5py.File(args.vme_data, 'r') as h5file:
        coinc = h5file['/vme/coinc'][:]
        logger.info('Total number of events is %d', coinc.shape[0])
        logger.info('Performing coincidence cut')
        coinccut = coincidence_cut(coinc)
        del coinc
        logger.info('%d events were cut', np.where(~coinccut)[0].shape[0])

        logger.info('Performing cut on events with no trigger signal')
        trig = h5file['/vme/trig'][:]
        trigcut = trigger_cut(trig)
        del trig
        logger.info('%d events were cut', np.where(~trigcut)[0].shape[0])

        logger.info('Performing cut on IC events that saturate the ADC')
        ic = h5file['/vme/ic'][:]
        satcut = saturation_cut(ic)
        logger.info('%d events were cut', np.where(~satcut)[0].shape[0])

        cut = np.all(np.column_stack((coinccut, trigcut, satcut)), axis=1)
        idx = np.where(cut)[0]
        logger.info('After cuts, %d good events remain (%d were cut)', idx.shape[0], np.where(~cut)[0].shape[0])

    logger.info('Finding single peaks for remaining events')
    goodvme = select_events(ic, idx)
    logger.info('Finished. Found %d single peaks.', len(goodvme))
    goodvme.to_hdf(args.outpath, 'single_ic_pk_heights', format='t')


if __name__ == '__main__':
    import signal
    import sys

    def handle_signal(signum, stack_frame):
        logger.critical('Received signal %d. Quitting.', signum)
        sys.stdout.flush()
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGQUIT, handle_signal)

    main()

"""
unpack_vme.py

A script for unpacking VME data.

This uses the vmedata module from pytpc to rewrite the VME data from its raw form
into HDF5 files. The resulting HDF5 files have a group called "vme" in their root,
with four datasets in that group. The datasets contain the ion chamber signals ('ic'),
the mesh signal ('mesh'), the trigger pulse ('trig'), and the coincidence register ('coinc').
"""

import numpy as np
import h5py
from clint.textui import progress
import pytpc.vmedata
import argparse


def parse_coinc_reg(cr):
    """Parse the integer coincidence register into an array of booleans.

    Parameters
    ----------
    cr : int
        The coincidence register from the VME data.

    Returns
    -------
    np.array
        The 16 boolean coincidence register bits.
    """
    return np.array([(cr & 1 << i) > 0 for i in range(16)])


def unpack(inpath, outpath):
    """Unpack the data from the vme file at `inpath` into the HDF5 file at `outpath`.

    Parameters
    ----------
    inpath : string
        Path to the raw input VME file. This will be parsed using the pytpc.vmedata module.
    outpath: string
        Path to an HDF5 file to write the output datasets to. This will be opened in 'append' mode,
        so if the file exists, a new group will be added to it for the VME data. If the
        VME datasets already exist in the file, this function will fail, so they should
        not be overwritten.
    """
    vmef = pytpc.vmedata.VMEFile(inpath)
    file_len = vmef.fp.seek(0, 2)
    vmef.fp.seek(0)

    with h5py.File(outpath, 'a') as h5file:
        vmegp = h5file.require_group('/vme')

        meshds = vmegp.create_dataset('mesh', (10, 512), maxshape=(None, 512), dtype='uint16',
                                      compression='gzip', shuffle=True)
        icds = vmegp.create_dataset('ic', (10, 512), maxshape=(None, 512), dtype='uint16',
                                    compression='gzip', shuffle=True)
        trigds = vmegp.create_dataset('trig', (10, 512), maxshape=(None, 512), dtype='uint16',
                                      compression='gzip', shuffle=True)
        coincds = vmegp.create_dataset('coinc', (10, 16), maxshape=(None, 16), dtype='uint8',
                                       compression='gzip', shuffle=True)

        print('Reading and unpacking events from file')
        max_evtid = 0
        with progress.Bar(label='File position: ', expected_size=file_len) as bar:
            for vmevt in vmef:
                if vmevt['type'] != 'adc':
                    continue

                evtid = vmevt['evt_num']
                max_evtid = max(evtid, max_evtid)
                adc = np.roll(vmevt['adc_data'], -vmevt['last_tb'], axis=1)
                mesh = adc[0]
                ic = adc[1]
                trig = adc[2]
                coinc = parse_coinc_reg(vmevt['coin_reg'])

                for ds, data in zip((meshds, icds, trigds, coincds), (mesh, ic, trig, coinc)):
                    if evtid > ds.shape[0] - 1:
                        current_size = ds.shape[0]
                        ds.resize(current_size + max(abs(evtid - current_size), 10), axis=0)

                    ds[evtid] = data

                bar.show(vmef.fp.tell())

            # Clean up extra rows
            for ds in (meshds, icds, trigds, coincds):
                if ds.shape[0] > max_evtid + 1:
                    ds.resize(max_evtid + 1, axis=0)


def main():
    parser = argparse.ArgumentParser('A program for unpacking VME data from the AT-TPC')
    parser.add_argument('input', help='Path to input VME file')
    parser.add_argument('output', help='Path to output HDF file')
    args = parser.parse_args()

    unpack(args.input, args.output)

if __name__ == '__main__':
    import signal
    import sys

    def handle_signal(signum, stack_frame):
        print('Received signal %d. Quitting.', signum)
        sys.stdout.flush()
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGQUIT, handle_signal)

    main()

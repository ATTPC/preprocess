
import pyclean 
import argparse
import pytpc
import sys
import numpy as np
import h5py
#import math
from math import sin, cos
from pytpc.constants import pi, degrees

import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='A script to clean data and write results to an HDF5 file')
    parser.add_argument('input', help='The input evt file')
    parser.add_argument('output', help='The output HDF5 file')
    args = parser.parse_args()

    v_drift = pytpc.simulation.drift_velocity_vector(-5.2,9000,1.75,0.10472)
    tmat = pytpc.utilities.tilt_matrix(-0.10472)
    inFile =  pytpc.HDFDataFile(args.input, 'r')
    pad_plane = pytpc.generate_pad_plane(rotation_angle=-108*degrees)
    un_tmat = pytpc.utilities.tilt_matrix(0.10472)
    un_rotate_ang = 108*degrees
    rot = [cos(un_rotate_ang), -sin(un_rotate_ang)],[sin(un_rotate_ang), cos(un_rotate_ang)]
    extra = 0
    print(inFile)
    with h5py.File(args.output, 'a') as outFile:
        
        gp = outFile.require_group('clean')
        n_evts = len(inFile)
        start = 0
        if(len(gp)>0):
            finished_evts = len(gp)
            print(finished_evts)
            evts_to_process = n_evts - finished_evts
            if evts_to_process > 0:
                print('Already processed {} events. Continuing from where we left off.'.format(finished_evts))
                start = finished_evts
            else:
                print('All events have already been processed.')
                sys.exit(0)
        else:
            evts_to_process = n_evts
        #for i in range(evts_to_process):
        while i < evts_to_process:
            try:
                evt = inFile[start+i+extra]
                if((start+i)%1000 == 0):
                    print(start+i)            
                    raw_xyz = evt.xyzs(pads=pad_plane, peaks_only=True, return_pads=True,cg_times=True)
                    
                    uvw = pytpc.evtdata.calibrate(raw_xyz,v_drift,12.5)
                    uvw = np.dot(tmat,uvw[:,:3].T).T
                    clean_uvw, center_uv = pyclean.clean(uvw)
                    nearest_neighbors = pyclean.nearest_neighbor_count(uvw,40) 
                    clean_xyz = np.column_stack((raw_xyz,nearest_neighbors,clean_uvw[:,-1]))
                    gp = outFile.require_group('clean')
                    deset = gp.create_dataset('{:d}'.format(evt.evt_id), data=clean_xyz, compression='gzip', shuffle=True)     
                    
                    cc = [center_uv[0],center_uv[1],0]
                    cc = np.dot(un_tmat,cc)  
                    cc = np.dot(rot,cc[:2].T).T
                    deset.attrs['center'] = cc
            except(KeyError):
                evts_to_process+=1
                continue
                #i-=1
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

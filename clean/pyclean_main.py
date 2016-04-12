
import pyclean 
import argparse
import pytpc
#import math
import numpy as np
import h5py
from pytpc.constants import pi, degrees

def main():
    parser = argparse.ArgumentParser(description='A script to clean data and write results to an HDF5 file')
    parser.add_argument('input', help='The input evt file')
    parser.add_argument('output', help='The output HDF5 file')
    args = parser.parse_args()

    
    #filename = '../../remove-noise/run_0150.evt'
    #efile = pytpc.EventFile(args.input,'r')
    #with pytpc.HDFDataFile(args.output, 'a') as hfile:
     #   gp = hfile.fp.require_group(hfile.get_group_name)
    v_drift = pytpc.simulation.drift_velocity_vector(-5.2,9000,1.75,0.10472)
    tmat = pytpc.utilities.tilt_matrix(-0.10472)
    inFile =  pytpc.HDFDataFile(args.input, 'r')
    pad_plane = pytpc.generate_pad_plane(rotation_angle=-108*degrees)
    print(inFile)
    with h5py.File(args.output, 'a') as outFile:
        
        #print(hfile.keys())
        #gp = hfile['get']
        #print(len(gp))
        gp = outFile.require_group('clean')
        n_evts = len(inFile)
        #print(gp.keys())
        start = 0
        if(len(gp)>0):
            print(len(gp))
            #finished_evts = set(int(k) for k in gp.keys() if k.isdigit())
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
        for i in range(evts_to_process):
            evt = inFile[start+i]
            print(start+i)
           # pad_plane = pytpc.generate_pad_plane(rotation_angle=-108*degrees)
        #print(len(gp))
            raw_xyz = evt.xyzs(pads=pad_plane, peaks_only=True, return_pads=True,cg_times=True)
            #v_drift = pytpc.simulation.drift_velocity_vector(-5.2,9000,1.75,0.10472)
            
            uvw = pytpc.evtdata.calibrate(raw_xyz,v_drift,12.5)
            #tmat = pytpc.utilities.tilt_matrix(-0.10472)
            uvw = np.dot(tmat,uvw[:,:3].T).T
            clean_uvw, center_uv = pyclean.clean(uvw)
            nearest_neighbors = pyclean.nearest_neighbor_count(uvw,40) 
            clean_xyz = np.column_stack((raw_xyz,nearest_neighbors,clean_uvw[:,-1]))
            gp = outFile.require_group('clean')
            deset = gp.create_dataset('{:d}'.format(evt.evt_id), data=clean_xyz, compression='gzip', shuffle=True)
            center_uvw = np.hstack((center_uv,0))
            #print(center_uvw)
            center_xyz = np.dot(tmat.T,center_uvw)
            #print(center_xyz)
            deset.attrs['center'] = center_xyz[:2]
 
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

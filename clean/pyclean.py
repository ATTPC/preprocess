#-*- mode: python -*-

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
This is because the relationship between (z,r*phi) is only approximately linear. Use care when using this value. E.g. for 46Ar experiment, 8mm was a good cutoff for this paramemter, giving quite clean results.
"""

import sys
import numpy as np
import pytpc
import math
import h5py
import argparse

"""
def smart_progress_bar(iterable, expected_size=None):
    if not sys.stdout.isatty():

        def print_progress(iterable, expected_size=None):
            from math import log10, floor

            if expected_size is None:
                max_ = len(iterable)
            else:
                max_ = expected_size

            if max_ > 100:
                interval = 10**floor(log10(max_ // 100))
            else:
                interval = 1

            for i, v in enumerate(iterable):
                if i % interval == 0:
                    print('At event {} / {}'.format(i, max_), flush=True)
                yield v

        return print_progress(iterable, expected_size)

    else:
        from clint.textui import progress
        return progress.bar(iterable, expected_size=expected_size)
"""
def hough_circle(xyzs):
    nbins = 200 #number of bins in r and theta for Hough space discretization
    xyz_order = xyzs[np.argsort(xyzs[:,2])]
    th = np.linspace(0,math.pi,nbins)
    Hrad = [0,0]
    xyz_order_f = xyz_order[5:]
    xyz_order = xyz_order[:-5]

    # sweep through theta, calculate r and bin histogram Hrad
    for theta in th:
        Radius = (xyz_order_f[:,0]**2 - xyz_order[:,0]**2+ xyz_order_f[:,1]**2- xyz_order[:,1]**2)/(2*((xyz_order_f[:,0]- xyz_order[:,0])*np.cos(theta)+(xyz_order_f[:,1]- xyz_order[:,1])*np.sin(theta)))
        index = 0
        for rr in Radius:  
            aRad = np.hstack((theta,Radius[index]))
            Hrad = np.vstack((Hrad,aRad))
            index +=1

    Hrad = Hrad[1:]
    
    # uncomment this line for viewing Hough space histogram (computationally slower)
    #countsRad, xedgesRad, yedgesRad, ImageRad = plt.hist2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-500,500]],cmap=plt.cm.jet) 

    # comment out this line when viewing Hough space histogram (this would be redundant with previous line)
    countsRad, xedgesRad, yedgesRad = np.histogram2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-500,500]])

    # max bin of histogram
    iRad,jRad = np.unravel_index(countsRad.argmax(), countsRad.shape)
    #convert max bin to theta, radius values
    tRad = iRad*math.pi/nbins
    rRad = jRad*1000/nbins - 500
   
    #convert theta, r to postions in xy space
    ax =  rRad*math.cos(tRad)  
    by =  rRad*math.sin(tRad)
  
    return ax,by
def min_distance(point, array):
    d = []
    for a in array:
        #print("a = ", a)
        d.append(math.sqrt((a[0]-point[0])**2 + (a[1]-point[1])**2))
        #i += 1
    return min(d)

def find_good_points(counts, r_th):
    N = 100

    ind = np.argpartition(counts.flatten(),-N)[-N:]
    indices = np.vstack(np.unravel_index(ind,counts.shape)).T

    r_signal = [0,0]
    r_signal2 = [0,0]
    r_temp = r_th

    nbins = 500
    tRad = indices[:,0]*math.pi/nbins
    rRad = indices[:,1]*4000/nbins - 2000   
    
    test = np.zeros(indices.shape)
    test[:,0] = indices[:,0]*math.pi/nbins
    test[:,1] = indices[:,1]*4000/nbins - 2000

    x = np.linspace(50,1600,100)
    y = np.zeros(100)
    plt.plot(r_th[:,0],r_th[:,1],'r.')
    sig_i = []
    sig_i2 = []
    for idx, rr in enumerate(rRad): 
        #if not((rr in rRad[:idx]) or (np.where(np.logical_and(rRad[:idx]>rr-3, rRad[:idx]<rr+3)))):
        #if not(((rRad[:idx] > rr-5).any() and (rRad[:idx] < rr+5).any()) ):
        mindist = 1000
        if(idx > 1):   
            d = test[:idx]
            current = (tRad[idx],rRad[idx])
            mindist = min_distance(current,d)
            #print("mindist = " , mindist)
        if(mindist > 25):
        #if not(((tRad[:idx] > tRad[idx]-5).any() and (tRad[:idx] < tRad[idx]+5).any()) ):
        #np.where((rRad[:idx] > rr-5) and (rRad[:idx] < rr+5)))
        #if not(rr in rRad[:idx]):
           # print(rr, tRad[idx])
            y_temp= (rRad[idx]-x*math.cos(tRad[idx]))/math.sin(tRad[idx])
            y = np.vstack((y,y_temp))
            #plt.plot(x,y_temp,'-')
            b = math.sin(tRad[idx])
            c = -rr
            a = math.cos(tRad[idx])

            y2 = (c/b)-a*x/b
            
            idxx = []
            idxx2 = []
            indd = 0;
            for rr in r_th:       
                yy = (a*(-b*rr[0]+a*rr[1])-b*c)/(a**2+b**2)
                xx = (b*(b*rr[0]-a*rr[1])-a*c)/(a**2+b**2)
    
                if(math.sqrt((xx-rr[0])**2 +(yy-rr[1])**2) < 25):
                    plt.plot(rr[0],rr[1],'g.')
                    idxx.append(indd)
                    if ((indd in sig_i) is False):
                        sig_i.append(indd)
                        r_signal = np.vstack((r_signal,rr))
                indd = indd+1
            
            mm = a/b
            A = np.vstack([r_th[idxx,0], np.ones(len(r_th[idxx,0]))]).T
            if(A.any()):
                m, yint = np.linalg.lstsq(A, r_th[idxx,1])[0]
                if(m>0):
                    plt.plot(x, m*x+ yint, '-r')
                    indd2 = 0;
                    for rr in r_th:       
                        yy = (rr[0]*m + rr[1]*m*m+yint)/(m*m+1)
                        xx = (rr[0]+rr[1]*m-yint*m)/(m*m+1)
                        if(math.sqrt((xx-rr[0])**2 +(yy-rr[1])**2) < 8):
                            idxx2.append(indd2)
                            if ((indd2 in sig_i2) is False):
                                sig_i2.append(indd2)
                                r_signal2 = np.vstack((r_signal2,rr))
                        indd2 = indd2+1
                    plt.plot(r_signal2[:,0],r_signal2[:,1],'g.')
               
    r_signal2 = r_signal2[1:]
    possible_inner = th_z
    possible_inner[:,0] = th_z[:,0]
    possible_inner[:,1] = th_z[:,1]-math.pi*rad_z[:,1]
    ordered = r_signal2[np.argsort(r_signal2[:,0])]
    A = np.vstack([ordered[:40,0], np.ones(len(ordered[:40,0]))]).T
    m, yint = np.linalg.lstsq(A, ordered[:40,1])[0]
    plt.plot(ordered[:40,0], ordered[:40,1],'.',possible_inner[:,0], possible_inner[:,1],'.')
    plt.plot(x, m*x+ yint, '-b')
    indd = 0;
    for rr in possible_inner:      
        if(rr[0] > (ordered[0,0] - 50) and rr[0] < (ordered[0,0] + 50)):              
            yy = (rr[0]*m + rr[1]*m*m+yint)/(m*m+1)
            xx = (rr[0]+rr[1]*m-yint*m)/(m*m+1)
            if(math.sqrt((xx-rr[0])**2 +(yy-rr[1])**2) < 8):
                idxx2.append(indd)
                if ((indd in sig_i2) is False):
                    sig_i2.append(indd)
                    r_signal2 = np.vstack((r_signal2,rr))
                
        indd = indd+1
    plt.plot(r_signal2[:,0],r_signal2[:,1],'g.')
    
    return sig_i2, rRad, tRad, r_signal2



def hough_line(xy):
    """ Finds lines in data

    """
    nbins = 500 # number of bins for r, theta histogram
    th = np.linspace(0,math.pi,nbins)
    Hrad = [0,0]

    for theta in th: #sweep over theta and bin radius r in Hrad
        Radius = xy[:,0]*math.cos(theta) + xy[:,1]*math.sin(theta)
        index = 0
        for rr in Radius:
            aRad = np.hstack((theta,Radius[index]))
            Hrad = np.vstack((Hrad,aRad))
            index +=1
 
    Hrad = Hrad[1:]
    
    #countsRad, xedgesRad, yedgesRad, ImageRad = plt.hist2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-500,500]],cmap=plt.cm.jet)
    #countsRad, xedgesRad, yedgesRad, ImageRad = plt.hist2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-2000,2000]],cmap=plt.cm.jet) 
    countsRad, xedgesRad, yedgesRad = np.histogram2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-2000,2000]]) 
    iRad,jRad = np.unravel_index(countsRad.argmax(), countsRad.shape)

    tRad = iRad*math.pi/nbins
    rRad = jRad*4000/nbins - 2000

    return rRad, tRad, countsRad


def clean(xyz):
    a,b = hough_circle(xyz)
    print("center found at ", a, b)
    rad_z = [0,0]
    th_z = [0,0]
    for xy in xyz:
        r_xy = np.sqrt((xy[0]-a)**2+(xy[1]-b)**2)
        rad_z = np.vstack((rad_z,[xy[2],r_xy]))
        th_xy = np.arctan((xy[1]-b)/(xy[0]-a))
        th_z = np.vstack((th_z,[xy[2],th_xy]))

    rad_z = rad_z[1:] 
    th_z = th_z[1:] 
 
    r_th = th_z
    r_th[:,1] = r_th[:,1]*rad_z[:,1]

    r,t, counts = hough_line(r_th)
    print("r, t = ", r, t)


def main():
    parser = argparse.ArgumentParser(description='A script to clean data and write results to an HDF5 file')
    parser.add_argument('input', help='The input evt file')
    parser.add_argument('output', help='The output HDF5 file')
    args = parser.parse_args()

    #filename = '../../remove-noise/run_0150.evt'
    efile = pytpc.EventFile(args.input,'r')
    #with pytpc.HDFDataFile(args.output, 'a') as hfile:
     #   gp = hfile.fp.require_group(hfile.get_group_name)
    gp = []

    all_evts = set(efile.evtids)

    if len(gp) > 0:
        finished_evts = set(int(k) for k in gp.keys() if k.isdigit())
        evts_to_process = all_evts - finished_evts
        if len(evts_to_process) > 0:
            print('Already processed {} events. Continuing from where we left off.'.format(len(finished_evts)))
        else:
            print('All events have already been processed.')
            sys.exit(0)
    else:
        evts_to_process = all_evts
        #for i in smart_progress_bar(evts_to_process):
        for i in evts_to_process:
            evt = efile.get_by_event_id(i)
            raw_xyz = evt.xyzs(peaks_only=True)
            v_drift = pytpc.simulation.drift_velocity_vector(-5.2,9000,1.75,0.10472)
            uvw = pytpc.evtdata.calibrate(raw_xyz,v_drift,12.5)
            tmat = pytpc.utilities.tilt_matrix(-0.10472)
            uvw = np.dot(tmat,uvw[:,:3].T).T
            clean(uvw)



if __name__ == '__main__':
    import signal

    def handle_signal(signum, stack_frame):
        logger.critical('Received signal %d. Quitting.', signum)
        sys.stdout.flush()
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGQUIT, handle_signal)

    main()

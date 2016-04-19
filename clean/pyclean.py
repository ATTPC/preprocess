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

def nearest_neighbor_count(data, radius):
    return np.sum((np.sqrt(np.sum((data[:, None, :3] - data[:, :3])**2, -1)) <= radius)
                  & ~np.eye(len(data), dtype='bool'), axis=0)

def hough_circle(xyzs):
    xyz_order = xyzs[np.argsort(xyzs[:, 2])]

    nbins = 200  # number of bins in r and theta for Hough space discretization
    ths = np.linspace(0, np.pi, nbins)

    npts = len(xyz_order) - 5
    sqdels = xyz_order[5:, :2]**2 - xyz_order[:-5, :2]**2
    deltas_tiled = np.tile(xyz_order[5:, :2] - xyz_order[:-5, :2], (nbins, 1))

    Hrad = np.empty((nbins*npts, 2))

    Hrad[:, 0] = np.repeat(ths, npts)
    Hrad[:, 1] = (np.tile(np.sum(sqdels, -1), nbins) /
                          (2 * (deltas_tiled[:, 0] * np.cos(Hrad[:, 0])
                                + deltas_tiled[:, 1] * np.sin(Hrad[:, 0]))))

    Hrad = Hrad[1:]

    # uncomment this line for viewing Hough space histogram (computationally slower)
    # countsRad, xedgesRad, yedgesRad, ImageRad = plt.hist2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-500,500]],cmap=plt.cm.jet)

    # comment out this line when viewing Hough space histogram (this would be redundant with previous line)
    countsRad, xedgesRad, yedgesRad = np.histogram2d(Hrad[:, 0], Hrad[:, 1], nbins, range=[[0, math.pi], [-500, 500]])

    # max bin of histogram
    iRad, jRad = np.unravel_index(countsRad.argmax(), countsRad.shape)
    # convert max bin to theta, radius values
    tRad = iRad*math.pi/nbins
    rRad = jRad*1000/nbins - 500

    # convert theta, r to postions in xy space
    ax = rRad*math.cos(tRad)
    by = rRad*math.sin(tRad)

    return ax, by

def min_distance(point, array):
    d = []
    for a in array:
        #print("a = ", a)
        d.append(math.sqrt((a[0]-point[0])**2 + (a[1]-point[1])**2))
        #i += 1
    return min(d)

def find_good_points(counts, thh, rad):
    N = 100

    ind = np.argpartition(counts.flatten(),-N)[-N:]
    indices = np.vstack(np.unravel_index(ind,counts.shape)).T

    r_signal = [0,0]
    r_signal2 = [0,0]
    r_temp = thh

    nbins = 500
    tRad = indices[:,0]*math.pi/nbins
    rRad = indices[:,1]*4000/nbins - 2000   
    
    test = np.zeros(indices.shape)
    test[:,0] = indices[:,0]*math.pi/nbins
    test[:,1] = indices[:,1]*4000/nbins - 2000

    x = np.linspace(50,1600,100)
    y = np.zeros(100)

    sig_i = []
    sig_i2 = []
 
    dist = np.zeros(thh[:,1].shape) 
    dist += 1000 #initialize all values to 1000
 
    for idx, rr in enumerate(rRad): 
        mindist = 1000
        if(idx > 1):   
            d = test[:idx]
            current = (tRad[idx],rRad[idx])
            mindist = min_distance(current,d)
        # want Hough lines sufficiently far away
        if(mindist > 25):
            y_temp= (rRad[idx]-x*math.cos(tRad[idx]))/math.sin(tRad[idx])
            y = np.vstack((y,y_temp))
            b = math.sin(tRad[idx])
            c = -rr
            a = math.cos(tRad[idx])

            y2 = (c/b)-a*x/b

            idxx = []
            idxx2 = []
            indd = 0;
            for rr in thh:       
                yy = (a*(-b*rr[0]+a*rr[1])-b*c)/(a**2+b**2)
                xx = (b*(b*rr[0]-a*rr[1])-a*c)/(a**2+b**2)
                d = math.sqrt((xx-rr[0])**2 +(yy-rr[1])**2)
               
                if(d < 25):
                    idxx.append(indd)
                    if ((indd in sig_i) is False):
                        sig_i.append(indd)
                        r_signal = np.vstack((r_signal,rr))
                indd = indd+1
            if(b==0):
                mm = 1000
            else:
                mm = a/b
            A = np.vstack([thh[idxx,0], np.ones(len(thh[idxx,0]))]).T
            if(A.any()):
                m, yint = np.linalg.lstsq(A, thh[idxx,1])[0]
                if(m>0):
                    indd2 = 0;
                    for rr in thh:       
                        yy = (rr[0]*m + rr[1]*m*m+yint)/(m*m+1)
                        xx = (rr[0]+rr[1]*m-yint*m)/(m*m+1)
                        d = math.sqrt((xx-rr[0])**2 +(yy-rr[1])**2)
                        if(d < dist[indd2]):
                             dist[indd2] = d
                        if(d < 8):
                            idxx2.append(indd2)
                            if ((indd2 in sig_i2) is False):
                                sig_i2.append(indd2)
                                r_signal2 = np.vstack((r_signal2,rr))
                        indd2 = indd2+1
    #print(len(r_signal2))
    if(len(r_signal2)>2):
        r_signal2 = r_signal2[1:]
 
        possible_inner = np.zeros(rad.shape)
        possible_inner[:,0] = rad[:,0]
        possible_inner[:,1] = thh[:,1]-math.pi*rad[:,1]
        possible_outer = np.zeros(rad.shape)
        possible_outer[:,0] = rad[:,0]
        possible_outer[:,1] = thh[:,1]+math.pi*rad[:,1]

        ordered = r_signal2[np.argsort(r_signal2[:,0])]
        A_end = np.vstack([ordered[:20,0], np.ones(len(ordered[:20,0]))]).T
        m_end, yint_end = np.linalg.lstsq(A_end, ordered[:20,1])[0]
        A_begin = np.vstack([ordered[-20:,0], np.ones(len(ordered[-20:,0]))]).T
        m_begin, yint_begin = np.linalg.lstsq(A_begin, ordered[-20:,1])[0]

        indd = 0;
        for rr in possible_inner:   
            try:
                if(rr[0] > (ordered[2,0] - 50) and rr[0] < (ordered[2,0] + 50)):      
                    yy = (rr[0]*m_end + rr[1]*m_end*m_end+yint_end)/(m_end*m_end+1)
                    xx = (rr[0]+rr[1]*m_end-yint_end*m_end)/(m_end*m_end+1)
                    d = math.sqrt((xx-rr[0])**2 +(yy-rr[1])**2)                 
                    if(d < dist[indd]):
                        dist[indd] = d
                        if(d < 22):
                            idxx2.append(indd)
                            if ((indd in sig_i2) is False):
                                sig_i2.append(indd)
                                indd = indd+1
            except IndexError:
                print("IndexError")
        indd = 0  
        for rr in possible_outer:  
            try:
                if(rr[0] > (ordered[-1,0] - 50) and rr[0] < (ordered[-1,0] + 50)):       
                    yy = (rr[0]*m_begin + rr[1]*m_begin*m_begin+yint_begin)/(m_begin*m_begin+1)
                    xx = (rr[0]+rr[1]*m_begin-yint_begin*m_begin)/(m_begin*m_begin+1)
                    d = math.sqrt((xx-rr[0])**2 +(yy-rr[1])**2)
                    if(d < dist[indd]):
                        dist[indd] = d
                        if(d < 22):
                            idxx2.append(indd)
                            if ((indd in sig_i2) is False):
                                sig_i2.append(indd)
                                r_signal2 = np.vstack((r_signal2,rr))
            
                                indd = indd+1
            except IndexError:
                print("IndexError")
    return sig_i2, rRad, tRad, dist

def hough_line(xy):
    """ Finds lines in data

    """
    nbins = 500 # number of bins for r, theta histogram
    max_val = 2000;
    th = np.linspace(0,math.pi,nbins)
    #Hrad = [0,0]
    Hrad = np.zeros((nbins*len(xy),2))
    index = 0
    for theta in th: #sweep over theta and bin radius r in Hrad
        Radius = xy[:,0]*math.cos(theta) + xy[:,1]*math.sin(theta)
        
        for rr in Radius:
            aRad = np.hstack((theta,rr))
            #Hrad = np.vstack((Hrad,aRad))
            Hrad[index,:] = aRad
            index +=1
 
    #Hrad = Hrad[1:]
    
    #countsRad, xedgesRad, yedgesRad, ImageRad = plt.hist2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-500,500]],cmap=plt.cm.jet)
    #countsRad, xedgesRad, yedgesRad, ImageRad = plt.hist2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-2000,2000]],cmap=plt.cm.jet) 
    countsRad, xedgesRad, yedgesRad = np.histogram2d(Hrad[:,0], Hrad[:,1], nbins,range=[[0,math.pi],[-max_val,max_val]]) 
    iRad,jRad = np.unravel_index(countsRad.argmax(), countsRad.shape)

    tRad = iRad*math.pi/nbins
    rRad = jRad*max_val*2/nbins - max_val

    return rRad, tRad, countsRad




def clean(xyz):
    a,b = hough_circle(xyz)
   # print("center found at ", a, b)

    rad_z = np.zeros((len(xyz),2))
    th_z = np.zeros((len(xyz),2))

    idx=0
    for xy in xyz:
        r_xy = np.sqrt((xy[0]-a)**2+(xy[1]-b)**2)
        rad_z[idx,:]=[xy[2],r_xy]
        th_xy = np.arctan((xy[1]-b)/(xy[0]-a))
        th_z[idx,:] =[xy[2],th_xy]
        idx +=1

    r_th = np.zeros(th_z.shape)

    r_th[:,0] = th_z[:,0]
    r_th[:,1] = th_z[:,1]*rad_z[:,1]   

    r,t, counts = hough_line(r_th)
    sig_i, rRad, tRad, dist = find_good_points(counts, th_z, rad_z)

    clean_xyz = np.column_stack((xyz,dist)) ###### append distances to data and return!
    center = [a,b]
 
    return clean_xyz, center

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

import numpy as np
import pytpc
import math
import h5py

print("working")

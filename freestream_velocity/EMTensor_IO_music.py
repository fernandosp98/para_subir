#!/usr/bin/env python3

from __future__ import division, print_function

import argparse

import numpy as np
import h5py

import freestream


def main():
    parser = argparse.ArgumentParser(
        description='Output a free streaming event to MUSIC format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    #Required inputs.
    parser.add_argument('input_file', help='initial profile input file')
    parser.add_argument('music_output_file', help='output file in MUSIC format')
    parser.add_argument('EMtensor_output_file', help='Energy Momentum output file')
    
    #optional output.
    parser.add_argument('--time', default=1.0, type=float,
                        help='time to freestream, in [fm/c]')
    parser.add_argument('--velocity', default=1.0, type=float, 
                        help='freestreaming velocity')
    parser.add_argument('--grid-max', default=10.0, type=float,
                        help='grid xy max [fm], same as in Trento')
    parser.add_argument('--renorm', default=1.0, type=float,
                        help='Renormalization factor to MULTIPLY the initial profile')
                       
    args = parser.parse_args()

    gfs = args.grid_max
    tfs = args.time
    vfs = args.velocity

    #Read in the initial profile
    initial = np.loadtxt( args.input_file, comments="#", delimiter=" ")

    initial = initial*args.renorm

    #Call the freestreamer.
    fs = freestream.FreeStreamer(initial, gfs, tfs, vfs)
    
    
    gev_to_invfm = 1.0/0.197326979
    
    xx = fs.grid()
    
    length = len(xx)
    
    dx = xx[2] - xx[1]
    
    e = fs.energy_density()
    u = fs.flow_velocity()
    pi = fs.shear_tensor()
    pi_eta_eta = fs.shear_tensor_eta_eta()
       
    #Write to file in MUSIC format.
    f = open( args.music_output_file, "w+")

    f.write( "# tau_in_fm %g etamax= 1 xmax= %d ymax= %d deta= 0 dx= %g dy= %g velocity/c= %g\n" %(tfs, length, length, dx, dx, vfs)  )
    
    for i in range(length):
        for j in range(length):
            f.write( ("0"+ 17*" %g"+"\n")
                    %(xx[i], xx[j], e[j,i], u[j,i,0], u[j,i,1], u[j,i,2], 0.0, 
                      gev_to_invfm*pi[j,i,0,0], gev_to_invfm*pi[j,i,0,1], gev_to_invfm*pi[j,i,0,2], 0.0, gev_to_invfm*pi[j,i,1,1],
                      gev_to_invfm*pi[j,i,1,2], 0.0, gev_to_invfm*pi[j,i,2,2], 0.0, gev_to_invfm*pi_eta_eta[j,i] ))        
    
    f.close
    
       
    #Write Tuv to file.
    f = open( args.EMtensor_output_file, "w+")
    
    T = fs.Tuv()
        
    for i in range(length):
        for j in range(length):
            f.write( (12*" %.12g"+"\n")
                    %(xx[j], xx[i], T[i,j,0,0], T[i,j,1,1], T[i,j,2,2], 0, -T[i,j,0,1], -T[i,j,0,2], 0, -T[i,j,1,2], 0, 0 ))        
        f.write( ("\n") )
    
    f.close    

if __name__ == "__main__":
    main()


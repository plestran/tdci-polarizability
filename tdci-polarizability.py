import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from scipy.fftpack import fft

#-----------------------------------------------------------------------------
def grabDipole(filename,direction,start):
    ''' Grab dipoles from log file '''
  
    print "filename = ", filename
  
    # search through whole log file
    dipole, field = [], []
    index = { 'x' : 1, 'y' : 3, 'z' : 5 }
    searchfile = open(filename, "r")
    for line in searchfile:
       
        # grab dipole values for proper direction
        if "X=" in line and "Z=" in line: 
            split = line.split()
            dipole.append(float( split[index[direction]] ))
        
        # grab field information for proper direction
        elif "Ex=" in line and "Ez=" in line:
            split = line.split()
            field.append(float( split[index[direction]] ))
    searchfile.close()
  
    # NOTE: field is not returned, but it can be if you want
    field  = np.asarray(field)[start:]
  
    return np.asarray(dipole)[start:]
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def grabTime(filename,start):
    ''' Grab time steps from log file '''
    print "filename = ", filename

    # search through whole log file
    time = []
    searchfile = open(filename, "r")
    for line in searchfile:
        # grab time information
        if "(t =" in line: 
            split = line.split()
            time.append(float(split[3]))
    searchfile.close()

    return np.asarray(time)[start:]
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def coeff_of_determination(truth,estimate):
    ''' Calculate the coefficient of determination '''

    Stot, Sres = 0, 0
    average = np.mean(truth)
    for t, e in zip(truth,estimate):
        Stot += (t - average)**2
        Sres += (t - e)**2

    return 1 - Sres/Stot
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def fit_polar(t,posAt,negAt,pos2At,neg2At,max_field,fsfreq):
    ''' Fit a function to extract the polarizability '''

    # build the polarizability 
    polarizability = (8*(posAt-negAt) - (pos2At-neg2At))/(12*max_field)

    # fit a function to it to extract the relevant value
    polar_max = max(polarizability)
    fit_func  = lambda x: x[0]*np.cos(t*fsfreq) - polarizability
    estimated = leastsq(fit_func, [polar_max])[0]

    # build the fitted function to plot later
    fitted = estimated*np.cos(t*fsfreq) 

    return polarizability, fitted, estimated
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def fit_hyper(t,posAt,negAt,pos2At,neg2At,max_field,fsfreq):
    ''' Fit a function to extract the hyperpolarizability '''

    # build the hyperpolarizability 
    initial = posAt[0]
    firsthyper = ( 16*(posAt+negAt) - (pos2At+neg2At) - 30*initial ) / \
                 ( 24*max_field**2 )

    # fit a function to it to extract the relevant values
    hyper_min = min(firsthyper) 
    hyper_avg = np.mean(firsthyper) 
    fit_func = lambda x: x[0]*np.cos(2*t*fsfreq) + x[1] - firsthyper
    est1, est2 = leastsq(fit_func, [hyper_min,hyper_avg])[0]

    # build the fitted function to plot later
    fitted = est1*np.cos(2*t*fsfreq) + est2

    return firsthyper, fitted, est1, est2
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def fit_2ndhyper(t,posAt,negAt,pos2At,neg2At,pos3At,neg3At,max_field,fsfreq):
    ''' Fit a function to extract the 2nd hyperpolarizability '''

    # build the 2nd hyperpolarizability 
    secondhyper = ( -13*(posAt-negAt) + 8*(pos2At-neg2At) - (pos3At-neg3At) ) / \
                  ( 48*max_field**3 )

    # fit a function to it to extract the relevant values
    hyper_min = min(secondhyper) 
    hyper_avg = np.mean(secondhyper) 
    fit_func = lambda x: x[0]*np.cos(3*t*fsfreq) + 3*x[1]*np.cos(t*fsfreq) - \
                         secondhyper
    est1, est2 = leastsq(fit_func, [hyper_min,hyper_avg])[0]
  
    # build the fitted function to plot later
    fitted= est1*np.cos(3*t*fsfreq) + 3*est2*np.cos(t*fsfreq)

    return secondhyper, fitted, est1, est2
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def GUGAPolarizability(f_base,direction,max_field,frequency,start):    
    '''
        (C) Patrick Lestrange 2018
        
        A post-processing script for computing the polarizability from
        Time Dependent CI jobs in Gaussian

    '''
   
    # determine names of log files
    f_posA  = baseFilename+'+A'+direction+'.log'
    f_negA  = baseFilename+'-A'+direction+'.log'
    f_pos2A = baseFilename+'+2A'+direction+'.log'
    f_neg2A = baseFilename+'-2A'+direction+'.log'
    f_pos3A = baseFilename+'+3A'+direction+'.log'
    f_neg3A = baseFilename+'-3A'+direction+'.log'

    # pull dipole moments from the log files
    posAt  = grabDipole(f_posA,direction,start)
    negAt  = grabDipole(f_negA,direction,start)
    pos2At = grabDipole(f_pos2A,direction,start)
    neg2At = grabDipole(f_neg2A,direction,start)
    pos3At = grabDipole(f_pos3A,direction,start)
    neg3At = grabDipole(f_neg3A,direction,start)

    # grab time and shift after removing ramping section
    t = grabTime(f_posA,start)
    t = t - t[0]
    fsfreq = frequency/27.211/0.0241888425
    dt = t[1] - t[0] # spacing between time samples; assumes constant time step

    # fit the polarizability
    polarizability, polar_fit, est_polar  = fit_polar(t,posAt,negAt,pos2At,
                                                      neg2At,max_field,fsfreq)
    R2 = coeff_of_determination(polarizability,polar_fit) 
    print "a(-w;w) = ", est_polar, "  R^2 = ", R2 

    # fit the first hyperpolarizability
    firsthyper, hyper_fit, hyper_est1, hyper_est2 = fit_hyper(t,posAt,negAt,
                                                     pos2At,neg2At,
                                                     max_field,fsfreq)
    R2 = coeff_of_determination(firsthyper,hyper_fit)
    print "b(-2w;w,w) = ", hyper_est1*4
    print "b(0;w,-w)  = ", hyper_est2*4, "  R^2 = ", R2

    # fit the second hyperpolarizability
    secondhyper, hyper2_fit, hyper2_est1, hyper2_est2= fit_2ndhyper(t,posAt,negAt,
                                                         pos2At,neg2At,pos3At,
                                                         neg3At,max_field,fsfreq)
    R2 = coeff_of_determination(secondhyper,hyper2_fit)
    print "g(-3w;w,w,w) = ", hyper2_est1*24
    print "g(-w;w,w,-w) = ", hyper2_est1*24, "  R^2 = ", R2

    # plot everything
    fig = plt.figure(figsize=(10, 8), dpi=100)
    plt.suptitle("Polarizability")
    plt.subplot(4,1,1)
    plt.plot(t,posAt,label=' +Az')
    plt.plot(t,negAt,label=' -Az')
    plt.plot(t,pos2At,label='+2Az')
    plt.plot(t,neg2At,label='-2Az')
    plt.plot(t,pos3At,label='+3Az')
    plt.plot(t,neg3At,label='-3Az')
    plt.ylabel('Dipole (au)')
    plt.xlabel('Time (fs)')
    plt.legend()

    plt.subplot(4,1,2)
    plt.plot(t,polarizability,label="Polarzability")
    plt.plot(t,polar_fit,label="Fit Curve")
    plt.ylabel('Polarizability')
    plt.xlabel('Time (fs)')
    plt.legend()

    plt.subplot(4,1,3)
    plt.plot(t,firsthyper,label="Hyperpolarzability")
    plt.plot(t,hyper_fit,label="Fit Curve")
    plt.ylabel('Hyperpolarizability')
    plt.xlabel('Time (fs)')
    plt.legend()

    plt.subplot(4,1,4)
    plt.plot(t,secondhyper,label="2nd Hyperpolarzability")
    plt.plot(t,hyper2_fit,label="Fit Curve")
    plt.ylabel('2nd Hyperpolarizability')
    plt.xlabel('Time (fs)')
    plt.legend()

    plt.show()
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
if __name__ == '__main__':

    baseFilename = 'h2'
    start = 4582  
  
    GUGAPolarizability(baseFilename,'z',0.001,1.9593,start)
    

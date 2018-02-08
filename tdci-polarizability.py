import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.fftpack import fft

#------------------------------------
def grabDipole(filename,direction):

  print "filename = ", filename

  if direction == 'x': 
    index = 1
  elif direction == 'y': 
    index = 3
  elif direction == 'z': 
    index = 5
  else: 
    print "Direction not recognized\n"
    exit()

  start = 4582
  dipole = []
  field  = []
  searchfile = open(filename, "r")
  for line in searchfile:
      if "X=" in line and "Z=" in line: 
        split = line.split()
        dipole.append(float(split[index]))
      elif "Ex=" in line and "Ez=" in line:
        split = line.split()
        field.append(float(split[index]))
  searchfile.close()
  dipole = np.asarray(dipole)
  field = np.asarray(field)
  initial = dipole[0]
  dipole = dipole[start:]
  field = field[start:]

  return dipole, initial
#------------------------------------

#------------------------------------
def grabTime(filename):

  print "filename = ", filename

  start = 4582
  time = []
  searchfile = open(filename, "r")
  for line in searchfile:
      if "(t =" in line: 
        split = line.split()
        time.append(float(split[3]))
  searchfile.close()
  time = np.asarray(time)
  time = time[start:]
  return time
#------------------------------------

#------------------------------------
def GUGAPolarizability(f_base,direction,max_field,frequency):    
    '''
        (C) Patrick Lestrange 2016
        
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
    posAt , initial = grabDipole(f_posA,direction)
    negAt , initial = grabDipole(f_negA,direction)
    pos2At, initial = grabDipole(f_pos2A,direction)
    neg2At, initial = grabDipole(f_neg2A,direction)
    pos3At, initial = grabDipole(f_pos3A,direction)
    neg3At, initial = grabDipole(f_neg3A,direction)

    # grab time and shift after removing ramping section
    t = grabTime(f_posA)
    t = t - t[0]
    fsfreq = frequency/27.211/0.0241888425
    dt = t[1] - t[0]          # spacing between time samples; assumes constant time step

    # fit the polarizability
    polarizability = (8*(posAt-negAt) - (pos2At-neg2At))/(12*max_field)
    polar_max = max(polarizability)
    fit_polar = lambda x: x[0]*np.cos(t*fsfreq) - polarizability
    est_polar = leastsq(fit_polar, [polar_max])[0]
    polar_fit = est_polar*np.cos(t*fsfreq) 
    Stot = 0
    Sres = 0
    average = np.mean(polarizability)
    for i in range(len(polarizability)):
      Stot += (polarizability[i] - average)**2
      Sres += (polarizability[i] - polar_fit[i])**2
    R = 1 - Sres/Stot
    print "a(-w;w) = ", est_polar, "  R^2 = ", R 

    # fit the first hyperpolarizability
    firsthyper = (16*(posAt+negAt) - (pos2At+neg2At) - 30*initial)/(24*max_field**2)
    hyper_min = min(firsthyper) 
    hyper_avg = np.mean(firsthyper) 
    fit_hyper = lambda x: x[0]*np.cos(2*t*fsfreq) + x[1] - firsthyper
    est_hyper, est_hyper2 = leastsq(fit_hyper, [hyper_min,hyper_avg])[0]
    hyper_fit = est_hyper*np.cos(2*t*fsfreq) + est_hyper2
    Stot = 0
    Sres = 0
    average = np.mean(firsthyper)
    for i in range(len(firsthyper)):
      Stot += (firsthyper[i] - average)**2
      Sres += (firsthyper[i] - hyper_fit[i])**2
    R = 1 - Sres/Stot
    print "b(-2w;w,w) = ", est_hyper*4
    print "b(0;w,-w)  = ", est_hyper2*4, "  R^2 = ", R 

    # fit the second hyperpolarizability
    secondhyper = (-13*(posAt-negAt) + 8*(pos2At-neg2At) - (pos3At-neg3At))/(48*max_field**3)
    hyper_min = min(secondhyper) 
    hyper_avg = np.mean(secondhyper) 
    fit_hyper = lambda x: x[0]*np.cos(3*t*fsfreq) + 3*x[1]*np.cos(t*fsfreq) - secondhyper
    est_hyper, est_hyper2 = leastsq(fit_hyper, [hyper_min,hyper_avg])[0]
    second_fit = est_hyper*np.cos(3*t*fsfreq) + 3*est_hyper2*np.cos(t*fsfreq)
    Stot = 0
    Sres = 0
    average = np.mean(secondhyper)
    for i in range(len(secondhyper)):
      Stot += (secondhyper[i] - average)**2
      Sres += (secondhyper[i] - second_fit[i])**2
    R = 1 - Sres/Stot
    print "g(-3w;w,w,w) = ", est_hyper*24
    print "g(-w;w,w,-w) = ", est_hyper2*24, "  R^2 = ", R 

    # fourier transform the dipole
    damp_const = 150.0
    tau = t*41.3413745758
    damp = np.exp(-(tau-tau[0])/damp_const)
    z = posAt# * damp
    fw = fft(z)
    n = len(z)                # number samples, including padding
    dt = tau[1] - tau[0]          # spacing between time samples; assumes constant time step
    period = (n-1)*dt - tau[0] 
    dw = 2.0 * np.pi / period # spacing between frequency samples, see above
    m = n / 2        # splitting (ignore negative freq)
    wmin = 0.0       # smallest energy/frequency value
    wmax = m * dw    # largest energy/frequency value
    fw_pos = fw[0:m]              # FFT values of positive frequencies (first half of output array)
    fw_re = np.real(fw_pos)       # the real positive FFT frequencies
    fw_im = (np.imag(fw_pos))     # the imaginary positive FFT frequencies
    fw_abs = abs(fw_pos)          # absolute value of positive frequencies
    w = np.linspace(wmin, wmax, m)  #positive frequency list
    w = (w*27.2114)    # give frequencies in eV

    fig = plt.figure(figsize=(10, 8), dpi=100)
    plt.suptitle("Polarizability (CIS)")
    plt.subplot(5,1,1)
    plt.plot(t,posAt,label=' +Az')
    plt.plot(t,negAt,label=' -Az')
    plt.plot(t,pos2At,label='+2Az')
    plt.plot(t,neg2At,label='-2Az')
    plt.plot(t,pos3At,label='+3Az')
    plt.plot(t,neg3At,label='-3Az')
    plt.ylabel('Dipole (au)')
    plt.xlabel('Time (fs)')
    plt.legend()

    plt.subplot(5,1,2)
    plt.plot(t,polarizability,label="Polarzability")
    plt.plot(t,polar_fit,label="Fit Curve")
    plt.ylabel('Polarizability')
    plt.xlabel('Time (fs)')
    plt.legend()

    plt.subplot(5,1,3)
    plt.plot(t,firsthyper,label="Hyperpolarzability")
    plt.plot(t,hyper_fit,label="Fit Curve")
    plt.ylabel('Hyperpolarizability')
    plt.xlabel('Time (fs)')
    plt.legend()

    plt.subplot(5,1,4)
    plt.plot(t,secondhyper,label="2nd Hyperpolarzability")
    plt.plot(t,second_fit,label="Fit Curve")
    plt.ylabel('2nd Hyperpolarizability')
    plt.xlabel('Time (fs)')
    plt.legend()

    plt.subplot(5,1,5)
    plt.plot(w,fw_abs,label="Spectrum")
    plt.ylabel('P(w)')
    plt.xlabel('Energy (eV)')
    plt.xlim(0,3)     # X range
    plt.legend()

    plt.show()
#------------------------------------

#------------------------------------
if __name__ == '__main__':

    baseFilename = 'h2'
    
    GUGAPolarizability(baseFilename,'z',0.001,1.9593)
    

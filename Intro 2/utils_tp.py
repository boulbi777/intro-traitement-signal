import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt


# def load_signal(name, Fe):
#   s = np.genfromtxt(name, delimiter=',')
#   t = np.arange(0,s.shape[0]/Fe,1./Fe)
#   return t,s

def fourier_transform(signal,Fe):
  FT = np.fft.fftshift(np.fft.fft(signal))
  F = np.linspace(-0.5,0.5,signal.shape[0])*Fe
  return F, FT

def fourier_transform_positif(signal,Fe):
  FT = np.fft.fft(signal)
  F = np.linspace(0,1,signal.shape[0])*Fe
  return F, FT


def inverse_fourier_transform(FT,Fe):
  s = np.fft.ifft(np.fft.fftshift(FT))
  return s.real
 
# def low_pass(Fc,Freq):
#   H = np.abs(Freq)<Fc
#   return H

# def high_pass(Fc,Freq):
#   H = np.abs(Freq)>Fc
#   return H

# def butterworth(Type,Order, Fc, time):
#   Fe = 1./(time[1]-time[0])
#   [b, a] = signal.butter(Order, Fc/(2*Fe), btype=Type)#, analog = True)
#   dirac =time*0.
#   dirac[0]=1.
#   h = signal.lfilter(b,a,dirac)
#   return h

# def convolution(Signal, Impulse_rep):
#   S_out = np.convolve(Signal,Impulse_rep,mode='full')
#   return S_out[0:Signal.shape[0]]

# def butterworth_order(F1,F2,A1,A2):
#   [n,Fc] = signal.buttord(F1,F2,A1,A2,analog='True')
#   return -n,Fc


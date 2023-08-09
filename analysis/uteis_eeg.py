#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:39:36 2021

@author: tercio
"""


from scipy.signal import butter, lfilter, firwin, hilbert, welch, hilbert

from scipy.stats import circmean, circstd
import math
import numpy as np
import scipy.fftpack
import mne
import random


def motifs_transf(x):
  motif = []
  for i in range(len(x)-2):
    if x[i]> x[i+1] and x[i+1]> x[i+2] and x[i]> x[i+2]:
      motif.append(1)
    if x[i]>= x[i+1] and x[i+1]< x[i+2] and x[i]> x[i+2]:
      motif.append(2)
    if x[i]<= x[i+1] and x[i+1]> x[i+2] and x[i]> x[i+2]:
      motif.append(3)
    if x[i] >= x[i+1] and x[i+1]< x[i+2] and x[i]< x[i+2]:
      motif.append(4)
    if x[i]<= x[i+1] and x[i+1]< x[i+2] and x[i]< x[i+2]:
      motif.append(5)
    if x[i]<= x[i+1] and x[i+1]> x[i+2] and x[i]< x[i+2]:
      motif.append(6)

  return np.array(motif)


def motif_appear (atrasos, x,y):
  Cxy = [] # receber os dados
  Cyx = []
  for t  in atrasos: #passar em todos atrasos
    cxy = 0
    cyx = 0
    for i in range(len(x)-t):
      if x[i]==y[i+t]:
        cxy+=1
      if y[i]==x[i+t]:
        cyx+=1


    Cxy.append(cxy)
    Cyx.append(cyx)


  return np.array(Cxy),np.array(Cyx)

def degree_synchronization (Cxy,Cyx,x):
    Qxy = max(max(Cxy),max(Cyx))/len(x)
    
    return Qxy

def synchronization_direction (Cxy,Cyx):
    '''Se for positivo o x influencia no y. 
     '''
    qxy = max(Cxy)- max(Cyx)
    return qxy

def tempo_com_maior_sincronizacao(Cxy,Cyx):
    '''Saber o t (indice) que obteve maior valor. 
     '''
    
    if max(Cxy) >= max(Cyx):
        for i in range(len(Cxy)):
            if Cxy[i]== max(Cxy):
                #print(i+1)
                break # Quando for ambiguo pegar o valor menor 
    else:
        for i in range(len(Cyx)):
            if Cyx[i] == max(Cyx):
                #print(i+1)
                break
    return i+1


def fir_band_filter(x, fs, cutoff1,cutoff2):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    
    norm_cutoff1 = cutoff1 / nyquist
    norm_cutoff2 = cutoff2 / nyquist

    # low cut filter
    fil = firwin(255, [norm_cutoff1,norm_cutoff2], pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x 


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a




def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y




def Variabilidade_diff_phase (f1,f2):
    
    try:
   
        hph1 = f1;
        hph2 = f2;       
        
        
        z= hilbert(hph1) #form the analytical signal
        inst_phase1 = np.unwrap(np.angle(z))#inst phase
    
        z= hilbert(hph2) #form the analytical signal
        inst_phase2 = np.unwrap(np.angle(z))#inst phase
    
        diff_phase = abs (np.cos(inst_phase1)*np.pi - np.cos(inst_phase2)*np.pi)
        
        Var = np.std(diff_phase)
       
        
        return Var, diff_phase
    except:
        return 'nan', 'nan'




def PLV_func (s1,s2):
    
    try:
   
        hph1 = s1;
        hph2 = s2;
        
        
       
        z= hilbert(hph1) #form the analytical signal
        inst_phase1 = np.unwrap(np.angle(z))#inst phase
    
        z= hilbert(hph2) #form the analytical signal
        inst_phase2 = np.unwrap(np.angle(z))#inst phase
        
        
        #discards 5% of samples of each side - filtering and hilbert border
        discsampl = int(len(inst_phase1)*0.01*5);
        inst_phase1 = inst_phase1[discsampl: len(hph1)-discsampl];
        inst_phase2 = inst_phase2[discsampl:len(hph2)-discsampl]; 
        
        inst_phase1 = np.cos(inst_phase1)*np.pi
        inst_phase2 = np.cos(inst_phase2)*np.pi
            
        
        phDif = inst_phase1 - inst_phase2
        
        
        PLV = np.sum(np.exp(1j*(phDif )))/len(phDif);   
        
        
        
        
        return abs(PLV)
    except:
        return 'nan'
    

def PLV_func2 (s1,s2):
    
    """ PLV com dados normalizados"""
   
    hph1 = s1;
    hph2 = s2;
    
    
   
    z= hilbert(hph1) #form the analytical signal
    inst_phase1 = np.unwrap(np.angle(z))#inst phase

    z= hilbert(hph2) #form the analytical signal
    inst_phase2 = np.unwrap(np.angle(z))#inst phase
    
    
    #discards 5% of samples of each side - filtering and hilbert border
    discsampl = int(len(inst_phase1)*0.01*5);
    inst_phase1 = inst_phase1[discsampl: len(hph1)-discsampl];
    inst_phase2 = inst_phase2[discsampl:len(hph2)-discsampl]; 
    
    inst_phase1 = np.cos(inst_phase1)*np.pi
    inst_phase2 = np.cos(inst_phase2)*np.pi
        
    
    phDif = inst_phase1 - inst_phase2
    
    
    PLV = np.sum(np.exp(1j*(phDif )))/len(phDif);
    
    
    #compute PLV for surrogate values and normalized PLV
    
    numsurrogate = 200
    surrogate_f2 = inst_phase2.copy()
    surrogates_PLV = np.zeros(200)
    
    
    for s in range(numsurrogate):        
        random.shuffle(surrogate_f2) 
        surrogatePhDif = inst_phase1 - surrogate_f2;
        surrogates_PLV[s] = np.sum(np.exp(1j*(surrogatePhDif))
                                    /len(surrogatePhDif))
   
    surr_PLVMean = np.mean(abs(surrogates_PLV))
    
    if abs(PLV) < abs(surr_PLVMean):
        PLV_norm = 0;
    else:
        PLV_norm = (abs(PLV) - (surr_PLVMean))/(1 - (surr_PLVMean));
    
    
    return abs(PLV), PLV_norm


def FFT (signal, fs):
    # Number of samplepoints
    N = len(signal)
    # sample spacing
    T = 1.0 / fs
    x = np.linspace(0.0, N*T, N)
  
    yf = scipy.fftpack.fft(signal)
    xf = np.linspace(0, int(1.0/(2.0*T)), int(N/2))
    spectrum = 2.0/N * np.abs(yf[:N//2])
    
    return xf, spectrum
    
 
    
def spectral_features (signal, window_size, fs):
        
    overlap = round(0.85*window_size) 
    w, Pxx = welch(signal, fs, nperseg = window_size,noverlap =overlap)
    
    PxxNorm = Pxx/sum(Pxx)#normalized spectrum

    Sent = -sum(PxxNorm*np.log2(PxxNorm))#spectral entropy
    Spow = np.mean(Pxx**2)#spectral power
    Cent = np.sum(w*PxxNorm) #frequency centroid
    Speak = np.max(Pxx) #peak amplitude
    Sfreq = w[np.argmax(PxxNorm)]# peak frequency
    
    
    
    return PxxNorm, Spow, Sent, Cent, Speak, Sfreq
     
 
def filterband(x,  fs, lowcut, highcut):
    y = []
    for i in range (9):
        y.append(fir_band_filter (x[i], fs, lowcut, highcut))
    return y




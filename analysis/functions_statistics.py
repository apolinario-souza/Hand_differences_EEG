#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:11:33 2020

@author: tercio
"""

import pandas as pd 
import numpy  as np
from scipy.spatial import distance
from scipy.optimize import least_squares
import scipy.interpolate
from matplotlib import patches
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy import stats
#################### CRIANDO FUNCOES      ##################

#Criei uma função para criar os blocos
class estatistica():
    def media_bl(x,n_tt_bl,n_bl):
        y = np.zeros(n_bl)
        contador = 0;
        for i in range(n_bl):
             y[i] = np.nanmean(x[contador:contador+n_tt_bl])
             contador +=n_tt_bl;
        return y
    def desvio_bl(x,n_tt_bl,n_bl):
        y = np.zeros(n_bl)
        contador = 0;
        for i in range(n_bl):
             y[i] = np.std(x[contador:contador+n_tt_bl])
             contador +=n_tt_bl;
        return y
    
    #Criei de out
    
    def outlier(x):
        from scipy.stats import iqr
        y = x
        ref = iqr (y)*1.5;
        quat1 = np.quantile(y,0.25);
        quat3 = np.quantile(y,0.75);
        lim_cima = quat3 + ref;
        lim_baixo = quat1 - ref;
        for i in range (len(y)):
                if y[i] >= lim_cima or y[i] <= lim_baixo:
                        y[i] =  np.nan;
                else:
                        y[i] = y[i];
        m = np.nanmean (y);
        for k in range (len(y)):
            teste = np.isnan (y[k]); #Testa para ve se o valo tem nan, se tiver retorna 1
            if teste == 1:
               y[k] = m;
            else:
               y[k] = y[k]; 
               
        return y
#Approximate_entropy.https://en.wikipedia.org/wiki/Approximate_entropy
def ApEn(U, m, r):
    U = np.array(U)
    N = U.shape[0]
            
    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i+m] for i in range(int(z))])
        X = np.repeat(x[:, np.newaxis], 1, axis=2)
        C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
        return np.log(C).sum() / z
    
    return abs(_phi(m + 1) - _phi(m))


def sampen(L, m, r):
    
    #https://en.wikipedia.org/wiki/Sample_entropy
    N = len(L)
    B = 0.0
    A = 0.0
    
    
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)


def mahalanobis(data):
    
    maha = []
    for i in range((data.shape[0])-1):
        x,y = data[i,:], data[i+1,:]
        arr = np.array([x,y])
        cov = np.cov(arr.T)
        maha.append(distance.mahalanobis(x,y,cov))
    return np.mean(maha)

def make_log (data):
    data_new = np.zeros((data.shape)); 
    for k in range (data_new.shape[1]):
        for i in range(data_new.shape[0]):
            data_new[i,k] = np.log2 (data [i,k]) 
    
    return data_new


def make_log_1d (data):
    data_new = np.zeros((data.shape));
    for i in range(data_new.shape[0]):
        data_new[i] = np.log2 (data [i]) 
    
    return data_new


def transf_to_1_1d (data):
    data_new = np.zeros((data.shape));
    max = np.max(data)
    for i in range(data_new.shape[0]):
        data_new[i] = (data [i]*100)/max
    
    return data_new/100

def transf_to_1 (data):
    data_new = np.zeros((data.shape));
    
    for k in range (data_new.shape[1]):
        for i in range(data_new.shape[0]):
            max = np.max(data[:,k])
            data_new[i,k] = (data [i,k]*100)/max
    
    return data_new/100

def plot_topomap(data, ax, fig, draw_cbar=True):
    '''
    Plot topographic plot of EEG data. This specialy design for Emotiv 14 electrode data. 
    This can be change for any other arrangement by changing ch_pos (channel position array)
    Input: data- 1D array 14 data values
           ax- Matplotlib subplot object to be plotted every thing
           fig- Matplot lib figure object to draw colormap
           draw_cbar- Visualize color bar in the plot
    '''
    N = 1000            
    xy_center = [2,2]  
    radius = 2 
        

    #Pz, Fz, Cz, C3, C4, F3, F4, P3, P4, 
    ch_pos = [[2,1],[2,4.35], [2,2], [1.5,2], 
             [2.5,2], [1.5,3.5], [2.5,3.5], [2.5,1], 
             [1.5,1]]
    x,y = [],[]
    for i in ch_pos:
        x.append(i[0])
        y.append(i[1])

    xi = np.linspace(-.5, 4.5, N)
    yi = np.linspace(-.5, 4.5, N)
    zi = scipy.interpolate.griddata((x, y), data, (xi[None,:], yi[:,None]), method='nearest')

    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
            if (r - dr/2) > radius:
                zi[j,i] = "nan"
    
    dist = ax.contourf(xi, yi, zi, 60, cmap = plt.get_cmap('coolwarm'), zorder = 2)
    #ax.contour(xi, yi, zi, 15, linewidths = 0.5,colors = "grey", zorder = 2)
    
    if draw_cbar:
        cbar = fig.colorbar(dist, ax=ax, format='%.1e')
        cbar.ax.tick_params(labelsize=8)

    #ax.scatter(x, y, marker = 'o', c = 'b', s = 15, zorder = 3)
    
    circle = patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none", zorder=4)
    ax.add_patch(circle)

    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
    
    ax.set_xticks([])
    ax.set_yticks([])

    circle = patches.Ellipse(xy = [0,2], width = 0.4, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = patches.Ellipse(xy = [4,2], width = 0.4, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    
    xy = [[1.6,3.6], [2,4.3],[2.4,3.6]]
    polygon = patches.Polygon(xy = xy, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(polygon) 
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)

    return ax

#Regressão linear
def linear (x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
    return slope, intercept, r_value, p_value, std_err    







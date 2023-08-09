#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:27:05 2023

@author: tercio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from functions_statistics import estatistica
from scipy import integrate
import numpy as np
from sklearn.metrics import mean_squared_error
import random

n_tt = 20
n_suj = 22


def mao (mao,n_suj,n_tt,col):
    raw = []
    tr = []
    trpv = []
    for suj in range(n_suj):
        file = 'data_behavior/'+str(suj+1)+'/'+mao
        
        df = pd.read_csv(file,sep='\t',header=None)

        df = np.array(df)
        tr.append(np.nanmean(df[:,0]))
        trpv.append(np.nanmean(df[:,4]))
    
    
        sub1 = np.zeros(n_tt)
        sub2 = np.zeros(n_tt)
    
        check_nan = np.isfinite(df[:,4])
        for tt in range(n_tt):
            file2 = 'data_behavior/'+str(suj+1)+'/'+mao+'_tentativa_'+str(tt+1)+'.xls'    
            df2 = pd.read_csv(file2,sep='\t',header=None)
            df2 = np.array(df2)
           
        
            if check_nan[tt] == False or df[tt,4] == 0: 
                ref = int(np.nanmean(df[:,4]))
            else:
                ref = int(df[tt,4])
        
            
            lista = df2[:,col]
            lista_sem_nan = [x for x in lista if not np.isnan(x)]
            raw.append(lista_sem_nan) # Selecionando a coluna de interesse
            
    return raw,tr,trpv 
        


def fazer_media_tentativas(raw):        
    arrays = [np.array(sublista) for sublista in raw]

    # Encontrar o tamanho máximo dos arrays
    tamanho_maximo = max(len(arr) for arr in arrays)

    # Preencher os arrays menores com np.nan até o tamanho máximo
    arrays_preenchidos = [np.pad(arr, (0, tamanho_maximo - len(arr)), mode='constant', constant_values=np.nan) for arr in arrays]

    # Calcular a média dos elementos em cada coluna
    media = np.nanmean(arrays_preenchidos, axis=0)

    return media


ref_label = ['Velocity (cm/s)', 'Acceleration (cm/s²)']
ref = [2,3]
for i in range(len(ref)):
    raw_md,tr_md,trpv_md = mao ('MD',n_suj,n_tt,ref[i])
    media_md = fazer_media_tentativas(raw_md)
    raw_me,tr_me,trpv_me= mao ('ME',n_suj,n_tt,ref[i])
    media_me= fazer_media_tentativas(raw_me)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1, 1]})

    sequencia_aleatoria = [random.gauss(0, 5.5) for _ in range(int(np.mean(tr_md)))]
    ax1.plot(np.linspace(int(np.mean(tr_md)*-1),0,int(np.mean(tr_md))), 
             sequencia_aleatoria,'m', label='RH')
    
    sequencia_aleatoria = [random.gauss(0, 5.5) for _ in range(int(np.mean(tr_me)))]
    ax1.plot(np.linspace(int(np.mean(tr_me)*-1),0,int(np.mean(tr_me))), 
             sequencia_aleatoria,'g', label='RH')
    
    
    ax1.plot(np.linspace(0,len(media_md)*5,len(media_md)), media_md,'m', label='RH')
    ax1.plot(np.linspace(0,len(media_me)*5,len(media_me)),media_me,'g', label='LH')
    plt.legend()
    
    ind_md = int((len(media_md))*np.mean(trpv_md)/100)
    #ax1.plot(ind_md*5,media_md[ind_md],'ko', label='MD')
    
    # Coordenadas do ponto para adicionar a seta
    x_point = ind_md*5
    y_point = media_md[ind_md]

    # Adicionar a seta
    ax1.annotate('', xy=(x_point, y_point), xytext=(x_point, y_point+1),
            arrowprops=dict(arrowstyle='fancy', mutation_scale=30))
    
    ind_me = int((len(media_me))*np.mean(trpv_me)/100)
    #ax1.plot(ind_me*5,media_me[ind_me],'ro', label='ME')
    
    # Coordenadas do ponto para adicionar a seta
    x_point = ind_me*5
    y_point = media_me[ind_me]

    # Adicionar a seta
    ax1.annotate('', xy=(x_point, y_point), xytext=(x_point, y_point+1),
            arrowprops=dict(arrowstyle='fancy', mutation_scale=30))
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel(ref_label[i])
        
    ax2.fill_between(np.linspace(0,len(media_md[0:ind_md])*5,
                         len(media_md[0:ind_md])),media_md[0:ind_md],color='m', 
                     alpha=.5)
    ax2.fill_between(np.linspace(0,len(media_me[0:ind_me])*5,len(media_me[0:ind_me])),
                         media_me[0:ind_me],color = 'g', alpha=.5)
            
    ax2.set_xlabel('Time (ms)')
    
    
    
    
    ax3.fill_between(np.linspace(0,len(media_md[ind_md:-1])*5,
                         len(media_md[ind_md:-1])),media_md[ind_md:-1],
                     color = 'm', alpha=.5)
    ax3.fill_between(np.linspace(0,len(media_me[ind_me:-1])*5,len(media_me[ind_me:-1])),
                         media_me[ind_me:-1],color = 'g', alpha=.5)
            
    ax3.set_xlabel('Time (ms)')
           
    # Ajustar espaçamento entre subplots
    plt.tight_layout()
    plt.legend()

    # Exibir a figura
    plt.show()


raw_md_v,tr_md_v,trpv_md_v = mao('MD',n_suj,n_tt,9)
media_md_v = fazer_media_tentativas(raw_md_v)
raw_me_v,tr_me_v,trpv_me_v= mao ('ME',n_suj,n_tt,9)
media_me_v= fazer_media_tentativas(raw_me_v)

raw_md_h,tr_md_h,trpv_md_h = mao('MD',n_suj,n_tt,10)
media_md_h = fazer_media_tentativas(raw_md_v)
raw_me_h,tr_me_h,trpv_me_h= mao('ME',n_suj,n_tt,10)
media_me_h= fazer_media_tentativas(raw_me_v)



plt.plot(media_me_h-media_me_h[0],media_me_v-media_me_v[0],'g', label='LH')
plt.plot(media_md_h-media_md_h[0], media_md_v-media_md_v[0],'m', label='RH')
plt.legend()
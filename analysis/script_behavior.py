#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 07:19:31 2022

@author: tercio
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from functions_statistics import estatistica
from scipy import integrate

tr, mt,tresp,er,trpv,pv,nc,pf_graus = [],[],[],[],[],[],[],[]
pf_cm,er_1sub,posi_1sub_cm, posi_1sub_graus = [],[],[],[]
tempo_primeiro_sub  = []
integral_1sub, integral_2sub  = [],[]


n_tt = 20
n_suj = 22
for suj in range(n_suj):
    file = 'data_behavior/'+str(suj+1)+'/MD'
    
    
    df = pd.read_csv(file,sep='\t',header=None)

    df = np.array(df)
    
    
    sub1 = np.zeros(n_tt)
    sub2 = np.zeros(n_tt)
    
    check_nan = np.isfinite(df[:,4])
    for tt in range(n_tt):
        file2 = 'data_behavior/'+str(suj+1)+'/MD_tentativa_'+str(tt+1)+'.xls'    
        df2 = pd.read_csv(file2,sep='\t',header=None)
        df2 = np.array(df2)
           
        
        if check_nan[tt] == False or df[tt,4] == 0: 
            ref = int(np.nanmean(df[:,4]))
        else:
            ref = int(df[tt,4])
            
        raw = df2[:,3] # Selecionando a coluna de interesse
        values = abs(raw) #Transformando em abs        
        indices = np.isfinite(values) #Marcando os valores com NAN
        raw  = values[indices] #selecionando apenas os dados sem NAN
        
        
        
        index = int(len(raw)*(ref/100))    
    
             
        sub1[tt] = float(integrate.trapz([raw[0:index]]))
        
        sub2[tt] = float(integrate.trapz([raw[index:-1]]))
        

    tr.append(np.mean(df[:,0])), mt.append(np.mean(df[:,1]))
    tresp.append(np.mean(df[:,2])), er.append(np.mean(df[:,3]))
    trpv.append(np.mean(df[:,4])), pv.append(np.mean(df[:,5]))
    nc.append(np.mean(df[:,6])),pf_graus.append(np.mean(abs(df[:,7]-45)))
    pf_cm.append(np.mean(df[:,8])), er_1sub.append(np.mean(df[:,9]))
    posi_1sub_cm.append(np.mean(df[:,10])), posi_1sub_graus.append(np.mean(abs(df[:,11]-45)))
    tempo_primeiro_sub.append(np.mean(df[:,12])),integral_1sub.append(np.mean(sub1)), integral_2sub.append(np.mean(sub2))
    



tr, mt,tresp,er = np.array(tr),np.array(mt),np.array(tresp),np.array(er) 
trpv,pv,nc,pf_graus = np.array(trpv),np.array(pv),np.array(nc),np.array(pf_graus)
pf_cm,er_1sub,posi_1sub_cm, posi_1sub_graus = np.array(pf_cm),np.array(er_1sub),np.array(posi_1sub_cm), np.array(posi_1sub_graus)
integral_1sub, integral_2sub = np.array(integral_1sub), np.array(integral_2sub)
tempo_primeiro_sub = np.array(tempo_primeiro_sub)
tempo_segundo_sub = mt-tempo_primeiro_sub
nc_erro = nc/abs(er_1sub-er)
nc_tempo = tempo_segundo_sub/nc

data = {'MAO': np.zeros(n_suj)+1,
        'TR': tr,'MT': mt,'Tresp': tresp, 'ER': er,
        'TRPV': trpv,'PV': pv,'NC': nc, 'PF_graus': pf_graus, 
        'ER1SUB': er_1sub, 'P1sub_graus': posi_1sub_graus,'T1sub':tempo_primeiro_sub,
        'T2sub':tempo_segundo_sub, 'NC_erro': nc_erro,'NC_tempo': nc_tempo,
        'length': pf_cm, 'length_1sub': posi_1sub_cm,'integral_1sub': integral_1sub ,'integral_2sub': integral_2sub}



data = pd.DataFrame(data)

'''
corr_matrix = data.corr()


mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
with sn.axes_style("white"):
    f, ax = plt.subplots(figsize=(10, 10))
    ax = sn.heatmap(corr_matrix, mask=mask, annot=True, square=True, 
                     cmap = 'coolwarm',linewidths =0.3)
'''
data_fill = data

col = ['TR','MT','Tresp','ER','TRPV','PV','NC','PF_graus', 'ER1SUB','P1sub_graus','T1sub', 'T2sub','NC_erro','NC_tempo',
       'length', 'length', 'integral_1sub','integral_2sub']

for i in col:
   data_fill[i] = estatistica.outlier(data_fill[i])
    

data_MD = data_fill


###################### MÂO ESQUERDA #################


tr, mt,tresp,er,trpv,pv,nc,pf_graus = [],[],[],[],[],[],[],[]
pf_cm,er_1sub,posi_1sub_cm, posi_1sub_graus = [],[],[],[]
tempo_primeiro_sub  = []
integral_1sub = []
integral_2sub = []



for suj in range(n_suj):
    file = 'data_behavior/'+str(suj+1)+'/ME'

    df = pd.read_csv(file,sep='\t',header=None)

    df = np.array(df)
    
    sub1 = np.zeros(n_tt)
    sub2 = np.zeros(n_tt)
    
    check_nan = np.isfinite(df[:,4])
    for tt in range(n_tt):
        file2 = 'data_behavior/'+str(suj+1)+'/ME_tentativa_'+str(tt+1)+'.xls'    
        df2 = pd.read_csv(file2,sep='\t',header=None)
        df2 = np.array(df2)
           
        
        if check_nan[tt] == False or df[tt,4] == 0: 
            ref = int(np.nanmean(df[:,4]))
        else:
            ref = int(df[tt,4])
        
        raw = df2[:,3] # Selecionando a coluna de interesse
        values = abs(raw) #Transformando em abs        
        indices = np.isfinite(values) #Marcando os valores com NAN
        raw  = values[indices] #selecionando apenas os dados sem NAN
        
        
        
        index = int(len(raw)*(ref/100))
    
    
                
        sub1[tt] = float(integrate.trapz([raw[0:index]]))
        
        
        sub2[tt] = float(integrate.trapz([raw[index:-1]]))
        
        

    tr.append(np.mean(df[:,0])), mt.append(np.mean(df[:,1]))
    tresp.append(np.mean(df[:,2])), er.append(np.mean(df[:,3]))
    trpv.append(np.mean(df[:,4])), pv.append(np.mean(df[:,5]))
    nc.append(np.mean(df[:,6])),pf_graus.append(np.mean(abs(df[:,7]-45)))
    pf_cm.append(np.mean(df[:,8])), er_1sub.append(np.mean(df[:,9]))
    posi_1sub_cm.append(np.mean(df[:,10])), posi_1sub_graus.append(np.mean(abs(df[:,11]-45)))
    tempo_primeiro_sub.append(np.mean(df[:,12])),integral_1sub.append(np.mean(sub1)), integral_2sub.append(np.mean(sub2))



tr, mt,tresp,er = np.array(tr),np.array(mt),np.array(tresp),np.array(er) 
trpv,pv,nc,pf_graus = np.array(trpv),np.array(pv),np.array(nc),np.array(pf_graus)
pf_cm,er_1sub,posi_1sub_cm, posi_1sub_graus = np.array(pf_cm),np.array(er_1sub),np.array(posi_1sub_cm), np.array(posi_1sub_graus)
integral_1sub, integral_2sub = np.array(integral_1sub), np.array(integral_2sub)
tempo_primeiro_sub = np.array(tempo_primeiro_sub)
tempo_segundo_sub = mt-tempo_primeiro_sub
nc_erro = nc/abs(er_1sub-er)
nc_tempo = tempo_segundo_sub/nc

data = {'MAO': np.zeros(n_suj),
        'TR': tr,'MT': mt,'Tresp': tresp, 'ER': er,
        'TRPV': trpv,'PV': pv,'NC': nc, 'PF_graus': pf_graus, 
        'ER1SUB': er_1sub, 'P1sub_graus': posi_1sub_graus, 'T1sub':tempo_primeiro_sub,
        'T2sub':tempo_segundo_sub, 'NC_erro': nc_erro,'NC_tempo': nc_tempo,
        'length': pf_cm, 'length_1sub': posi_1sub_cm, 'integral_1sub': integral_1sub ,'integral_2sub': integral_2sub}


data = pd.DataFrame(data)

'''
corr_matrix = data.corr()


mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
with sn.axes_style("white"):
    f, ax = plt.subplots(figsize=(10, 10))
    ax = sn.heatmap(corr_matrix, mask=mask, annot=True, square=True, 
                     cmap = 'coolwarm',linewidths =0.3)


'''

data_fill = data

col = ['TR','MT','Tresp','ER','TRPV','PV','NC','PF_graus', 'ER1SUB','P1sub_graus','T1sub', 'T2sub','NC_erro','NC_tempo',
       'length', 'length', 'integral_1sub','integral_2sub']

for i in col:
   data_fill[i] = estatistica.outlier(data_fill[i])


data_ME = data_fill




data = pd.concat([data_MD, data_ME])

data['TRPV'] = data['TRPV'].fillna(data['TRPV'].median()) # para retirar o NAN
data['MT'] = data['MT'].fillna(data['MT'].median()) # para retirar o NAN



def grafico (medida):
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    axis = [ax1, ax2, ax3, ax4]


    for i in range(len(medida)):
        sn.despine(bottom=True, left=False)

        sn.stripplot(x="MAO",y=medida[i],
            hue="MAO", palette=["g", "m"],size=5,
            data=data, dodge=True, alpha=.5, zorder=1, legend=False, ax=axis[i])

        sn.pointplot(
            data=data, x="MAO", y=medida[i],
            join=False, dodge=.8 - .8 / 1, palette=["g", "m"],
            markers="d", scale=1, errorbar=None, ax=axis[i])





#medida = ['TR', 'MT', 'Tresp', 'ER']
#grafico (medida)

#medida = ['TRPV', 'PV', 'NC', 'PF_graus']
#grafico (medida)

#medida = ['ER1SUB', 'P1sub_graus','length', 'length_1sub']
#grafico (medida)

#medida = ['integral_1sub','integral_2sub']
#grafico (medida)

#medida = ['T1sub', 'T2sub','NC_erro', 'NC_tempo']
#grafico (medida)

#medida = ['ER1SUB', 'P1sub_graus','NC_erro', 'NC_tempo']
#grafico (medida)

#medida = ['integral_1sub','integral_2sub']
#grafico (medida)

medida = ['MT','integral_2sub']
grafico (medida)

#sn.relplot(data=data, x="MT", y="ER",  hue="MAO")
#sn.relplot(data=data, x="PV", y="ER",  hue="MAO")

#### analise de similaridade entre as matrizes de correção


data_salvar = pd.concat([data_MD, data_ME], axis=1)



data_salvar.to_csv('data_behavior.csv', sep=',', float_format=None, columns=None)



#############################3









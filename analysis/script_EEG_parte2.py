

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 13:33:11 2021

@author: tercio
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import seaborn as sn
from functions_statistics import estatistica
from statsmodels.stats.contingency_tables import mcnemar


n_tt = 20;


freq = 'alpha_lower'#theta (4–7 Hz), alpha_lower (8–10 Hz), alpha_upper (10–12 Hz), beta (13–30), gamma (36–44 Hz)


def extrair_medida(file):
    mao ='MD'
    df_MD = pickle.load(open(file+freq+"_"+mao+".pkl","rb"))
    df_MD = df_MD[0] 

    mao ='ME' 
    df_ME = pickle.load(open(file+freq+"_"+mao+".pkl","rb"))
    df_ME = df_ME[0] 

    df = pd.DataFrame()
    for i in range(36):
        # Criação do DataFrame vazio    
        nome_colunas = df_ME.columns
        b = nome_colunas[i][0]+'-'+nome_colunas[i][1]      
        col1_name, col2_name = ['ME_'+b,'MD_'+b]

        # Inserindo as duas novas colunas ao mesmo tempo
        df.insert(len(df.columns), col1_name, df_ME[nome_colunas[i]])
        df.insert(len(df.columns), col2_name, df_MD[nome_colunas[i]])

    return df

def outliers (data):
    data_fill = data
    nomes_colunas = sincro_maos.columns
    for i in nomes_colunas:
        data_fill[i] = estatistica.outlier(data[i])
    return data_fill



file = "data_eeg/sincronizacao/sincronizacao_"
sincro_maos = extrair_medida(file)
sincro_maos = outliers(sincro_maos)
sincro_maos.to_csv('dados_sincronizacao_'+freq+'.csv', index=False)


file = "data_eeg/sincronizacao/tempo_"
tempo_maos = extrair_medida(file)
tempo_maos = outliers(tempo_maos)
tempo_maos.to_csv('dados_tempo_'+freq+'.csv', index=False)

file = "data_eeg/sincronizacao/direcao_"
direcao_maos = extrair_medida(file)


for col in range (direcao_maos.shape[1]):
    for lin in range (direcao_maos.shape[0]):
        if direcao_maos.iloc[lin,col] == ['Positivo', 'Negativo'] or direcao_maos.iloc[lin,col] == ['Negativo', 'Positivo']:
            direcao_maos.iloc[lin,col] = np.nan

direcao_maos.to_csv('dados_direcao_'+freq+'.csv', index=False)

################# Graficos 


def gerar_comparacao(i, medias):
    if i % 2 == 0: # fazer par de comparacao
       c = medias[i],medias[i+1] #quando for par, o par e atual e a proxima
       v = max(medias[i],medias[i+1])             
    else:
       c = medias[i],medias[i-1] #quando for impar, o par e atual e a anterior
       v = max(medias[i],medias[i-1])
    
    if medias[i]/v == 1:
        b = medias[i]/v
    else:
        b = (medias[i]/v)*.7 #retirei 30% para evidenciar mais as diferenças
    
    
    return b



# Informações dos canais desejados (nome e posição 3D)
canais_desejados = {
    'F3': [-0.07, 0.08, 0.09],
    'Fz': [0.00, 0.08, 0.09],
    'F4': [0.07, 0.08, 0.09],
    
    'C3': [-0.07, 0.0, 0.09],
    'Cz': [0.00, 0.0, 0.09],
    'C4': [0.07, 0.0, 0.09],
    
    'P3': [-0.07, -0.08, 0.09],
    'POz': [0.00, -0.08, 0.09],
    'P4': [0.07, -0.08, 0.09],
 }

# Criar a montagem personalizada com os canais desejados
montage_personalizada = mne.channels.make_dig_montage(ch_pos=canais_desejados, coord_frame='head')

medias = sincro_maos.mean()
nomes_colunas = sincro_maos.columns
fig, axs = plt.subplots(9, 8, figsize=(12, 6), sharex=True, sharey=True)
cont = 1

for j in range(9): 
    for k in range(8):
        mne.channels.DigMontage.plot(montage_personalizada,show_names=False, axes=axs[j,k])
        #intensidade = ((medias[nomes_colunas[k]]*100)/(medias.max()))/100
        linha = 5
        intensidade = gerar_comparacao(k+(cont-1), medias)
        
        
        # Obter as coordenadas dos eletrodos C3 e C4 diretamente do dicionário canais_desejados
        coords1 = canais_desejados[nomes_colunas[k+(cont-1)][3:-3]][:2]*0.7 
        try:
            coords2 = canais_desejados[nomes_colunas[k+(cont-1)][7:]][:2]*0.7  # Pegando apenas as coordenadas x e y
        except:
            coords2 = canais_desejados[nomes_colunas[k+(cont-1)][6:]][:2]*0.7 
    
        # Traçar uma linha unindo C3 e C4
        if k % 2 == 0:        
            cor ='g'        
        else:
            cor ='m'
       
        axs[j,k].plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], lw=linha, alpha=intensidade, color=cor)
      
    # Use a função subplots_adjust() para ajustar o espaçamento manualmente
    plt.subplots_adjust(left=0.0, right=1, bottom=0.0, top=1, wspace=0.0, hspace=0.0)

    plt.show()
    #plt.tight_layout()
    cont +=8




# Criando a figura
fig = plt.figure(figsize=(20, 10))


# Lista de nomes das colunas
nomes_colunas = list(sincro_maos.keys())

cont = 0
for i in range(4):
    for j in range(9):
        ax = fig.add_subplot(4, 9, i * 9 + j + 1)  # Adicionando o subplot individualmente

        df = pd.DataFrame()
        df.insert(0, 'Left', sincro_maos[nomes_colunas[cont]])
        df.insert(1, 'Right', sincro_maos[nomes_colunas[cont + 1]])

        sn.stripplot(data=df, palette=["g", "m"], size=5,
                     dodge=True, alpha=.5, zorder=1, legend=False, ax=ax)
        sn.pointplot(data=df, join=False,
                     dodge=.8 - .8 / 1, palette=["g", "m"],
                     markers="d", scale=1, errorbar=None, ax=ax)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Definindo o limite do eixo y para cada subplot
        ax.set_ylim(df.min().min(), df.max().max())
        ax.set_title(nomes_colunas[cont][3:])

        cont += 2

plt.tight_layout()  # Ajusta automaticamente o layout para evitar sobreposição de títulos e eixos
plt.show()



##### 

def contar_valores(array, valor_procurado):
    contador = 0
    for valor in array:
        if valor == valor_procurado:
            contador += 1
    return contador


#### Primeiro dado da estistica precisa ser assim

esquerda = direcao_maos.iloc[:,0]
direita = direcao_maos.iloc[:,1]

esquerda_positivo = contar_valores(esquerda, ['Positivo'])/(len(esquerda)-esquerda.isna().sum()) # Contagem de ocorrências dos valores "Positivo" e "Negativo" antes da intervenção
esquerda_negativo = contar_valores(esquerda, ['Negativo'])/(len(esquerda)-esquerda.isna().sum()) # Contagem de ocorrências dos valores "Positivo" e "Negativo" antes da intervenção

direita_positivo = contar_valores(direita, ['Positivo'])/(len(direita)-direita.isna().sum())
direita_negativo = contar_valores(direita, ['Negativo'])/(len(direita)-direita.isna().sum())

tabela = np.array([[int(esquerda_positivo*100), int(direita_positivo*100)],   # Número de casos concordantes (antes e depois iguais)
              [int(esquerda_negativo*100), int(direita_negativo*100)]])  # Número de casos discordantes (antes e depois diferentes)
# Realizando o teste de McNemar
resultado_mcnemar = mcnemar(tabela)

ch1 = nomes_colunas[0][3:-3][:3]
try:
    ch2 = nomes_colunas[0][7:][:2]
except:
    ch2 = nomes_colunas[0][7:][:2]



resultados_df = pd.DataFrame()


resultados_df.insert(0, 'Direction', value= [ch1+'-'+ch2])
resultados_df.insert(1, 'Hands %', value= ['LH '+str(tabela[0,0])+'% | '+'RH '+str(tabela[0,1])+'%'])                                             
resultados_df.insert(2,'Statistic', ["{:.1f}".format(resultado_mcnemar.statistic)])
resultados_df.insert(3,'p-value', ["{:.3f}".format(resultado_mcnemar.pvalue)])

nova_linha = pd.Series({'Direction': ch2+'-'+ch1,'Hands %':'LH '+str(tabela[1,0])+'% | '+'RH '+str(tabela[1,1])+'%',
                        'Statistic':"" ,'p-value':"" })
resultados_df = resultados_df.append(nova_linha, ignore_index=True)
nova_linha = pd.Series({'Direction': "",'Hands %':"",'Statistic':"" ,'p-value':"" })# Linha em branco
resultados_df = resultados_df.append(nova_linha, ignore_index=True)





cont = 2

for i in range(35):
    
    esquerda = direcao_maos.iloc[:,cont]
    direita = direcao_maos.iloc[:,cont+1]

    esquerda_positivo = contar_valores(esquerda, ['Positivo'])/(len(esquerda)-esquerda.isna().sum()) # Contagem de ocorrências dos valores "Positivo" e "Negativo" antes da intervenção
    esquerda_negativo = contar_valores(esquerda, ['Negativo'])/(len(esquerda)-esquerda.isna().sum()) # Contagem de ocorrências dos valores "Positivo" e "Negativo" antes da intervenção

    direita_positivo = contar_valores(direita, ['Positivo'])/(len(direita)-direita.isna().sum())
    direita_negativo = contar_valores(direita, ['Negativo'])/(len(direita)-direita.isna().sum())

    tabela = np.array([[int(esquerda_positivo*100), int(direita_positivo*100)],   # Número de casos concordantes (antes e depois iguais)
                  [int(esquerda_negativo*100), int(direita_negativo*100)]])  # Número de casos discordantes (antes e depois diferentes)
    # Realizando o teste de McNemar
    resultado_mcnemar = mcnemar(tabela)

    ch1 = nomes_colunas[cont][3:-3][:3]
    try:
        ch2 = nomes_colunas[cont][6:][:4]
    except:
        ch2 = nomes_colunas[cont][6:][:4]


    nova_linha = pd.Series({'Direction': ch1+'-'+ch2,'Hands %':'LH '+str(tabela[0,0])+'% | '+'RH '+str(tabela[0,1])+'%',
                        'Statistic':"{:.1f}".format(resultado_mcnemar.statistic),'p-value':"{:.3f}".format(resultado_mcnemar.pvalue)})
    resultados_df = resultados_df.append(nova_linha, ignore_index=True)
    nova_linha = pd.Series({'Direction': ch2+'-'+ch1,'Hands %':'LH '+str(tabela[1,0])+'% | '+'RH '+str(tabela[1,1])+'%',
                        'Statistic':"",'p-value':"" })
    resultados_df = resultados_df.append(nova_linha, ignore_index=True)   
    
    nova_linha = pd.Series({'Direction': "",'Hands %':"",'Statistic':"" ,'p-value':"" })# Linha em branco
    resultados_df = resultados_df.append(nova_linha, ignore_index=True)
    
    cont += 2


resultados_df.to_csv('estatistica_direcao_'+freq+'.csv', index=False)

   
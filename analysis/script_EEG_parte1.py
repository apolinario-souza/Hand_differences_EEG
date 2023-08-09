
# Carregando pacotes
from funcoes_uteis import sep_horas, time_to_sec, convert_time_sec, criar_eventos
from funcoes_uteis import selecionar_intervalos, selecionar_tentativas
import numpy as np
import mne
from mne.time_frequency import psd_array_multitaper
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from uteis_eeg import motifs_transf, motif_appear
from uteis_eeg import degree_synchronization, synchronization_direction,tempo_com_maior_sincronizacao 
from collections import Counter
from uteis_eeg import filterband
import pickle
 
n_tt = 20;
sujeitos = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22];

lowcut, highcut = 36,44; #theta (4–7 Hz), alpha_lower (8–10 Hz), alpha_upper (10–12 Hz), beta (13–30), gamma (36–44 Hz)
freq = 'beta'#

#Colunas de referencia para cortar as tentativas
ref_col1 = 14
ref_col2 = 19

atrasos = np.arange(1,11,1) #sugestao 10 atrasos, comenando pelo 1

def fazer_moda(dados_nominais):
    contador = Counter(dados_nominais)
    moda = [item for item, contagem in contador.items() if contagem == contador.most_common(1)[0][1]]
    return moda

def princial_mao(mao):
    nivel_sincro_final =[]
    direcao_final = []
    tempo_final = []

    for suj in sujeitos:
        nome_arquivo = 'data_eeg/preprocess/'+str(suj)+mao+'.fif'
        raw = mne.io.read_raw_fif(nome_arquivo, preload=True)

  #Inserindo o sistema 10-20
        easycap_montage = mne.channels.make_standard_montage("easycap-M1")
        raw.set_montage(easycap_montage)

        #Filtrando os dados
    
        y = raw.copy()
        x = y.get_data()
        channels = y.ch_names 
        raw_fill = filterband (x, y.info['sfreq'],lowcut, highcut)
    
        #Abrindo os dados comportamentais
        nome_arquivo = 'data_behavior/'+str(suj)+'/'+mao
        dados_comp = pd.read_csv(nome_arquivo,sep='\t',header=None)

        #Pegando a hora que iniciou a gravação do eeg e transformando em segudos

        h, m,s = raw.info['meas_date'].hour, raw.info['meas_date'].minute, raw.info['meas_date'].second
        T0 = h * 3600 + m * 60 + s

        #Definindo os parêmtros para as combinações
        channels_num = np.arange(0,len(channels))

        combinations_size_2 =  list(itertools.combinations(channels, 2))

        combinations_size_2_num = list(itertools.combinations(channels_num, 2))
    
        #Gerando o array com inicio e fim dos eventos
        eventos = convert_time_sec (dados_comp,n_tt,T0,ref_col1,ref_col2)
    
    
        nivel_sincro_tt = []
        direcao_tt = []
        tempo_tt = [] 
        for comb in range(len(combinations_size_2_num)):
            nivel_sincro = []
            direcao = []
            tempo = []  
            for tt in range(n_tt):
                # Separar os dados por tentativas    
                w = selecionar_intervalos(np.array(raw_fill),eventos[tt],y.info['sfreq'])          
        
        
        
                         
        
                sinalX_raw = w[combinations_size_2_num[comb][0],:]
                sinalY_raw = w[combinations_size_2_num[comb][1],:]



                motifs_x = motifs_transf(sinalX_raw)
                motifs_y = motifs_transf(sinalY_raw)            

                Cxy,Cyx = motif_appear(atrasos, motifs_x,motifs_y)
                Qxy = degree_synchronization (Cxy,Cyx,sinalX_raw)
            
                nivel_sincro.append(Qxy)


                qxy = synchronization_direction(Cxy,Cyx)
                if qxy >= 0:
                    direcao.append('Positivo')
                else:
                    direcao.append('Negativo')


                tempo.append(tempo_com_maior_sincronizacao(Cxy,Cyx))
        
        
            nivel_sincro_tt.append(np.nanmean(nivel_sincro))
            direcao_tt.append(fazer_moda(direcao))
            tempo_tt.append(np.nanmean(tempo))
        
        nivel_sincro_final.append(nivel_sincro_tt)
        direcao_final.append(direcao_tt)
        tempo_final.append(tempo_tt)
        
    #inserindo tudo dentro de um array       
    nivel_sincro_final = np.array(nivel_sincro_final)
    tempo_final = np.array(tempo_final)
    direcao_final = np.array(direcao_final)

    # Para não ficar perdido eu coloquei um dataframe com o nome das combinacoes
    nivel_sincro_final = pd.DataFrame(nivel_sincro_final)
    nivel_sincro_final.columns = combinations_size_2


    tempo_final = pd.DataFrame(tempo_final)
    tempo_final.columns = combinations_size_2

    direcao_final = pd.DataFrame(direcao_final)
    direcao_final.columns = combinations_size_2

    return nivel_sincro_final, tempo_final, direcao_final



mao = 'MD'
nivel_sincro_final, tempo_final, direcao_final = princial_mao(mao)



pickle.dump([nivel_sincro_final], open("data_eeg/sincronizacao/sincronizacao_"+freq+"_"+mao+".pkl", "wb"))

pickle.dump([tempo_final], open("data_eeg/sincronizacao/tempo_"+freq+"_"+mao+".pkl", "wb"))

pickle.dump([direcao_final], open("data_eeg/sincronizacao/direcao_"+freq+"_"+mao+".pkl", "wb"))



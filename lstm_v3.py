#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import csv
# import torchvision as tv
from matplotlib import pyplot as plt
import random


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
    # return X, y
    
# definición del modelo
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100):
        super().__init__()
        self.output_size = 1
        # self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, self.output_size)
            
    def forward(self,x):
        # x, (h,c) = self.lstm(x)
        x, h = self.gru(x)
        return self.fc(h)


if __name__ == '__main__':
    
    
    datos =[]
    # with open('NNPOC.txt') as data:
    with open('NNPOC_v2.csv') as data:
        line_count=0
        for line in csv.reader(data):
            if line_count != 0 and line_count<300: #solamente el primer pozo
            # if line_count != 0:
                # datos.append(line[3:])
                datos.append(list(line[i] for i in [6,9,12,15,18,19])) #solamente agua+oil y oil tiempo anterior
                # datos.append(list(line[i] for i in [6,9,12,15,18])) #solamente agua+oil
            line_count += 1
            
    datosAorig = np.array(datos, dtype='f')
    # np.random.shuffle(datosA)
    mean = datosAorig.mean()
    std = datosAorig.std()
    datosA = (datosAorig-mean)/std
    
    # Prueba datos normales random
    # dataO = np.random.normal(0,1,(100,5))
    # datosA=dataO
        
    N = len(datosA)
    n = 4
    
    datosA_trn = datosA[:-N//n]
    datosA_tst = datosA[-N//n:]
    # datosA_trn = datosA[:-2]
    # datosA_tst = datosA[-2:]
    
    # con cuántas filas se predice la siguiente
    n_steps = 3
    
    
    # se separan los datos para aprendizaje supervisado.
    X_trn, y_trn = split_sequences(datosA_trn, n_steps)
    X_tst, y_tst = split_sequences(datosA_tst, n_steps)
    
    # entero_trn = list(zip(X_trn,y_trn))
    # random.shuffle(entero_trn)
    # X_trn, y_trn = zip(*entero_trn)
    
    
    batch_size_trn = X_trn.shape[0]
    seq_len_trn = X_trn.shape[1]
    batch_size_tst = X_tst.shape[0]
    seq_len_tst = X_tst.shape[1]
    input_size = X_trn.shape[2]
    
    # pasar a tensor de torch
    x_trn = torch.FloatTensor(X_trn).view(batch_size_trn,seq_len_trn,input_size)
    # transform_train = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize([4000], [1000])])
    # x_trn = transform_train(X_trn)
    # x_trn = torch.reshape(x_trn, (batch_size_trn,seq_len_trn,input_size))

    labels_trn = torch.FloatTensor(y_trn)

    x_tst = torch.FloatTensor(X_tst).view(batch_size_tst,seq_len_tst,input_size)
    # x_tst = transform_train(X_tst)
    # x_tst = torch.reshape(x_tst, (batch_size_tst,seq_len_tst,input_size))

    labels_tst = torch.FloatTensor(y_tst)
    
    # B_trn=1024 #tamaño del batch
    B_trn=batch_size_trn #tamaño del batch
    trn_data = TensorDataset(x_trn, labels_trn)
    trn_load = DataLoader(trn_data, shuffle=True, batch_size=B_trn)
    B_tst=1
    tst_data = TensorDataset(x_tst, labels_tst)
    tst_load = DataLoader(tst_data, shuffle=True, batch_size=B_tst)
    
    model = LSTM(input_size)
    # model = MV_LSTM(input_size,n_steps)
    costF = torch.nn.MSELoss() 
    optim = torch.optim.Adam(model.parameters())#, lr=1e-3)

    T = 500 #épocas de entrenamiento
    model.train()
    for t in range(T+1):
        for data, label in trn_load:
            # reinicializo el gradiente
            optim.zero_grad()
            
            # calculo predicción por modelo
            out = model(data)
            out=out.squeeze()

            # comparo contra target verdadero
            label = label.squeeze()
            error = costF(out, label)
            
            # gradiente por back prop
            error.backward()
            
            # paso optimización
            optim.step()
            # print(data)
        if t%50==0 or t==T:
            print(t)
            print(error.item())
            # print(out)
            # print(label)

    
    # predicción: mejoró bastante.
    model.eval()
    errores = []
    # predicciones = list(datosAorig[:241,-1])
    predicciones = []
    targets = []
    with torch.no_grad():
        for data, label in tst_load:
    #         print(data)
    # # #         # outx, outh = model(data)
    # # #         # if data.shape[0]==16:
    # #         model.init_hidden(data.size(0))
            out = model(data)
            print('TEST')
            print('modelo :',out.squeeze()*std+mean)
            print('ground truth:', label.squeeze()*std+mean)
            out = out.squeeze()*std+mean
            predicciones.append(out.item())
            label = label.squeeze()*std+mean
            targets.append(label.item())
            errorEste = abs(out.item()-label.item())/out.item()
            # print('Error: ', errorEste)
            errores.append(errorEste)
        print('Error promedio %: ', np.array(errores).mean())
        print('Error devío %: ', np.array(errores).std())
        # print('Error máximo %: ', np.array(errores).max())
        plt.hist(errores,bins=50)
        # plt.plot(datosAorig[:,-1][::-1])
        predicciones = np.array(predicciones)
        # plt.plot(predicciones[::-1])
        # plt.ylim(0, 15000)
        # plt.plot(np.concatenate((y_trn,y_tst))*std+mean)
        # plt.plot(np.concatenate((y_trn*std+mean,predicciones)))
        plt.plot(predicciones, label='predicción')
        plt.plot(targets, label='target')
        plt.legend()
        plt.show()
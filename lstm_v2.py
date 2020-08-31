#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import csv

# multivariate data preparation
from numpy import array
from numpy import hstack


# para separar datos para aprendizaje supervisado
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
    
# definición del modelo
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100):
        super().__init__()
        self.output_size = 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        # self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, self.output_size)
            
    def forward(self,x):
        x, (h,c) = self.lstm(x)
        return self.fc(h)


class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 100 # number of hidden states
        self.n_layers = 1 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*seq_length, 1)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out, (h, c) = self.l_lstm(x)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        # y = self.l_linear(h)
        return self.l_linear(x)
        # return y    


if __name__ == '__main__':
    
    # define input sequence
    in_seq1 = np.array([x for x in range(0,200,10)])
    in_seq2 = np.array([x for x in range(5,205,10)])
    out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    # horizontally stack columns
    dataset = hstack((in_seq1, in_seq2, out_seq))
  
    # matriz de datos artificiales. Cada fila es un tiempo.
    # dataOrig = np.linspace(start=(0,5,10),stop=(100,105,110), num=16,dtype=int)
    
    datos =[]
    with open('NNPOC.txt') as data:
    # with open('NNPOC_v2.csv') as data:
        line_count=0
        for line in csv.reader(data):
            if line_count != 0 and line_count<299:
                # datos.append(line[3:])
                datos.append(list(line[i] for i in [6,9,12,15,18]))
            line_count += 1
            
    # datosA = np.array(datos, dtype='f')
    # datosA=dataset
    # np.random.shuffle(datosA)
    
    dataO = np.random.normal(0,1,(100,4))
    datosA=dataO
        
    N = len(datosA)
    n=4
    
    datosA_trn = datosA[:-N//n]
    datosA_tst = datosA[-N//n:]
    # datosA_trn = datosA[:-4]
    # datosA_tst = datosA[-4:]
    
    # con cuántas filas se predice la siguiente
    n_steps = 3
    
    
    # se separan los datos para aprendizaje supervisado.
    # X_trn, y_trn = split_sequence(datosA_trn, n_steps)
    # X_tst, y_tst = split_sequence(datosA_tst, n_steps)
    X_trn, y_trn = split_sequences(datosA_trn, n_steps)
    X_tst, y_tst = split_sequences(datosA_tst, n_steps)
    
    batch_size_trn = X_trn.shape[0]
    seq_len_trn = X_trn.shape[1]
    batch_size_tst = X_tst.shape[0]
    seq_len_tst = X_tst.shape[1]
    input_size = X_trn.shape[2]
    
    # pasar a tensor de torch
    x_trn = torch.FloatTensor(X_trn).view(batch_size_trn,seq_len_trn,input_size)
    labels_trn = torch.FloatTensor(y_trn)
    x_tst = torch.FloatTensor(X_tst).view(batch_size_tst,seq_len_tst,input_size)
    labels_tst = torch.FloatTensor(y_tst)
    
    B_trn=64 #tamaño del batch
    trn_data = TensorDataset(x_trn, labels_trn)
    trn_load = DataLoader(trn_data, shuffle=True, batch_size=B_trn)
    B_tst=1
    tst_data = TensorDataset(x_tst, labels_tst)
    tst_load = DataLoader(tst_data, shuffle=True, batch_size=B_tst)
    
    model = LSTM(input_size)
    # model = MV_LSTM(input_size,n_steps)
    costF = torch.nn.MSELoss() 
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    T = 500 #épocas de entrenamiento
    model.train()
    for t in range(T+1):
        for data, label in trn_load:
            # reinicializo el gradiente
            optim.zero_grad()
            
            # calculo predicción por modelo
            # model.init_hidden(data.size(0))
            out = model(data)
            # outx, outh = model(data)
            # outh = outh.squeeze()
            out=out.squeeze()
            # comparo contra target verdadero
            label = label.squeeze()
            error = costF(out, label)
            
            # gradiente por back prop
            error.backward()
            
            # paso optimización
            optim.step()
            
        if t%100==0 or t==T:
            print(t)
            print(error.item())
            # print(out)
            # print(label)

    
    # predicción: por ahora da muy mal. Aprende solamente "una" cosa.
    model.eval()
    with torch.no_grad():
        for data, label in tst_load:
    #         print(data)
    # # #         # outx, outh = model(data)
    # # #         # if data.shape[0]==16:
    # #         model.init_hidden(data.size(0))
            out = model(data)
            print('TEST')
            print('modelo :',out.squeeze())
            print('ground truth:', label.squeeze())
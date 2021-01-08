'''
House price prediction using Regression from pytorch
'''

import numpy as np
import pandas as pd

df = pd.read_csv("Real estate.csv")
y = df['house price of unit area']
df.drop('No', inplace=True, axis=1)
df.drop('transaction date', inplace=True, axis=1)

print(df.head())

inputs = df.iloc[:,:-1].values
output = df.iloc[:,-1:].values

import torch
import math
inp = torch.from_numpy(inputs).float()
oup = torch.from_numpy(output).float()

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
tensor_dataset = TensorDataset(inp,oup)
batch_size = 5

train = DataLoader(tensor_dataset, batch_size, shuffle=True)
model = nn.Linear(5,1)

import torch.nn.functional as F
loss_fn = F.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=0.0000000001)

def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred,yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
        
        # Print the progress
        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

fit(5000, model, loss_fn, opt, train)

sample_input_tensor = torch.tensor([[10., 101., 12., 30., 122.]])

print("Sample input : -")
print(sample_input_tensor)
#sample data
print(model(sample_input_tensor))
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import math

import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from lstm_autoencoder_modeling import *

batch_size = 256
lr = 0.0001
use_cuda = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() & use_cuda) else "cpu")
print("Loading dataset ... ")
np_data_train = np.load("./pretrain/data/sequence_dataset_train.npy").astype(np.float32)
np_data_val = np.load("./pretrain/data/sequence_dataset_val.npy").astype(np.float32)
print("Data length: ", len(np_data_train), len(np_data_val))

tensor_data_train = torch.from_numpy(np_data_train)
tensor_data_val = torch.from_numpy(np_data_val)
train_dataset = TensorDataset(tensor_data_train, tensor_data_train)
valid_dataset = TensorDataset(tensor_data_val, tensor_data_val)

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)

# Define hyperparameters
input_size = 44  # Size of each input vector
hidden_size = 20  # Size of the LSTM hidden state
num_layers = 2  # Number of LSTM layers
max_seq_length = 16  # Maximum sequence length

# Create the Bidirectional LSTM Autoencoder model
autoencoder = LSTMAutoencoder(input_size, hidden_size, num_layers, max_seq_length).to(device)

# Train
train_log_loss = []
valid_log_loss = []
best_valid_loss = math.inf

# Loss function (MSE loss)
loss_function = nn.MSELoss().to(device)

# Optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

sequence_length = 16

# Training loop
num_epochs = 700

for epoch in range(num_epochs):
    train_loss = 0.0
    autoencoder.train()

    for data, _ in train_dataloader:
        inputs = data.view(-1, sequence_length, 44).to(device) 
            
        autoencoder.zero_grad()
        outputs = autoencoder(inputs)

        loss = loss_function(outputs, inputs)
        loss.backward()
        
        optimizer.step()
        
        train_loss+=loss

    train_log_loss.append(train_loss.item()/len(train_dataloader))
    print("Epoch ", epoch, " (train loss): ", train_loss.item()/len(train_dataloader))

    valid_loss = 0.0
    autoencoder.eval()

    for data, _ in valid_dataloader:
        inputs = data.view(-1, sequence_length, 44).to(device)
        outputs = autoencoder(inputs)
        loss = loss_function(outputs, inputs)
        valid_loss += loss.item() 

    valid_log_loss.append(valid_loss/len(valid_dataloader))
    print("Epoch ", epoch, " (valid loss): ", valid_loss/len(valid_dataloader))
    print()

    if best_valid_loss > valid_loss:
        best_valid_loss = valid_loss
        # Saving State Dict
        # torch.save(autoencoder.state_dict(), "pretrain/model/lstm_autoencoder_perm100_005drop.pt")
        torch.save(autoencoder.state_dict(), "pretrain/model/test.pt")

cpu_train_log_loss = np.array(torch.tensor(train_log_loss, device = 'cpu'))
cpu_valid_log_loss = np.array(torch.tensor(valid_log_loss, device = 'cpu'))
print("finished training lstm autoencoder 2 layers perm100 (0.05 dropout)")
plt.plot(cpu_train_log_loss)
plt.plot(cpu_valid_log_loss)
plt.yscale("log")
plt.legend(["Train loss", "Valid loss"])
# plt.savefig('pretrain/plots/lstm_autoencoder_perm100_2layer_005drop.png')
plt.savefig('pretrain/plots/test.png')

import torch
import torch.nn as nn

# Define the LSTM Autoencoder model
# Define the Bidirectional LSTM Autoencoder model with regularization in the encoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, max_seq_length):
        super(LSTMAutoencoder, self).__init__()
        # Maximum sequence length
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Encoder (Bidirectional LSTM)
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Regularization layer (e.g., Dropout)
        self.regularization = nn.Dropout(0.05)  # Adjust the dropout rate as needed
        
        # Decoder (Bidirectional LSTM)
        #self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True, bidirectional=False)
        self.decoder = nn.Linear(hidden_size*2*max_seq_length, input_size*max_seq_length)
        
    def flatten_parameters(self):
        self.encoder.flatten_parameters()

    def forward(self, x):
        # Encoding
        encoded, _ = self.encoder(x)
        
        # Regularization
        regulized = self.regularization(encoded.reshape(-1, self.hidden_size*2*self.max_seq_length))
        
        # Decoding
        #decoded, _ = self.decoder(encoded)
        decoded = self.decoder(regulized)
        
        return decoded.view(-1, self.max_seq_length, self.input_size)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
autoencoder = LSTMAutoencoder(input_size=44, hidden_size=20, num_layers=2, max_seq_length=16)
autoencoder.load_state_dict(torch.load('./models/lstm_autoencoder_perm100_005drop.pt', map_location = device))
autoencoder.to(device)
autoencoder.flatten_parameters()
autoencoder.eval()
encoder = autoencoder.encoder
encoder.eval()

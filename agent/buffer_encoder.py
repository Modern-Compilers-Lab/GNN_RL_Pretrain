import torch.nn as nn
import torch

class NN(nn.Module):
    def __init__(self) -> None:
        super(NN,self).__init__()
        self.encoder1 = nn.Linear(702,100)
        self.encoder2 = nn.Linear(100,100)
        self.encoder3 = nn.Linear(100,20)
        self.decoder1 = nn.Linear(20,100)
        self.decoder2 = nn.Linear(100,702)

    def encode(self, x): 
        y = nn.functional.selu(self.encoder1(x))
        y = nn.functional.selu(self.encoder2(y))
        y = nn.functional.selu(self.encoder3(y))
        return y

    def forward(self, x):
        y = nn.functional.selu(self.encoder1(x))
        y = nn.functional.selu(self.encoder2(y))
        y = nn.functional.selu(self.encoder3(y))
        y = nn.functional.selu(self.decoder1(y))
        y = self.decoder2(y)
        return y 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
encoder = NN()
encoder.load_state_dict(torch.load("./models/rw_buf_encoder.pt"), map_location=device)
encoder.to(device)
encoder = encoder.eval()
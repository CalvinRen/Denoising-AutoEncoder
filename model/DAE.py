import torch 
import torch.nn as nn
import torch.nn.functional as F


class DAE(nn.Module):
    def __init__(self, args):
        super(DAE, self).__init__()
        self.linear1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, args.output_dim)
        self.linear3 = nn.Linear(args.output_dim, args.hidden_dim)
        self.linear4 = nn.Linear(args.hidden_dim, args.input_dim)
    
    def encode(self, x):
        hidden = F.relu(self.linear1(x))
        z = F.relu(self.linear2(hidden))
        return z
    
    def decode(self, x):
        hidden = F.relu(self.linear3(x))
        output = torch.sigmoid(self.linear4(hidden))
        return output
    
    def forward(self, x):
        z = self.encode(x)
        output = self.decode(z)
        return output
import torch 
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.linear1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, args.output_dim)
        self.linear3 = nn.Linear(args.hidden_dim, args.output_dim)
        self.linear4 = nn.Linear(args.output_dim, args.hidden_dim)
        self.linear5 = nn.Linear(args.hidden_dim, args.input_dim)

    def encode(self, x):
        hidden = F.relu(self.linear1(x))
        return self.linear2(hidden), self.linear3(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        hidden = F.relu(self.linear4(z))
        output = torch.sigmoid(self.linear5(hidden))
        return output
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar
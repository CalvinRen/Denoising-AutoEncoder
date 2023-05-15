import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from model import DAE, DCAE, VAE


# Arguments Parser
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--input_dim', type=int, default=784, help='Input dimension')
parser.add_argument('--hidden_dim', type=int, default=400, help='Hidden dimension')
parser.add_argument('--output_dim', type=int, default=20, help='Laten Space dimension')
parser.add_argument('--ae_model_type', type=str, default='DAE', help='AutoEncoder network')
parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
parser.add_argument('--model_save_path', type=str, default='None', help='Model save path')
parser.add_argument('--device', type=str, default='0', help='Device to use')
args = parser.parse_args()


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.device


# Dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


# Model
if args.ae_model_type == 'DAE':
    model = DAE(args).to(device)
elif args.ae_model_type == 'DCAE':
    model = DCAE(args).to(device)
elif args.ae_model_type == 'VAE':
    model = VAE(args).to(device)
else:
    raise NotImplementedError


# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# Tensorboard
writer = SummaryWriter(args.log_dir)


# Train
for epoch in tqdm(range(args.num_epochs)):
    for i, (images, _) in enumerate(dataloader):
        if args.ae_model_type == 'DCAE':
            images = images.view(-1, 1, 28, 28).to(device)
        else:
            images = images.view(-1, 28*28).to(device)

        # add noise
        noisy_images = images + 0.2 * torch.randn(images.shape).to(device)
        noisy_images = torch.clamp(noisy_images, 0., 1.)

        # backward 
        if args.ae_model_type == 'VAE':
            outputs, mu, logvar = model(noisy_images)
            reconst_loss = F.binary_cross_entropy(outputs, images, size_average=False)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconst_loss + kl_divergence
            
        else:
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # output 
        if (i+1) % 100 == 0:
            # print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            writer.add_scalar('training loss', loss.item(), epoch * len(dataloader) + i)

# Save model
if args.model_save_path != 'None':
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    torch.save(model.state_dict(), os.path.join(args.model_save_path, f'{args.ae_model_type}.pth'))

# Close Tensorboard
writer.close()
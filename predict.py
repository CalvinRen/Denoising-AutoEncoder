import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model import DAE, DCAE, VAE


# Arguments Parser
parser = argparse.ArgumentParser()
parser.add_argument('--load_model_path', type=str, default='None', help='Model load path')
parser.add_argument('--input_dim', type=int, default=784, help='Input dimension')
parser.add_argument('--hidden_dim', type=int, default=400, help='Hidden dimension')
parser.add_argument('--output_dim', type=int, default=20, help='Laten Space dimension')
parser.add_argument('--ae_model_type', type=str, default='DAE', help='AutoEncoder network')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--device', type=str, default='0', help='Device to use')
parser.add_argument('--out_dir', type=str, default='out_dir', help='Output directory')

args = parser.parse_args()


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.device


# Dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


# Model
if args.ae_model_type == 'DAE':
    model = DAE(args).to(device)
elif args.ae_model_type == 'DCAE':
    model = DCAE(args).to(device)
elif args.ae_model_type == 'VAE':
    model = VAE(args).to(device)
else:
    raise NotImplementedError

model.load_state_dict(torch.load(args.load_model_path))


# Test
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

model.eval()
with torch.no_grad():
    images = next(iter(dataloader))[0].to(device)

    if args.ae_model_type == 'DCAE':
        input_images = images.to(device)
    else:
        input_images = images.view(-1, 784).to(device)

    noisy_images = input_images + 0.5 * torch.randn(input_images.size()).to(device)
    output = model(noisy_images)

    output_images = output.view(-1, 1, 28, 28)
    images = images.view(-1, 1, 28, 28)
    noisy_images = noisy_images.view(-1, 1, 28, 28)

    save_image(images, os.path.join(args.out_dir, 'original.png'), nrow=8)
    save_image(noisy_images, os.path.join(args.out_dir, 'noisy.png'), nrow=8)
    save_image(output_images, os.path.join(args.out_dir, 'reconstructed.png'), nrow=8)
    










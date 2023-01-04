import torch
import torch.nn as nn
import torch.optim as optim


from GAN_dataset import return_dataloader
from GAN_model import Discriminator, Generator


device = "cuda" if torch.cuda.is_available() else "cpu"

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optim_g = optim.Adam(generator.parameters(), lr=0.0002)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)

criterion = nn.BCELoss()

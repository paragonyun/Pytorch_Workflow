import torch
import torch.nn as nn
import torch.optim as optim


from GAN_dataset import return_dataloader
from GAN_model import Discriminator, Generator
from train import GANTrainer

train_loader = return_dataloader()

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optim_g = optim.Adam(generator.parameters(), lr=0.0002)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)

criterion = nn.BCELoss()

gan = GANTrainer(
    discriminator=discriminator,
    generator=generator,
    criterion=criterion,
    optim_g=optim_g,
    optim_d=optim_d,
    epochs=100,
    train_loader=train_loader,
)
gan.trainer()

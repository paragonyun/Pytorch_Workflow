import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from utils import save_generator_image

class GANTrainer:
    def __init__(self,     
                discriminator,
                generator,
                criterion,
                optim_g,
                optim_d,
                epochs,
                train_loader,
                device="cuda" if torch.cuda.is_available() else "cpu",
        ):
        self.discriminator = discriminator
        self.generator = generator
        self.criterion = criterion
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.device = device
        self.epochs = epochs
        self.train_loader = train_loader

    def train_discriminator(self, data_real, data_fake):
        b_size = data_real.size(0)  # Batch Size

        real_label = torch.ones(b_size, 1).to(self.device)  # 진짜는 1로 라벨 저장
        fake_label = torch.zeros(b_size, 1).to(self.device)  # 가짜는 0으로 라벨 저장

        self.optim_d.zero_grad()

        output_real = self.discriminator(data_real)  # 진짜 데이터에 대한 판단
        loss_real = self.criterion(output_real, real_label)  # loss 구하기

        output_fake = self.discriminator(data_fake)  # 가짜 데이터에 대한 판단
        loss_fake = self.criterion(output_fake, fake_label)  # Loss 구하기

        loss_real.backward()
        loss_fake.backward()

        self.optim_d.step()

        return loss_real + loss_fake


    def train_generator(self, data_fake):

        b_size = data_fake.size(0)
        real_label = torch.ones(b_size, 1).to(self.device)

        self.optim_g.zero_grad()

        output = self.discriminator(data_fake)

        loss = self.criterion(output, real_label)

        loss.backward()
        self.optim_g.step()

        return loss

    def trainer(self,):
        losses_g = []
        losses_d = []
        images = []

        k = 1 # 판별자를 실행시키는 횟수
        nz = 128

        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.epochs):
            loss_g = 0.0
            loss_d = 0.0

            for idx, data in tqdm(enumerate(self.train_loader)):
                img, _ = data # label은 우리가 train에서 정했음
                img = img.to(self.device)
                b_size = len(img)

                for step in range(k):
                    """
                    생성자에서 image 생성에 필요한 noise 데이터가 생성되는 곳
                    randn으로 생성한다.
                    randn은 평균이 0, 표준편차가 1인 noise를 생성해줌.
                    """
                    fake_data = self.generator(torch.randn(b_size, nz).to(self.device)).detach() # detach는 잘라내기(crtl+x)와 같은 기능으로 활용됨
                    data_real = img
                    loss_d += self.train_discriminator(data_fake=fake_data, data_real=data_real)

                data_fake = self.generator(torch.randn(b_size, nz).to(self.device))
                loss_g += self.train_generator(data_fake=data_fake)

            generated_img = self.generator(torch.randn(b_size, nz).to(self.device)).cpu().detach()
            generated_img = make_grid(generated_img)

            save_generator_image(generated_img, f"./generated_img_{epoch}.png")
            images.append(generated_img)

            epoch_loss_g = loss_g / idx
            epoch_loss_d = loss_d / idx

            losses_g.append(epoch_loss_g)
            losses_d.append(epoch_loss_d)

            print(f"EPOCH {epoch}/{self.epochs}")
            print(f"Generator Loss: {epoch_loss_g:.4f}\tDiscriminator Loss: {epoch_loss_d:.4f}")
import torch

losses_g = []
logges_d = []
images = []


def train_discriminator(
    discriminator,
    criterion,
    optimizer,
    data_real,
    data_fake,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    b_size = data_real.size(0)  # Batch Size

    real_label = torch.ones(b_size, 1).to(device)  # 진짜는 1로 라벨 저장
    fake_label = torch.zeros(b_size, 0).to(device)  # 가짜는 0으로 라벨 저장

    optimizer.zero_grad()

    output_real = discriminator(data_real)  # 진짜 데이터에 대한 판단
    loss_real = criterion(output_real, real_label)  # loss 구하기

    output_fake = discriminator(data_fake)  # 가짜 데이터에 대한 판단
    loss_fake = criterion(output_fake, fake_label)  # Loss 구하기

    loss_real.backward()
    loss_fake.backward()

    optimizer.step()

    return loss_real + loss_fake


def train_generator(
    discriminator,
    criterion,
    optimizer,
    data_fake,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    b_size = data_fake.size(0)
    real_label = torch.ones(b_size, 1).to(device)

    optimizer.zero_grad()

    output = discriminator(data_fake)

    loss = criterion(output, real_label)

    loss.backward()
    optimizer.step()

    return loss
